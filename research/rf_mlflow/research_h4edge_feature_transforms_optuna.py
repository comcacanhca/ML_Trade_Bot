from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from config import CFG, ensure_dirs
from plots import plot_model_diagnostics, plot_threshold_curve
from research_h4_time_edge_mlop import FEATURE_FAMILY as BASE_FEATURE_FAMILY, GATE_COLS, _load_edge_frames, _load_scaled_split, _quantiles
from research_random50_initial_features import _mlflow, _plot_importance, _plot_shap, _threshold_table, _total_at_threshold, build_cache
from tune_h4edge_lgbm_01_optuna import _select_threshold_guarded


FEATURE_FAMILY = "h4edge_feature_transforms_optuna"
SOURCE_RUN_ID = "1788886130"
SOURCE_MODEL_NAME = "h4edge_lgbm_01"
SOURCE_SPEC_PATH = CFG.outputs_dir / f"{BASE_FEATURE_FAMILY}_{SOURCE_RUN_ID}" / "model_specs.json"
GATE = "bb_h4_weak_london"
THRESHOLDS = tuple(np.round(np.arange(0.50, 0.701, 0.01), 2))
PROTECTED = {
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "session_london_ny",
    "session_ny",
    "session_asia",
    "sig_bb_reversion_buy",
    "sig_pullback_trend_buy",
    "sig_momentum_buy",
    "buy_signal",
}


def _load_source_features() -> list[str]:
    specs = json.loads(SOURCE_SPEC_PATH.read_text(encoding="utf-8"))
    for spec in specs:
        if spec.get("model_name") == SOURCE_MODEL_NAME:
            return [c for c in spec["features"] if c != "buy_signal"]
    raise KeyError(f"Cannot find {SOURCE_MODEL_NAME} in {SOURCE_SPEC_PATH}")


def _metric_row(split: str, y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "split": split,
        "samples": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(np.unique(y)) == 2 else float("nan"),
    }


def _as_float_frame(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame[cols].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rank_features_by_train_signal(x: pd.DataFrame, y: np.ndarray, max_mi_rows: int, seed: int) -> pd.DataFrame:
    numeric = [c for c in x.columns if c not in PROTECTED]
    rows = []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    if len(idx) > max_mi_rows:
        idx = rng.choice(idx, size=max_mi_rows, replace=False)
    for col in numeric:
        s = x[col].to_numpy("float32", copy=False)
        corr = np.corrcoef(np.nan_to_num(s, nan=0.0), y)[0, 1] if np.nanstd(s) > 1e-9 else 0.0
        rows.append({"feature": col, "abs_label_corr": float(abs(corr)) if np.isfinite(corr) else 0.0})
    rank = pd.DataFrame(rows)
    if numeric:
        mi = mutual_info_classif(x.loc[idx, numeric].to_numpy("float32"), y[idx], discrete_features=False, random_state=seed)
        mi_map = dict(zip(numeric, mi))
        rank["mutual_info"] = rank["feature"].map(mi_map).fillna(0.0)
    else:
        rank["mutual_info"] = 0.0
    rank["rank_score"] = rank["abs_label_corr"].rank(pct=True) + rank["mutual_info"].rank(pct=True)
    return rank.sort_values("rank_score", ascending=False).reset_index(drop=True)


def _safe_name(text: str) -> str:
    return text.replace("/", "_div_").replace("*", "_x_").replace("+", "_plus_").replace("-", "_minus_")


def _add_univariate(x: pd.DataFrame, cols: list[str], methods: tuple[str, ...]) -> pd.DataFrame:
    out = x.copy()
    for col in cols:
        v = x[col].to_numpy("float32", copy=False)
        if "abs" in methods:
            out[f"{col}__abs"] = np.abs(v).astype("float32")
        if "square" in methods:
            out[f"{col}__square"] = np.clip(v * v, 0, 25).astype("float32")
        if "signed_log1p" in methods:
            out[f"{col}__signed_log1p"] = (np.sign(v) * np.log1p(np.abs(v))).astype("float32")
        if "tanh" in methods:
            out[f"{col}__tanh"] = np.tanh(v).astype("float32")
        if "hinge" in methods:
            out[f"{col}__pos"] = np.maximum(v, 0).astype("float32")
            out[f"{col}__neg"] = np.maximum(-v, 0).astype("float32")
    return out


def _add_pairwise_products(x: pd.DataFrame, cols: list[str], max_pairs: int) -> pd.DataFrame:
    out = x.copy()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pairs.append((a, b))
    for a, b in pairs[:max_pairs]:
        out[f"{a}__x__{b}"] = np.clip(x[a].to_numpy("float32") * x[b].to_numpy("float32"), -25, 25).astype("float32")
    return out


def _add_diffs_ratios(x: pd.DataFrame, cols: list[str], max_pairs: int) -> pd.DataFrame:
    out = x.copy()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pairs.append((a, b))
    for a, b in pairs[:max_pairs]:
        av = x[a].to_numpy("float32")
        bv = x[b].to_numpy("float32")
        out[f"{a}__minus__{b}"] = np.clip(av - bv, -10, 10).astype("float32")
        out[f"{a}__ratio__{b}"] = np.clip(av / (np.abs(bv) + 1.0), -10, 10).astype("float32")
    return out


def _add_time_signal_interactions(x: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    out = x.copy()
    anchors = [c for c in ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "session_ny", "sig_bb_reversion_buy", "sig_momentum_buy"] if c in x.columns]
    for a in anchors:
        av = x[a].to_numpy("float32")
        for col in numeric_cols:
            out[f"{col}__x__{a}"] = np.clip(x[col].to_numpy("float32") * av, -10, 10).astype("float32")
    return out


def _transform_frames(
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    test_x: pd.DataFrame,
    variant: str,
    rank: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    top_cols = [c for c in rank["feature"].head(top_n).tolist() if c in train_x.columns]
    if variant == "base":
        return train_x, valid_x, test_x, {"variant": variant, "top_cols": []}
    if variant == "abs_square":
        methods = ("abs", "square")
        return (
            _add_univariate(train_x, top_cols, methods),
            _add_univariate(valid_x, top_cols, methods),
            _add_univariate(test_x, top_cols, methods),
            {"variant": variant, "top_cols": top_cols, "methods": methods},
        )
    if variant == "signed_log_tanh":
        methods = ("signed_log1p", "tanh")
        return (
            _add_univariate(train_x, top_cols, methods),
            _add_univariate(valid_x, top_cols, methods),
            _add_univariate(test_x, top_cols, methods),
            {"variant": variant, "top_cols": top_cols, "methods": methods},
        )
    if variant == "hinge":
        methods = ("hinge",)
        return (
            _add_univariate(train_x, top_cols, methods),
            _add_univariate(valid_x, top_cols, methods),
            _add_univariate(test_x, top_cols, methods),
            {"variant": variant, "top_cols": top_cols, "methods": methods},
        )
    if variant == "products":
        return (
            _add_pairwise_products(train_x, top_cols, max_pairs=80),
            _add_pairwise_products(valid_x, top_cols, max_pairs=80),
            _add_pairwise_products(test_x, top_cols, max_pairs=80),
            {"variant": variant, "top_cols": top_cols, "max_pairs": 80},
        )
    if variant == "diff_ratio":
        return (
            _add_diffs_ratios(train_x, top_cols, max_pairs=60),
            _add_diffs_ratios(valid_x, top_cols, max_pairs=60),
            _add_diffs_ratios(test_x, top_cols, max_pairs=60),
            {"variant": variant, "top_cols": top_cols, "max_pairs": 60},
        )
    if variant == "time_signal_interactions":
        return (
            _add_time_signal_interactions(train_x, top_cols),
            _add_time_signal_interactions(valid_x, top_cols),
            _add_time_signal_interactions(test_x, top_cols),
            {"variant": variant, "top_cols": top_cols},
        )
    if variant == "mixed_compact":
        methods = ("abs", "square", "signed_log1p", "tanh")
        tr = _add_univariate(train_x, top_cols, methods)
        va = _add_univariate(valid_x, top_cols, methods)
        te = _add_univariate(test_x, top_cols, methods)
        pair_cols = top_cols[: min(8, len(top_cols))]
        tr = _add_pairwise_products(tr, pair_cols, max_pairs=28)
        va = _add_pairwise_products(va, pair_cols, max_pairs=28)
        te = _add_pairwise_products(te, pair_cols, max_pairs=28)
        return tr, va, te, {"variant": variant, "top_cols": top_cols, "pair_cols": pair_cols, "methods": methods}
    raise ValueError(f"Unknown variant: {variant}")


def _suggest_params(trial: optuna.Trial) -> dict:
    max_depth = trial.suggest_int("max_depth", 2, 5)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 180, 720, step=60),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 3, min(31, 2**max_depth - 1)),
        "max_depth": max_depth,
        "min_child_samples": trial.suggest_int("min_child_samples", 160, 880, step=40),
        "subsample": trial.suggest_float("subsample", 0.65, 0.95, step=0.05),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.95, step=0.05),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 0.8, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 5.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.08, step=0.01),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _score_valid(valid_frame: pd.DataFrame, p_valid: np.ndarray, y_valid: np.ndarray, min_resolved: int, target_resolved: int, max_resolved: int) -> tuple[float, float, dict, float]:
    auc = float(roc_auc_score(y_valid, p_valid)) if len(np.unique(y_valid)) == 2 else float("nan")
    curve = _threshold_table(valid_frame[["label", "year", "session_london_ny"]], p_valid, THRESHOLDS)
    threshold = _select_threshold_guarded(curve, min_resolved=min_resolved, target_resolved=target_resolved, max_resolved=max_resolved)
    total = _total_at_threshold(curve, threshold)
    resolved = float(total["resolved"])
    if resolved < min_resolved:
        return float(-100.0 + resolved / max(1.0, min_resolved)), threshold, total, auc
    deal_score = min(1.0, resolved / float(target_resolved))
    too_many_penalty = max(0.0, resolved - float(max_resolved)) / float(max_resolved)
    score = float(total["winrate"]) + 2.0 * auc + 2.0 * deal_score - too_many_penalty
    return float(score), float(threshold), total, auc


def _fit_eval_variant(
    variant: str,
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_meta: pd.DataFrame,
    valid_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    rank: pd.DataFrame,
    top_n: int,
    n_trials: int,
    seed: int,
    min_valid_resolved: int,
    target_valid_resolved: int,
    max_valid_resolved: int,
    out_dir: Path,
) -> tuple[dict, lgb.LGBMClassifier, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    print({"phase": "variant_start", "variant": variant}, flush=True)
    tr_x, va_x, te_x, transform_meta = _transform_frames(train_x, valid_x, test_x, variant, rank, top_n)
    cols = list(tr_x.columns)
    trial_rows = []

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            random_state=seed + trial.number,
            n_jobs=1,
            verbose=-1,
            deterministic=True,
            force_col_wise=True,
            **params,
        )
        model.fit(
            tr_x.to_numpy("float32", copy=False),
            y_train,
            eval_set=[(va_x.to_numpy("float32", copy=False), y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        p_valid = model.predict_proba(va_x.to_numpy("float32", copy=False))[:, 1]
        score, threshold, total, auc = _score_valid(valid_meta, p_valid, y_valid, min_valid_resolved, target_valid_resolved, max_valid_resolved)
        row = {"variant": variant, "trial": trial.number, "objective": score, "valid_auc": auc, "valid_wr": total["winrate"], "valid_resolved": total["resolved"], "threshold": threshold, **params}
        trial_rows.append(row)
        print({"phase": "trial", "variant": variant, "trial": trial.number, "score": round(score, 5), "auc": round(auc, 5), "wr": total["winrate"], "resolved": total["resolved"], "threshold": threshold}, flush=True)
        del model, p_valid
        gc.collect()
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True), study_name=f"{FEATURE_FAMILY}_{variant}")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    best_params = dict(study.best_params)
    best_params["subsample_freq"] = 1
    model = lgb.LGBMClassifier(objective="binary", boosting_type="gbdt", random_state=seed + int(study.best_trial.number), n_jobs=1, verbose=-1, deterministic=True, force_col_wise=True, **best_params)
    model.fit(
        tr_x.to_numpy("float32", copy=False),
        y_train,
        eval_set=[(va_x.to_numpy("float32", copy=False), y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    p_valid = model.predict_proba(va_x.to_numpy("float32", copy=False))[:, 1]
    p_test = model.predict_proba(te_x.to_numpy("float32", copy=False))[:, 1]
    valid_metrics = _metric_row("valid", y_valid, p_valid)
    test_metrics = _metric_row("test", y_test, p_test)
    valid_curve = _threshold_table(valid_meta[["label", "year", "session_london_ny"]], p_valid, THRESHOLDS)
    test_curve = _threshold_table(test_meta[["label", "year", "session_london_ny"]], p_test, THRESHOLDS)
    score, threshold, valid_total, _ = _score_valid(valid_meta, p_valid, y_valid, min_valid_resolved, target_valid_resolved, max_valid_resolved)
    test_total = _total_at_threshold(test_curve, threshold)
    imp = pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    summary = {
        "variant": variant,
        "n_features": len(cols),
        "n_trials": n_trials,
        "best_trial": int(study.best_trial.number),
        "objective_valid": float(score),
        "selected_threshold": float(threshold),
        "valid_auc": valid_metrics["auc"],
        "valid_wr": valid_total["winrate"],
        "valid_resolved": valid_total["resolved"],
        "test_auc": test_metrics["auc"],
        "test_wr": test_total["winrate"],
        "test_resolved": test_total["resolved"],
        "test_min_year_deals": test_total["min_year_deals"],
        "best_params": best_params,
        "top10_features": "|".join(imp.head(10)["feature"].tolist()),
        "transform_meta": transform_meta,
    }
    var_dir = out_dir / variant
    var_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_rows).sort_values("objective", ascending=False).to_csv(var_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    valid_curve.to_csv(var_dir / "threshold_valid.csv", index=False, encoding="utf-8-sig")
    test_curve.to_csv(var_dir / "threshold_test.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(var_dir / "lgbm_importance.csv", index=False, encoding="utf-8-sig")
    Path(var_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"phase": "variant_done", "variant": variant, "valid_auc": round(summary["valid_auc"], 5), "valid_wr": summary["valid_wr"], "test_auc": round(summary["test_auc"], 5), "test_wr": summary["test_wr"], "test_resolved": summary["test_resolved"]}, flush=True)
    return summary, model, imp, valid_curve, test_curve, cols


def run(
    variants: list[str],
    n_trials: int,
    max_train_rows: int,
    top_n: int,
    max_mi_rows: int,
    seed: int,
    shap_rows: int,
    enable_mlflow: bool,
    min_valid_resolved: int,
    target_valid_resolved: int,
    max_valid_resolved: int,
) -> dict:
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"{FEATURE_FAMILY}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    build_cache(force=False)
    base_cols = _load_source_features()
    train_edge, _, _ = _load_edge_frames(base_cols + list(GATE_COLS))
    gate_quantiles = _quantiles(train_edge, [c for c in GATE_COLS if c not in {"label", "year"}])
    del train_edge
    gc.collect()

    print({"phase": "load_scaled_data", "features": len(base_cols), "gate": GATE}, flush=True)
    train = _load_scaled_split(CFG.split.train_years, base_cols, GATE_COLS, GATE, gate_quantiles)
    if len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)
    valid = _load_scaled_split(CFG.split.valid_years, base_cols, GATE_COLS, GATE, gate_quantiles)
    test = _load_scaled_split(CFG.split.test_years, base_cols, GATE_COLS, GATE, gate_quantiles)
    y_train = train["label"].to_numpy("int8", copy=True)
    y_valid = valid["label"].to_numpy("int8", copy=True)
    y_test = test["label"].to_numpy("int8", copy=True)
    train_x = _as_float_frame(train, base_cols)
    valid_x = _as_float_frame(valid, base_cols)
    test_x = _as_float_frame(test, base_cols)
    train_meta = train[["label", "year", "session_london_ny"]].copy()
    valid_meta = valid[["label", "year", "session_london_ny"]].copy()
    test_meta = test[["label", "year", "session_london_ny"]].copy()
    print({"phase": "data_ready", "train": len(train), "valid": len(valid), "test": len(test), "base_features": len(base_cols)}, flush=True)
    del train, valid, test
    gc.collect()

    rank = _rank_features_by_train_signal(train_x, y_train, max_mi_rows=max_mi_rows, seed=seed)
    rank.to_csv(out_dir / "base_feature_rank.csv", index=False, encoding="utf-8-sig")
    all_summaries = []
    best_payload = None
    for i, variant in enumerate(variants):
        summary, model, imp, valid_curve, test_curve, cols = _fit_eval_variant(
            variant,
            train_x,
            valid_x,
            test_x,
            train_meta,
            valid_meta,
            test_meta,
            y_train,
            y_valid,
            y_test,
            rank,
            top_n,
            n_trials,
            seed + 1000 * i,
            min_valid_resolved,
            target_valid_resolved,
            max_valid_resolved,
            out_dir,
        )
        all_summaries.append(summary)
        if best_payload is None or summary["objective_valid"] > best_payload["summary"]["objective_valid"]:
            best_payload = {"summary": summary, "model": model, "imp": imp, "valid_curve": valid_curve, "test_curve": test_curve, "cols": cols}
        else:
            del model
        gc.collect()
        pd.DataFrame(all_summaries).sort_values("objective_valid", ascending=False).to_csv(out_dir / "summary_checkpoint.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(all_summaries).sort_values("objective_valid", ascending=False)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    best = best_payload["summary"]
    best_variant = best["variant"]
    best_dir = out_dir / best_variant
    chart_paths = []
    yv = y_valid
    yt = y_test
    # Rebuild best transformed matrices for diagnostics/SHAP only.
    best_train_x, best_valid_x, best_test_x, _ = _transform_frames(train_x, valid_x, test_x, best_variant, rank, top_n)
    p_valid = best_payload["model"].predict_proba(best_valid_x.to_numpy("float32", copy=False))[:, 1]
    p_test = best_payload["model"].predict_proba(best_test_x.to_numpy("float32", copy=False))[:, 1]
    chart_paths += plot_model_diagnostics(yv, p_valid, best_dir, "best_valid")
    chart_paths += plot_model_diagnostics(yt, p_test, best_dir, "best_test")
    chart_paths.append(plot_threshold_curve(best_payload["valid_curve"], best_dir / "best_threshold_valid.png", f"{best_variant} valid threshold"))
    chart_paths.append(plot_threshold_curve(best_payload["test_curve"], best_dir / "best_threshold_test.png", f"{best_variant} test threshold"))
    chart_paths.append(_plot_importance(best_payload["imp"], best_dir / "best_lgbm_importance_top25.png", f"{best_variant} LightGBM importance"))
    if shap_rows > 0:
        chart_paths += _plot_shap(best_payload["model"], best_valid_x.to_numpy("float32", copy=False), best_payload["cols"], best_dir, "best", shap_rows, seed)

    bundle_path = CFG.model_dir / f"{FEATURE_FAMILY}_{run_id}_{best_variant}_bundle.joblib"
    joblib.dump(
        {
            "model": best_payload["model"],
            "model_type": "LightGBM",
            "feature_columns": best_payload["cols"],
            "base_feature_columns": base_cols,
            "feature_family": FEATURE_FAMILY,
            "source": f"{SOURCE_MODEL_NAME}_{SOURCE_RUN_ID}",
            "gate": GATE,
            "selected_threshold": best["selected_threshold"],
            "params": best["best_params"],
            "transform_meta": best["transform_meta"],
            "gate_quantiles": gate_quantiles,
        },
        bundle_path,
        compress=3,
    )
    final = {
        "run_id": run_id,
        "feature_family": FEATURE_FAMILY,
        "source": f"{SOURCE_MODEL_NAME}_{SOURCE_RUN_ID}",
        "gate": GATE,
        "variants": variants,
        "n_trials_per_variant": n_trials,
        "max_train_rows": max_train_rows,
        "actual_train_rows": int(len(y_train)),
        "best_variant": best_variant,
        "best": best,
        "bundle_path": str(bundle_path),
        "out_dir": str(out_dir),
        "elapsed_sec": round(time.time() - started, 2),
    }
    summary_json_path = out_dir / "summary.json"
    summary_json_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow = _mlflow() if enable_mlflow else None
    if mlflow:
        with mlflow.start_run(run_name=f"{FEATURE_FAMILY}_{run_id}"):
            mlflow.log_params(
                {
                    "feature_family": FEATURE_FAMILY,
                    "source_model": SOURCE_MODEL_NAME,
                    "source_run_id": SOURCE_RUN_ID,
                    "gate": GATE,
                    "variants": ",".join(variants),
                    "n_trials_per_variant": n_trials,
                    "max_train_rows": max_train_rows,
                    "actual_train_rows": int(len(y_train)),
                    "best_variant": best_variant,
                    "best_n_features": best["n_features"],
                    "threshold": best["selected_threshold"],
                }
            )
            for key in ["objective_valid", "valid_auc", "valid_wr", "valid_resolved", "test_auc", "test_wr", "test_resolved", "test_min_year_deals"]:
                value = best.get(key)
                if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                    mlflow.log_metric(key, value)
            for path in [summary_path, summary_json_path, out_dir / "summary_checkpoint.csv", out_dir / "base_feature_rank.csv", bundle_path, *chart_paths]:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
            for variant in variants:
                var_dir = out_dir / variant
                for path in var_dir.glob("*"):
                    if path.is_file():
                        mlflow.log_artifact(str(path), artifact_path=f"variants/{variant}")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="base,abs_square,signed_log_tanh,hinge,products,diff_ratio,time_signal_interactions,mixed_compact")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--max-train-rows", type=int, default=1000000)
    parser.add_argument("--top-n", type=int, default=14)
    parser.add_argument("--max-mi-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=91403)
    parser.add_argument("--shap-rows", type=int, default=120)
    parser.add_argument("--min-valid-resolved", type=int, default=3000)
    parser.add_argument("--target-valid-resolved", type=int, default=4200)
    parser.add_argument("--max-valid-resolved", type=int, default=9000)
    parser.add_argument("--enable-mlflow", action="store_true")
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    run(
        variants=variants,
        n_trials=args.n_trials,
        max_train_rows=args.max_train_rows,
        top_n=args.top_n,
        max_mi_rows=args.max_mi_rows,
        seed=args.seed,
        shap_rows=args.shap_rows,
        enable_mlflow=args.enable_mlflow,
        min_valid_resolved=args.min_valid_resolved,
        target_valid_resolved=args.target_valid_resolved,
        max_valid_resolved=args.max_valid_resolved,
    )


if __name__ == "__main__":
    main()
