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
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from config import CFG, ensure_dirs
from plots import plot_model_diagnostics, plot_threshold_curve
from research_h4_time_edge_mlop import (
    FEATURE_FAMILY as BASE_FEATURE_FAMILY,
    GATE_COLS,
    _gate_mask,
    _load_edge_frames,
    _load_scaled_split,
    _quantiles,
)
from research_random50_initial_features import _mlflow, _plot_importance, _plot_shap, _select_threshold, _threshold_table, _total_at_threshold, build_cache


FEATURE_FAMILY = "h4edge_lgbm_01_optuna"
SOURCE_RUN_ID = "1788886130"
SOURCE_MODEL_NAME = "h4edge_lgbm_01"
SOURCE_SPEC_PATH = CFG.outputs_dir / f"{BASE_FEATURE_FAMILY}_{SOURCE_RUN_ID}" / "model_specs.json"
DEFAULT_THRESHOLD_GRID = tuple(np.round(np.arange(0.50, 0.701, 0.01), 2))


def _metric_row(split: str, y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "split": split,
        "samples": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(np.unique(y)) == 2 else float("nan"),
    }


def _load_source_spec() -> dict:
    specs = json.loads(SOURCE_SPEC_PATH.read_text(encoding="utf-8"))
    for spec in specs:
        if spec.get("model_name") == SOURCE_MODEL_NAME:
            return spec
    raise KeyError(f"Cannot find {SOURCE_MODEL_NAME} in {SOURCE_SPEC_PATH}")


def _select_threshold_guarded(curve: pd.DataFrame, min_resolved: int, target_resolved: int, max_resolved: int) -> float:
    grouped = curve.groupby("threshold", as_index=False).agg(resolved=("resolved", "sum"), wins=("wins", "sum"))
    grouped["winrate"] = grouped["wins"] * 100 / grouped["resolved"].replace(0, np.nan)
    eligible = grouped[grouped["resolved"] >= min_resolved].copy()
    if eligible.empty:
        return float(grouped.sort_values(["resolved", "winrate"], ascending=[False, False]).iloc[0]["threshold"])
    bounded = eligible[eligible["resolved"] <= max_resolved].copy()
    if bounded.empty:
        bounded = eligible
    bounded["score"] = bounded["winrate"] - (bounded["resolved"] - target_resolved).abs() / target_resolved
    return float(bounded.sort_values(["score", "winrate"], ascending=[False, False]).iloc[0]["threshold"])


def _objective_from_curve(curve: pd.DataFrame, auc: float, min_resolved: int, target_resolved: int, max_resolved: int) -> tuple[float, float, dict]:
    threshold = _select_threshold_guarded(curve, min_resolved=min_resolved, target_resolved=target_resolved, max_resolved=max_resolved)
    total = _total_at_threshold(curve, threshold)
    wr = float(total["winrate"])
    resolved = float(total["resolved"])
    if resolved < min_resolved:
        return float(-100.0 + resolved / max(1.0, min_resolved)), float(threshold), total
    # Valid 2023 is only one year. Avoid tuning to tiny high-WR pockets.
    deal_score = min(1.0, resolved / float(target_resolved))
    sparse_penalty = max(0.0, float(min_resolved) - resolved) / 600.0
    too_many_penalty = max(0.0, resolved - float(max_resolved)) / float(max_resolved)
    objective = wr + 2.0 * float(auc) + 2.0 * deal_score - sparse_penalty - too_many_penalty
    return float(objective), float(threshold), total


def _suggest_params(trial: optuna.Trial) -> dict:
    max_depth = trial.suggest_int("max_depth", 2, 5)
    max_leaves_by_depth = max(3, min(31, (2**max_depth) - 1))
    return {
        "n_estimators": trial.suggest_int("n_estimators", 180, 900, step=60),
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.06, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 3, max_leaves_by_depth),
        "max_depth": max_depth,
        "min_child_samples": trial.suggest_int("min_child_samples", 120, 880, step=40),
        "subsample": trial.suggest_float("subsample", 0.60, 0.95, step=0.05),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0, step=0.05),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 0.5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 5.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.08, step=0.01),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def run(
    n_trials: int,
    max_train_rows: int,
    seed: int,
    shap_rows: int,
    enable_mlflow: bool,
    timeout_sec: int | None,
    min_valid_resolved: int,
    target_valid_resolved: int,
    max_valid_resolved: int,
) -> dict:
    ensure_dirs()
    started = time.time()
    run_id = int(started)
    out_dir = CFG.outputs_dir / f"{FEATURE_FAMILY}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = out_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)

    build_cache(force=False)
    source_spec = _load_source_spec()
    cols = [c for c in source_spec["features"] if c != "buy_signal"]
    gate = str(source_spec["gate"])

    train_edge, _, _ = _load_edge_frames(cols + list(GATE_COLS))
    gate_quantiles = _quantiles(train_edge, [c for c in GATE_COLS if c not in {"label", "year"}])
    del train_edge
    gc.collect()

    print({"phase": "load_scaled_data", "gate": gate, "features": len(cols)}, flush=True)
    train = _load_scaled_split(CFG.split.train_years, cols, GATE_COLS, gate, gate_quantiles)
    train = train if len(train) <= max_train_rows else train.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)
    valid = _load_scaled_split(CFG.split.valid_years, cols, GATE_COLS, gate, gate_quantiles)
    test = _load_scaled_split(CFG.split.test_years, cols, GATE_COLS, gate, gate_quantiles)

    x_train = train[cols].to_numpy("float32", copy=True)
    y_train = train["label"].to_numpy("int8", copy=True)
    x_valid = valid[cols].to_numpy("float32", copy=True)
    y_valid = valid["label"].to_numpy("int8", copy=True)
    x_test = test[cols].to_numpy("float32", copy=True)
    y_test = test["label"].to_numpy("int8", copy=True)

    print({"phase": "data_ready", "train": len(y_train), "valid": len(y_valid), "test": len(y_test)}, flush=True)
    trial_rows: list[dict] = []

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
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        p_valid = model.predict_proba(x_valid)[:, 1]
        valid_auc = float(roc_auc_score(y_valid, p_valid)) if len(np.unique(y_valid)) == 2 else float("nan")
        curve = _threshold_table(valid[["label", "year", "session_london_ny"]], p_valid, DEFAULT_THRESHOLD_GRID)
        score, threshold, total = _objective_from_curve(
            curve,
            valid_auc,
            min_resolved=min_valid_resolved,
            target_resolved=target_valid_resolved,
            max_resolved=max_valid_resolved,
        )
        row = {
            "trial": trial.number,
            "objective": score,
            "valid_auc": valid_auc,
            "valid_wr": total["winrate"],
            "valid_resolved": total["resolved"],
            "threshold": threshold,
            "best_iteration": getattr(model, "best_iteration_", None),
            **params,
        }
        trial_rows.append(row)
        print({"phase": "trial", "trial": trial.number, "score": round(score, 5), "auc": round(valid_auc, 5), "wr": total["winrate"], "resolved": total["resolved"], "threshold": threshold}, flush=True)
        del model, p_valid
        gc.collect()
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=max(8, min(12, n_trials // 4))),
        study_name=f"{FEATURE_FAMILY}_{run_id}",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, gc_after_trial=True)

    trials_df = pd.DataFrame(trial_rows).sort_values("objective", ascending=False)
    trials_path = out_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_path, index=False, encoding="utf-8-sig")

    best_params = dict(study.best_params)
    # Optuna stores None categorical correctly, keep LGBM static params outside study if missing.
    best_params["subsample_freq"] = 1
    best_model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        random_state=seed + int(study.best_trial.number),
        n_jobs=1,
        verbose=-1,
        deterministic=True,
        force_col_wise=True,
        **best_params,
    )
    best_model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    p_valid = best_model.predict_proba(x_valid)[:, 1]
    p_test = best_model.predict_proba(x_test)[:, 1]
    valid_metrics = _metric_row("valid", y_valid, p_valid)
    test_metrics = _metric_row("test", y_test, p_test)
    valid_curve = _threshold_table(valid[["label", "year", "session_london_ny"]], p_valid, DEFAULT_THRESHOLD_GRID)
    test_curve = _threshold_table(test[["label", "year", "session_london_ny"]], p_test, DEFAULT_THRESHOLD_GRID)
    objective_score, threshold, valid_total = _objective_from_curve(
        valid_curve,
        float(valid_metrics["auc"]),
        min_resolved=min_valid_resolved,
        target_resolved=target_valid_resolved,
        max_resolved=max_valid_resolved,
    )
    test_total = _total_at_threshold(test_curve, threshold)

    imp = pd.DataFrame({"feature": cols, "importance": best_model.feature_importances_}).sort_values("importance", ascending=False)
    imp_path = best_dir / "best_lgbm_importance.csv"
    valid_curve_path = best_dir / "best_threshold_valid.csv"
    test_curve_path = best_dir / "best_threshold_test.csv"
    metrics_path = best_dir / "best_metrics.json"
    imp.to_csv(imp_path, index=False, encoding="utf-8-sig")
    valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
    test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")
    chart_paths = []
    chart_paths += plot_model_diagnostics(y_valid, p_valid, best_dir, "best_valid")
    chart_paths += plot_model_diagnostics(y_test, p_test, best_dir, "best_test")
    chart_paths.append(plot_threshold_curve(valid_curve, best_dir / "best_threshold_valid.png", "Optuna best valid threshold"))
    chart_paths.append(plot_threshold_curve(test_curve, best_dir / "best_threshold_test.png", "Optuna best test threshold"))
    chart_paths.append(_plot_importance(imp, best_dir / "best_lgbm_importance_top25.png", "Optuna best LightGBM importance"))
    if shap_rows > 0:
        chart_paths += _plot_shap(best_model, x_valid, cols, best_dir, "best", shap_rows, seed)

    bundle_path = CFG.model_dir / f"{FEATURE_FAMILY}_{run_id}_bundle.joblib"
    scaling_contract = {
        "scaler": "rolling_zscore",
        "source_run_id": SOURCE_RUN_ID,
        "source_model_name": SOURCE_MODEL_NAME,
        "gate": gate,
        "gate_quantiles": gate_quantiles,
    }
    joblib.dump(
        {
            "model": best_model,
            "model_type": "LightGBM",
            "feature_columns": cols,
            "feature_family": FEATURE_FAMILY,
            "source_feature_family": BASE_FEATURE_FAMILY,
            "source_run_id": SOURCE_RUN_ID,
            "source_model_name": SOURCE_MODEL_NAME,
            "gate": gate,
            "selected_threshold": threshold,
            "params": best_params,
            "scaling": scaling_contract,
        },
        bundle_path,
        compress=3,
    )

    summary = {
        "run_id": run_id,
        "feature_family": FEATURE_FAMILY,
        "source": f"{SOURCE_MODEL_NAME}_{SOURCE_RUN_ID}",
        "gate": gate,
        "n_trials": len(study.trials),
        "n_features": len(cols),
        "max_train_rows": max_train_rows,
        "min_valid_resolved": min_valid_resolved,
        "target_valid_resolved": target_valid_resolved,
        "max_valid_resolved": max_valid_resolved,
        "best_trial": int(study.best_trial.number),
        "best_objective_valid": float(study.best_value),
        "selected_threshold": threshold,
        "valid": valid_metrics,
        "valid_total": valid_total,
        "test": test_metrics,
        "test_total": test_total,
        "best_params": best_params,
        "top10_features": imp.head(10)["feature"].tolist(),
        "bundle_path": str(bundle_path),
        "out_dir": str(out_dir),
        "elapsed_sec": round(time.time() - started, 2),
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow = _mlflow() if enable_mlflow else None
    if mlflow:
        with mlflow.start_run(run_name=f"{FEATURE_FAMILY}_{run_id}"):
            mlflow.log_params(
                {
                    "feature_family": FEATURE_FAMILY,
                    "source_model": SOURCE_MODEL_NAME,
                    "source_run_id": SOURCE_RUN_ID,
                    "model_type": "LightGBM",
                    "gate": gate,
                    "n_trials": len(study.trials),
                    "n_features": len(cols),
                    "max_train_rows": max_train_rows,
                    "min_valid_resolved": min_valid_resolved,
                    "target_valid_resolved": target_valid_resolved,
                    "max_valid_resolved": max_valid_resolved,
                    "threshold": threshold,
                    **{f"best_{k}": v for k, v in best_params.items() if v is not None},
                }
            )
            for prefix, metrics in [("valid", valid_metrics), ("test", test_metrics)]:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                        mlflow.log_metric(f"{prefix}_{key}", value)
            mlflow.log_metric("valid_total_wr", valid_total["winrate"])
            mlflow.log_metric("valid_total_resolved", valid_total["resolved"])
            mlflow.log_metric("test_total_wr", test_total["winrate"])
            mlflow.log_metric("test_total_resolved", test_total["resolved"])
            mlflow.log_metric("test_min_year_deals", test_total["min_year_deals"])
            mlflow.log_metric("best_objective_valid", float(study.best_value))
            for path in [trials_path, summary_path, metrics_path, imp_path, valid_curve_path, test_curve_path, bundle_path, *chart_paths]:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=88011)
    parser.add_argument("--shap-rows", type=int, default=120)
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--min-valid-resolved", type=int, default=3000)
    parser.add_argument("--target-valid-resolved", type=int, default=4200)
    parser.add_argument("--max-valid-resolved", type=int, default=9000)
    parser.add_argument("--enable-mlflow", action="store_true")
    args = parser.parse_args()
    run(
        n_trials=args.n_trials,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
        shap_rows=args.shap_rows,
        enable_mlflow=args.enable_mlflow,
        timeout_sec=args.timeout_sec if args.timeout_sec > 0 else None,
        min_valid_resolved=args.min_valid_resolved,
        target_valid_resolved=args.target_valid_resolved,
        max_valid_resolved=args.max_valid_resolved,
    )


if __name__ == "__main__":
    main()
