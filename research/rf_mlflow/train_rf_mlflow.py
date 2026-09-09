from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from config import CFG, PROJECT_ROOT, ensure_dirs
from data_io import DATE, load_years
from features import RobustClipScaler, build_features, feature_columns
from labels import buy_labels_next_open
from mophong_adapter import orders_from_scores, simulate_orders


def _mlflow():
    try:
        import mlflow

        db_path = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def _years_all() -> tuple[int, ...]:
    return CFG.split.train_years + CFG.split.valid_years + CFG.split.test_years


def _cache_file(name: str) -> Path:
    return CFG.cache_dir / name


def build_cached_dataset(force: bool = False) -> pd.DataFrame:
    ensure_dirs()
    cache_file = _cache_file("dataset_features_labels.joblib")
    if cache_file.exists() and not force:
        return joblib.load(cache_file)
    data = load_years(_years_all())
    frame = build_features(data)
    frame = buy_labels_next_open(frame)
    frame["year"] = pd.to_datetime(frame[DATE]).dt.year.astype(int)
    joblib.dump(frame, cache_file, compress=3)
    return frame


def _candidate_dataset(frame: pd.DataFrame, cols: list[str], years: tuple[int, ...]):
    mask = (
        frame["year"].isin(years)
        & (frame["buy_signal"] == 1)
        & frame["label"].notna()
        & frame[cols].notna().all(axis=1)
    )
    x = frame.loc[mask, cols].to_numpy(float)
    y = frame.loc[mask, "label"].astype(int).to_numpy()
    meta = frame.loc[mask, [DATE, "year", "label", "buy_signal"]].copy()
    return x, y, meta


def _fit_model(x_train, y_train) -> RandomForestClassifier:
    params = CFG.rf
    model = RandomForestClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        min_samples_leaf=params.min_samples_leaf,
        min_samples_split=params.min_samples_split,
        max_features=params.max_features,
        max_samples=params.max_samples,
        class_weight=params.class_weight,
        random_state=params.random_state,
        n_jobs=params.n_jobs,
        bootstrap=True,
    )
    model.fit(x_train, y_train)
    return model


def _metrics(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    out = {
        "samples": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else 0.0,
        "accuracy": float(accuracy_score(y, pred)) if len(y) else 0.0,
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["brier"] = float(brier_score_loss(y, p))
        out["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    return out


def _score_full_frame(frame: pd.DataFrame, cols: list[str], scaler: RobustClipScaler, model: RandomForestClassifier) -> np.ndarray:
    valid = frame[cols].notna().all(axis=1).to_numpy()
    scores = np.full(len(frame), np.nan, dtype=float)
    x = scaler.transform(frame.loc[valid, cols].to_numpy(float))
    scores[valid] = model.predict_proba(x)[:, 1]
    return scores


def _proxy_threshold_table(frame: pd.DataFrame, scores: np.ndarray, years: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for threshold in CFG.threshold_grid:
        orders = orders_from_scores(frame, scores, threshold, require_signal=True, session_filter="london_ny")
        for year in years:
            ym = frame["year"].to_numpy() == year
            selected = (orders == 1) & ym & frame["label"].notna().to_numpy()
            wins = int(np.count_nonzero(selected & (frame["label"].to_numpy(float) == 1.0)))
            losses = int(np.count_nonzero(selected & (frame["label"].to_numpy(float) == 0.0)))
            resolved = wins + losses
            rows.append(
                {
                    "threshold": threshold,
                    "year": year,
                    "resolved": resolved,
                    "wins": wins,
                    "losses": losses,
                    "winrate": round(wins * 100 / resolved, 4) if resolved else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _select_threshold(valid_proxy: pd.DataFrame) -> float:
    grouped = valid_proxy.groupby("threshold", as_index=False).agg(
        resolved=("resolved", "sum"),
        wins=("wins", "sum"),
        min_year_deals=("resolved", "min"),
    )
    grouped["winrate"] = grouped["wins"] * 100 / grouped["resolved"].replace(0, np.nan)
    grouped["distance_to_3000"] = (grouped["resolved"] - 3000).abs()
    candidates = grouped[grouped["resolved"] >= 2200].copy()
    if candidates.empty:
        candidates = grouped.copy()
    candidates = candidates.sort_values(["winrate", "resolved"], ascending=[False, False])
    return float(candidates.iloc[0]["threshold"])


def _simulate_yearly(frame: pd.DataFrame, scores: np.ndarray, threshold: float, years: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for year in years:
        chunk = frame[frame["year"] == year].copy().reset_index(drop=True)
        score_chunk = scores[frame["year"].to_numpy() == year]
        orders = orders_from_scores(chunk, score_chunk, threshold, require_signal=True, session_filter="london_ny")
        summary, results = simulate_orders(chunk, orders, tinh_tien=True)
        rows.append({"year": year, "threshold": threshold, **summary})
        pd.DataFrame(results).to_csv(CFG.outputs_dir / f"mophong_deals_{year}.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def _sample_frame(x: np.ndarray, y: np.ndarray | None, cols: list[str], max_rows: int, seed: int) -> pd.DataFrame:
    out = pd.DataFrame(x, columns=cols)
    if y is not None:
        out["label"] = y
    if len(out) > max_rows:
        out = out.sample(max_rows, random_state=seed)
    return out.reset_index(drop=True)


def _add_train_explainability(
    model: RandomForestClassifier,
    feature_cols: list[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    out_dir: Path,
    pairplot_rows: int = 1000,
    shap_rows: int = 800,
    seed: int = 271828,
) -> list[Path]:
    """Create explainability charts for the trained RF model.

    Inputs are already transformed/scaled using train-fitted scaler. This avoids
    leakage because no fitting is performed on validation/test data here.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_path = out_dir / "rf_feature_importance.csv"
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
    paths.append(importance_path)

    top = importance.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("RF feature importance top 20")
    ax.set_xlabel("Gini importance")
    fig.tight_layout()
    rf_plot = out_dir / "rf_feature_importance_top20.png"
    fig.savefig(rf_plot, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(rf_plot)

    top10 = importance.head(10)["feature"].tolist()
    pair_df = _sample_frame(x_train, y_train, feature_cols, pairplot_rows, seed)
    pair_df["label"] = pair_df["label"].map({0: "Lose", 1: "Win"})
    grid = sns.pairplot(
        pair_df[top10 + ["label"]],
        vars=top10,
        hue="label",
        corner=True,
        plot_kws={"s": 8, "alpha": 0.35},
        diag_kws={"common_norm": False},
    )
    grid.fig.suptitle("Pair plot top 10 RF features - train sample", y=1.02)
    pairplot_path = out_dir / "pairplot_top10_train.png"
    grid.fig.savefig(pairplot_path, dpi=120, bbox_inches="tight")
    plt.close(grid.fig)
    paths.append(pairplot_path)

    try:
        import shap

        shap_df = _sample_frame(x_valid, None, feature_cols, shap_rows, seed)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(shap_df)
        if isinstance(shap_values, list):
            values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif getattr(shap_values, "ndim", 0) == 3:
            values = shap_values[:, :, 1]
        else:
            values = shap_values

        shap_importance = pd.DataFrame(
            {"feature": feature_cols, "mean_abs_shap": np.abs(values).mean(axis=0)}
        ).sort_values("mean_abs_shap", ascending=False)
        shap_importance_path = out_dir / "shap_importance.csv"
        shap_importance.to_csv(shap_importance_path, index=False, encoding="utf-8-sig")
        paths.append(shap_importance_path)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(values, shap_df, plot_type="bar", show=False, max_display=20)
        shap_bar = out_dir / "shap_summary_bar.png"
        plt.tight_layout()
        plt.savefig(shap_bar, dpi=140, bbox_inches="tight")
        plt.close()
        paths.append(shap_bar)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(values, shap_df, show=False, max_display=20)
        shap_bee = out_dir / "shap_beeswarm.png"
        plt.tight_layout()
        plt.savefig(shap_bee, dpi=140, bbox_inches="tight")
        plt.close()
        paths.append(shap_bee)

        for feature in shap_importance.head(3)["feature"]:
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(feature, values, shap_df, show=False, interaction_index=None)
            dep_path = out_dir / f"shap_dependence_{feature}.png"
            plt.tight_layout()
            plt.savefig(dep_path, dpi=140, bbox_inches="tight")
            plt.close()
            paths.append(dep_path)
    except Exception as exc:
        error_path = out_dir / "shap_error.txt"
        error_path.write_text(repr(exc), encoding="utf-8")
        paths.append(error_path)

    pd.DataFrame({"artifact": [str(path) for path in paths]}).to_csv(
        out_dir / "explainability_manifest.csv", index=False, encoding="utf-8-sig"
    )
    paths.append(out_dir / "explainability_manifest.csv")
    return paths


def main(force: bool = False, explainability: bool = False) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    started = time.time()
    frame = build_cached_dataset(force=force)
    cols = feature_columns(frame)

    x_train_raw, y_train, _ = _candidate_dataset(frame, cols, CFG.split.train_years)
    x_valid_raw, y_valid, _ = _candidate_dataset(frame, cols, CFG.split.valid_years)
    x_test_raw, y_test, _ = _candidate_dataset(frame, cols, CFG.split.test_years)

    scaler = RobustClipScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_valid = scaler.transform(x_valid_raw)
    x_test = scaler.transform(x_test_raw)
    model = _fit_model(x_train, y_train)

    p_train = model.predict_proba(x_train)[:, 1]
    p_valid = model.predict_proba(x_valid)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    metrics = {
        "train": _metrics(y_train, p_train),
        "valid": _metrics(y_valid, p_valid),
        "test_candidate_reference": _metrics(y_test, p_test),
    }

    scores = _score_full_frame(frame, cols, scaler, model)
    valid_proxy = _proxy_threshold_table(frame, scores, CFG.split.valid_years)
    test_proxy = _proxy_threshold_table(frame, scores, CFG.split.test_years)
    threshold = _select_threshold(valid_proxy)
    mophong_yearly = _simulate_yearly(frame, scores, threshold, CFG.split.test_years)

    model_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": cols,
        "selected_threshold": threshold,
        "config": {
            "trade": CFG.trade.__dict__,
            "split": CFG.split.__dict__,
            "rf": CFG.rf.__dict__,
        },
    }
    model_file = CFG.model_dir / "rf_buy_winlose_bundle.joblib"
    joblib.dump(model_bundle, model_file, compress=3)

    valid_proxy.to_csv(CFG.outputs_dir / "threshold_proxy_valid_2023.csv", index=False, encoding="utf-8-sig")
    test_proxy.to_csv(CFG.outputs_dir / "threshold_proxy_test_2024_2026.csv", index=False, encoding="utf-8-sig")
    mophong_yearly.to_csv(CFG.outputs_dir / "mophong_yearly_2024_2026.csv", index=False, encoding="utf-8-sig")
    with open(CFG.outputs_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "selected_threshold": threshold}, f, ensure_ascii=False, indent=2)

    explainability_paths: list[Path] = []
    if explainability:
        explainability_paths = _add_train_explainability(
            model=model,
            feature_cols=cols,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            out_dir=CFG.outputs_dir / "train_explainability",
        )

    mlflow = _mlflow()
    if mlflow is not None:
        with mlflow.start_run(run_name=f"rf_buy_winlose_{int(started)}"):
            mlflow.log_params({f"rf_{k}": v for k, v in CFG.rf.__dict__.items()})
            mlflow.log_params({f"trade_{k}": v for k, v in CFG.trade.__dict__.items()})
            mlflow.log_param("selected_threshold", threshold)
            for split, split_metrics in metrics.items():
                for key, value in split_metrics.items():
                    mlflow.log_metric(f"{split}_{key}", value)
            for _, row in mophong_yearly.iterrows():
                year = int(row["year"])
                mlflow.log_metric(f"mophong_{year}_winrate", float(row["winrate"]))
                mlflow.log_metric(f"mophong_{year}_deals", float(row["deals"]))
            mlflow.log_artifact(str(model_file))
            for name in ["threshold_proxy_valid_2023.csv", "threshold_proxy_test_2024_2026.csv", "mophong_yearly_2024_2026.csv", "metrics.json"]:
                mlflow.log_artifact(str(CFG.outputs_dir / name))
            for path in explainability_paths:
                artifact_path = "explainability/shap" if "shap" in path.name else "explainability"
                mlflow.log_artifact(str(path), artifact_path=artifact_path)

    result = {
        "model_file": str(model_file),
        "selected_threshold": threshold,
        "metrics": metrics,
        "mophong_yearly": mophong_yearly.to_dict(orient="records"),
        "explainability_artifacts": [str(path) for path in explainability_paths],
        "elapsed_sec": round(time.time() - started, 2),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--explainability", action="store_true")
    args = parser.parse_args()
    main(force=args.force, explainability=args.explainability)
