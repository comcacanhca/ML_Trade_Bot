from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from config import CFG, PROJECT_ROOT, ensure_dirs
from train_rf_mlflow import _candidate_dataset, build_cached_dataset


DEFAULT_MODEL = CFG.model_dir / "fs03_fixed_params_1788583848_bundle.joblib"


def _mlflow():
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(CFG.experiment_name)
        return mlflow
    except Exception:
        return None


def _sample_df(x: np.ndarray, y: np.ndarray | None, cols: list[str], rows: int, seed: int) -> pd.DataFrame:
    df = pd.DataFrame(x, columns=cols)
    if y is not None:
        df["label"] = y
    if len(df) > rows:
        df = df.sample(rows, random_state=seed)
    return df.reset_index(drop=True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rf_importance(model, cols: list[str], out_dir: Path, top_n: int) -> tuple[Path, Path]:
    imp = pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    csv_path = out_dir / "rf_feature_importance.csv"
    imp.to_csv(csv_path, index=False, encoding="utf-8-sig")

    top = imp.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(f"RF feature importance top {top_n}")
    ax.set_xlabel("Gini importance")
    png_path = _save(fig, out_dir / "rf_feature_importance_top.png")
    return csv_path, png_path


def plot_permutation_importance(model, x_valid: np.ndarray, y_valid: np.ndarray, cols: list[str], out_dir: Path, rows: int, seed: int, top_n: int) -> tuple[Path, Path]:
    if len(y_valid) > rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(len(y_valid)), rows, replace=False)
        x = x_valid[idx]
        y = y_valid[idx]
    else:
        x = x_valid
        y = y_valid

    result = permutation_importance(
        model,
        x,
        y,
        scoring="roc_auc",
        n_repeats=5,
        random_state=seed,
        n_jobs=-1,
    )
    imp = pd.DataFrame(
        {
            "feature": cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    csv_path = out_dir / "permutation_importance_valid_auc.csv"
    imp.to_csv(csv_path, index=False, encoding="utf-8-sig")

    top = imp.head(top_n).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_title(f"Permutation importance on validation AUC top {top_n}")
    ax.set_xlabel("AUC decrease")
    png_path = _save(fig, out_dir / "permutation_importance_valid_auc_top.png")
    return csv_path, png_path


def plot_pairplot(model, x_train: np.ndarray, y_train: np.ndarray, cols: list[str], out_dir: Path, rows: int, seed: int) -> Path:
    top10 = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False).head(10).index.tolist()
    df = _sample_df(x_train, y_train, cols, rows, seed)
    df["label"] = df["label"].map({0: "Lose", 1: "Win"})
    grid = sns.pairplot(
        df[top10 + ["label"]],
        vars=top10,
        hue="label",
        corner=True,
        plot_kws={"s": 8, "alpha": 0.35},
        diag_kws={"common_norm": False},
    )
    grid.fig.suptitle("Pair plot top 10 RF features - train sample", y=1.02)
    path = out_dir / "pairplot_top10_train.png"
    grid.fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(grid.fig)
    return path


def plot_shap(model, x_valid: np.ndarray, cols: list[str], out_dir: Path, rows: int, seed: int) -> list[Path]:
    import shap

    df = _sample_df(x_valid, None, cols, rows, seed)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif getattr(shap_values, "ndim", 0) == 3:
        values = shap_values[:, :, 1]
    else:
        values = shap_values

    paths: list[Path] = []
    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, df, plot_type="bar", show=False, max_display=20)
    bar = out_dir / "shap_summary_bar.png"
    plt.tight_layout()
    plt.savefig(bar, dpi=140, bbox_inches="tight")
    plt.close()
    paths.append(bar)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(values, df, show=False, max_display=20)
    bee = out_dir / "shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(bee, dpi=140, bbox_inches="tight")
    plt.close()
    paths.append(bee)

    shap_imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    shap_csv = out_dir / "shap_importance.csv"
    shap_imp.to_csv(shap_csv, index=False, encoding="utf-8-sig")
    paths.append(shap_csv)

    for feature in shap_imp.head(3)["feature"]:
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(feature, values, df, show=False, interaction_index=None)
        dep = out_dir / f"shap_dependence_{feature}.png"
        plt.tight_layout()
        plt.savefig(dep, dpi=140, bbox_inches="tight")
        plt.close()
        paths.append(dep)
    return paths


def plot_pdp(model, x_valid: np.ndarray, cols: list[str], out_dir: Path, rows: int, seed: int, top_n: int) -> list[Path]:
    df = _sample_df(x_valid, None, cols, rows, seed)
    top_features = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False).head(top_n).index.tolist()
    paths: list[Path] = []
    for feature in top_features:
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(model, df, [feature], ax=ax, kind="average", grid_resolution=30)
        ax.set_title(f"Partial dependence: {feature}")
        paths.append(_save(fig, out_dir / f"pdp_{feature}.png"))
    return paths


def run(model_file: Path, out_name: str, train_rows: int, valid_rows: int, shap_rows: int, seed: int) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    ensure_dirs()
    bundle = joblib.load(model_file)
    model = bundle["model"]
    scaler = bundle["scaler"]
    cols = bundle["feature_columns"]

    frame = build_cached_dataset(force=False)
    x_train_raw, y_train, _ = _candidate_dataset(frame, cols, CFG.split.train_years)
    x_valid_raw, y_valid, _ = _candidate_dataset(frame, cols, CFG.split.valid_years)
    x_train = scaler.transform(x_train_raw)
    x_valid = scaler.transform(x_valid_raw)

    out_dir = CFG.outputs_dir / out_name / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    rf_csv, rf_png = plot_rf_importance(model, cols, out_dir, top_n=16)
    paths += [rf_csv, rf_png]
    perm_csv, perm_png = plot_permutation_importance(model, x_valid, y_valid, cols, out_dir, valid_rows, seed, top_n=16)
    paths += [perm_csv, perm_png]
    paths.append(plot_pairplot(model, x_train, y_train, cols, out_dir, train_rows, seed))
    paths += plot_shap(model, x_valid, cols, out_dir, shap_rows, seed)
    paths += plot_pdp(model, x_valid, cols, out_dir, valid_rows, seed, top_n=6)

    manifest = pd.DataFrame({"artifact": [str(p) for p in paths]})
    manifest_path = out_dir / "explainability_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    paths.append(manifest_path)

    mlflow = _mlflow()
    if mlflow:
        with mlflow.start_run(run_name=f"explainability_{out_name}"):
            mlflow.log_param("model_file", str(model_file))
            mlflow.log_param("feature_count", len(cols))
            mlflow.log_param("train_rows_pairplot", train_rows)
            mlflow.log_param("valid_rows_permutation_pdp", valid_rows)
            mlflow.log_param("shap_rows", shap_rows)
            for p in paths:
                artifact_path = "explainability/shap" if "shap" in p.name else "explainability"
                mlflow.log_artifact(str(p), artifact_path=artifact_path)

    result = {"out_dir": str(out_dir), "artifacts": [str(p) for p in paths]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=str(DEFAULT_MODEL))
    parser.add_argument("--out-name", default="reproduce_fs03_fixed_params_1788583848")
    parser.add_argument("--train-rows", type=int, default=1000)
    parser.add_argument("--valid-rows", type=int, default=5000)
    parser.add_argument("--shap-rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1003)
    args = parser.parse_args()
    run(Path(args.model_file), args.out_name, args.train_rows, args.valid_rows, args.shap_rows, args.seed)
