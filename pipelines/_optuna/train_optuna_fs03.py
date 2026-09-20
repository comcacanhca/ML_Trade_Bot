from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import optuna
import pandas as pd
from fs03_common import (
    CFG,
    FEATURE_SET,
    dump_json,
    load_fs03_splits,
    metrics_from_scores,
    mlflow_client,
    new_run_dir,
    threshold_curves,
)


def suggest_lgbm_params(trial: optuna.Trial) -> dict:
    max_depth = trial.suggest_int("max_depth", 2, 8)
    max_leaves_by_depth = max(3, min(63, (2**max_depth) - 1))
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 400, step=60),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 3, max_leaves_by_depth),
        "max_depth": max_depth,
        "min_child_samples": trial.suggest_int("min_child_samples", 80, 300, step=40),
        "subsample": trial.suggest_float("subsample", 0.60, 0.95, step=0.05),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.0, step=0.1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 8.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.10, step=0.01),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def make_model(params: dict, seed: int, n_jobs: int = -1) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
        deterministic=True,
        force_col_wise=True,
        **params,
    )


def run(
    n_trials: int,
    seed: int,
    force_dataset: bool,
    max_train_rows: int | None,
    enable_mlflow: bool,
    output_root: Path | None = None,
) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    started = time.time()
    run_dir = new_run_dir("fs03_optuna_lgbm", output_root)
    data = load_fs03_splits(force_dataset=force_dataset, max_train_rows=max_train_rows)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgbm_params(trial)
        model = make_model(params, seed + trial.number, n_jobs=1)
        model.fit(
            data["x_train"],
            data["y_train"],
            eval_set=[(data["x_valid"], data["y_valid"])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        p_valid = model.predict_proba(data["x_valid"])[:, 1]
        metrics = metrics_from_scores(data["y_valid"], p_valid)
        trial.set_user_attr("valid_metrics", metrics)
        trial.set_user_attr("best_iteration", getattr(model, "best_iteration_", None))
        return float(metrics.get("auc", 0.0))

    study = optuna.create_study(
        direction="maximize",
        study_name=f"fs03_optuna_lgbm_{int(started)}",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=max(5, min(10, n_trials // 3))),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_trial.params)
    best_params["subsample_freq"] = 1
    best_model = make_model(best_params, seed)
    best_model.fit(
        data["x_train"],
        data["y_train"],
        eval_set=[(data["x_valid"], data["y_valid"])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    p_valid = best_model.predict_proba(data["x_valid"])[:, 1]
    p_test = best_model.predict_proba(data["x_test"])[:, 1]
    valid_metrics = metrics_from_scores(data["y_valid"], p_valid)
    test_metrics = metrics_from_scores(data["y_test"], p_test)
    _, valid_curve, test_curve = threshold_curves(data["frame"], data["feature_columns"], data["scaler"], best_model)

    model_path = CFG.model_dir / f"fs03_optuna_lgbm_{int(started)}_bundle.joblib"
    valid_curve_path = run_dir / "fs03_optuna_lgbm_threshold_valid.csv"
    test_curve_path = run_dir / "fs03_optuna_lgbm_threshold_test.csv"
    trials_path = run_dir / "optuna_trials.csv"
    summary_path = run_dir / "summary.json"

    joblib.dump(
        {
            "model": best_model,
            "scaler": data["scaler"],
            "feature_columns": data["feature_columns"],
            "feature_set": FEATURE_SET,
            "model_type": "lightgbm_lgbmclassifier",
            "params": best_params,
            "best_iteration": getattr(best_model, "best_iteration_", None),
            "seed": seed,
        },
        model_path,
        compress=3,
    )
    valid_curve.to_csv(valid_curve_path, index=False, encoding="utf-8-sig")
    test_curve.to_csv(test_curve_path, index=False, encoding="utf-8-sig")
    study.trials_dataframe().to_csv(trials_path, index=False, encoding="utf-8-sig")

    summary = {
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "feature_set": FEATURE_SET,
        "model_type": "lightgbm_lgbmclassifier",
        "seed": seed,
        "n_trials": n_trials,
        "best_trial": int(study.best_trial.number),
        "best_value_valid_auc": float(study.best_value),
        "best_params": best_params,
        "best_iteration": getattr(best_model, "best_iteration_", None),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "valid_curve": str(valid_curve_path),
            "test_curve": str(test_curve_path),
            "trials": str(trials_path),
        },
        "elapsed_sec": round(time.time() - started, 2),
    }
    dump_json(summary_path, summary)

    mlflow = mlflow_client(enable_mlflow)
    if mlflow:
        with mlflow.start_run(run_name=f"fs03_optuna_lgbm_{int(started)}"):
            mlflow.log_param("pipeline", "pipelines/_optuna/train_optuna_fs03.py")
            mlflow.log_param("model_type", "lightgbm_lgbmclassifier")
            mlflow.log_param("feature_set", FEATURE_SET)
            mlflow.log_param("seed", seed)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            for k, v in valid_metrics.items():
                mlflow.log_metric(f"valid_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            for artifact in (model_path, valid_curve_path, test_curve_path, trials_path, summary_path):
                mlflow.log_artifact(str(artifact))

    print(summary_path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI/CD Optuna LightGBM pipeline for fs03_lags_cycle.")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1003)
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    run(args.n_trials, args.seed, args.force_dataset, args.max_train_rows, args.enable_mlflow, args.output_root)
