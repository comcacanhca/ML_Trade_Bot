from __future__ import annotations

from kfp import dsl


DEFAULT_IMAGE = "ghcr.io/OWNER/REPO/ml-trade-bot:latest"


@dsl.component(base_image=DEFAULT_IMAGE)
def build_cache(data_dir: str, mlflow_tracking_uri: str) -> None:
    import os
    import subprocess

    os.environ["ML_TRADE_DATA_DIR"] = data_dir
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    os.environ["PYTHONPATH"] = "research/rf_mlflow:vendor/scj"
    subprocess.run(
        ["python", "research/rf_mlflow/mlops_build_cache.py"],
        check=True,
    )


@dsl.component(base_image=DEFAULT_IMAGE)
def train_transform_model(
    data_dir: str,
    mlflow_tracking_uri: str,
    variant: str,
    top_n: int,
    n_trials: int,
    max_train_rows: int,
    seed: int,
) -> str:
    import os
    import subprocess

    os.environ["ML_TRADE_DATA_DIR"] = data_dir
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    os.environ["PYTHONPATH"] = "research/rf_mlflow:vendor/scj"
    cmd = [
        "python",
        "research/rf_mlflow/research_h4edge_feature_transforms_optuna.py",
        "--variants",
        variant,
        "--top-n",
        str(top_n),
        "--n-trials",
        str(n_trials),
        "--max-train-rows",
        str(max_train_rows),
        "--seed",
        str(seed),
        "--enable-mlflow",
    ]
    subprocess.run(cmd, check=True)
    return "completed"


@dsl.component(base_image=DEFAULT_IMAGE)
def register_candidate(status: str, model_name: str) -> str:
    # Placeholder: replace with model registry promotion logic after DVC/MLflow remote is configured.
    return f"{model_name}:{status}"


@dsl.pipeline(name="ml-trade-bot-train-evaluate-deploy")
def train_evaluate_deploy_pipeline(
    data_dir: str = "/mnt/data_handler/1M",
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
    variant: str = "products",
    top_n: int = 10,
    n_trials: int = 50,
    max_train_rows: int = 1000000,
    seed: int = 92101,
    model_name: str = "h4edge-products-lgbm",
):
    cache_task = build_cache(data_dir=data_dir, mlflow_tracking_uri=mlflow_tracking_uri)
    train_task = train_transform_model(
        data_dir=data_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        variant=variant,
        top_n=top_n,
        n_trials=n_trials,
        max_train_rows=max_train_rows,
        seed=seed,
    ).after(cache_task)
    register_candidate(status=train_task.output, model_name=model_name)


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=train_evaluate_deploy_pipeline,
        package_path="pipelines/kubeflow/ml_trade_bot_pipeline.yaml",
    )
