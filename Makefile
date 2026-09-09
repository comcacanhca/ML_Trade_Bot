.PHONY: install test lint dvc-status compile-kfp train-transform

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest -q

lint:
	ruff check tests pipelines

dvc-status:
	dvc status --no-commit

compile-kfp:
	python pipelines/kubeflow/pipeline.py

train-transform:
	cd research/rf_mlflow && python research_h4edge_feature_transforms_optuna.py --variants products --top-n 10 --n-trials 50 --max-train-rows 1000000 --seed 92101 --enable-mlflow
