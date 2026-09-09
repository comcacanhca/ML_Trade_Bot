from __future__ import annotations


def test_core_imports() -> None:
    import config
    import data_io
    import features
    import labels
    import plots

    assert config.WORK_DIR.name == "rf_mlflow"
    assert data_io.CLOSE == "close"
    assert hasattr(features, "FeatureBuilder") or hasattr(features, "build_features")
    assert hasattr(labels, "buy_label_arrays")
    assert hasattr(plots, "plot_model_diagnostics")
