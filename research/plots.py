from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_feature_set_summary(summary: pd.DataFrame, path: Path) -> Path:
    top = summary.sort_values(["objective_score"], ascending=False).head(10).copy()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(top))
    ax1.bar(x - 0.2, top["test_total_deals"], width=0.4, label="test deals")
    ax1.set_ylabel("Deals")
    ax1.set_xticks(x)
    ax1.set_xticklabels(top["feature_set"], rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, top["test_total_wr"], marker="o", color="tab:red", label="test WR")
    ax2.axhline(58, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Winrate %")
    ax1.set_title("Top feature-set candidates: deals vs winrate")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    return _save(fig, path)


def plot_threshold_curve(curve: pd.DataFrame, path: Path, title: str) -> Path:
    grouped = curve.groupby("threshold", as_index=False).agg(
        resolved=("resolved", "sum"),
        wins=("wins", "sum"),
    )
    grouped["winrate"] = grouped["wins"] * 100 / grouped["resolved"].replace(0, np.nan)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(grouped["threshold"], grouped["resolved"], marker="o", label="resolved/deals proxy")
    ax1.set_ylabel("Resolved")
    ax1.set_xlabel("Threshold")
    ax2 = ax1.twinx()
    ax2.plot(grouped["threshold"], grouped["winrate"], marker="o", color="tab:red", label="winrate")
    ax2.axhline(58, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Winrate %")
    ax1.set_title(title)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    return _save(fig, path)


def plot_yearly_mophong(yearly: pd.DataFrame, path: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(yearly))
    ax1.bar(x, yearly["deals"], label="deals")
    ax1.set_ylabel("Deals")
    ax1.set_xticks(x)
    ax1.set_xticklabels(yearly["year"].astype(str))
    ax2 = ax1.twinx()
    ax2.plot(x, yearly["winrate"], marker="o", color="tab:red", label="winrate")
    ax2.axhline(58, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Winrate %")
    ax1.set_title("MoPhongDeals yearly validation")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    return _save(fig, path)


def plot_model_diagnostics(y_true, prob, path_dir: Path, prefix: str) -> list[Path]:
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, prob, ax=ax)
    ax.set_title(f"{prefix} ROC")
    paths.append(_save(fig, path_dir / f"{prefix}_roc.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    pred = (prob >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_true, pred, ax=ax, colorbar=False)
    ax.set_title(f"{prefix} confusion @0.5")
    paths.append(_save(fig, path_dir / f"{prefix}_confusion.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title(f"{prefix} calibration")
    paths.append(_save(fig, path_dir / f"{prefix}_calibration.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(prob[y_true == 0], bins=40, alpha=0.55, label="Lose")
    ax.hist(prob[y_true == 1], bins=40, alpha=0.55, label="Win")
    ax.set_xlabel("Predicted P(Win)")
    ax.set_title(f"{prefix} score distribution")
    ax.legend()
    paths.append(_save(fig, path_dir / f"{prefix}_score_distribution.png"))
    return paths
