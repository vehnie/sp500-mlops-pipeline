"""Generate report-quality visualizations from the project's real artifacts.

Reads the feature data, model metrics, and saved test predictions and writes a
cohesive set of figures to ``report/figures/`` for inclusion in the final
six-page report.

Run from the project root with the project virtual environment active:

    python scripts/generate_report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import auc, confusion_matrix, roc_curve

# --------------------------------------------------------------------------- #
# Paths and shared style
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
FEATURE_CSV = ROOT / "data" / "04_feature" / "sp500_feature_data.csv"
OUT_DIR = ROOT / "report" / "figures"

LR_PRED = ROOT / "data" / "07_model_output" / "test_predictions.csv"
RF_PRED = ROOT / "data" / "07_model_output" / "random_forest_test_predictions.csv"
LR_TEST = ROOT / "data" / "07_model_output" / "test_metrics.json"
RF_TEST = ROOT / "data" / "07_model_output" / "random_forest_test_metrics.json"

# Consistent identity for the two models across every figure.
CHAMPION = "Logistic Regression"
CHALLENGER = "Random Forest"
C_CHAMPION = "#1F6FB2"   # blue  -> champion
C_CHALLENGER = "#E07B39"  # orange -> challenger
C_NEUTRAL = "#6C757D"
C_UP = "#2E8B6F"
C_DOWN = "#C0504D"

FEATURES = [
    "simple_return",
    "log_return",
    "sma_10",
    "sma_20",
    "sma_ratio_10",
    "rsi_14",
    "volatility_10",
    "volume_change",
]
TARGET = "target_next_day_up"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "font.size": 12,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.autolayout": False,
    }
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# 1. S&P 500 close price with chronological train/validation/test regions
# --------------------------------------------------------------------------- #
def fig_price_splits(feat: pd.DataFrame) -> None:
    n = len(feat)
    i_train = int(n * 0.70)
    i_val = int(n * 0.85)
    d = feat["Date"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(d, feat["Close"], color="#11304E", lw=1.3)

    spans = [
        (0, i_train, "#1F6FB2", "Train (70%)"),
        (i_train, i_val, "#E6A817", "Validation (15%)"),
        (i_val, n, "#C0504D", "Test (15%)"),
    ]
    for start, end, color, label in spans:
        ax.axvspan(d.iloc[start], d.iloc[end - 1], color=color, alpha=0.12, label=label)

    ax.set_title("S&P 500 (^GSPC) Close Price and Chronological Data Split")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close price (index points)")
    handles = [Patch(facecolor=c, alpha=0.30, label=l) for _, _, c, l in spans]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=11)
    ax.margins(x=0.01)
    _save(fig, "01_sp500_price_splits.png")


# --------------------------------------------------------------------------- #
# 2. Target class balance
# --------------------------------------------------------------------------- #
def fig_target_balance(feat: pd.DataFrame) -> None:
    counts = feat[TARGET].value_counts().sort_index()
    labels = ["Down / flat (0)", "Up (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.bar(labels, values, color=[C_DOWN, C_UP], width=0.6, edgecolor="white")
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + total * 0.01,
            f"{v:,}\n({v / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_title("Next-Day Direction Target Balance")
    ax.set_ylabel("Number of trading days")
    ax.set_ylim(0, max(values) * 1.18)
    _save(fig, "02_target_balance.png")


# --------------------------------------------------------------------------- #
# 3. Feature correlation heatmap
# --------------------------------------------------------------------------- #
def fig_correlation(feat: pd.DataFrame) -> None:
    cols = FEATURES + [TARGET]
    corr = feat[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Pearson correlation"},
        annot_kws={"size": 9},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (with Target)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    _save(fig, "03_feature_correlation.png")


# --------------------------------------------------------------------------- #
# 4. Model comparison on the test set
# --------------------------------------------------------------------------- #
def fig_model_comparison(lr: dict, rf: dict) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    lr_vals = [lr[m] for m in metrics]
    rf_vals = [rf[m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w / 2, lr_vals, w, label=f"{CHAMPION} (champion)", color=C_CHAMPION)
    b2 = ax.bar(x + w / 2, rf_vals, w, label=f"{CHALLENGER} (challenger)", color=C_CHALLENGER)

    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.012,
                f"{b.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax.axhline(0.5, ls="--", lw=1.4, color=C_NEUTRAL, alpha=0.9,
               label="Random baseline (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Champion vs Challenger — Test-Set Performance")
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    _save(fig, "04_model_comparison.png")


# --------------------------------------------------------------------------- #
# 5. ROC curves
# --------------------------------------------------------------------------- #
def fig_roc(lr_pred: pd.DataFrame, rf_pred: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for pred, name, color in [
        (lr_pred, CHAMPION, C_CHAMPION),
        (rf_pred, CHALLENGER, C_CHALLENGER),
    ]:
        fpr, tpr, _ = roc_curve(pred["actual_target"], pred["probability_up"])
        ax.plot(fpr, tpr, color=color, lw=2.4, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], ls="--", color=C_NEUTRAL, lw=1.4, label="Random (AUC = 0.500)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    ax.set_aspect("equal")
    _save(fig, "05_roc_curves.png")


# --------------------------------------------------------------------------- #
# 6. Confusion matrices (side by side)
# --------------------------------------------------------------------------- #
def fig_confusion(lr_pred: pd.DataFrame, rf_pred: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    tick = ["Down (0)", "Up (1)"]
    for ax, pred, name, cmap in [
        (axes[0], lr_pred, CHAMPION, "Blues"),
        (axes[1], rf_pred, CHALLENGER, "Oranges"),
    ]:
        cm = confusion_matrix(pred["actual_target"], pred["predicted_target"])
        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap=cmap,
            cbar=False,
            square=True,
            linewidths=0.5,
            linecolor="white",
            xticklabels=tick,
            yticklabels=tick,
            annot_kws={"size": 15, "weight": "bold"},
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices — Test Set", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "06_confusion_matrices.png")


# --------------------------------------------------------------------------- #
# 7. Champion predicted-probability distribution by actual class
# --------------------------------------------------------------------------- #
def fig_probability(lr_pred: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bins = np.linspace(0.3, 0.7, 31)
    for cls, color, label in [(0, C_DOWN, "Actual: Down (0)"), (1, C_UP, "Actual: Up (1)")]:
        sub = lr_pred.loc[lr_pred["actual_target"] == cls, "probability_up"]
        ax.hist(sub, bins=bins, alpha=0.6, color=color, label=label, edgecolor="white")
    ax.axvline(0.5, ls="--", color=C_NEUTRAL, lw=1.5, label="Decision threshold (0.5)")
    ax.set_title(f"{CHAMPION}: Predicted P(up) by Actual Class — Test Set")
    ax.set_xlabel("Predicted probability of an up day")
    ax.set_ylabel("Number of trading days")
    ax.legend(frameon=True, fontsize=11)
    _save(fig, "07_probability_distribution.png")


# --------------------------------------------------------------------------- #
# 8. Feature distributions (small multiples)
# --------------------------------------------------------------------------- #
def fig_feature_distributions(feat: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, col in zip(axes.ravel(), FEATURES):
        ax.hist(feat[col].dropna(), bins=50, color=C_CHAMPION, alpha=0.85, edgecolor="white")
        ax.set_title(col, fontsize=12)
        ax.tick_params(labelsize=9)
    fig.suptitle("Engineered Feature Distributions", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "08_feature_distributions.png")


def main() -> None:
    print("Generating report figures...")
    feat = pd.read_csv(FEATURE_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    lr_pred = pd.read_csv(LR_PRED)
    rf_pred = pd.read_csv(RF_PRED)
    lr_test = _load_json(LR_TEST)
    rf_test = _load_json(RF_TEST)

    fig_price_splits(feat)
    fig_target_balance(feat)
    fig_correlation(feat)
    fig_model_comparison(lr_test, rf_test)
    fig_roc(lr_pred, rf_pred)
    fig_confusion(lr_pred, rf_pred)
    fig_probability(lr_pred)
    fig_feature_distributions(feat)
    print(f"Done. Figures saved to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
