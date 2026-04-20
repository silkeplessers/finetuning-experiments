"""Generate evaluation charts and save them as PNGs."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

SCORE_COLS = ["grammar_score", "fluency_score", "vocabulary_score", "instruction_following_score"]
SCORE_LABELS = ["Grammar", "Fluency", "Vocabulary", "Instruction\nFollowing"]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart: %s", path)


def _score_distribution(df_scores: pd.DataFrame, charts_dir: Path, label: str) -> None:
    """Histogram of each score dimension."""
    fig, axes = plt.subplots(1, len(SCORE_COLS), figsize=(4 * len(SCORE_COLS), 4), sharey=True)
    for ax, col, name in zip(axes, SCORE_COLS, SCORE_LABELS):
        vals = pd.to_numeric(df_scores[col], errors="coerce").dropna()
        ax.hist(vals, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], rwidth=0.8, color="#4C72B0", edgecolor="white")
        ax.set_title(name)
        ax.set_xlabel("Score")
        ax.set_xticks([1, 2, 3, 4, 5])
    axes[0].set_ylabel("Count")
    fig.suptitle(f"Score Distribution — {label}", fontsize=14, y=1.02)
    _save(fig, charts_dir / "score_distribution.png")


def _baseline_vs_finetuned_bars(
    agg: dict, charts_dir: Path,
    df_baseline_scores: pd.DataFrame | None = None,
) -> None:
    """Side-by-side bar chart comparing mean scores."""
    if df_baseline_scores is None:
        return

    ft_means = [agg.get(f"mean_{c}") for c in SCORE_COLS]
    bl_means = []
    for c in SCORE_COLS:
        s = pd.to_numeric(df_baseline_scores[c], errors="coerce").dropna()
        bl_means.append(round(float(s.mean()), 3) if len(s) > 0 else None)

    if any(v is None for v in ft_means + bl_means):
        return

    import numpy as np
    x = np.arange(len(SCORE_COLS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, bl_means, w, label="Baseline", color="#4C72B0")
    ax.bar(x + w / 2, ft_means, w, label=agg.get("model_label", "Finetuned"), color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(SCORE_LABELS)
    ax.set_ylabel("Mean Score")
    ax.set_ylim(0, 5.5)
    ax.legend()
    ax.set_title("Baseline vs Finetuned — Mean Scores")

    # Annotate delta
    for i, (b, f) in enumerate(zip(bl_means, ft_means)):
        delta = f - b
        color = "green" if delta > 0 else ("red" if delta < 0 else "gray")
        ax.annotate(f"{delta:+.2f}", (x[i] + w / 2, f + 0.1), ha="center", fontsize=9, color=color)

    _save(fig, charts_dir / "baseline_vs_finetuned.png")


def _pairwise_chart(agg: dict, charts_dir: Path) -> None:
    """Win/tie/loss stacked bar chart for pairwise comparison."""
    dims = []
    for dim, label in [("pairwise_quality", "Language Quality"), ("pairwise_instruction", "Instruction Following")]:
        w = agg.get(f"{dim}_win")
        if w is not None:
            dims.append((label, w, agg.get(f"{dim}_tie", 0), agg.get(f"{dim}_loss", 0)))

    if not dims:
        return

    import numpy as np
    labels = [d[0] for d in dims]
    wins = [d[1] for d in dims]
    ties = [d[2] for d in dims]
    losses = [d[3] for d in dims]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(x, wins, color="#55A868", label="Finetuned wins")
    ax.barh(x, ties, left=wins, color="#C4C4C4", label="Tie")
    ax.barh(x, losses, left=[w + t for w, t in zip(wins, ties)], color="#C44E52", label="Baseline wins")
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count")
    ax.legend(loc="lower right")
    ax.set_title("Pairwise Comparison — Win / Tie / Loss")

    # Annotate counts
    for i, (w, t, l) in enumerate(zip(wins, ties, losses)):
        total = w + t + l
        ax.text(total + 1, i, f"W:{w}  T:{t}  L:{l}", va="center", fontsize=9)

    _save(fig, charts_dir / "pairwise_comparison.png")


def _language_mixing_chart(
    agg: dict, charts_dir: Path,
    df_baseline_scores: pd.DataFrame | None = None,
) -> None:
    """Bar chart comparing language mixing rates."""
    rates = {"Finetuned": agg.get("language_mixing_rate", 0)}
    if df_baseline_scores is not None:
        rates["Baseline"] = agg.get("baseline_language_mixing_rate", 0)

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(list(rates.keys()), [v * 100 for v in rates.values()],
                  color=["#DD8452", "#4C72B0"][:len(rates)], width=0.5)
    ax.set_ylabel("% of samples with language mixing")
    ax.set_title("Language Mixing Rate")
    ax.set_ylim(0, max(100, max(v * 100 for v in rates.values()) + 10))
    for bar, val in zip(bars, rates.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1%}", ha="center", fontsize=10)
    _save(fig, charts_dir / "language_mixing.png")


def generate_charts(
    agg: dict,
    df_scores: pd.DataFrame,
    charts_dir: Path,
    df_baseline_scores: pd.DataFrame | None = None,
    df_pairwise: pd.DataFrame | None = None,
) -> None:
    """Generate all evaluation charts."""
    label = agg.get("model_label", "model")
    _score_distribution(df_scores, charts_dir, label)
    _baseline_vs_finetuned_bars(agg, charts_dir, df_baseline_scores)
    _language_mixing_chart(agg, charts_dir, df_baseline_scores)
    if df_pairwise is not None:
        _pairwise_chart(agg, charts_dir)
    logger.info("All charts saved to %s", charts_dir)
