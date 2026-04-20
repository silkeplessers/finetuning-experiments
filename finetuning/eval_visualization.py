"""Generate evaluation charts and save them as PNGs.

Charts are produced per judge (j1, j2) and combined. Includes an inter-judge
agreement heatmap when both judges are present.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCORE_COLS = [
    "grammar_score",
    "fluency_score",
    "vocabulary_score",
    "instruction_following_score",
    "correctness_score",
]
SCORE_LABELS = ["Grammar", "Fluency", "Vocabulary", "Instruction\nFollowing", "Correctness"]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart: %s", path)


# ── Per-judge score distribution ──────────────────────────────────────────────


def _score_distribution(
    df_scores: pd.DataFrame,
    charts_dir: Path,
    label: str,
    prefix: str,
    judge_name: str,
) -> None:
    """Histogram of each score dimension for a single judge."""
    fig, axes = plt.subplots(
        1, len(SCORE_COLS), figsize=(4 * len(SCORE_COLS), 4), sharey=True
    )
    for ax, col, name in zip(axes, SCORE_COLS, SCORE_LABELS):
        pcol = f"{prefix}_{col}"
        if pcol not in df_scores.columns:
            continue
        vals = pd.to_numeric(df_scores[pcol], errors="coerce").dropna()
        ax.hist(
            vals,
            bins=[0.5 + i for i in range(11)],
            rwidth=0.8,
            color="#4C72B0",
            edgecolor="white",
        )
        ax.set_title(name)
        ax.set_xlabel("Score")
        ax.set_xticks(range(1, 11))
    axes[0].set_ylabel("Count")
    fig.suptitle(f"Score Distribution — {label} ({judge_name})", fontsize=14, y=1.02)
    _save(fig, charts_dir / f"score_distribution_{prefix}.png")


# ── Baseline vs finetuned bars (per judge) ────────────────────────────────────


def _baseline_vs_finetuned_bars(
    agg: dict,
    charts_dir: Path,
    prefix: str,
    judge_name: str,
    df_baseline_scores: pd.DataFrame | None = None,
) -> None:
    """Side-by-side bar chart comparing mean scores for one judge."""
    if df_baseline_scores is None:
        return

    ft_means = [agg.get(f"{prefix}_mean_{c}") for c in SCORE_COLS]
    bl_means = []
    for c in SCORE_COLS:
        pcol = f"{prefix}_{c}"
        if pcol in df_baseline_scores.columns:
            s = pd.to_numeric(df_baseline_scores[pcol], errors="coerce").dropna()
            bl_means.append(round(float(s.mean()), 3) if len(s) > 0 else None)
        else:
            bl_means.append(None)

    if any(v is None for v in ft_means + bl_means):
        return

    x = np.arange(len(SCORE_COLS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, bl_means, w, label="Baseline", color="#4C72B0")
    ax.bar(
        x + w / 2,
        ft_means,
        w,
        label=agg.get("model_label", "Finetuned"),
        color="#DD8452",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SCORE_LABELS)
    ax.set_ylabel("Mean Score")
    ax.set_ylim(0, 10.5)
    ax.legend()
    ax.set_title(f"Baseline vs Finetuned — Mean Scores ({judge_name})")

    for i, (b, f) in enumerate(zip(bl_means, ft_means)):
        delta = f - b
        color = "green" if delta > 0 else ("red" if delta < 0 else "gray")
        ax.annotate(
            f"{delta:+.2f}",
            (x[i] + w / 2, f + 0.1),
            ha="center",
            fontsize=9,
            color=color,
        )

    _save(fig, charts_dir / f"baseline_vs_finetuned_{prefix}.png")


# ── Pairwise win/tie/loss (per judge) ────────────────────────────────────────


def _pairwise_chart(
    agg: dict,
    charts_dir: Path,
    prefix: str,
    judge_name: str,
) -> None:
    """Win/tie/loss stacked bar chart for pairwise comparison for one judge."""
    dims = []
    for dim, label in [
        ("pairwise_quality", "Language Quality"),
        ("pairwise_instruction", "Instruction Following"),
    ]:
        w = agg.get(f"{prefix}_{dim}_win")
        if w is not None:
            dims.append(
                (
                    label,
                    w,
                    agg.get(f"{prefix}_{dim}_tie", 0),
                    agg.get(f"{prefix}_{dim}_loss", 0),
                )
            )

    if not dims:
        return

    labels = [d[0] for d in dims]
    wins = [d[1] for d in dims]
    ties = [d[2] for d in dims]
    losses = [d[3] for d in dims]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(x, wins, color="#55A868", label="Finetuned wins")
    ax.barh(x, ties, left=wins, color="#C4C4C4", label="Tie")
    ax.barh(
        x,
        losses,
        left=[w + t for w, t in zip(wins, ties)],
        color="#C44E52",
        label="Baseline wins",
    )
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count")
    ax.legend(loc="lower right")
    ax.set_title(f"Pairwise — Win / Tie / Loss ({judge_name})")

    for i, (w, t, l) in enumerate(zip(wins, ties, losses)):
        total = w + t + l
        ax.text(total + 1, i, f"W:{w}  T:{t}  L:{l}", va="center", fontsize=9)

    _save(fig, charts_dir / f"pairwise_{prefix}.png")


# ── Language mixing chart (per judge) ─────────────────────────────────────────


def _language_mixing_chart(
    agg: dict,
    charts_dir: Path,
    prefix: str,
    judge_name: str,
    has_baseline: bool = False,
) -> None:
    """Bar chart comparing language mixing rates for one judge."""
    rate = agg.get(f"{prefix}_language_mixing_rate", 0)
    rates = {"Finetuned": rate}
    if has_baseline:
        bl_rate = agg.get(f"{prefix}_baseline_language_mixing_rate", 0)
        rates["Baseline"] = bl_rate

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(
        list(rates.keys()),
        [v * 100 for v in rates.values()],
        color=["#DD8452", "#4C72B0"][: len(rates)],
        width=0.5,
    )
    ax.set_ylabel("% of samples with language mixing")
    ax.set_title(f"Language Mixing Rate ({judge_name})")
    ax.set_ylim(0, max(100, max(v * 100 for v in rates.values()) + 10))
    for bar, val in zip(bars, rates.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1%}",
            ha="center",
            fontsize=10,
        )
    _save(fig, charts_dir / f"language_mixing_{prefix}.png")


# ── Inter-judge agreement heatmap ─────────────────────────────────────────────


def _agreement_heatmap(agg: dict, charts_dir: Path) -> None:
    """Heatmap showing agreement rate and Cohen's Kappa between judges."""
    dimensions = SCORE_COLS + ["language_mixing"]
    dim_labels = [c.replace("_score", "").replace("_", " ").title() for c in dimensions]

    # Add pairwise dimensions if present
    for dim in ["pairwise_quality", "pairwise_instruction"]:
        if f"agreement_{dim}" in agg:
            dimensions.append(dim)
            dim_labels.append(dim.replace("_", " ").title())

    agreement = [agg.get(f"agreement_{d}", 0) for d in dimensions]
    kappa = [agg.get(f"kappa_{d}", 0) for d in dimensions]

    fig, axes = plt.subplots(1, 2, figsize=(10, max(3, len(dimensions) * 0.6)))

    for ax, values, title in [
        (axes[0], agreement, "Agreement Rate"),
        (axes[1], kappa, "Cohen's Kappa"),
    ]:
        data = np.array(values).reshape(-1, 1)
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(range(len(dim_labels)))
        ax.set_yticklabels(dim_labels)
        ax.set_xticks([])
        ax.set_title(title)
        for i, v in enumerate(values):
            ax.text(
                0,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if v < 0.4 else "black",
            )

    fig.suptitle("Inter-Judge Agreement (J1 vs J2)", fontsize=13)
    fig.tight_layout()
    _save(fig, charts_dir / "agreement_heatmap.png")


# ── Main entry point ─────────────────────────────────────────────────────────


def generate_charts(
    agg: dict,
    df_scores: pd.DataFrame,
    charts_dir: Path,
    df_baseline_scores: pd.DataFrame | None = None,
    df_pairwise: pd.DataFrame | None = None,
) -> None:
    """Generate all evaluation charts (per judge + combined agreement)."""
    label = agg.get("model_label", "model")
    has_baseline = df_baseline_scores is not None

    for prefix, judge_name in [("j1", "Judge 1"), ("j2", "Judge 2")]:
        _score_distribution(df_scores, charts_dir, label, prefix, judge_name)
        _baseline_vs_finetuned_bars(
            agg, charts_dir, prefix, judge_name, df_baseline_scores
        )
        _language_mixing_chart(agg, charts_dir, prefix, judge_name, has_baseline)
        if df_pairwise is not None:
            _pairwise_chart(agg, charts_dir, prefix, judge_name)

    # Combined agreement heatmap
    _agreement_heatmap(agg, charts_dir)

    logger.info("All charts saved to %s", charts_dir)
