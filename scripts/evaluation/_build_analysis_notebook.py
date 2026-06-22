"""Build pairwise_analysis.ipynb — run once to (re)generate the notebook."""
from pathlib import Path
import nbformat as nbf

NB_PATH = Path(__file__).resolve().parent.parent.parent / "pairwise_analysis.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}

cells = []

def md(src: str):
    cells.append(nbf.v4.new_markdown_cell(src))

def py(src: str):
    cells.append(nbf.v4.new_code_cell(src))


md("""# Pairwise & per-row evaluation analysis

Deep-dive companion to the Streamlit dashboard. Reads the row-level result
JSONLs from blob storage and produces a manual-review-friendly analysis:

1. Aggregate snapshot
2. Win / Tie / Loss summary (horizontal bar)
3. **Where the baseline beat the finetuned model** — full text, both
   judges, both A/B verdicts and justifications
4. Position-bias flip rate — interpretation, distribution and the
   exact pairs that flipped
5. Inter-judge disagreement — absolute score and pairwise verdict
6. Per-row regressions — biggest score drops vs baseline
7. Score distributions — finetuned vs baseline per dimension
8. Language mixing — flagged rows with examples
9. Cross-experiment comparison — all experiments side by side

Change `MODEL_LABEL` in the config cell to focus on a different run.
All data is pulled live from blob storage via `finetuning.blob_storage`.
""")

py("""# ── Imports & config ─────────────────────────────────────────────────────────
from __future__ import annotations

import sys, json, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path so the `finetuning` package resolves
PROJECT_ROOT = Path.cwd()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "finetuning").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finetuning.blob_storage import (
    download_blob_file,
    list_blob_prefixes,
    read_blob_json,
)

pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_rows", 100)

# ── Storage config ───────────────────────────────────────────────────────────
STORAGE_ACCOUNT = "llmaml5615532443"
EVAL_CONTAINER = "azureml-blobstore-4c704101-7a51-4680-bcf8-f13966bf69b4"
EVAL_BLOB_PREFIX = "eval-results"

# ── Pick the experiment to analyse ───────────────────────────────────────────
MODEL_LABEL = "mistral_r16_a16_e1_b16_w30"  # change me
BASELINE_LABEL = "baseline"

SCORE_COLS = [
    "grammar_score",
    "fluency_score",
    "vocabulary_score",
    "instruction_following_score",
    "correctness_score",
]
SCORE_LABELS = {
    "grammar_score": "Grammar",
    "fluency_score": "Fluency",
    "vocabulary_score": "Vocabulary",
    "instruction_following_score": "Instruction Following",
    "correctness_score": "Correctness",
}
DIMENSIONS = [
    ("pairwise_quality", "Language Quality"),
    ("pairwise_instruction", "Instruction Following"),
]
JUDGES = [("j1", "Judge 1"), ("j2", "Judge 2")]
""")

py("""# ── Helpers ──────────────────────────────────────────────────────────────────
def _download_jsonl(blob_name: str) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp = f.name
    found = download_blob_file(STORAGE_ACCOUNT, EVAL_CONTAINER, blob_name, tmp)
    if not found:
        Path(tmp).unlink(missing_ok=True)
        raise FileNotFoundError(blob_name)
    df = pd.read_json(tmp, lines=True)
    Path(tmp).unlink(missing_ok=True)
    return df


def _download_json(blob_name: str) -> dict | None:
    return read_blob_json(STORAGE_ACCOUNT, EVAL_CONTAINER, blob_name)


def truncate(text: str, n: int = 220) -> str:
    if not isinstance(text, str):
        return ""
    return text if len(text) <= n else text[: n - 1] + "…"
""")

py("""# ── Load all artefacts for the chosen experiment ─────────────────────────────
agg = _download_json(f"{EVAL_BLOB_PREFIX}/{MODEL_LABEL}/aggregate.json")
ft_scores = _download_jsonl(f"{EVAL_BLOB_PREFIX}/{MODEL_LABEL}/row_scores.jsonl")
bl_scores = _download_jsonl(f"{EVAL_BLOB_PREFIX}/{BASELINE_LABEL}/row_scores.jsonl")
pairwise = _download_jsonl(f"{EVAL_BLOB_PREFIX}/{MODEL_LABEL}/pairwise.jsonl")

print(f"Experiment        : {MODEL_LABEL}")
print(f"Baseline scores   : {len(bl_scores)} rows")
print(f"Finetuned scores  : {len(ft_scores)} rows")
print(f"Pairwise verdicts : {len(pairwise)} rows")
""")

md("""## 1. Aggregate snapshot

The aggregate JSON is what the dashboard reads. Below we surface the
headline numbers in one compact frame so you can confirm what you are
looking at before drilling in.""")

py("""def _aggregate_snapshot(agg: dict) -> pd.DataFrame:
    rows = []
    for col in SCORE_COLS:
        rows.append({
            "Dimension": SCORE_LABELS[col],
            "J1 mean": agg.get(f"j1_mean_{col}"),
            "J2 mean": agg.get(f"j2_mean_{col}"),
            "Combined mean": agg.get(f"combined_mean_{col}"),
            "J1 Δ vs baseline": agg.get(f"j1_mean_delta_{col}"),
            "J2 Δ vs baseline": agg.get(f"j2_mean_delta_{col}"),
            "Cohen's κ (j1 vs j2)": agg.get(f"kappa_{col}"),
            "Within-1 %": (agg.get(f"within_1_{col}") or 0) * 100,
        })
    return pd.DataFrame(rows).set_index("Dimension")


snap = _aggregate_snapshot(agg)
display(snap.style.format({
    "J1 mean": "{:.2f}",
    "J2 mean": "{:.2f}",
    "Combined mean": "{:.2f}",
    "J1 Δ vs baseline": "{:+.2f}",
    "J2 Δ vs baseline": "{:+.2f}",
    "Cohen's κ (j1 vs j2)": "{:.2f}",
    "Within-1 %": "{:.0f}%",
}, na_rep="—"))
""")

py("""# Pairwise headline (W/T/L + flip rate per judge per dim)
rows = []
for prefix, jlabel in JUDGES:
    for dim, dlabel in DIMENSIONS:
        w = agg.get(f"{prefix}_{dim}_win")
        t = agg.get(f"{prefix}_{dim}_tie")
        l = agg.get(f"{prefix}_{dim}_loss")
        flip = agg.get(f"{prefix}_{dim}_flip_rate")
        if w is None:
            continue
        total = (w + t + l) or 1
        rows.append({
            "Judge": jlabel,
            "Dimension": dlabel,
            "Win": w, "Tie": t, "Loss": l,
            "Win %": round(w / total * 100, 1),
            "Tie %": round(t / total * 100, 1),
            "Loss %": round(l / total * 100, 1),
            "Flip %": round(flip * 100, 1) if flip is not None else None,
        })
pw_summary = pd.DataFrame(rows).set_index(["Judge", "Dimension"])
display(pw_summary)
""")

md("""## 2. Win / Tie / Loss — horizontal stacked bar

The break-even line (50%) is drawn dotted. The chart is the same one used
in the dashboard, replicated here so this notebook is self-sufficient.""")

py("""def horizontal_wtl(pw_df: pd.DataFrame, title: str) -> go.Figure:
    pw_df = pw_df.assign(row=pw_df["Judge"] + " · " + pw_df["Dimension"])
    fig = go.Figure()
    for label, col, color in [
        ("Win",  "Win %",  "#2ca02c"),
        ("Tie",  "Tie %",  "#bcbcbc"),
        ("Loss", "Loss %", "#d62728"),
    ]:
        fig.add_bar(
            y=pw_df["row"], x=pw_df[col], name=label, orientation="h",
            marker_color=color,
            text=pw_df[col].map(lambda v: f"{v:.0f}%"),
            textposition="inside",
            hovertemplate=f"%{{y}}<br>{label}: %{{x:.1f}}%<extra></extra>",
        )
    fig.update_layout(
        barmode="stack", title=title,
        xaxis=dict(title="% of pairs", range=[0, 100], ticksuffix="%"),
        yaxis=dict(autorange="reversed", title=""),
        height=80 + 50 * len(pw_df),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    fig.add_vline(x=50, line_dash="dot", line_color="#888")
    return fig


horizontal_wtl(pw_summary.reset_index(), f"{MODEL_LABEL} vs baseline").show()
""")

md("""## 3. Where the baseline beat the finetuned model

A row is listed if **at least one judge** declared the baseline the winner
on that dimension. The table shows the input prompt and the resolved
winners; the expander below dumps the full text of every losing pair so
you can read the prompt, both responses, and both judges' justifications
side by side.""")

py("""# Build a tidy per-row table of resolved winners per (dim, judge)
def winner_table(pairwise: pd.DataFrame) -> pd.DataFrame:
    out = pairwise[["input", "baseline_output", "finetuned_output"]].copy()
    for prefix, jlabel in JUDGES:
        for dim, dlabel in DIMENSIONS:
            out[f"{jlabel} · {dlabel}"] = pairwise[f"{prefix}_{dim}_winner"]
    return out


winners = winner_table(pairwise)

# Filter rows where any judge×dim picked baseline
verdict_cols = [c for c in winners.columns if " · " in c]
mask = (winners[verdict_cols] == "baseline").any(axis=1)
baseline_wins = winners[mask].copy()

print(f"{len(baseline_wins)} / {len(winners)} pairs have at least one baseline win.")
display(baseline_wins[["input"] + verdict_cols].head(50))
""")

py("""# Drill-down: for the first N baseline-winning pairs, print full text + justifications
N_DRILL = 5

def show_pair(idx: int) -> None:
    row = pairwise.iloc[idx]
    print("=" * 100)
    print(f"Pair #{idx}")
    print("=" * 100)
    print("PROMPT:")
    print(row["input"])
    print()
    print("--- BASELINE OUTPUT ---")
    print(row["baseline_output"])
    print()
    print("--- FINETUNED OUTPUT ---")
    print(row["finetuned_output"])
    for prefix, jlabel in JUDGES:
        for dim, dlabel in DIMENSIONS:
            w = row[f"{prefix}_{dim}_winner"]
            wab = row[f"{prefix}_{dim}_winner_ab"]
            wba = row[f"{prefix}_{dim}_winner_ba"]
            jab = row[f"{prefix}_{dim}_justification_ab"]
            jba = row[f"{prefix}_{dim}_justification_ba"]
            flipped = row[f"{prefix}_{dim}_flipped"]
            tag = "🔁 FLIPPED" if flipped else ""
            print()
            print(f"  [{jlabel} — {dlabel}] resolved={w!r}  (A/B={wab!r}, B/A={wba!r}) {tag}")
            print(f"     A/B justification: {jab}")
            print(f"     B/A justification: {jba}")


for i in baseline_wins.index[:N_DRILL]:
    show_pair(i)
""")

md("""## 4. Position-bias flip rate

A flip means the judge picked a different non-tie winner when the A/B
positions were swapped. The metric measures **how much position drives
the verdict** — lower is better.

| Flip rate | Interpretation |
|-----------|----------------|
| **< 10 %** | ✅ Solid — judge mostly decides on content. |
| **10–20 %** | ⚠️ Acceptable but noisy — many borderline pairs; treat narrow win-rate gaps as ties. |
| **> 20 %** | ❌ Unreliable — position is driving the call; do not trust pairwise on this dimension. |

**Is 10 % high?** It is at the upper edge of what is normally considered
acceptable. MT-Bench / AlpacaEval typically report 5–15 % for strong
judges. For a Dutch fine-tuning study with subtle stylistic deltas, 10 %
is normal but not great — roughly 1 in 10 of your wins would flip on a
re-roll.

**Levers to tighten it:**

1. Use a stronger / larger judge (smaller models flip 2-3× more).
2. Add explicit *"position is not informative"* wording to the pairwise
   system prompt and force a `tie` when the gap is small.
3. Widen the tie band in the rubric — many flips are photo-finish pairs
   that should never have been a non-tie.
4. Switch from a 2-run AND protocol to a 3-run majority vote (more API
   cost; cuts flip rate roughly in half on borderline pairs).
5. Use a reasoning model (`o4-mini`, `gpt-5` with reasoning) so the
   verdict is grounded in chain-of-thought rather than vibe.""")

py("""# Flip rate per judge × dimension, with thresholds
flip_rows = []
for prefix, jlabel in JUDGES:
    for dim, dlabel in DIMENSIONS:
        col = f"{prefix}_{dim}_flipped"
        if col in pairwise.columns:
            flipped = pairwise[col].apply(lambda x: bool(x) and str(x).lower() != "false")
            flip_rate = flipped.mean()
            flip_rows.append({
                "Judge": jlabel, "Dimension": dlabel,
                "Flips": int(flipped.sum()),
                "Total": int(len(pairwise)),
                "Flip %": round(flip_rate * 100, 1),
                "Verdict": ("✅ solid" if flip_rate < 0.10
                            else "⚠️ noisy" if flip_rate < 0.20
                            else "❌ unreliable"),
            })
flip_df = pd.DataFrame(flip_rows)
display(flip_df)
""")

py("""# Distribution: how often each judge flipped per dimension
fig = px.bar(
    flip_df, x="Judge", y="Flip %", color="Dimension", barmode="group",
    text="Flip %", title="Position-bias flip rate",
)
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig.update_yaxes(range=[0, max(25, flip_df["Flip %"].max() * 1.2)])
fig.add_hrect(y0=0,  y1=10, fillcolor="#2ca02c", opacity=0.08, line_width=0)
fig.add_hrect(y0=10, y1=20, fillcolor="#ff9f0a", opacity=0.08, line_width=0)
fig.add_hrect(y0=20, y1=100, fillcolor="#d62728", opacity=0.08, line_width=0)
fig.update_layout(height=380, legend=dict(orientation="h", y=-0.2),
                  margin=dict(l=10, r=10, t=50, b=40))
fig.show()
""")

py("""# List of pairs that flipped (any judge × dimension)
flip_cols = [
    f"{prefix}_{dim}_flipped" for prefix, _ in JUDGES for dim, _ in DIMENSIONS
]
flip_cols = [c for c in flip_cols if c in pairwise.columns]

flipped_mask = pairwise[flip_cols].apply(
    lambda s: s.map(lambda x: bool(x) and str(x).lower() != "false")
).any(axis=1)

flipped_rows = pairwise[flipped_mask]
print(f"{len(flipped_rows)} / {len(pairwise)} pairs flipped on at least one judge × dimension.")

view = pd.DataFrame({"input": flipped_rows["input"].map(lambda s: truncate(s, 120))})
for prefix, jlabel in JUDGES:
    for dim, dlabel in DIMENSIONS:
        col = f"{prefix}_{dim}_flipped"
        if col in flipped_rows.columns:
            view[f"{jlabel}·{dlabel}"] = flipped_rows[col].map(
                lambda x: "🔁" if (bool(x) and str(x).lower() != "false") else ""
            )
display(view.head(30))
""")

py("""# Show the first few flipped pairs in full so you can see why the judge wavered
for i in flipped_rows.index[:3]:
    show_pair(i)
""")

md("""## 5. Inter-judge disagreement

Two angles:

* **Absolute scores** — for each dimension, how often did `j1` and `j2`
  give the same number, and which rows had the largest gap?
* **Pairwise verdicts** — on which pairs did the two judges disagree
  about the winner?""")

py("""# Per-dimension absolute-score agreement summary
rows = []
merged = ft_scores.copy()  # j1_*/j2_* are already on the same row
for col in SCORE_COLS:
    j1 = pd.to_numeric(merged[f"j1_{col}"], errors="coerce")
    j2 = pd.to_numeric(merged[f"j2_{col}"], errors="coerce")
    diff = (j1 - j2).abs()
    rows.append({
        "Dimension": SCORE_LABELS[col],
        "Mean |J1 − J2|": round(float(diff.mean()), 2),
        "Exact agree %": round((diff == 0).mean() * 100, 1),
        "Within-1 %": round((diff <= 1).mean() * 100, 1),
        "Cohen's κ": agg.get(f"kappa_{col}"),
    })
display(pd.DataFrame(rows).set_index("Dimension"))
""")

py("""# Top absolute disagreements per dimension (rows where j1 and j2 are farthest apart)
TOP_K = 5
for col in SCORE_COLS:
    diff = (pd.to_numeric(ft_scores[f"j1_{col}"], errors="coerce")
            - pd.to_numeric(ft_scores[f"j2_{col}"], errors="coerce")).abs()
    top = ft_scores.assign(diff=diff).nlargest(TOP_K, "diff")
    if top["diff"].iloc[0] == 0:
        continue
    print(f"\\n=== {SCORE_LABELS[col]} — top {TOP_K} judge disagreements ===")
    out = top[[
        "input",
        f"j1_{col}", f"j1_{col.replace('_score', '_justification')}",
        f"j2_{col}", f"j2_{col.replace('_score', '_justification')}",
        "diff",
    ]].copy()
    out.columns = ["input", "J1 score", "J1 justification",
                   "J2 score", "J2 justification", "|Δ|"]
    out["input"] = out["input"].map(lambda s: truncate(s, 140))
    out["J1 justification"] = out["J1 justification"].map(lambda s: truncate(s, 200))
    out["J2 justification"] = out["J2 justification"].map(lambda s: truncate(s, 200))
    display(out)
""")

py("""# Pairwise winner disagreement: rows where j1_winner != j2_winner
for dim, dlabel in DIMENSIONS:
    j1c, j2c = f"j1_{dim}_winner", f"j2_{dim}_winner"
    if j1c not in pairwise.columns:
        continue
    disagree = pairwise[pairwise[j1c] != pairwise[j2c]]
    print(f"\\n=== {dlabel}: {len(disagree)}/{len(pairwise)} pairs where j1 and j2 disagree on the winner ===")
    if len(disagree) == 0:
        continue
    view = pd.DataFrame({
        "input": disagree["input"].map(lambda s: truncate(s, 180)),
        "j1 winner": disagree[j1c],
        "j2 winner": disagree[j2c],
        "j1 just. (A/B)": disagree[f"j1_{dim}_justification_ab"].map(lambda s: truncate(s, 160)),
        "j2 just. (A/B)": disagree[f"j2_{dim}_justification_ab"].map(lambda s: truncate(s, 160)),
    })
    display(view.head(20))
""")

md("""## 6. Per-row regressions vs baseline

For every dimension and judge, compute the per-row delta
`finetuned − baseline`. Negative deltas are regressions. We then rank
rows by combined-judge regression to surface the worst drops first.""")

py("""# Align baseline and finetuned on the input prompt (positional alignment is what the eval pipeline used)
assert len(bl_scores) == len(ft_scores), (
    f"Row count mismatch: baseline={len(bl_scores)} ft={len(ft_scores)}"
)

deltas = pd.DataFrame({"input": ft_scores["input"]})
for col in SCORE_COLS:
    for prefix, jlabel in JUDGES:
        bl = pd.to_numeric(bl_scores[f"{prefix}_{col}"], errors="coerce")
        ft = pd.to_numeric(ft_scores[f"{prefix}_{col}"], errors="coerce")
        deltas[f"Δ {jlabel} {SCORE_LABELS[col]}"] = ft - bl

# Combined delta = mean across both judges and all 5 dimensions
delta_cols = [c for c in deltas.columns if c.startswith("Δ ")]
deltas["Δ macro (mean)"] = deltas[delta_cols].mean(axis=1)

print("Macro Δ summary:")
display(deltas["Δ macro (mean)"].describe().to_frame().T)
""")

py("""# Worst regressions (negative macro Δ first)
worst = deltas.nsmallest(10, "Δ macro (mean)").copy()
worst["input"] = worst["input"].map(lambda s: truncate(s, 160))
display(worst.style.background_gradient(
    cmap="RdYlGn", subset=delta_cols + ["Δ macro (mean)"], vmin=-3, vmax=3
).format({c: "{:+.1f}" for c in delta_cols + ["Δ macro (mean)"]}))
""")

py("""# Biggest improvements (positive macro Δ first)
best = deltas.nlargest(10, "Δ macro (mean)").copy()
best["input"] = best["input"].map(lambda s: truncate(s, 160))
display(best.style.background_gradient(
    cmap="RdYlGn", subset=delta_cols + ["Δ macro (mean)"], vmin=-3, vmax=3
).format({c: "{:+.1f}" for c in delta_cols + ["Δ macro (mean)"]}))
""")

md("""## 7. Score distributions

Side-by-side box plots per dimension, baseline vs finetuned, per judge.
A noticeable shift of the median is what you want to see; a wide spread
means the model is inconsistent on that dimension.""")

py("""# Build long-format frame for plotly box plots
records = []
for col in SCORE_COLS:
    for prefix, jlabel in JUDGES:
        for label, frame in [("baseline", bl_scores), (MODEL_LABEL, ft_scores)]:
            for v in pd.to_numeric(frame[f"{prefix}_{col}"], errors="coerce").dropna():
                records.append({
                    "Dimension": SCORE_LABELS[col],
                    "Judge": jlabel,
                    "Model": label,
                    "Score": int(v),
                })
long_df = pd.DataFrame(records)

fig = px.box(
    long_df, x="Dimension", y="Score", color="Model",
    facet_col="Judge", points=False, category_orders={"Model": ["baseline", MODEL_LABEL]},
)
fig.update_yaxes(range=[0.5, 5.5], dtick=1)
fig.update_layout(height=460, legend=dict(orientation="h", y=-0.2),
                  margin=dict(l=10, r=10, t=40, b=40))
fig.show()
""")

py("""# Per-dimension score histograms (stacked) — useful when distributions are heavily skewed
hist_fig = px.histogram(
    long_df, x="Score", color="Model", facet_col="Dimension", facet_row="Judge",
    barmode="overlay", opacity=0.65, nbins=5,
)
hist_fig.update_layout(height=520, legend=dict(orientation="h", y=-0.1),
                       margin=dict(l=10, r=10, t=40, b=20))
hist_fig.update_xaxes(dtick=1)
hist_fig.show()
""")

md("""## 8. Language mixing

A flagged row contains non-Dutch tokens according to the judge. The
expander below shows examples plus the snippets the judge highlighted.""")

py("""for prefix, jlabel in JUDGES:
    flag_col = f"{prefix}_language_mixing"
    ex_col = f"{prefix}_language_mixing_examples"
    if flag_col not in ft_scores.columns:
        continue
    flagged = ft_scores[ft_scores[flag_col].apply(
        lambda x: bool(x) and str(x).lower() != "false"
    )]
    print(f"=== {jlabel}: {len(flagged)}/{len(ft_scores)} rows flagged for language mixing ===")
    if len(flagged) == 0:
        continue
    view = pd.DataFrame({
        "input": flagged["input"].map(lambda s: truncate(s, 140)),
        "predicted_output": flagged["predicted_output"].map(lambda s: truncate(s, 220)),
        "examples": flagged[ex_col].map(lambda s: truncate(str(s), 200)),
    })
    display(view.head(15))
""")

md("""## 9. Cross-experiment comparison

Pulls **every** aggregate.json from blob storage and lays out a compact
comparison so you can sanity-check the chosen experiment against its
siblings.""")

py("""blobs = list_blob_prefixes(
    STORAGE_ACCOUNT, EVAL_CONTAINER, EVAL_BLOB_PREFIX, suffix="aggregate.json"
)
all_aggs = [a for a in (read_blob_json(STORAGE_ACCOUNT, EVAL_CONTAINER, b) for b in blobs) if a]
all_df = pd.DataFrame(all_aggs)
all_df["_sort"] = all_df["model_label"].apply(lambda x: (0, x) if x == BASELINE_LABEL else (1, x))
all_df = all_df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

key_cols = ["model_label", "n_samples"]
for col in SCORE_COLS:
    key_cols.append(f"combined_mean_{col}")
for prefix, _ in JUDGES:
    for dim, _ in DIMENSIONS:
        key_cols.append(f"{prefix}_{dim}_win_rate")
        key_cols.append(f"{prefix}_{dim}_flip_rate")
key_cols = [c for c in key_cols if c in all_df.columns]

display(all_df[key_cols].set_index("model_label"))
""")

py("""# Heatmap: combined mean scores per experiment × dimension
combined_cols = [
    f"combined_mean_{c}" for c in SCORE_COLS if f"combined_mean_{c}" in all_df.columns
]
mat = all_df.set_index("model_label")[combined_cols].rename(
    columns={f"combined_mean_{c}": SCORE_LABELS[c] for c in SCORE_COLS}
)
fig = px.imshow(
    mat.values, x=list(mat.columns), y=list(mat.index),
    color_continuous_scale="RdYlGn", zmin=1, zmax=5, text_auto=".2f", aspect="auto",
    title="Combined mean score — all experiments",
)
fig.update_layout(height=80 + 40 * len(mat), margin=dict(l=10, r=10, t=50, b=10))
fig.show()
""")

md("""---

**Reading guide for manual analysis**

- Section 3 is where you'll spend the most time: real losses to the
  baseline tell you exactly where the fine-tune is hurting.
- Section 5 (judge disagreement) tells you whether to *trust* a verdict
  in the first place — high disagreement on a row means take it with a
  pinch of salt.
- Section 4 puts the *flip rate* in context — combine it with section 3
  to spot losses that are also flips (those are the most fragile
  verdicts and probably the first ones worth spot-checking).
""")

nb["cells"] = cells
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NB_PATH)
print(f"Wrote {NB_PATH}  ({len(cells)} cells)")
