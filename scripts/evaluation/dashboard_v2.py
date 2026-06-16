"""Streamlit dashboard v2 — compact, plotly-based, tabbed.

Reads all aggregate.json files from blob storage under eval-results/ and
renders an auto-refreshing dashboard organized in tabs so judges/dimensions
can be compared at a glance instead of scrolling through long tables.

Usage:
    streamlit run scripts/evaluation/dashboard_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from finetuning.blob_storage import list_blob_prefixes, read_blob_json

# ── Constants ─────────────────────────────────────────────────────────────────

STORAGE_ACCOUNT = "llmaml5615532443"
EVAL_CONTAINER = "azureml-blobstore-4c704101-7a51-4680-bcf8-f13966bf69b4"
EVAL_BLOB_PREFIX = "eval-results"

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
    "instruction_following_score": "Instr. Following",
    "correctness_score": "Correctness",
}
JUDGES = [("j1", "Judge 1"), ("j2", "Judge 2")]
DIMENSIONS = [
    ("pairwise_quality", "Language Quality"),
    ("pairwise_instruction", "Instruction Following"),
]

# Win / Tie / Loss colour palette (colour-blind friendly)
WIN_COLOR = "#2ca02c"   # green
TIE_COLOR = "#bcbcbc"   # grey
LOSS_COLOR = "#d62728"  # red


# ── Data loading (cached with TTL for auto-refresh) ──────────────────────────


@st.cache_data(ttl=120)
def load_all_experiments() -> pd.DataFrame:
    blobs = list_blob_prefixes(
        STORAGE_ACCOUNT, EVAL_CONTAINER, EVAL_BLOB_PREFIX, suffix="aggregate.json"
    )
    records = []
    for blob_name in blobs:
        agg = read_blob_json(STORAGE_ACCOUNT, EVAL_CONTAINER, blob_name)
        if agg:
            records.append(agg)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["_sort"] = df["model_label"].apply(
        lambda x: (0, x) if x == "baseline" else (1, x)
    )
    df = df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return df


# ── Plot helpers ──────────────────────────────────────────────────────────────


def horizontal_wtl_bar(pw_df: pd.DataFrame, dim_label: str) -> go.Figure:
    """Horizontal stacked Win/Tie/Loss % bar chart, one row per (Experiment, Judge)."""
    pw_df = pw_df.sort_values(["Experiment", "Judge"])
    # Use a single row label "Experiment — Judge" to keep the chart compact.
    pw_df = pw_df.assign(row=pw_df["Experiment"] + " · " + pw_df["Judge"])

    fig = go.Figure()
    fig.add_bar(
        y=pw_df["row"],
        x=pw_df["Win %"],
        name="Win",
        orientation="h",
        marker_color=WIN_COLOR,
        text=pw_df["Win %"].map(lambda v: f"{v:.0f}%"),
        textposition="inside",
        hovertemplate="%{y}<br>Win: %{x:.1f}%<extra></extra>",
    )
    fig.add_bar(
        y=pw_df["row"],
        x=pw_df["Tie %"],
        name="Tie",
        orientation="h",
        marker_color=TIE_COLOR,
        text=pw_df["Tie %"].map(lambda v: f"{v:.0f}%"),
        textposition="inside",
        hovertemplate="%{y}<br>Tie: %{x:.1f}%<extra></extra>",
    )
    fig.add_bar(
        y=pw_df["row"],
        x=pw_df["Loss %"],
        name="Loss",
        orientation="h",
        marker_color=LOSS_COLOR,
        text=pw_df["Loss %"].map(lambda v: f"{v:.0f}%"),
        textposition="inside",
        hovertemplate="%{y}<br>Loss: %{x:.1f}%<extra></extra>",
    )
    fig.update_layout(
        barmode="stack",
        title=f"Win / Tie / Loss — {dim_label}",
        xaxis=dict(title="% of pairs", range=[0, 100], ticksuffix="%"),
        yaxis=dict(title="", autorange="reversed"),
        height=max(220, 40 * len(pw_df) + 120),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    # Add 50% reference line (neutral break-even)
    fig.add_vline(x=50, line_dash="dot", line_color="#888")
    return fig


def grouped_score_bars(df_sel: pd.DataFrame, score_prefix: str, title: str) -> go.Figure:
    """Grouped bar chart: x = score dimension, colour = experiment."""
    rows = []
    for _, r in df_sel.iterrows():
        for col in SCORE_COLS:
            v = r.get(f"{score_prefix}_{col}")
            if v is not None:
                rows.append(
                    {
                        "Experiment": r["model_label"],
                        "Dimension": SCORE_LABELS[col],
                        "Score": v,
                    }
                )
    if not rows:
        return None
    long_df = pd.DataFrame(rows)
    fig = px.bar(
        long_df,
        x="Dimension",
        y="Score",
        color="Experiment",
        barmode="group",
        title=title,
        text_auto=".2f",
    )
    fig.update_layout(
        yaxis=dict(range=[0, 5], title="Score (1–5)"),
        height=380,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def score_heatmap(df_sel: pd.DataFrame, score_prefix: str, title: str) -> go.Figure:
    """Heatmap of mean scores: rows = experiments, columns = dimensions."""
    cols = [f"{score_prefix}_{c}" for c in SCORE_COLS if f"{score_prefix}_{c}" in df_sel.columns]
    if not cols:
        return None
    mat = df_sel.set_index("model_label")[cols].rename(
        columns={f"{score_prefix}_{c}": SCORE_LABELS[c] for c in SCORE_COLS}
    )
    fig = px.imshow(
        mat.values,
        x=list(mat.columns),
        y=list(mat.index),
        color_continuous_scale="RdYlGn",
        zmin=1,
        zmax=5,
        text_auto=".2f",
        aspect="auto",
        title=title,
    )
    fig.update_layout(
        height=max(220, 38 * len(mat) + 120),
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_colorbar=dict(title="Score"),
    )
    return fig


def delta_diverging_bar(finetuned: pd.DataFrame, score_prefix: str, title: str) -> go.Figure:
    """Diverging horizontal bar: Δ score vs baseline, per dimension per experiment."""
    rows = []
    for _, r in finetuned.iterrows():
        for col in SCORE_COLS:
            key = f"{score_prefix}_mean_delta_{col}" if score_prefix != "combined" else None
            if score_prefix == "combined":
                # Compute combined delta on the fly from j1/j2 deltas average
                d1 = r.get(f"j1_mean_delta_{col}")
                d2 = r.get(f"j2_mean_delta_{col}")
                delta = (
                    (d1 + d2) / 2 if d1 is not None and d2 is not None else None
                )
            else:
                delta = r.get(key)
            if delta is not None:
                rows.append(
                    {
                        "Experiment": r["model_label"],
                        "Dimension": SCORE_LABELS[col],
                        "Δ vs baseline": delta,
                    }
                )
    if not rows:
        return None
    long_df = pd.DataFrame(rows)
    fig = px.bar(
        long_df,
        x="Δ vs baseline",
        y="Dimension",
        color="Experiment",
        orientation="h",
        barmode="group",
        title=title,
        text_auto=".2f",
    )
    fig.update_layout(
        height=max(260, 60 * len(long_df["Dimension"].unique()) + 120),
        xaxis=dict(title="Δ score (positive = finetuned better)"),
        yaxis=dict(title=""),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#666")
    return fig


def flip_rate_bar(flip_df: pd.DataFrame) -> go.Figure:
    """Grouped bar: x = experiment, colour = judge, faceted by dimension."""
    fig = px.bar(
        flip_df,
        x="Experiment",
        y="Flip %",
        color="Judge",
        barmode="group",
        facet_col="Dimension",
        text_auto=".1f",
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=60, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_yaxes(range=[0, max(25, flip_df["Flip %"].max() * 1.2)])
    # Threshold bands: green <10, yellow 10-20, red >20
    # NOTE: add_hrect appends " domain" to xref internally — pass the bare
    # axis ref (e.g. "x2"), not "x2 domain", or you get "x2 domain domain".
    for ax_idx in range(len(flip_df["Dimension"].unique())):
        suffix = "" if ax_idx == 0 else str(ax_idx + 1)
        fig.add_hrect(
            y0=0, y1=10, fillcolor="#2ca02c", opacity=0.07, line_width=0,
            xref=f"x{suffix}", yref=f"y{suffix}",
        )
        fig.add_hrect(
            y0=10, y1=20, fillcolor="#ff9f0a", opacity=0.07, line_width=0,
            xref=f"x{suffix}", yref=f"y{suffix}",
        )
        fig.add_hrect(
            y0=20, y1=100, fillcolor="#d62728", opacity=0.07, line_width=0,
            xref=f"x{suffix}", yref=f"y{suffix}",
        )
    return fig


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Eval Dashboard", page_icon="📊", layout="wide")
st.title("Evaluation Dashboard")
st.caption(
    "Auto-refreshes every 2 minutes — click Refresh to force a reload. "
    "All views respect the experiment filter in the sidebar."
)

if st.button("🔄 Refresh"):
    st.cache_data.clear()

df = load_all_experiments()

if df.empty:
    st.warning("No experiments found in blob storage.")
    st.stop()

experiments = df["model_label"].tolist()
st.sidebar.header("Experiments")
st.sidebar.write(f"**{len(experiments)}** experiments found")
selected = st.sidebar.multiselect(
    "Select experiments to compare",
    experiments,
    default=experiments,
)
df_sel = df[df["model_label"].isin(selected)].copy()
finetuned = df_sel[df_sel["model_label"] != "baseline"].copy()

if df_sel.empty:
    st.info("Select at least one experiment.")
    st.stop()

# Sidebar: judge focus + score view toggle
st.sidebar.header("View options")
score_view = st.sidebar.radio(
    "Score view",
    options=["Combined (J1+J2 avg)", "Per-judge bars", "Per-judge heatmap"],
    index=0,
)
show_raw = st.sidebar.checkbox("Show raw aggregate JSON", value=False)


# ── Headline KPIs (top of page, always visible) ──────────────────────────────

kpi_cols = st.columns(min(4, max(1, len(df_sel))))
for i, (_, r) in enumerate(df_sel.iterrows()):
    with kpi_cols[i % len(kpi_cols)]:
        # Combined macro score = mean of all 5 combined dimension means
        means = [
            r.get(f"combined_mean_{c}") for c in SCORE_COLS
            if r.get(f"combined_mean_{c}") is not None
        ]
        macro = round(sum(means) / len(means), 2) if means else None

        # Best/worst headline numbers
        st.metric(
            label=f"**{r['model_label']}** (n={int(r.get('n_samples', 0))})",
            value=f"{macro:.2f} / 5" if macro is not None else "—",
        )
        # Pairwise quality vs baseline (averaged across judges)
        if r["model_label"] != "baseline":
            wins = []
            for prefix, _ in JUDGES:
                w = r.get(f"{prefix}_pairwise_quality_win")
                t = r.get(f"{prefix}_pairwise_quality_tie")
                l = r.get(f"{prefix}_pairwise_quality_loss")
                if w is not None and (w + t + l) > 0:
                    wins.append(w / (w + t + l))
            if wins:
                avg_win = sum(wins) / len(wins) * 100
                st.caption(f"Pairwise quality win rate: **{avg_win:.0f}%** (avg of judges)")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_scores, tab_pairwise, tab_flip, tab_lang, tab_agree, tab_detail = st.tabs([
    "📊 Scores",
    "⚔️ Pairwise (W/T/L)",
    "🎯 Flip rate",
    "🌐 Language mixing",
    "🤝 Inter-judge agreement",
    "🔍 Experiment detail",
])


# ── Tab 1: Scores ─────────────────────────────────────────────────────────────

with tab_scores:
    st.subheader("Mean scores per dimension")
    if score_view == "Combined (J1+J2 avg)":
        fig = grouped_score_bars(df_sel, "combined_mean", "Combined mean (J1 + J2 averaged)")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        # Heatmap below for quick scanning
        h = score_heatmap(df_sel, "combined_mean", "Heatmap — combined mean (1=red, 5=green)")
        if h is not None:
            st.plotly_chart(h, use_container_width=True)
    elif score_view == "Per-judge bars":
        c1, c2 = st.columns(2)
        with c1:
            f1 = grouped_score_bars(df_sel, "j1_mean", "Judge 1 — mean scores")
            if f1 is not None:
                st.plotly_chart(f1, use_container_width=True)
        with c2:
            f2 = grouped_score_bars(df_sel, "j2_mean", "Judge 2 — mean scores")
            if f2 is not None:
                st.plotly_chart(f2, use_container_width=True)
    else:  # heatmap
        c1, c2 = st.columns(2)
        with c1:
            h1 = score_heatmap(df_sel, "j1_mean", "Judge 1 — mean scores")
            if h1 is not None:
                st.plotly_chart(h1, use_container_width=True)
        with c2:
            h2 = score_heatmap(df_sel, "j2_mean", "Judge 2 — mean scores")
            if h2 is not None:
                st.plotly_chart(h2, use_container_width=True)

    st.divider()
    st.subheader("Δ vs baseline (positive = finetuned better)")
    if not finetuned.empty:
        if score_view == "Combined (J1+J2 avg)":
            fig = delta_diverging_bar(finetuned, "combined", "Combined Δ vs baseline")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                f1 = delta_diverging_bar(finetuned, "j1", "Judge 1 — Δ vs baseline")
                if f1 is not None:
                    st.plotly_chart(f1, use_container_width=True)
            with c2:
                f2 = delta_diverging_bar(finetuned, "j2", "Judge 2 — Δ vs baseline")
                if f2 is not None:
                    st.plotly_chart(f2, use_container_width=True)
    else:
        st.info("Δ vs baseline requires at least one finetuned experiment in the selection.")


# ── Tab 2: Pairwise W/T/L ────────────────────────────────────────────────────

with tab_pairwise:
    st.subheader("Win / Tie / Loss vs baseline")
    st.caption(
        "Two-run protocol: each pair is judged twice with A/B positions swapped. "
        "A verdict only counts as a win when both runs agree; otherwise the pair "
        "is resolved as a tie. The dotted line marks 50% — break-even."
    )

    if finetuned.empty:
        st.info("No pairwise results — only baseline is selected.")
    else:
        for dim, dim_label in DIMENSIONS:
            pw_data = []
            for _, row in finetuned.iterrows():
                for prefix, jlabel in [("j1", "J1"), ("j2", "J2")]:
                    w = row.get(f"{prefix}_{dim}_win")
                    t = row.get(f"{prefix}_{dim}_tie")
                    l = row.get(f"{prefix}_{dim}_loss")
                    if w is None:
                        continue
                    total = (w + t + l) or 1
                    pw_data.append(
                        {
                            "Experiment": row["model_label"],
                            "Judge": jlabel,
                            "Win %": round(w / total * 100, 1),
                            "Tie %": round(t / total * 100, 1),
                            "Loss %": round(l / total * 100, 1),
                            "Win": int(w),
                            "Tie": int(t),
                            "Loss": int(l),
                            "Flip %": (
                                round(row.get(f"{prefix}_{dim}_flip_rate", 0) * 100, 1)
                                if row.get(f"{prefix}_{dim}_flip_rate") is not None
                                else None
                            ),
                        }
                    )
            if not pw_data:
                continue
            pw_df = pd.DataFrame(pw_data)
            st.plotly_chart(horizontal_wtl_bar(pw_df, dim_label), use_container_width=True)

            with st.expander(f"Show counts table — {dim_label}"):
                st.dataframe(
                    pw_df.set_index(["Experiment", "Judge"]),
                    use_container_width=True,
                )


# ── Tab 3: Flip rate ──────────────────────────────────────────────────────────

with tab_flip:
    st.subheader("Position-bias flip rate")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("✅ **< 10 %** — solid")
        st.caption("Judge is largely deciding on content.")
    with c2:
        st.markdown("⚠️ **10–20 %** — acceptable but noisy")
        st.caption("Borderline cases dominate; treat small win-rate gaps as ties.")
    with c3:
        st.markdown("❌ **> 20 %** — unreliable")
        st.caption("Position is driving the verdict; do not trust pairwise on this dimension.")

    st.markdown(
        """
        **Is 10 % high?** It is *the upper edge of what most evaluation papers call
        acceptable* (MT-Bench / AlpacaEval typically report 5–15 % for strong judges).
        For a Dutch fine-tuning study with subtle stylistic differences, 10 % is
        normal but not great — half of your wins could be flipped if you re-rolled.
        Tighten it by:

        1. **Stronger / larger judge** (GPT-5 / Claude Opus / Grok-4).
           Smaller judges flip 2-3× more often.
        2. **Pairwise prompt hygiene** — explicitly tell the judge that *position
           is not informative*, ask it to think about A and B in *both* orders
           internally before answering, and force a `tie` when the gap is small.
        3. **Wider tie band** in the rubric (e.g. *"call it a tie when the
           differences are stylistic only"*) — a lot of flips are 'photo-finish'
           pairs that should never have been a non-tie.
        4. **Three-run majority vote** instead of two-run AND — more API cost,
           but cuts the flip rate roughly in half on borderline pairs.
        5. **Reasoning trace** — switch the judge to a reasoning model (`o3`,
           `o4-mini`, or `gpt-5` with reasoning) so the verdict is grounded in
           an internal chain of thought rather than a vibe call.
        """
    )

    if finetuned.empty:
        st.info("Select at least one finetuned experiment to see flip rates.")
    else:
        flip_data = []
        for _, row in finetuned.iterrows():
            for prefix, jlabel in [("j1", "J1"), ("j2", "J2")]:
                for dim, dim_label in DIMENSIONS:
                    flip = row.get(f"{prefix}_{dim}_flip_rate")
                    if flip is not None:
                        flip_data.append(
                            {
                                "Experiment": row["model_label"],
                                "Judge": jlabel,
                                "Dimension": dim_label,
                                "Flip %": round(flip * 100, 1),
                            }
                        )
        if flip_data:
            flip_df = pd.DataFrame(flip_data)
            st.plotly_chart(flip_rate_bar(flip_df), use_container_width=True)

            # Compact pivoted table: rows = (Experiment, Dimension), cols = judges
            pivot = flip_df.pivot_table(
                index=["Experiment", "Dimension"], columns="Judge", values="Flip %"
            )

            def _color(v):
                if pd.isna(v):
                    return ""
                if v < 10:
                    return "background-color:#d4f5d4"
                if v < 20:
                    return "background-color:#fde7c4"
                return "background-color:#f7c1c1"

            st.dataframe(pivot.style.map(_color).format("{:.1f}%"), use_container_width=True)
        else:
            st.info("Flip rate not available — re-run pairwise eval with the swap protocol.")


# ── Tab 4: Language mixing ────────────────────────────────────────────────────

with tab_lang:
    st.subheader("Language mixing rate")
    st.caption("Fraction of responses where the judge detected non-Dutch tokens (lower = better).")

    lm_rows = []
    for _, row in df_sel.iterrows():
        for prefix, jlabel in JUDGES:
            rate = row.get(f"{prefix}_language_mixing_rate")
            if rate is not None:
                lm_rows.append(
                    {
                        "Experiment": row["model_label"],
                        "Judge": jlabel,
                        "Rate %": round(rate * 100, 1),
                    }
                )
    if lm_rows:
        lm_df = pd.DataFrame(lm_rows)
        fig = px.bar(
            lm_df,
            x="Experiment",
            y="Rate %",
            color="Judge",
            barmode="group",
            text_auto=".1f",
        )
        fig.update_layout(
            height=380,
            yaxis=dict(title="% rows with mixing"),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=10, r=10, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 5: Inter-judge agreement ──────────────────────────────────────────────

with tab_agree:
    st.subheader("Inter-judge agreement")
    st.caption(
        "Cohen's κ on a 1-5 ordinal scale (quadratic-weighted). "
        "Rule of thumb: κ > 0.8 excellent, 0.6–0.8 substantial, "
        "0.4–0.6 moderate, < 0.4 weak."
    )
    rows = []
    for _, r in df_sel.iterrows():
        for col in SCORE_COLS:
            k = r.get(f"kappa_{col}")
            a = r.get(f"agreement_{col}")
            w1 = r.get(f"within_1_{col}")
            if k is not None:
                rows.append(
                    {
                        "Experiment": r["model_label"],
                        "Dimension": SCORE_LABELS[col],
                        "Cohen's κ": k,
                        "Exact agree %": round(a * 100, 1) if a is not None else None,
                        "Within-1 %": round(w1 * 100, 1) if w1 is not None else None,
                    }
                )
        for dim, dlabel in DIMENSIONS:
            k = r.get(f"kappa_{dim}")
            a = r.get(f"agreement_{dim}")
            if k is not None:
                rows.append(
                    {
                        "Experiment": r["model_label"],
                        "Dimension": f"Pairwise — {dlabel}",
                        "Cohen's κ": k,
                        "Exact agree %": round(a * 100, 1) if a is not None else None,
                        "Within-1 %": None,
                    }
                )
    if rows:
        agree_df = pd.DataFrame(rows)
        kappa_pivot = agree_df.pivot_table(
            index="Experiment", columns="Dimension", values="Cohen's κ"
        )
        fig = px.imshow(
            kappa_pivot.values,
            x=list(kappa_pivot.columns),
            y=list(kappa_pivot.index),
            color_continuous_scale="RdYlGn",
            zmin=-0.1,
            zmax=1.0,
            text_auto=".2f",
            aspect="auto",
            title="Cohen's κ — judges agreement per dimension",
        )
        fig.update_layout(
            height=max(220, 38 * len(kappa_pivot) + 100),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show full agreement table (κ, exact %, within-1 %)"):
            st.dataframe(
                agree_df.set_index(["Experiment", "Dimension"]),
                use_container_width=True,
            )


# ── Tab 6: Experiment detail ──────────────────────────────────────────────────

with tab_detail:
    st.subheader("Experiment detail")
    detail_exp = st.selectbox("Select experiment", experiments)
    if detail_exp:
        detail_row = df[df["model_label"] == detail_exp].iloc[0]

        # Compact KPI strip for the selected experiment
        st.markdown(f"### {detail_exp}")
        cols = st.columns(5)
        for i, c in enumerate(SCORE_COLS):
            v = detail_row.get(f"combined_mean_{c}")
            d_vals = [
                detail_row.get(f"j1_mean_delta_{c}"),
                detail_row.get(f"j2_mean_delta_{c}"),
            ]
            d_vals = [x for x in d_vals if x is not None]
            delta = round(sum(d_vals) / len(d_vals), 2) if d_vals else None
            cols[i].metric(
                label=SCORE_LABELS[c],
                value=f"{v:.2f}" if v is not None else "—",
                delta=f"{delta:+.2f}" if delta is not None else None,
            )

        if show_raw:
            st.json(detail_row.dropna().to_dict())
