"""Streamlit dashboard for cross-experiment evaluation comparison.

Reads all aggregate.json files from blob storage under eval-results/ and
renders an auto-refreshing dashboard with comparison charts and tables.

Usage:
    streamlit run scripts/evaluation/dashboard.py
"""

import sys
from pathlib import Path

import pandas as pd
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
]
SCORE_LABELS = {
    "grammar_score": "Grammar",
    "fluency_score": "Fluency",
    "vocabulary_score": "Vocabulary",
    "instruction_following_score": "Instruction Following",
}

# ── Data loading (cached with TTL for auto-refresh) ──────────────────────────


@st.cache_data(ttl=120)
def load_all_experiments() -> pd.DataFrame:
    """Scan blob storage for all aggregate.json files and return a DataFrame."""
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
    # Put baseline first, then sort the rest alphabetically
    df["_sort"] = df["model_label"].apply(
        lambda x: (0, x) if x == "baseline" else (1, x)
    )
    df = df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return df


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Eval Dashboard", page_icon="📊", layout="wide")
st.title("Evaluation Dashboard")
st.caption("Auto-refreshes every 2 minutes. Click the button below to refresh now.")

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

if df_sel.empty:
    st.info("Select at least one experiment.")
    st.stop()


# ── Section 1: Summary table ─────────────────────────────────────────────────

st.header("Summary")

summary_cols = ["model_label", "n_samples"]
for prefix in ["j1", "j2"]:
    for col in SCORE_COLS:
        summary_cols.append(f"{prefix}_mean_{col}")
    summary_cols.append(f"{prefix}_language_mixing_rate")
for col in SCORE_COLS:
    summary_cols.append(f"combined_mean_{col}")

available = [c for c in summary_cols if c in df_sel.columns]
st.dataframe(
    df_sel[available].set_index("model_label"),
    use_container_width=True,
)


# ── Section 2: Combined mean scores comparison ───────────────────────────────

st.header("Combined Mean Scores")

combined_cols = [
    f"combined_mean_{c}" for c in SCORE_COLS if f"combined_mean_{c}" in df_sel.columns
]
if combined_cols:
    chart_df = df_sel.set_index("model_label")[combined_cols].rename(
        columns={f"combined_mean_{c}": SCORE_LABELS[c] for c in SCORE_COLS}
    )
    st.bar_chart(chart_df, height=400)
else:
    st.info("Combined scores not available (need both judges).")


# ── Section 3: Per-judge scores ───────────────────────────────────────────────

st.header("Per-Judge Scores")
col_j1, col_j2 = st.columns(2)

for col_container, prefix, label in [
    (col_j1, "j1", "Judge 1"),
    (col_j2, "j2", "Judge 2"),
]:
    with col_container:
        st.subheader(label)
        judge_cols = [
            f"{prefix}_mean_{c}"
            for c in SCORE_COLS
            if f"{prefix}_mean_{c}" in df_sel.columns
        ]
        if judge_cols:
            jdf = df_sel.set_index("model_label")[judge_cols].rename(
                columns={f"{prefix}_mean_{c}": SCORE_LABELS[c] for c in SCORE_COLS}
            )
            st.bar_chart(jdf, height=350)


# ── Section 4: Score deltas vs baseline ───────────────────────────────────────

st.header("Score Deltas vs Baseline")

finetuned = df_sel[df_sel["model_label"] != "baseline"]
if not finetuned.empty:
    for prefix, label in [
        ("j1", "Judge 1"),
        ("j2", "Judge 2"),
        ("combined", "Combined"),
    ]:
        if prefix == "combined":
            delta_cols = {f"combined_mean_{c}": SCORE_LABELS[c] for c in SCORE_COLS}
            # Compute deltas from baseline combined
            baseline_row = df_sel[df_sel["model_label"] == "baseline"]
            if baseline_row.empty:
                continue
            bl_vals = {
                SCORE_LABELS[c]: baseline_row.iloc[0].get(f"combined_mean_{c}")
                for c in SCORE_COLS
            }
            delta_df = finetuned.set_index("model_label")[
                [f"combined_mean_{c}" for c in SCORE_COLS]
            ].rename(
                columns={f"combined_mean_{c}": SCORE_LABELS[c] for c in SCORE_COLS}
            )
            for col_name, bl_val in bl_vals.items():
                if bl_val is not None and col_name in delta_df.columns:
                    delta_df[col_name] = delta_df[col_name] - bl_val
            st.subheader(f"Δ {label}")
            st.bar_chart(delta_df, height=300)
        else:
            delta_cols_list = [
                f"{prefix}_mean_delta_{c}"
                for c in SCORE_COLS
                if f"{prefix}_mean_delta_{c}" in finetuned.columns
            ]
            if delta_cols_list:
                st.subheader(f"Δ {label}")
                ddf = finetuned.set_index("model_label")[delta_cols_list].rename(
                    columns={
                        f"{prefix}_mean_delta_{c}": SCORE_LABELS[c] for c in SCORE_COLS
                    }
                )
                st.bar_chart(ddf, height=300)
else:
    st.info("No finetuned experiments to compare against baseline.")


# ── Section 5: Pairwise win rates ────────────────────────────────────────────

st.header("Pairwise Win Rates")

pw_experiments = finetuned
if not pw_experiments.empty:
    for dim, dim_label in [
        ("pairwise_quality", "Language Quality"),
        ("pairwise_instruction", "Instruction Following"),
    ]:
        st.subheader(dim_label)
        pw_data = []
        for _, row in pw_experiments.iterrows():
            for prefix, jlabel in [("j1", "J1"), ("j2", "J2")]:
                w = row.get(f"{prefix}_{dim}_win")
                t = row.get(f"{prefix}_{dim}_tie")
                l = row.get(f"{prefix}_{dim}_loss")
                if w is not None:
                    total = w + t + l
                    pw_data.append(
                        {
                            "Experiment": row["model_label"],
                            "Judge": jlabel,
                            "Win %": round(w / total * 100, 1) if total else 0,
                            "Tie %": round(t / total * 100, 1) if total else 0,
                            "Loss %": round(l / total * 100, 1) if total else 0,
                            "Win": int(w),
                            "Tie": int(t),
                            "Loss": int(l),
                        }
                    )
        if pw_data:
            pw_df = pd.DataFrame(pw_data)
            st.dataframe(
                pw_df.set_index(["Experiment", "Judge"]), use_container_width=True
            )

            # Win rate chart
            win_chart = pw_df.pivot(index="Experiment", columns="Judge", values="Win %")
            st.bar_chart(win_chart, height=300)
else:
    st.info("No pairwise results (baseline-only).")


# ── Section 6: Language mixing rates ──────────────────────────────────────────

st.header("Language Mixing Rate")

lm_data = []
for _, row in df_sel.iterrows():
    for prefix, jlabel in [("j1", "Judge 1"), ("j2", "Judge 2")]:
        rate = row.get(f"{prefix}_language_mixing_rate")
        if rate is not None:
            lm_data.append(
                {
                    "Experiment": row["model_label"],
                    "Judge": jlabel,
                    "Rate (%)": round(rate * 100, 1),
                }
            )
if lm_data:
    lm_df = pd.DataFrame(lm_data)
    lm_chart = lm_df.pivot(index="Experiment", columns="Judge", values="Rate (%)")
    st.bar_chart(lm_chart, height=300)


# ── Section 7: Inter-judge agreement ─────────────────────────────────────────

st.header("Inter-Judge Agreement")

agreement_dims = SCORE_COLS + ["language_mixing"]
pw_agreement_dims = ["pairwise_quality", "pairwise_instruction"]

agree_data = []
for _, row in df_sel.iterrows():
    entry = {"Experiment": row["model_label"]}
    for dim in agreement_dims:
        entry[f"{SCORE_LABELS.get(dim, dim.replace('_', ' ').title())} (agree)"] = (
            row.get(f"agreement_{dim}")
        )
        entry[f"{SCORE_LABELS.get(dim, dim.replace('_', ' ').title())} (κ)"] = row.get(
            f"kappa_{dim}"
        )
    for dim in pw_agreement_dims:
        label = dim.replace("_", " ").title()
        entry[f"{label} (agree)"] = row.get(f"agreement_{dim}")
        entry[f"{label} (κ)"] = row.get(f"kappa_{dim}")
    agree_data.append(entry)

if agree_data:
    agree_df = pd.DataFrame(agree_data).set_index("Experiment")
    # Filter to only columns that have data
    agree_df = agree_df.dropna(axis=1, how="all")
    if not agree_df.empty:
        st.dataframe(agree_df, use_container_width=True)


# ── Section 8: Experiment detail drill-down ───────────────────────────────────

st.header("Experiment Detail")

detail_exp = st.selectbox("Select experiment", experiments)
if detail_exp:
    detail_row = df[df["model_label"] == detail_exp].iloc[0]
    st.json(detail_row.dropna().to_dict())
