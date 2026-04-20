"""Evaluation helpers for scoring inference results with AI judge models.

Design:
  - Both JUDGE_LLM_1 and JUDGE_LLM_2 score every row (columns prefixed j1_/j2_).
  - 2 API calls per row per judge: dutch_quality (merged) + instruction_following.
  - 1 pairwise API call per row per judge (quality + instruction in one prompt).
  - Baseline row-level scores are cached and reused across experiments.
  - Aggregates are computed per judge, then combined with inter-judge agreement
    (Cohen's Kappa and simple agreement rate).
  - MLflow is used as the experiment tracker: one MLflow run per evaluation.
"""

import json
import logging
import random
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from finetuning.blob_storage import download_blob_directory, download_blob_file, upload_directory_to_blob, upload_file_to_blob
from finetuning.judge_prompts import (
    DUTCH_QUALITY_SYSTEM,
    INSTRUCTION_FOLLOWING_SYSTEM,
    PAIRWISE_SYSTEM,
)

logger = logging.getLogger(__name__)

# Column keys produced by each judge call
QUALITY_KEYS = [
    "grammar_score", "grammar_justification",
    "fluency_score", "fluency_justification",
    "vocabulary_score", "vocabulary_justification",
    "language_mixing", "language_mixing_examples",
]
INSTRUCTION_KEYS = ["instruction_following_score", "instruction_following_justification"]
PAIRWISE_KEYS = [
    "quality_winner", "quality_justification",
    "instruction_winner", "instruction_justification",
]
SCORE_COLS = ["grammar_score", "fluency_score", "vocabulary_score", "instruction_following_score"]

# ── Client builder ────────────────────────────────────────────────────────────


def build_judge_client(azure_endpoint: str):
    """Create an Azure OpenAI client using Entra ID (DefaultAzureCredential)."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    azure_endpoint = azure_endpoint.split("/openai/")[0].rstrip("/")

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2025-01-01-preview",
    )


# ── Response parsing ─────────────────────────────────────────────────────────


def _parse_judge_response(text: str, expected_keys: list[str]) -> dict:
    """Best-effort extraction of judge JSON from LLM output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {k: (None if "score" in k else text.strip()) for k in expected_keys}


def _judge_call(client, model: str, system: str, user_msg: str, expected_keys: list[str]) -> dict:
    """Generic single judge API call."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )
    return _parse_judge_response(response.choices[0].message.content, expected_keys)


# ── Judge calls (2 per row for absolute, 1 for pairwise) ─────────────────────


def evaluate_dutch_quality(client, model: str, prompt: str, response_text: str) -> dict:
    """Single call returning grammar, fluency, vocabulary, and language mixing."""
    user_msg = f"Prompt:\n{prompt}\n\nModel response:\n{response_text}"
    return _judge_call(client, model, DUTCH_QUALITY_SYSTEM, user_msg, QUALITY_KEYS)


def evaluate_instruction_following(client, model: str, prompt: str, expected: str, response_text: str) -> dict:
    user_msg = (
        f"Prompt:\n{prompt}\n\n"
        f"Expected response:\n{expected}\n\n"
        f"Model response:\n{response_text}"
    )
    return _judge_call(client, model, INSTRUCTION_FOLLOWING_SYSTEM, user_msg, INSTRUCTION_KEYS)


def evaluate_row(client, model: str, prompt: str, expected: str, response_text: str) -> dict:
    """2 API calls: dutch_quality + instruction_following."""
    result = {}
    result.update(evaluate_dutch_quality(client, model, prompt, response_text))
    result.update(evaluate_instruction_following(client, model, prompt, expected, response_text))
    return result


# ── Pairwise ──────────────────────────────────────────────────────────────────


def _build_pairwise_msg(
    prompt: str, response_a: str, response_b: str, expected: str,
) -> tuple[str, bool]:
    """Build pairwise user message. Returns (msg, swapped).

    swapped=True means A=finetuned, B=baseline (position randomised).
    """
    swapped = random.random() < 0.5
    if swapped:
        a_text, b_text = response_b, response_a
    else:
        a_text, b_text = response_a, response_b
    parts = [
        f"Prompt:\n{prompt}",
        f"Expected response:\n{expected}",
        f"Response A:\n{a_text}",
        f"Response B:\n{b_text}",
    ]
    return "\n\n".join(parts), swapped


def _map_winner(raw_winner: str, swapped: bool) -> str:
    """Map A/B/tie back to baseline/finetuned/tie."""
    w = raw_winner.strip().upper()
    if w == "TIE":
        return "tie"
    if (w == "A" and not swapped) or (w == "B" and swapped):
        return "baseline"
    return "finetuned"


def evaluate_pairwise(
    client, model: str, prompt: str, expected: str,
    baseline_text: str, finetuned_text: str,
) -> dict:
    """Single pairwise call returning quality_winner + instruction_winner."""
    user_msg, swapped = _build_pairwise_msg(prompt, baseline_text, finetuned_text, expected)
    raw = _judge_call(client, model, PAIRWISE_SYSTEM, user_msg, PAIRWISE_KEYS)
    return {
        "pairwise_quality_winner": _map_winner(str(raw.get("quality_winner", "")), swapped),
        "pairwise_quality_justification": raw.get("quality_justification", ""),
        "pairwise_instruction_winner": _map_winner(str(raw.get("instruction_winner", "")), swapped),
        "pairwise_instruction_justification": raw.get("instruction_justification", ""),
    }


# ── Data loading ──────────────────────────────────────────────────────────────


def load_inference_results(
    model_label: str,
    storage_account: str,
    results_container: str,
    local_path: str | None = None,
) -> pd.DataFrame:
    """Load inference results from blob storage (or a local file override)."""
    if local_path:
        return pd.read_json(local_path, lines=True)

    blob_prefix = f"{model_label}/inference_results.jsonl"
    with tempfile.TemporaryDirectory() as tmp_dir:
        download_blob_directory(
            storage_account, results_container, blob_prefix, tmp_dir
        )
        downloaded = Path(tmp_dir) / "inference_results.jsonl"
        if not downloaded.exists():
            candidates = list(Path(tmp_dir).rglob("*.jsonl"))
            if not candidates:
                raise FileNotFoundError(f"No JSONL found after downloading {blob_prefix}")
            downloaded = candidates[0]
        return pd.read_json(downloaded, lines=True)


# ── Concurrent runners ───────────────────────────────────────────────────────


def _run_concurrent(func, items: list, max_workers: int, label: str) -> list:
    """Run func(idx, item) concurrently and return ordered results."""
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, i, item): i for i, item in enumerate(items)}
        done = 0
        total = len(futures)
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            done += 1
            if done % 10 == 0 or done == total:
                logger.info("[%s] %d/%d", label, done, total)
    return results


def _prefix_dict(d: dict, prefix: str) -> dict:
    """Add a prefix to all dictionary keys."""
    return {f"{prefix}{k}": v for k, v in d.items()}


def run_absolute_evaluation(
    df: pd.DataFrame,
    client,
    judge_model: str,
    judge_label: str,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Run absolute scoring with a single judge. Columns are prefixed with judge_label (e.g. j1_)."""
    rows = list(df.iterrows())

    def _eval(i, item):
        _, row = item
        result = evaluate_row(client, judge_model, row["input"], row["expected_output"], row["predicted_output"])
        return i, result

    results = _run_concurrent(_eval, rows, max_workers, f"absolute-{judge_label}")

    df = df.copy()
    all_keys = QUALITY_KEYS + INSTRUCTION_KEYS
    for key in all_keys:
        df[f"{judge_label}_{key}"] = [r.get(key) for r in results]
    return df


def run_pairwise_evaluation(
    df_baseline: pd.DataFrame,
    df_finetuned: pd.DataFrame,
    client,
    judge_model: str,
    judge_label: str,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Run pairwise comparison with a single judge. Columns prefixed with judge_label."""
    pairs = list(zip(df_baseline.iterrows(), df_finetuned.iterrows()))

    def _eval(i, item):
        (_, b_row), (_, f_row) = item
        result = evaluate_pairwise(
            client, judge_model, b_row["input"], b_row["expected_output"],
            b_row["predicted_output"], f_row["predicted_output"],
        )
        return i, result

    results = _run_concurrent(_eval, pairs, max_workers, f"pairwise-{judge_label}")

    df = df_baseline[["input", "expected_output"]].copy()
    df["baseline_output"] = df_baseline["predicted_output"].values
    df["finetuned_output"] = df_finetuned["predicted_output"].values
    pairwise_cols = [
        "pairwise_quality_winner", "pairwise_quality_justification",
        "pairwise_instruction_winner", "pairwise_instruction_justification",
    ]
    for key in pairwise_cols:
        df[f"{judge_label}_{key}"] = [r.get(key) for r in results]
    return df


# ── Inter-judge agreement ────────────────────────────────────────────────────


def _cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Compute Cohen's Kappa for two lists of categorical labels."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    categories = sorted(set(labels_a) | set(labels_b))
    if len(categories) <= 1:
        return 1.0

    # Observed agreement
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n

    # Expected agreement
    p_e = 0.0
    for cat in categories:
        p_a = sum(1 for x in labels_a if x == cat) / n
        p_b = sum(1 for x in labels_b if x == cat) / n
        p_e += p_a * p_b

    if p_e == 1.0:
        return 1.0
    return round((p_o - p_e) / (1 - p_e), 4)


def _agreement_rate(labels_a: list, labels_b: list) -> float:
    """Simple agreement rate (fraction of exact matches)."""
    if not labels_a:
        return 0.0
    return round(sum(1 for a, b in zip(labels_a, labels_b) if a == b) / len(labels_a), 4)


# ── Aggregate metrics ─────────────────────────────────────────────────────────


def _compute_judge_aggregate(
    df_scores: pd.DataFrame,
    prefix: str,
    df_baseline_scores: pd.DataFrame | None = None,
    baseline_prefix: str | None = None,
    df_pairwise: pd.DataFrame | None = None,
) -> dict:
    """Compute aggregate for a single judge (identified by prefix, e.g. 'j1')."""
    n = len(df_scores)
    agg: dict = {}

    # Mean absolute scores
    for col in SCORE_COLS:
        pcol = f"{prefix}_{col}"
        series = pd.to_numeric(df_scores[pcol], errors="coerce").dropna()
        agg[f"{prefix}_mean_{col}"] = round(float(series.mean()), 3) if len(series) > 0 else None

    # Language mixing rate
    lm_col = f"{prefix}_language_mixing"
    lm = df_scores[lm_col].apply(lambda x: x is True or str(x).lower() == "true")
    agg[f"{prefix}_language_mixing_rate"] = round(float(lm.mean()), 3)

    # Score deltas vs baseline
    if df_baseline_scores is not None and baseline_prefix and len(df_baseline_scores) == n:
        for col in SCORE_COLS:
            ft = pd.to_numeric(df_scores[f"{prefix}_{col}"], errors="coerce")
            bl = pd.to_numeric(df_baseline_scores[f"{baseline_prefix}_{col}"], errors="coerce")
            delta = (ft - bl).dropna()
            agg[f"{prefix}_mean_delta_{col}"] = round(float(delta.mean()), 3) if len(delta) > 0 else None

        bl_lm = df_baseline_scores[f"{baseline_prefix}_language_mixing"].apply(
            lambda x: x is True or str(x).lower() == "true"
        )
        agg[f"{prefix}_baseline_language_mixing_rate"] = round(float(bl_lm.mean()), 3)
        agg[f"{prefix}_delta_language_mixing_rate"] = round(
            agg[f"{prefix}_language_mixing_rate"] - agg[f"{prefix}_baseline_language_mixing_rate"], 3
        )

    # Pairwise win/tie/loss
    if df_pairwise is not None:
        for dim in ["pairwise_quality_winner", "pairwise_instruction_winner"]:
            pcol = f"{prefix}_{dim}"
            counts = df_pairwise[pcol].value_counts()
            short = dim.replace("_winner", "")
            agg[f"{prefix}_{short}_win"] = int(counts.get("finetuned", 0))
            agg[f"{prefix}_{short}_tie"] = int(counts.get("tie", 0))
            agg[f"{prefix}_{short}_loss"] = int(counts.get("baseline", 0))
            agg[f"{prefix}_{short}_win_rate"] = round(agg[f"{prefix}_{short}_win"] / n, 3) if n > 0 else None

    return agg


def compute_aggregate(
    df_scores: pd.DataFrame,
    model_label: str,
    df_pairwise: pd.DataFrame | None = None,
    df_baseline_scores: pd.DataFrame | None = None,
) -> dict:
    """Compute per-judge aggregates + combined inter-judge agreement metrics."""
    n = len(df_scores)
    agg: dict = {"model_label": model_label, "n_samples": n}

    # Per-judge aggregates
    baseline_prefix = "j1" if df_baseline_scores is not None else None
    for prefix in ["j1", "j2"]:
        bp = prefix if df_baseline_scores is not None else None
        agg.update(_compute_judge_aggregate(
            df_scores, prefix, df_baseline_scores, bp, df_pairwise,
        ))

    # Combined mean (average of both judges)
    for col in SCORE_COLS:
        j1 = agg.get(f"j1_mean_{col}")
        j2 = agg.get(f"j2_mean_{col}")
        if j1 is not None and j2 is not None:
            agg[f"combined_mean_{col}"] = round((j1 + j2) / 2, 3)

    # Inter-judge agreement on absolute scores (binned to same integer)
    for col in SCORE_COLS:
        j1_vals = pd.to_numeric(df_scores[f"j1_{col}"], errors="coerce").dropna().astype(int).tolist()
        j2_vals = pd.to_numeric(df_scores[f"j2_{col}"], errors="coerce").dropna().astype(int).tolist()
        min_len = min(len(j1_vals), len(j2_vals))
        if min_len > 0:
            agg[f"agreement_{col}"] = _agreement_rate(j1_vals[:min_len], j2_vals[:min_len])
            agg[f"kappa_{col}"] = _cohens_kappa(j1_vals[:min_len], j2_vals[:min_len])

    # Inter-judge agreement on language mixing (boolean)
    j1_lm = df_scores["j1_language_mixing"].apply(lambda x: x is True or str(x).lower() == "true").tolist()
    j2_lm = df_scores["j2_language_mixing"].apply(lambda x: x is True or str(x).lower() == "true").tolist()
    agg["agreement_language_mixing"] = _agreement_rate(j1_lm, j2_lm)
    agg["kappa_language_mixing"] = _cohens_kappa(
        [str(x) for x in j1_lm], [str(x) for x in j2_lm]
    )

    # Inter-judge agreement on pairwise winners
    if df_pairwise is not None:
        for dim in ["pairwise_quality_winner", "pairwise_instruction_winner"]:
            j1_w = df_pairwise[f"j1_{dim}"].tolist()
            j2_w = df_pairwise[f"j2_{dim}"].tolist()
            short = dim.replace("_winner", "")
            agg[f"agreement_{short}"] = _agreement_rate(j1_w, j2_w)
            agg[f"kappa_{short}"] = _cohens_kappa(j1_w, j2_w)

    return agg


# ── Persistence helpers (blob storage) ────────────────────────────────────────


def save_row_scores(
    df: pd.DataFrame,
    storage_account: str,
    container: str,
    blob_prefix: str,
    filename: str = "row_scores.jsonl",
) -> None:
    """Save row-level scores to blob storage."""
    import tempfile

    blob_name = f"{blob_prefix}/{filename}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        df.to_json(f, orient="records", lines=True, force_ascii=False)
        tmp_path = f.name
    upload_file_to_blob(storage_account, container, blob_name, tmp_path)
    Path(tmp_path).unlink()
    logger.info("Saved row scores to blob: %s/%s", container, blob_name)


def load_row_scores(
    storage_account: str,
    container: str,
    blob_prefix: str,
    filename: str = "row_scores.jsonl",
) -> pd.DataFrame | None:
    """Load cached row-level scores from blob, or None if not found."""
    import tempfile

    blob_name = f"{blob_prefix}/{filename}"
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
    found = download_blob_file(storage_account, container, blob_name, tmp_path)
    if found:
        logger.info("Loaded cached row scores from blob: %s/%s", container, blob_name)
        df = pd.read_json(tmp_path, lines=True)
        Path(tmp_path).unlink()
        return df
    Path(tmp_path).unlink(missing_ok=True)
    return None


def save_aggregate(
    agg: dict,
    storage_account: str,
    container: str,
    blob_prefix: str,
    filename: str = "aggregate.json",
) -> None:
    """Save aggregate metrics to blob storage."""
    import tempfile

    blob_name = f"{blob_prefix}/{filename}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(agg, f, indent=2)
        tmp_path = f.name
    upload_file_to_blob(storage_account, container, blob_name, tmp_path)
    Path(tmp_path).unlink()
    logger.info("Saved aggregate metrics to blob: %s/%s", container, blob_name)


def save_charts_to_blob(
    charts_dir: Path,
    storage_account: str,
    container: str,
    blob_prefix: str,
) -> None:
    """Upload all chart files from a local directory to blob."""
    if charts_dir.exists():
        upload_directory_to_blob(
            storage_account, container, f"{blob_prefix}/charts", str(charts_dir)
        )


# ── MLflow logging ────────────────────────────────────────────────────────────


def log_to_mlflow(
    agg: dict,
    model_label: str,
    experiment_name: str,
    storage_account: str,
    container: str,
    blob_prefix: str,
) -> None:
    """Log aggregate metrics, params, and artifacts to MLflow."""
    import mlflow

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"eval-{model_label}"):
        # Log all aggregate metrics
        metrics = {k: v for k, v in agg.items() if isinstance(v, (int, float)) and v is not None}
        mlflow.log_metrics(metrics)

        # Log model label and sample count as params
        mlflow.log_param("model_label", model_label)
        mlflow.log_param("n_samples", agg.get("n_samples"))

        # Download eval results from blob to a temp dir, then log as artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                download_blob_directory(storage_account, container, blob_prefix, tmp_dir)
                for f in Path(tmp_dir).rglob("*"):
                    if f.is_file():
                        mlflow.log_artifact(str(f), artifact_path=str(f.relative_to(tmp_dir).parent))
            except FileNotFoundError:
                logger.warning("No blob artifacts found under %s to log to MLflow", blob_prefix)


def print_summary(agg: dict) -> None:
    """Log a human-readable summary of aggregate metrics."""
    label = agg.get("model_label", "?")
    lines = [f"=== Evaluation summary: {label} (n={agg.get('n_samples', '?')}) ==="]

    # Per-judge scores
    for prefix, name in [("j1", "Judge 1"), ("j2", "Judge 2")]:
        lines.append(f"\n  --- {name} ---")
        for col in SCORE_COLS:
            val = agg.get(f"{prefix}_mean_{col}")
            delta = agg.get(f"{prefix}_mean_delta_{col}")
            line = f"    {col}: {val:.2f}" if val is not None else f"    {col}: N/A"
            if delta is not None:
                line += f"  (Δ {delta:+.2f})"
            lines.append(line)
        lm = agg.get(f"{prefix}_language_mixing_rate")
        lines.append(f"    language_mixing_rate: {lm}")
        for dim in ["pairwise_quality", "pairwise_instruction"]:
            w = agg.get(f"{prefix}_{dim}_win")
            if w is not None:
                t = agg.get(f"{prefix}_{dim}_tie", 0)
                l = agg.get(f"{prefix}_{dim}_loss", 0)
                wr = agg.get(f"{prefix}_{dim}_win_rate", 0)
                lines.append(f"    {dim}: W={w} / T={t} / L={l}  (win_rate={wr:.1%})")

    # Combined + agreement
    lines.append("\n  --- Combined ---")
    for col in SCORE_COLS:
        val = agg.get(f"combined_mean_{col}")
        line = f"    {col}: {val:.2f}" if val is not None else f"    {col}: N/A"
        agree = agg.get(f"agreement_{col}")
        kappa = agg.get(f"kappa_{col}")
        if agree is not None:
            line += f"  (agree={agree:.1%}, κ={kappa:.3f})"
        lines.append(line)

    agree_lm = agg.get("agreement_language_mixing")
    kappa_lm = agg.get("kappa_language_mixing")
    if agree_lm is not None:
        lines.append(f"    language_mixing agreement={agree_lm:.1%}, κ={kappa_lm:.3f}")

    for dim in ["pairwise_quality", "pairwise_instruction"]:
        agree = agg.get(f"agreement_{dim}")
        kappa = agg.get(f"kappa_{dim}")
        if agree is not None:
            lines.append(f"    {dim} agreement={agree:.1%}, κ={kappa:.3f}")

    logger.info("\n".join(lines))
