"""Evaluation helpers for scoring inference results with AI judge models.

Design:
  - Both JUDGE_LLM_1 and JUDGE_LLM_2 score every row (columns prefixed j1_/j2_).
  - 3 API calls per row per judge: dutch_quality (merged) + instruction_following + correctness.
  - 4 pairwise API calls per row per judge: 2 dimensions × 2 position orderings
    (two-run swap protocol). Verdicts only count when both orderings agree;
    otherwise the pair is resolved as a tie. The flip rate (fraction of pairs
    where the two orderings disagree on a non-tie winner) is logged per judge
    as a direct measurement of position bias.
  - All judge calls use structured outputs (Pydantic response models) — no regex parsing.
  - Both judges run in parallel (row × judge tasks submitted to a single thread pool).
  - Baseline row-level scores are cached and reused across experiments.
  - Aggregates are computed per judge, then combined with inter-judge agreement
    (Cohen's Kappa via sklearn and simple agreement rate).
  - MLflow is used as the experiment tracker: one MLflow run per evaluation.
"""

import asyncio
import json
import logging
import re
import tempfile
from pathlib import Path

import pandas as pd
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from sklearn.metrics import cohen_kappa_score
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from finetuning.blob_storage import (
    download_blob_directory,
    download_blob_file,
    upload_directory_to_blob,
    upload_file_to_blob,
)
from finetuning.judge_prompts import (
    CORRECTNESS_SYSTEM,
    DUTCH_QUALITY_SYSTEM,
    INSTRUCTION_FOLLOWING_SYSTEM,
    PAIRWISE_INSTRUCTION_SYSTEM,
    PAIRWISE_QUALITY_SYSTEM,
)
from finetuning.schemas import (
    CorrectnessResult,
    DutchQualityResult,
    InstructionFollowingResult,
    PairwiseSingleResult,
)

logger = logging.getLogger(__name__)

SCORE_COLS = [
    "grammar_score",
    "fluency_score",
    "vocabulary_score",
    "instruction_following_score",
    "correctness_score",
]

# ── Client builder ────────────────────────────────────────────────────────────


def build_judge_client(azure_endpoint: str):
    """Create an async Azure OpenAI client using Entra ID (DefaultAzureCredential)."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AsyncAzureOpenAI

    azure_endpoint = azure_endpoint.split("/openai/")[0].rstrip("/")

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2025-01-01-preview",
    )


# ── Structured judge calls ───────────────────────────────────────────────────


@retry(
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    ),
    wait=wait_random_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _judge_call(client, model: str, system: str, user_msg: str, response_format):
    """Single async judge API call with structured output. Returns a Pydantic model instance.

    Retries on rate limits and transient errors with exponential backoff (2s -> 60s,
    up to 8 attempts ~ several minutes of waiting before giving up).
    """
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content or ""
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Judge response did not contain JSON: {content!r}")

    json_text = content[start : end + 1]
    return response_format.model_validate_json(json_text)


async def evaluate_dutch_quality(
    client, model: str, prompt: str, response_text: str
) -> DutchQualityResult:
    """Single call returning grammar, fluency, vocabulary, and language mixing."""
    user_msg = f"Prompt:\n{prompt}\n\nModel response:\n{response_text}"
    return await _judge_call(
        client, model, DUTCH_QUALITY_SYSTEM, user_msg, DutchQualityResult
    )


async def evaluate_instruction_following(
    client, model: str, prompt: str, response_text: str
) -> InstructionFollowingResult:
    user_msg = (
        f"Prompt:\n{prompt}\n\n"
        f"Model response:\n{response_text}"
    )
    return await _judge_call(
        client,
        model,
        INSTRUCTION_FOLLOWING_SYSTEM,
        user_msg,
        InstructionFollowingResult,
    )


async def evaluate_correctness(
    client, model: str, prompt: str, response_text: str
) -> CorrectnessResult:
    user_msg = (
        f"Prompt:\n{prompt}\n\n"
        f"Model response:\n{response_text}"
    )
    return await _judge_call(
        client,
        model,
        CORRECTNESS_SYSTEM,
        user_msg,
        CorrectnessResult,
    )


async def evaluate_row(
    client, model: str, prompt: str, response_text: str
) -> dict:
    """3 API calls in parallel: dutch_quality + instruction_following + correctness.

    Instruction following and correctness are scored by separate judge calls
    (single-focus rubrics) to avoid halo effects between the two dimensions.
    Returns a flat dict merging all three structured responses.
    """
    quality, instruction, correctness = await asyncio.gather(
        evaluate_dutch_quality(client, model, prompt, response_text),
        evaluate_instruction_following(client, model, prompt, response_text),
        evaluate_correctness(client, model, prompt, response_text),
    )
    return {
        **quality.model_dump(),
        **instruction.model_dump(),
        **correctness.model_dump(),
    }


# ── Pairwise ──────────────────────────────────────────────────────────────────


def _build_pairwise_msg(
    prompt: str,
    baseline_text: str,
    finetuned_text: str,
    swapped: bool,
) -> str:
    """Build pairwise user message with an explicit A/B orientation.

    swapped=False -> A=baseline,  B=finetuned
    swapped=True  -> A=finetuned, B=baseline
    """
    if swapped:
        a_text, b_text = finetuned_text, baseline_text
    else:
        a_text, b_text = baseline_text, finetuned_text
    parts = [
        f"Prompt:\n{prompt}",
        f"Response A:\n{a_text}",
        f"Response B:\n{b_text}",
    ]
    return "\n\n".join(parts)


def _map_winner(raw_winner: str, swapped: bool) -> str:
    """Map raw A/B/tie verdict back to baseline/finetuned/tie."""
    w = raw_winner.strip().upper()
    if w == "TIE":
        return "tie"
    if (w == "A" and not swapped) or (w == "B" and swapped):
        return "baseline"
    return "finetuned"


def _resolve_two_run(verdict_ab: str, verdict_ba: str) -> tuple[str, bool]:
    """Combine two-run verdicts using the 'both must agree, else tie' rule.

    Returns (resolved_winner, flipped) where ``flipped`` is True only when
    the two runs picked different non-tie winners (a position-bias flip).
    Disagreements involving a 'tie' verdict are conservatively resolved as
    tie but NOT counted as flips.
    """
    if verdict_ab == verdict_ba:
        return verdict_ab, False
    if verdict_ab == "tie" or verdict_ba == "tie":
        return "tie", False
    # Both non-tie but disagree -> position flip
    return "tie", True


async def _evaluate_pairwise_single(
    client,
    model: str,
    system_prompt: str,
    prompt: str,
    baseline_text: str,
    finetuned_text: str,
    swapped: bool,
) -> tuple[str, str]:
    """One pairwise call for a single dimension at a fixed A/B orientation.

    Returns (winner_in_baseline_finetuned_tie, justification).
    """
    user_msg = _build_pairwise_msg(
        prompt, baseline_text, finetuned_text, swapped
    )
    result = await _judge_call(
        client, model, system_prompt, user_msg, PairwiseSingleResult
    )
    return _map_winner(result.winner.value, swapped), result.justification


async def _evaluate_pairwise_dimension(
    client,
    model: str,
    system_prompt: str,
    prompt: str,
    baseline_text: str,
    finetuned_text: str,
) -> dict:
    """Two-run pairwise for ONE dimension (baseline=A, then finetuned=A).

    Runs both orderings in parallel and resolves with the 'both must agree,
    else tie' rule. Returns a flat dict with the resolved winner, the two raw
    verdicts, both justifications, and a per-row ``flipped`` flag.
    """
    (w_ab, j_ab), (w_ba, j_ba) = await asyncio.gather(
        _evaluate_pairwise_single(
            client,
            model,
            system_prompt,
            prompt,
            baseline_text,
            finetuned_text,
            swapped=False,
        ),
        _evaluate_pairwise_single(
            client,
            model,
            system_prompt,
            prompt,
            baseline_text,
            finetuned_text,
            swapped=True,
        ),
    )
    resolved, flipped = _resolve_two_run(w_ab, w_ba)
    return {
        "winner": resolved,
        "winner_ab": w_ab,
        "winner_ba": w_ba,
        "justification_ab": j_ab,
        "justification_ba": j_ba,
        "flipped": flipped,
    }


async def evaluate_pairwise(
    client,
    model: str,
    prompt: str,
    baseline_text: str,
    finetuned_text: str,
    seed: int = 0,  # kept for backward-compat with run_pairwise_evaluation; unused
) -> dict:
    """Four parallel pairwise calls (2 dimensions × 2 position orderings).

    Each dimension is judged in its own single-focus prompt (avoiding halo
    between dimensions) AND in both A/B orientations (avoiding position bias).
    A verdict counts as a win only when both orderings agree; otherwise the
    pair is resolved as a tie and marked as a position flip.
    """
    del seed  # no longer used; orientation is now forced rather than randomised
    quality_res, instruction_res = await asyncio.gather(
        _evaluate_pairwise_dimension(
            client,
            model,
            PAIRWISE_QUALITY_SYSTEM,
            prompt,
            baseline_text,
            finetuned_text,
        ),
        _evaluate_pairwise_dimension(
            client,
            model,
            PAIRWISE_INSTRUCTION_SYSTEM,
            prompt,
            baseline_text,
            finetuned_text,
        ),
    )
    return {
        "pairwise_quality_winner": quality_res["winner"],
        "pairwise_quality_winner_ab": quality_res["winner_ab"],
        "pairwise_quality_winner_ba": quality_res["winner_ba"],
        "pairwise_quality_justification_ab": quality_res["justification_ab"],
        "pairwise_quality_justification_ba": quality_res["justification_ba"],
        "pairwise_quality_flipped": quality_res["flipped"],
        "pairwise_instruction_winner": instruction_res["winner"],
        "pairwise_instruction_winner_ab": instruction_res["winner_ab"],
        "pairwise_instruction_winner_ba": instruction_res["winner_ba"],
        "pairwise_instruction_justification_ab": instruction_res["justification_ab"],
        "pairwise_instruction_justification_ba": instruction_res["justification_ba"],
        "pairwise_instruction_flipped": instruction_res["flipped"],
    }


# ── Data loading ──────────────────────────────────────────────────────────────


def load_inference_results(
    model_label: str,
    storage_account: str,
    container: str,
    inference_prefix: str = "inference-results",
    local_path: str | None = None,
) -> pd.DataFrame:
    """Load inference results from blob storage (or a local file override).

    Blob path: {inference_prefix}/{model_label}_results.jsonl
    """
    if local_path:
        return pd.read_json(local_path, lines=True)

    blob_name = f"{inference_prefix}/{model_label}_results.jsonl"
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
    found = download_blob_file(storage_account, container, blob_name, tmp_path)
    if not found:
        Path(tmp_path).unlink(missing_ok=True)
        raise FileNotFoundError(
            f"Inference results not found: {container}/{blob_name}"
        )
    df = pd.read_json(tmp_path, lines=True)
    Path(tmp_path).unlink()
    return df


# ── Parallel dual-judge runners ──────────────────────────────────────────────


async def run_absolute_evaluation(
    df: pd.DataFrame,
    client,
    judges: list[tuple[str, str]],
    max_workers: int = 8,
) -> pd.DataFrame:
    """Run absolute scoring with all judges in parallel (async).

    Args:
        judges: list of (judge_label, judge_model) e.g. [("j1", "gpt-5"), ("j2", "grok-4")].
        max_workers: max concurrent API calls (controlled via asyncio.Semaphore).

    Each row's 3 judge calls (dutch_quality + instruction_following + correctness)
    also run in parallel within evaluate_row via asyncio.gather, so effective
    concurrency is up to 3× max_workers API calls in flight.
    """
    rows = list(df.iterrows())
    n_tasks = len(rows) * len(judges)
    sem = asyncio.Semaphore(max_workers)
    done = 0

    results: dict[tuple[int, str], dict] = {}

    async def _eval(row_idx, row, judge_label, judge_model):
        nonlocal done
        async with sem:
            result = await evaluate_row(
                client,
                judge_model,
                row["input"],
                row["predicted_output"],
            )
        results[(row_idx, judge_label)] = result
        done += 1
        if done % 20 == 0 or done == n_tasks:
            logger.info("[absolute] %d/%d", done, n_tasks)

    tasks = [
        _eval(i, row, jl, jm)
        for i, (_, row) in enumerate(rows)
        for jl, jm in judges
    ]
    await asyncio.gather(*tasks)

    # Build output DataFrame
    df = df.copy()
    for jl, _ in judges:
        keys = (
            list(DutchQualityResult.model_fields)
            + list(InstructionFollowingResult.model_fields)
            + list(CorrectnessResult.model_fields)
        )
        for key in keys:
            df[f"{jl}_{key}"] = [results[(i, jl)].get(key) for i in range(len(rows))]
    return df


async def run_pairwise_evaluation(
    df_baseline: pd.DataFrame,
    df_finetuned: pd.DataFrame,
    client,
    judges: list[tuple[str, str]],
    max_workers: int = 8,
) -> pd.DataFrame:
    """Run pairwise comparison with all judges in parallel (async).

    Uses a two-run swap protocol: each (dimension, judge) pair issues 2 calls
    (baseline=A then finetuned=A), and the verdict only counts as a win when
    both orderings agree. Total API calls per row per judge = 2 dimensions ×
    2 orderings = 4.
    """
    pairs = list(zip(df_baseline.iterrows(), df_finetuned.iterrows()))
    n_tasks = len(pairs) * len(judges)
    sem = asyncio.Semaphore(max_workers)
    done = 0

    results: dict[tuple[int, str], dict] = {}

    async def _eval(pair_idx, b_row, f_row, judge_label, judge_model):
        nonlocal done
        async with sem:
            result = await evaluate_pairwise(
                client,
                judge_model,
                b_row["input"],
                b_row["predicted_output"],
                f_row["predicted_output"],
                seed=pair_idx,
            )
        results[(pair_idx, judge_label)] = result
        done += 1
        if done % 20 == 0 or done == n_tasks:
            logger.info("[pairwise] %d/%d", done, n_tasks)

    tasks = [
        _eval(i, b_row, f_row, jl, jm)
        for i, ((_, b_row), (_, f_row)) in enumerate(pairs)
        for jl, jm in judges
    ]
    await asyncio.gather(*tasks)

    # Build output DataFrame
    df = df_baseline[["input"]].copy()
    df["baseline_output"] = df_baseline["predicted_output"].values
    df["finetuned_output"] = df_finetuned["predicted_output"].values
    pairwise_cols = [
        "pairwise_quality_winner",
        "pairwise_quality_winner_ab",
        "pairwise_quality_winner_ba",
        "pairwise_quality_justification_ab",
        "pairwise_quality_justification_ba",
        "pairwise_quality_flipped",
        "pairwise_instruction_winner",
        "pairwise_instruction_winner_ab",
        "pairwise_instruction_winner_ba",
        "pairwise_instruction_justification_ab",
        "pairwise_instruction_justification_ba",
        "pairwise_instruction_flipped",
    ]
    for jl, _ in judges:
        for key in pairwise_cols:
            df[f"{jl}_{key}"] = [results[(i, jl)].get(key) for i in range(len(pairs))]
    return df


# ── Inter-judge agreement ────────────────────────────────────────────────────


def _agreement_rate(labels_a: list, labels_b: list) -> float:
    """Simple agreement rate (fraction of exact matches)."""
    if not labels_a:
        return 0.0
    return round(
        sum(1 for a, b in zip(labels_a, labels_b) if a == b) / len(labels_a), 4
    )


def _safe_kappa(labels_a: list, labels_b: list, weights: str = "quadratic") -> float:
    """Weighted Cohen's Kappa via sklearn, returning 1.0 when all labels agree.

    Uses quadratic weighting by default, which is appropriate for ordinal scales
    (a 1-point difference is penalised much less than a 5-point difference).
    For nominal labels (e.g. pairwise winners), pass weights=None.
    """
    if not labels_a:
        return 0.0
    if len(set(labels_a) | set(labels_b)) <= 1:
        return 1.0
    return round(float(cohen_kappa_score(labels_a, labels_b, weights=weights)), 4)


def _within_n_agreement(labels_a: list, labels_b: list, n: int = 1) -> float:
    """Fraction of pairs where |a - b| <= n (for ordinal scores)."""
    if not labels_a:
        return 0.0
    return round(
        sum(1 for a, b in zip(labels_a, labels_b) if abs(a - b) <= n)
        / len(labels_a),
        4,
    )


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

    for col in SCORE_COLS:
        pcol = f"{prefix}_{col}"
        series = pd.to_numeric(df_scores[pcol], errors="coerce").dropna()
        agg[f"{prefix}_mean_{col}"] = (
            round(float(series.mean()), 3) if len(series) > 0 else None
        )

    lm_col = f"{prefix}_language_mixing"
    lm = df_scores[lm_col].apply(lambda x: x is True or str(x).lower() == "true")
    agg[f"{prefix}_language_mixing_rate"] = round(float(lm.mean()), 3)

    if (
        df_baseline_scores is not None
        and baseline_prefix
        and len(df_baseline_scores) == n
    ):
        for col in SCORE_COLS:
            ft = pd.to_numeric(df_scores[f"{prefix}_{col}"], errors="coerce")
            bl = pd.to_numeric(
                df_baseline_scores[f"{baseline_prefix}_{col}"], errors="coerce"
            )
            delta = (ft - bl).dropna()
            agg[f"{prefix}_mean_delta_{col}"] = (
                round(float(delta.mean()), 3) if len(delta) > 0 else None
            )

        bl_lm = df_baseline_scores[f"{baseline_prefix}_language_mixing"].apply(
            lambda x: x is True or str(x).lower() == "true"
        )
        agg[f"{prefix}_baseline_language_mixing_rate"] = round(float(bl_lm.mean()), 3)
        agg[f"{prefix}_delta_language_mixing_rate"] = round(
            agg[f"{prefix}_language_mixing_rate"]
            - agg[f"{prefix}_baseline_language_mixing_rate"],
            3,
        )

    if df_pairwise is not None:
        for dim in ["pairwise_quality_winner", "pairwise_instruction_winner"]:
            pcol = f"{prefix}_{dim}"
            counts = df_pairwise[pcol].value_counts()
            short = dim.replace("_winner", "")
            agg[f"{prefix}_{short}_win"] = int(counts.get("finetuned", 0))
            agg[f"{prefix}_{short}_tie"] = int(counts.get("tie", 0))
            agg[f"{prefix}_{short}_loss"] = int(counts.get("baseline", 0))
            agg[f"{prefix}_{short}_win_rate"] = (
                round(agg[f"{prefix}_{short}_win"] / n, 3) if n > 0 else None
            )

            # Position-bias flip rate: fraction of pairs where the two A/B
            # orderings disagreed on a non-tie winner (lower is better).
            flip_col = f"{prefix}_{short}_flipped"
            if flip_col in df_pairwise.columns:
                flipped = df_pairwise[flip_col].apply(
                    lambda x: x is True or str(x).lower() == "true"
                )
                agg[f"{prefix}_{short}_flip_rate"] = (
                    round(float(flipped.mean()), 3) if len(flipped) > 0 else None
                )

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

    for prefix in ["j1", "j2"]:
        bp = prefix if df_baseline_scores is not None else None
        agg.update(
            _compute_judge_aggregate(
                df_scores,
                prefix,
                df_baseline_scores,
                bp,
                df_pairwise,
            )
        )

    for col in SCORE_COLS:
        j1 = agg.get(f"j1_mean_{col}")
        j2 = agg.get(f"j2_mean_{col}")
        if j1 is not None and j2 is not None:
            agg[f"combined_mean_{col}"] = round((j1 + j2) / 2, 3)

    # Inter-judge agreement on absolute scores
    for col in SCORE_COLS:
        j1_vals = (
            pd.to_numeric(df_scores[f"j1_{col}"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        j2_vals = (
            pd.to_numeric(df_scores[f"j2_{col}"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        min_len = min(len(j1_vals), len(j2_vals))
        if min_len > 0:
            agg[f"agreement_{col}"] = _agreement_rate(
                j1_vals[:min_len], j2_vals[:min_len]
            )
            agg[f"within_1_{col}"] = _within_n_agreement(
                j1_vals[:min_len], j2_vals[:min_len], n=1
            )
            agg[f"kappa_{col}"] = _safe_kappa(
                j1_vals[:min_len], j2_vals[:min_len], weights="quadratic"
            )

    # Inter-judge agreement on language mixing
    j1_lm = (
        df_scores["j1_language_mixing"]
        .apply(lambda x: x is True or str(x).lower() == "true")
        .tolist()
    )
    j2_lm = (
        df_scores["j2_language_mixing"]
        .apply(lambda x: x is True or str(x).lower() == "true")
        .tolist()
    )
    agg["agreement_language_mixing"] = _agreement_rate(j1_lm, j2_lm)
    agg["kappa_language_mixing"] = _safe_kappa(
        [str(x) for x in j1_lm], [str(x) for x in j2_lm], weights=None
    )

    # Inter-judge agreement on pairwise winners
    if df_pairwise is not None:
        for dim in ["pairwise_quality_winner", "pairwise_instruction_winner"]:
            j1_w = df_pairwise[f"j1_{dim}"].tolist()
            j2_w = df_pairwise[f"j2_{dim}"].tolist()
            short = dim.replace("_winner", "")
            agg[f"agreement_{short}"] = _agreement_rate(j1_w, j2_w)
            agg[f"kappa_{short}"] = _safe_kappa(j1_w, j2_w, weights=None)

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
        metrics = {
            k: v
            for k, v in agg.items()
            if isinstance(v, (int, float)) and v is not None
        }
        mlflow.log_metrics(metrics)

        # Log model label and sample count as params
        mlflow.log_param("model_label", model_label)
        mlflow.log_param("n_samples", agg.get("n_samples"))

        # Download eval results from blob to a temp dir, then log as artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                download_blob_directory(
                    storage_account, container, blob_prefix, tmp_dir
                )
                for f in Path(tmp_dir).rglob("*"):
                    if f.is_file():
                        mlflow.log_artifact(
                            str(f), artifact_path=str(f.relative_to(tmp_dir).parent)
                        )
            except FileNotFoundError:
                logger.warning(
                    "No blob artifacts found under %s to log to MLflow", blob_prefix
                )


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
                line = f"    {dim}: W={w} / T={t} / L={l}  (win_rate={wr:.1%})"
                flip = agg.get(f"{prefix}_{dim}_flip_rate")
                if flip is not None:
                    line += f"  flip_rate={flip:.1%}"
                lines.append(line)

    # Combined + agreement
    lines.append("\n  --- Combined ---")
    for col in SCORE_COLS:
        val = agg.get(f"combined_mean_{col}")
        line = f"    {col}: {val:.2f}" if val is not None else f"    {col}: N/A"
        agree = agg.get(f"agreement_{col}")
        within1 = agg.get(f"within_1_{col}")
        kappa = agg.get(f"kappa_{col}")
        if agree is not None:
            line += f"  (agree={agree:.1%}, within-1={within1:.1%}, κ_w={kappa:.3f})"
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
