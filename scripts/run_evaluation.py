"""Evaluate inference results using Azure AI Foundry judge LLMs.

Assesses:
  1. Dutch language quality  (grammar, fluency, vocabulary)
  2. Instruction following    (faithfulness to expected output)

Usage:
    python scripts/run_evaluation.py \
        --config configs/qlora_config.json \
        --model-label baseline \
        --azure-endpoint https://finetuning-foundry.openai.azure.com \
        --judge-model grok-4-fast-reasoning

    # Or evaluate the finetuned run:
    python scripts/run_evaluation.py \
        --config configs/qlora_config.json \
        --model-label run_r16_a16_e1_b16-TEST \
        --azure-endpoint https://finetuning-foundry.openai.azure.com \
        --judge-model grok-4-fast-reasoning
"""

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from finetuning.blob_storage import download_blob_directory, upload_file_to_blob
from finetuning.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STORAGE_ACCOUNT = "llmaml5615532443"
RESULTS_CONTAINER = "inference-results"

# ── Judge system prompts ──────────────────────────────────────────────────────

DUTCH_QUALITY_SYSTEM = """\
You are an expert evaluator of Dutch language quality.
You will receive a prompt (in Dutch) and an AI-generated response.
Evaluate ONLY the Dutch language quality of the response.

Scoring rubric (1-5):
  1 - Very poor: major grammar errors, largely incomprehensible or not Dutch.
  2 - Poor: frequent grammar mistakes, unnatural phrasing.
  3 - Acceptable: understandable but contains noticeable errors.
  4 - Good: mostly fluent with only minor mistakes.
  5 - Excellent: fluent, grammatically correct, natural vocabulary.

Reply with ONLY a JSON object (no markdown fences):
{"score": <int 1-5>, "justification": "<one-sentence explanation>"}"""

INSTRUCTION_FOLLOWING_SYSTEM = """\
You are an expert evaluator assessing how well an AI model follows instructions.
You will receive the original prompt, the expected (reference) answer, and the model's actual response.
Evaluate how faithfully and completely the response follows the instruction compared to the reference.

Scoring rubric (1-5):
  1 - Completely irrelevant or fails to address the instruction.
  2 - Partially addresses the instruction but misses key elements.
  3 - Addresses the instruction with notable omissions or inaccuracies.
  4 - Follows instructions well with only minor deviations.
  5 - Perfectly follows instructions; comprehensive and accurate.

Reply with ONLY a JSON object (no markdown fences):
{"score": <int 1-5>, "justification": "<one-sentence explanation>"}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_judge_client(azure_endpoint: str):
    """Create an Azure OpenAI client using Entra ID (DefaultAzureCredential)."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2025-01-01-preview",
    )


def _parse_judge_response(text: str) -> dict:
    """Best-effort extraction of {"score": int, "justification": str} from LLM output."""
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to regex extraction
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"score": None, "justification": text.strip()}


def judge_single(client, model: str, system_prompt: str, user_message: str) -> dict:
    """Call the judge LLM and parse the structured response."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return _parse_judge_response(response.choices[0].message.content)


def evaluate_dutch_quality(client, model: str, prompt: str, response_text: str) -> dict:
    user_msg = f"Prompt:\n{prompt}\n\nResponse:\n{response_text}"
    return judge_single(client, model, DUTCH_QUALITY_SYSTEM, user_msg)


def evaluate_instruction_following(
    client, model: str, prompt: str, expected: str, response_text: str,
) -> dict:
    user_msg = (
        f"Prompt:\n{prompt}\n\n"
        f"Expected response:\n{expected}\n\n"
        f"Model response:\n{response_text}"
    )
    return judge_single(client, model, INSTRUCTION_FOLLOWING_SYSTEM, user_msg)


# ── Main logic ────────────────────────────────────────────────────────────────

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
        download_blob_directory(storage_account, results_container, blob_prefix, tmp_dir)
        downloaded = Path(tmp_dir) / "inference_results.jsonl"
        if not downloaded.exists():
            # Blob might have been downloaded flat (prefix = full blob name)
            candidates = list(Path(tmp_dir).rglob("*.jsonl"))
            if not candidates:
                raise FileNotFoundError(f"No JSONL found after downloading {blob_prefix}")
            downloaded = candidates[0]
        return pd.read_json(downloaded, lines=True)


def run_evaluation(
    df: pd.DataFrame,
    client,
    judge_model: str,
    request_delay: float = 0.5,
) -> pd.DataFrame:
    """Run both evaluation criteria on every row and return an enriched DataFrame."""
    dutch_scores, dutch_justifications = [], []
    instr_scores, instr_justifications = [], []

    total = len(df)
    for idx, row in df.iterrows():
        logger.info("Evaluating %d/%d ...", idx + 1, total)

        # --- Dutch quality ---
        dq = evaluate_dutch_quality(client, judge_model, row["input"], row["predicted_output"])
        dutch_scores.append(dq.get("score"))
        dutch_justifications.append(dq.get("justification", ""))

        time.sleep(request_delay)

        # --- Instruction following ---
        inf = evaluate_instruction_following(
            client, judge_model, row["input"], row["expected_output"], row["predicted_output"],
        )
        instr_scores.append(inf.get("score"))
        instr_justifications.append(inf.get("justification", ""))

        time.sleep(request_delay)

    df = df.copy()
    df["dutch_quality_score"] = dutch_scores
    df["dutch_quality_justification"] = dutch_justifications
    df["instruction_following_score"] = instr_scores
    df["instruction_following_justification"] = instr_justifications
    return df


def print_summary(df: pd.DataFrame, model_label: str) -> None:
    dq_mean = df["dutch_quality_score"].dropna().mean()
    inf_mean = df["instruction_following_score"].dropna().mean()
    logger.info(
        "=== %s === Dutch quality: %.2f | Instruction following: %.2f  (n=%d)",
        model_label, dq_mean, inf_mean, len(df),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inference results with an AI judge")
    parser.add_argument("--config", type=str, required=True, help="Path to qlora_config.json")
    parser.add_argument(
        "--model-label", type=str, required=True,
        help="Model label to evaluate (e.g. 'baseline' or the wandb run_name)",
    )
    parser.add_argument("--azure-endpoint", type=str, required=True, help="Azure OpenAI endpoint URL")
    parser.add_argument("--judge-model", type=str, default="grok-4-fast-reasoning", help="Deployment name of the judge model")
    parser.add_argument("--local-results", type=str, default=None, help="Optional local JSONL file instead of downloading from blob")
    parser.add_argument("--storage-account", type=str, default=STORAGE_ACCOUNT)
    parser.add_argument("--results-container", type=str, default=RESULTS_CONTAINER)
    parser.add_argument("--request-delay", type=float, default=0.5, help="Seconds between judge API calls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = load_config(args.config)  # validate config exists

    # Load inference results
    logger.info("Loading inference results for model: %s", args.model_label)
    df = load_inference_results(
        args.model_label, args.storage_account, args.results_container, args.local_results,
    )
    logger.info("Loaded %d inference results", len(df))

    # Build judge client
    client = build_judge_client(args.azure_endpoint)

    # Run evaluation
    eval_df = run_evaluation(df, client, args.judge_model, args.request_delay)

    # Print summary
    print_summary(eval_df, args.model_label)

    # Upload evaluation results to blob
    blob_name = f"{args.model_label}/evaluation_results.jsonl"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
        eval_df.to_json(f, orient="records", lines=True)

    try:
        url = upload_file_to_blob(args.storage_account, args.results_container, blob_name, tmp_path)
        logger.info("Evaluation results uploaded: %s", url)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
