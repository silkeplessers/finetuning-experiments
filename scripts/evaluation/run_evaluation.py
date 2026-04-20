"""Evaluate inference results using Azure AI Foundry judge LLMs.

Both JUDGE_LLM_1 and JUDGE_LLM_2 score every row independently (2 API calls
per row per judge for absolute scoring, 1 per row per judge for pairwise).
Results are stored with j1_/j2_ prefixes and aggregated per judge plus combined
with inter-judge agreement (Cohen's Kappa).

Baseline row-level scores are cached in blob storage and reused across
experiments. Only pairwise comparison is re-run each time.

Endpoint and judge model deployments are read from .env:
    ENDPOINT, JUDGE_LLM_1, JUDGE_LLM_2

Usage:
    python scripts/evaluation/run_evaluation.py \
        --config configs/qlora_config.json \
        --model-label baseline

    python scripts/evaluation/run_evaluation.py \
        --config configs/qlora_config.json \
        --model-label mistral_r16_a16_e1_b16_w30
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")

from finetuning.config import load_config
from finetuning.evaluation import (
    build_judge_client,
    compute_aggregate,
    load_inference_results,
    load_row_scores,
    log_to_mlflow,
    print_summary,
    run_absolute_evaluation,
    run_pairwise_evaluation,
    save_aggregate,
    save_charts_to_blob,
    save_row_scores,
)
from finetuning.eval_visualization import generate_charts

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STORAGE_ACCOUNT = "llmaml5615532443"
INFERENCE_CONTAINER = "inference-results"
EVAL_CONTAINER = "azureml-blobstore-4c704101-7a51-4680-bcf8-f13966bf69b4"
EVAL_BLOB_PREFIX = "eval-results"


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate inference results with AI judges"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to qlora_config.json"
    )
    parser.add_argument(
        "--model-label",
        type=str,
        required=True,
        help="Model label to evaluate (e.g. 'baseline' or the wandb run_name)",
    )
    parser.add_argument(
        "--local-results",
        type=str,
        default=None,
        help="Optional local JSONL file instead of downloading from blob",
    )
    parser.add_argument("--storage-account", type=str, default=STORAGE_ACCOUNT)
    parser.add_argument("--inference-container", type=str, default=INFERENCE_CONTAINER)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent judge API calls (default: 4)",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Skip logging to MLflow",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    azure_endpoint = os.environ["ENDPOINT"]
    judge_llm_1 = os.environ["JUDGE_LLM_1"]
    judge_llm_2 = os.environ["JUDGE_LLM_2"]
    experiment_name = config["wandb"]["project"]
    is_baseline = args.model_label == "baseline"

    storage = args.storage_account
    eval_container = EVAL_CONTAINER
    eval_prefix = f"{EVAL_BLOB_PREFIX}/{args.model_label}"
    baseline_prefix = f"{EVAL_BLOB_PREFIX}/baseline"

    # Load inference results
    logger.info("Loading inference results for model: %s", args.model_label)
    df = load_inference_results(
        args.model_label,
        storage,
        args.inference_container,
        args.local_results,
    )
    logger.info("Loaded %d inference results", len(df))

    client = build_judge_client(azure_endpoint)
    judges = [("j1", judge_llm_1), ("j2", judge_llm_2)]
    logger.info("Judge 1: %s | Judge 2: %s", judge_llm_1, judge_llm_2)

    # ── Absolute scoring (both judges) ────────────────────────────────────
    # For baseline: reuse cached scores if they exist in blob
    df_scores = load_row_scores(storage, eval_container, eval_prefix) if is_baseline else None

    if df_scores is None:
        df_scores = df.copy()
        for judge_label, judge_model in judges:
            logger.info("Running absolute evaluation [%s=%s] for %s ...", judge_label, judge_model, args.model_label)
            df_scores = run_absolute_evaluation(
                df_scores, client, judge_model, judge_label, args.max_workers
            )
        save_row_scores(df_scores, storage, eval_container, eval_prefix)
    else:
        logger.info("Reusing cached row scores from blob")

    # ── Pairwise comparison (finetuned only, both judges) ─────────────────
    df_pairwise = None
    df_baseline_scores = None

    if not is_baseline:
        logger.info("Loading baseline inference results for pairwise comparison ...")
        df_baseline = load_inference_results("baseline", storage, args.inference_container)
        df_baseline_scores = load_row_scores(storage, eval_container, baseline_prefix)

        if df_baseline_scores is None:
            logger.info("Baseline row scores not cached — running baseline evaluation first ...")
            df_baseline_scores = df_baseline.copy()
            for judge_label, judge_model in judges:
                logger.info("Running absolute evaluation [%s=%s] for baseline ...", judge_label, judge_model)
                df_baseline_scores = run_absolute_evaluation(
                    df_baseline_scores, client, judge_model, judge_label, args.max_workers
                )
            save_row_scores(df_baseline_scores, storage, eval_container, baseline_prefix)

        # Run pairwise for each judge, merging columns into one DF
        df_pairwise = df_baseline[["input", "expected_output"]].copy()
        df_pairwise["baseline_output"] = df_baseline["predicted_output"].values
        df_pairwise["finetuned_output"] = df["predicted_output"].values

        for judge_label, judge_model in judges:
            logger.info("Running pairwise evaluation [%s=%s] ...", judge_label, judge_model)
            df_pw = run_pairwise_evaluation(
                df_baseline, df, client, judge_model, judge_label, args.max_workers
            )
            # Merge judge-specific pairwise columns into the combined DF
            for col in df_pw.columns:
                if col.startswith(judge_label):
                    df_pairwise[col] = df_pw[col].values

        save_row_scores(df_pairwise, storage, eval_container, eval_prefix, "pairwise.jsonl")

    # ── Aggregate metrics ─────────────────────────────────────────────────
    agg = compute_aggregate(df_scores, args.model_label, df_pairwise, df_baseline_scores)
    save_aggregate(agg, storage, eval_container, eval_prefix)
    print_summary(agg)

    # ── Charts ────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as charts_dir:
        charts_path = Path(charts_dir)
        generate_charts(agg, df_scores, charts_path, df_baseline_scores, df_pairwise)
        save_charts_to_blob(charts_path, storage, eval_container, eval_prefix)

    # ── MLflow ────────────────────────────────────────────────────────────
    if not args.skip_mlflow:
        log_to_mlflow(agg, args.model_label, experiment_name, storage, eval_container, eval_prefix)
        logger.info("Logged to MLflow experiment: %s", experiment_name)

    logger.info("Evaluation complete. Results in blob: %s/%s", eval_container, eval_prefix)


if __name__ == "__main__":
    main()
