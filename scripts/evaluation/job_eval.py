"""Azure ML job entry point for model evaluation (baseline vs finetuned).

This script runs inside an Azure ML job and orchestrates the evaluation pipeline:
1. Optionally runs baseline absolute evaluation (or loads cached scores)
2. Runs finetuned model absolute evaluation
3. Runs pairwise comparison between baseline and finetuned
4. Aggregates metrics and generates visualizations
5. Uploads results to blob storage and logs to MLflow

Invoked by: submit_eval_job.py

Environment variables (from .env or passed to job):
    ENDPOINT: Azure OpenAI endpoint
    JUDGE_LLM_1: First judge model deployment name
    JUDGE_LLM_2: Second judge model deployment name
"""

import argparse
import asyncio
import json
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
from finetuning.eval_visualization import generate_charts
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STORAGE_ACCOUNT = "llmaml5615532443"
CONTAINER = "azureml-blobstore-4c704101-7a51-4680-bcf8-f13966bf69b4"
INFERENCE_PREFIX = "inference-results"
EVAL_PREFIX = "eval-results"


def parse_args() -> argparse.Namespace:
    """Parse Azure ML job arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate inference results with AI judges (Azure ML job)"
    )
    parser.add_argument(
        "--model-label",
        type=str,
        required=True,
        help="Model label to evaluate (e.g., 'mistral_r16_a16_e1_b16_w30')",
    )
    parser.add_argument(
        "--baseline-eval",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Whether to run baseline evaluation (true) or use cached baseline scores (false)",
    )
    parser.add_argument(
        "--qlora-config",
        type=str,
        required=True,
        help="Path to qlora_config.json (for judge endpoints and settings)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent judge API calls (default: 4)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Optional: evaluate only N samples. Skips blob save and MLflow.",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Skip logging to MLflow",
    )
    parser.add_argument(
        "--storage-account",
        type=str,
        default=STORAGE_ACCOUNT,
        help="Azure Storage account name",
    )
    parser.add_argument(
        "--container",
        type=str,
        default=CONTAINER,
        help="Azure Storage container name",
    )
    return parser.parse_args()


async def main() -> None:
    """Orchestrate evaluation job: baseline (if requested) + finetuned + pairwise."""
    args = parse_args()

    # Load configuration
    config = load_config(args.qlora_config)

    # Get Azure OpenAI credentials from environment
    azure_endpoint = os.environ.get("ENDPOINT")
    judge_llm_1 = os.environ.get("JUDGE_LLM_1")
    judge_llm_2 = os.environ.get("JUDGE_LLM_2")

    if not (azure_endpoint and judge_llm_1 and judge_llm_2):
        logger.warning(
            "Missing judge credentials. Set ENDPOINT, JUDGE_LLM_1, JUDGE_LLM_2 in .env"
        )
        if not azure_endpoint:
            raise RuntimeError("ENDPOINT environment variable not set")
        if not judge_llm_1:
            raise RuntimeError("JUDGE_LLM_1 environment variable not set")
        if not judge_llm_2:
            raise RuntimeError("JUDGE_LLM_2 environment variable not set")

    experiment_name = config.get("wandb", {}).get("project", "dutch-mistral")
    model_label = args.model_label
    baseline_eval_flag = args.baseline_eval.lower() == "true"
    dry_run = args.test_size

    storage = args.storage_account
    container = args.container
    eval_prefix = f"{EVAL_PREFIX}/{model_label}"
    baseline_prefix = f"{EVAL_PREFIX}/baseline"

    judges = [("j1", judge_llm_1), ("j2", judge_llm_2)]
    logger.info("═" * 70)
    logger.info("Azure ML Evaluation Job")
    logger.info("═" * 70)
    logger.info(f"Model label: {model_label}")
    logger.info(f"Baseline evaluation: {baseline_eval_flag}")
    logger.info(f"Dry-run mode: {dry_run is not None} (samples: {dry_run})")
    logger.info(f"Judge 1: {judge_llm_1} | Judge 2: {judge_llm_2}")

    # Load inference results for finetuned model
    logger.info("")
    logger.info("▶ Loading finetuned model inference results ...")
    df = load_inference_results(
        model_label,
        storage,
        container,
        INFERENCE_PREFIX,
    )
    logger.info(f"  Loaded {len(df)} inference results")

    if dry_run is not None:
        df = df.head(dry_run)
        logger.info(f"  Subsampled to {len(df)} rows for dry-run")

    # Initialize judge client
    client = build_judge_client(azure_endpoint)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Baseline evaluation (conditional based on --baseline-eval flag)
    # ─────────────────────────────────────────────────────────────────────────
    df_baseline_scores = None
    df_baseline = None

    if baseline_eval_flag:
        logger.info("")
        logger.info("▶ Phase 1: Running baseline model evaluation ...")
        logger.info("  Loading baseline inference results ...")
        df_baseline = load_inference_results(
            "baseline",
            storage,
            container,
            INFERENCE_PREFIX,
        )
        logger.info(f"  Loaded {len(df_baseline)} baseline inference results")

        if dry_run is not None:
            df_baseline = df_baseline.head(dry_run)
            logger.info(f"  Subsampled to {len(df_baseline)} rows")

        logger.info("  Running absolute evaluation for baseline ...")
        df_baseline_scores = await run_absolute_evaluation(
            df_baseline, client, judges, args.max_workers
        )
        logger.info(f"  Evaluated {len(df_baseline_scores)} baseline samples")

        if dry_run is None:
            save_row_scores(df_baseline_scores, storage, container, baseline_prefix)
            logger.info(f"  Saved baseline scores to blob: {baseline_prefix}")
    else:
        logger.info("")
        logger.info("▶ Phase 1: Loading cached baseline scores ...")
        try:
            df_baseline_scores = load_row_scores(storage, container, baseline_prefix)
            logger.info(f"  Loaded cached baseline scores ({len(df_baseline_scores)} rows)")

            # Also need baseline inference results for pairwise comparison
            logger.info("  Loading baseline inference results ...")
            df_baseline = load_inference_results(
                "baseline",
                storage,
                container,
                INFERENCE_PREFIX,
            )
            logger.info(f"  Loaded {len(df_baseline)} baseline inference results")

            if dry_run is not None:
                df_baseline_scores = df_baseline_scores.head(dry_run)
                df_baseline = df_baseline.head(dry_run)
                logger.info(f"  Subsampled to {len(df_baseline_scores)} rows")

        except FileNotFoundError:
            logger.error(
                f"Baseline scores not cached at {baseline_prefix}. "
                "Re-run with --baseline-eval true or provide cached scores."
            )
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Finetuned model absolute evaluation
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("▶ Phase 2: Running finetuned model absolute evaluation ...")
    df_scores = await run_absolute_evaluation(df, client, judges, args.max_workers)
    logger.info(f"  Evaluated {len(df_scores)} finetuned samples")

    if dry_run is None:
        save_row_scores(df_scores, storage, container, eval_prefix)
        logger.info(f"  Saved finetuned scores to blob: {eval_prefix}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Pairwise comparison (baseline vs finetuned)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("▶ Phase 3: Running pairwise comparison (baseline vs finetuned) ...")
    df_pairwise = await run_pairwise_evaluation(
        df_baseline, df, client, judges, args.max_workers
    )
    logger.info(f"  Evaluated {len(df_pairwise)} pairwise comparisons")

    if dry_run is None:
        save_row_scores(df_pairwise, storage, container, eval_prefix, "pairwise.jsonl")
        logger.info(f"  Saved pairwise scores to blob: {eval_prefix}/pairwise.jsonl")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4: Aggregate metrics and generate visualizations
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("▶ Phase 4: Computing aggregates and generating visualizations ...")
    agg = compute_aggregate(df_scores, model_label, df_pairwise, df_baseline_scores)
    logger.info("  Computed per-judge metrics and inter-judge agreement")

    if dry_run is None:
        save_aggregate(agg, storage, container, eval_prefix)
        logger.info(f"  Saved aggregate metrics to blob: {eval_prefix}/aggregate.json")
    
    print_summary(agg)

    # Generate charts
    if dry_run is None:
        with tempfile.TemporaryDirectory() as charts_dir:
            charts_path = Path(charts_dir)
            generate_charts(
                agg, df_scores, charts_path, df_baseline_scores, df_pairwise
            )
            save_charts_to_blob(charts_path, storage, container, eval_prefix)
            logger.info(f"  Generated and saved charts to blob: {eval_prefix}/charts")
    else:
        # Save dry-run results locally for analysis
        dry_dir = _PROJECT_ROOT / "outputs" / "eval_dry_run"
        dry_dir.mkdir(parents=True, exist_ok=True)
        scores_path = dry_dir / f"{model_label}_row_scores.jsonl"
        df_scores.to_json(
            scores_path, orient="records", lines=True, force_ascii=False
        )
        logger.info(f"  Dry-run row scores: {scores_path}")

        if df_pairwise is not None:
            pairwise_path = dry_dir / f"{model_label}_pairwise.jsonl"
            df_pairwise.to_json(
                pairwise_path, orient="records", lines=True, force_ascii=False
            )
            logger.info(f"  Dry-run pairwise: {pairwise_path}")

        agg_path = dry_dir / f"{model_label}_aggregate.json"
        agg_path.write_text(json.dumps(agg, indent=2))
        logger.info(f"  Dry-run aggregate: {agg_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 5: Log to MLflow
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_mlflow and dry_run is None:
        logger.info("")
        logger.info("▶ Phase 5: Logging to MLflow ...")
        log_to_mlflow(agg, model_label, experiment_name, storage, container, eval_prefix)
        logger.info(f"  Logged metrics to experiment: {experiment_name}")

    logger.info("")
    logger.info("═" * 70)
    logger.info("✓ Evaluation job complete!")
    logger.info("═" * 70)
    logger.info(f"Results location: {storage}/{container}/{eval_prefix}")


if __name__ == "__main__":
    asyncio.run(main())
