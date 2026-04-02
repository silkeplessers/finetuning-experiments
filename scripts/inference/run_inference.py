"""Run inference on the test dataset using the baseline and/or finetuned model.

Results are uploaded to Azure Blob Storage.

Usage:
    # Baseline only
    python scripts/inference/run_inference.py --config configs/qlora_config.json --test-data datasets/alpaca_test.jsonl --mode baseline

    # Finetuned only
    python scripts/inference/run_inference.py --config configs/qlora_config.json --test-data datasets/alpaca_test.jsonl --mode finetuned

    # Both
    python scripts/inference/run_inference.py --config configs/qlora_config.json --test-data datasets/alpaca_test.jsonl --mode both
"""

import argparse
import gc
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

# Ensure project root is on sys.path so `finetuning` package is importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from finetuning.blob_storage import download_blob_directory, upload_file_to_blob
from finetuning.config import load_config
from finetuning.data import load_jsonl, merge_instruction_into_input
from finetuning.inference import run_inference
from finetuning.model import load_base_model, load_finetuned_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STORAGE_ACCOUNT = "llmaml5615532443"
ADAPTER_CONTAINER = "finetuning-output"
RESULTS_CONTAINER = "inference-results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on the test set")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to qlora_config.json"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test JSONL (e.g. datasets/alpaca_test.jsonl)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "finetuned", "both"],
        default="both",
        help="Which model(s) to run inference with (default: both)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1000,
        help="Max tokens to generate per example",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit test set to N random samples (default: use all)",
    )
    parser.add_argument(
        "--storage-account",
        type=str,
        default=STORAGE_ACCOUNT,
        help="Azure storage account name",
    )
    parser.add_argument(
        "--results-container",
        type=str,
        default=RESULTS_CONTAINER,
        help="Blob container for inference results",
    )
    parser.add_argument(
        "--adapter-container",
        type=str,
        default=ADAPTER_CONTAINER,
        help="Blob container with LoRA adapters",
    )
    return parser.parse_args()


def prepare_test_data(path: str) -> pd.DataFrame:
    """Load test data and merge instruction+input into a single column."""
    df = load_jsonl(path)
    df = merge_instruction_into_input(df)
    return df


def save_and_upload_results(
    df: pd.DataFrame,
    model_label: str,
    storage_account: str,
    container: str,
) -> None:
    """Save results to a temp file and upload to blob storage."""
    blob_name = f"{model_label}/inference_results.jsonl"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
        df.to_json(f, orient="records", lines=True)

    try:
        url = upload_file_to_blob(storage_account, container, blob_name, tmp_path)
        logger.info("Results uploaded: %s", url)
    finally:
        os.unlink(tmp_path)


def run_baseline(
    config: dict,
    test_df: pd.DataFrame,
    max_new_tokens: int,
    batch_size: int,
    storage_account: str,
    results_container: str,
) -> None:
    """Load the base model, run inference, and upload results."""
    model_cfg = config["model"]
    logger.info("Loading base model: %s", model_cfg["name"])

    model, tokenizer = load_base_model(
        model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    predictions = run_inference(
        model, tokenizer, test_df["prompt"].tolist(), max_new_tokens, batch_size
    )

    results = test_df[["prompt", "output"]].copy()
    results.rename(
        columns={"prompt": "input", "output": "expected_output"}, inplace=True
    )
    results["predicted_output"] = predictions
    results["model"] = "baseline"

    save_and_upload_results(results, "baseline", storage_account, results_container)

    # Free GPU memory before loading next model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def run_finetuned(
    config: dict,
    test_df: pd.DataFrame,
    max_new_tokens: int,
    batch_size: int,
    storage_account: str,
    adapter_container: str,
    results_container: str,
) -> None:
    """Download LoRA adapters from blob, load finetuned model, run inference, upload results."""
    model_cfg = config["model"]
    run_name = config["wandb"]["run_name"]
    final_model_subdir = config["training"]["final_model_subdir"]
    blob_prefix = f"{run_name}/{final_model_subdir}/"

    logger.info(
        "Downloading LoRA adapters from %s/%s ...", adapter_container, blob_prefix
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        adapter_local = download_blob_directory(
            storage_account,
            adapter_container,
            blob_prefix,
            tmp_dir,
        )

        logger.info("Loading finetuned model from adapter: %s", adapter_local)
        model, tokenizer = load_finetuned_model(
            adapter_local,
            max_seq_length=model_cfg["max_seq_length"],
            load_in_4bit=model_cfg["load_in_4bit"],
        )

        predictions = run_inference(
            model, tokenizer, test_df["prompt"].tolist(), max_new_tokens, batch_size
        )

    results = test_df[["prompt", "output"]].copy()
    results.rename(
        columns={"prompt": "input", "output": "expected_output"}, inplace=True
    )
    results["predicted_output"] = predictions
    results["model"] = run_name

    save_and_upload_results(results, run_name, storage_account, results_container)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    test_df = prepare_test_data(args.test_data)
    if args.max_samples and args.max_samples < len(test_df):
        test_df = test_df.head(args.max_samples).reset_index(drop=True)
    logger.info("Loaded %d test examples", len(test_df))

    if args.mode in ("baseline", "both"):
        logger.info("--- Running baseline inference ---")
        run_baseline(
            config,
            test_df,
            args.max_new_tokens,
            args.batch_size,
            args.storage_account,
            args.results_container,
        )

    if args.mode in ("finetuned", "both"):
        logger.info("--- Running finetuned inference ---")
        run_finetuned(
            config,
            test_df,
            args.max_new_tokens,
            args.batch_size,
            args.storage_account,
            args.adapter_container,
            args.results_container,
        )

    logger.info("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
