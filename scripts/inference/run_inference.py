"""Run inference on the test dataset using the baseline and/or finetuned model.

Results are saved to the directory specified by --output-dir. When submitted
via Azure ML, that directory is a blob-mounted output path.

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
import json
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

from finetuning.blob_storage import download_blob_directory
from finetuning.config import load_config
from finetuning.data import load_jsonl, merge_instruction_into_input
from finetuning.inference import run_inference
from finetuning.model import load_base_model, load_finetuned_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STORAGE_ACCOUNT = "llmaml5615532443"
ADAPTER_CONTAINER = "azureml-blobstore-4c704101-7a51-4680-bcf8-f13966bf69b4"


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
        default=512,
        help=(
            "Max tokens to generate per example. Default 512 covers the p99 "
            "length of training outputs (~305 tokens) with headroom; raising "
            "this above the training-length distribution triggers degenerate "
            "repetition on creative-writing prompts."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
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
        "--adapter-container",
        type=str,
        default=ADAPTER_CONTAINER,
        help="Blob container with LoRA adapters",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save inference results (default: from config training.output_dir)",
    )
    return parser.parse_args()


def prepare_test_data(path: str) -> pd.DataFrame:
    """Load test data and merge instruction+input into a single column."""
    df = load_jsonl(path)
    df = merge_instruction_into_input(df)
    return df


def save_results(results: pd.DataFrame, output_dir: str, label: str) -> None:
    """Save inference results to a JSONL file inside *output_dir*."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path / f"{label}_results.jsonl"
    results.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    logger.info("Saved results to %s", jsonl_path)


BASELINE_SYSTEM_PROMPT = (
    "Je bent een behulpzame assistent. "
    "Beantwoord de volgende vraag volledig in het Nederlands. "
    "Wees beknopt en relevant."
)


def run_baseline(
    config: dict,
    test_df: pd.DataFrame,
    max_new_tokens: int,
    batch_size: int,
    output_dir: str,
) -> None:
    """Load the base model, run inference, and save results."""
    model_cfg = config["model"]
    logger.info("Loading base model: %s", model_cfg["name"])

    model, tokenizer = load_base_model(
        model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    predictions = run_inference(
        model,
        tokenizer,
        test_df["prompt"].tolist(),
        max_new_tokens,
        batch_size=batch_size,
        system_prompt=BASELINE_SYSTEM_PROMPT,
        max_seq_length=model_cfg["max_seq_length"],
    )

    results = test_df[["prompt", "output"]].copy()
    results.rename(
        columns={"prompt": "input", "output": "expected_output"}, inplace=True
    )
    results["predicted_output"] = predictions
    results["model"] = "baseline"

    save_results(results, output_dir, "baseline")

    # Free GPU memory before loading next model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def run_finetuned(
    config: dict,
    test_df: pd.DataFrame,
    max_new_tokens: int,
    batch_size: int,
    output_dir: str,
    storage_account: str,
    adapter_container: str,
    managed_identity_client_id: str | None = None,
) -> None:
    """Download LoRA adapters from blob, load finetuned model, run inference, save results."""
    model_cfg = config["model"]
    run_name = config["wandb"]["run_name"]
    final_model_subdir = config["training"]["final_model_subdir"]

    # Extract the blob path prefix from the output_uri (e.g. "finetuning-output")
    output_uri = config.get("azureml", {}).get(
        "output_uri",
        "azureml://datastores/workspaceblobstore/paths/finetuning-output/",
    )
    blob_path_prefix = ""
    if "/paths/" in output_uri:
        blob_path_prefix = output_uri.split("/paths/", 1)[1].strip("/")

    blob_prefix = (
        f"{blob_path_prefix}/{run_name}/{final_model_subdir}/"
        if blob_path_prefix
        else f"{run_name}/{final_model_subdir}/"
    )

    logger.info(
        "Downloading LoRA adapters from %s/%s ...", adapter_container, blob_prefix
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        adapter_local = download_blob_directory(
            storage_account,
            adapter_container,
            blob_prefix,
            tmp_dir,
            managed_identity_client_id=managed_identity_client_id,
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

    save_results(results, output_dir, run_name)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or config["training"]["output_dir"]
    test_df = prepare_test_data(args.test_data)
    if args.max_samples and args.max_samples < len(test_df):
        test_df = test_df.head(args.max_samples).reset_index(drop=True)
    logger.info("Loaded %d test examples", len(test_df))
    logger.info("Inference results will be saved to %s", output_dir)

    if args.mode in ("baseline", "both"):
        logger.info("--- Running baseline inference ---")
        run_baseline(
            config,
            test_df,
            args.max_new_tokens,
            args.batch_size,
            output_dir,
        )

    if args.mode in ("finetuned", "both"):
        logger.info("--- Running finetuned inference ---")
        run_finetuned(
            config,
            test_df,
            args.max_new_tokens,
            args.batch_size,
            output_dir,
            args.storage_account,
            args.adapter_container,
            managed_identity_client_id=config.get("azureml", {}).get(
                "managed_identity_client_id"
            ),
        )

    logger.info("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
