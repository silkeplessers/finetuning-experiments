"""
Merge synthetic data into the existing train/test sets.

Splits the synthetic dataset 70/30, appends to the existing alpaca train/test
sets, re-indexes IDs, and uploads the merged sets to blob storage.

Usage:
    python scripts/merge_synthetic_data.py
    python scripts/merge_synthetic_data.py --synthetic datasets/synthetic_dutch.jsonl --no-upload
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from finetuning.blob_storage import upload_file_to_blob

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STORAGE_ACCOUNT = os.environ["STORAGE_ACCOUNT"]
CONTAINER_NAME = os.environ["CONTAINER_NAME"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge synthetic data into train/test sets and upload to blob storage."
    )
    parser.add_argument(
        "--synthetic",
        type=str,
        default="datasets/synthetic_dutch.jsonl",
        help="Path to synthetic JSONL dataset",
    )
    parser.add_argument(
        "--train",
        type=str,
        default="datasets/alpaca_train.jsonl",
        help="Path to existing train set",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="datasets/alpaca_test.jsonl",
        help="Path to existing test set",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of synthetic data to add to train (default: 0.7)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible split (default: 42)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to blob storage",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load datasets
    synthetic = pd.read_json(args.synthetic, lines=True)
    train = pd.read_json(args.train, lines=True)
    test = pd.read_json(args.test, lines=True)

    logger.info(f"Existing train: {len(train):,} rows")
    logger.info(f"Existing test:  {len(test):,} rows")
    logger.info(f"Synthetic:      {len(synthetic):,} rows")

    # Split synthetic 70/30
    syn_train = synthetic.sample(frac=args.train_frac, random_state=args.random_state)
    syn_test = synthetic.drop(syn_train.index)

    logger.info(f"Synthetic -> train: {len(syn_train):,}, test: {len(syn_test):,}")

    # Append
    merged_train = pd.concat([train, syn_train], ignore_index=True)
    merged_test = pd.concat([test, syn_test], ignore_index=True)

    # Re-index IDs
    merged_train["id"] = range(1, len(merged_train) + 1)
    merged_test["id"] = range(1, len(merged_test) + 1)

    # Ensure consistent column order
    cols = ["id", "instruction", "input", "output", "prompt"]
    merged_train = merged_train[[c for c in cols if c in merged_train.columns]]
    merged_test = merged_test[[c for c in cols if c in merged_test.columns]]

    # Save locally (overwrite existing)
    merged_train.to_json(args.train, orient="records", lines=True, force_ascii=False)
    merged_test.to_json(args.test, orient="records", lines=True, force_ascii=False)

    logger.info(f"Merged train: {len(merged_train):,} rows -> {args.train}")
    logger.info(f"Merged test:  {len(merged_test):,} rows -> {args.test}")

    # Upload to blob storage
    if not args.no_upload:
        for local_path in [args.train, args.test]:
            blob_name = Path(local_path).name
            try:
                url = upload_file_to_blob(
                    storage_account=STORAGE_ACCOUNT,
                    container_name=CONTAINER_NAME,
                    blob_name=blob_name,
                    local_path=local_path,
                )
                logger.info(f"Uploaded {blob_name} -> {url}")
            except Exception as e:
                logger.error(f"Failed to upload {blob_name}: {e}")
    else:
        logger.info("Skipping blob storage upload (--no-upload)")


if __name__ == "__main__":
    main()
