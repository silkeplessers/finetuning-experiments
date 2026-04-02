"""Split a raw JSONL dataset into train and test sets.

Usage:
    python scripts/data/split_dataset.py --data datasets/alpaca_data_cleaned-dutch.jsonl
    python scripts/data/split_dataset.py --data datasets/alpaca_data_cleaned-dutch.jsonl --train-frac 0.9
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from finetuning.data import load_jsonl, merge_instruction_into_input, split_train_test

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a JSONL dataset, merge instruction+input, and split into train/test."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the raw JSONL dataset",
    )
    parser.add_argument(
        "--train-out",
        type=str,
        default="datasets/alpaca_train.jsonl",
        help="Output path for the train split (default: datasets/alpaca_train.jsonl)",
    )
    parser.add_argument(
        "--test-out",
        type=str,
        default="datasets/alpaca_test.jsonl",
        help="Output path for the test split (default: datasets/alpaca_test.jsonl)",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_jsonl(args.data)
    logger.info("Loaded %d rows from %s", len(data), args.data)

    data = merge_instruction_into_input(data)

    train, test = split_train_test(
        data, train_frac=args.train_frac, random_state=args.random_state
    )

    train.to_json(args.train_out, orient="records", lines=True)
    test.to_json(args.test_out, orient="records", lines=True)

    logger.info("Train: %d rows -> %s", len(train), args.train_out)
    logger.info("Test:  %d rows -> %s", len(test), args.test_out)


if __name__ == "__main__":
    main()
