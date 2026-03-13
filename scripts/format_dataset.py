"""Format a train JSONL split into a HuggingFace Dataset with chat-template prompts.

Usage:
    python scripts/format_dataset.py --data datasets/alpaca_train.jsonl --config configs/qlora_config.json
    python scripts/format_dataset.py --data datasets/alpaca_train.jsonl --config configs/qlora_config.json --output datasets/alpaca_train_formatted
"""

import argparse
import json
import logging

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from formatting import format_prompt_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHAT_TEMPLATE = """\
Beantwoord de volgende vraag zo goed mogelijk.

### Vraag:
{INPUT}

### Antwoord:
{OUTPUT}
"""


def format_dataset(
    df: pd.DataFrame,
    chat_template: str,
    eos_token: str,
    input_column: str = "input",
    output_column: str = "output",
) -> Dataset:
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset.map(
        lambda batch: format_prompt_batch(
            batch, chat_template, input_column, output_column, eos_token,
        ),
        batched=True,
        remove_columns=hf_dataset.column_names,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a chat template to a JSONL split and save as a HuggingFace Dataset.",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the train JSONL file (output of split_dataset.py)",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the qlora config JSON (used to resolve the tokenizer)",
    )
    parser.add_argument(
        "--output", type=str, default="datasets/alpaca_train_formatted",
        help="Output directory for the formatted HF dataset (default: datasets/alpaca_train_formatted)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config["model"]["name"]

    data = pd.read_json(args.data, lines=True)
    logger.info("Loaded %d rows from %s", len(data), args.data)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    eos_token = tokenizer.eos_token

    formatted = format_dataset(data, CHAT_TEMPLATE, eos_token)
    formatted.save_to_disk(args.output)

    logger.info("Saved %d formatted examples -> %s", len(formatted), args.output)
    logger.info("Sample:\n%s...", formatted[0]["text"][:200])


if __name__ == "__main__":
    main()
