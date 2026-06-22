"""Build the v2 train/test datasets.

Targets (everything written to NEW filenames; v1 files are not overwritten):
    - 5000 alpaca high-quality + 7000 synthetic = 12,000 train rows
    - 1000 alpaca high-quality + 500 synthetic  =  1,500 test rows

Steps:
    1. Re-select top 6000 alpaca rows from the existing
       `datasets/subsample_scoring_log.jsonl` (sorted by `llm_total`, then
       `total` as tiebreaker). Materialize full rows from
       `datasets/alpaca_data_cleaned-dutch-clean.jsonl`. Write to
       `datasets/alpaca_high_quality_6k.jsonl`. NO API calls.
    2. Combine existing 2000-row `datasets/synthetic_dutch.jsonl` with the
       newly generated `datasets/synthetic_dutch_extra5500.jsonl` -> 7500-row
       `datasets/synthetic_dutch_7500.jsonl`. (Generation is run separately
       via `generate_synthetic_data.py`.)
    3. Split alpaca 5000/1000 and synthetic 7000/500 (seed 42), concat,
       reindex IDs, write `datasets/alpaca_train_v2.jsonl` and
       `datasets/alpaca_test_v2.jsonl`.
    4. Optionally upload new files to blob storage.

Usage:
    python scripts/data/build_dataset_v2.py
    python scripts/data/build_dataset_v2.py --no-upload
    python scripts/data/build_dataset_v2.py --skip-select  # if 6k file already exists
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from finetuning.blob_storage import upload_file_to_blob
from finetuning.data import merge_instruction_into_input

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- paths ------------------------------------------------------------------
SCORING_LOG = Path("datasets/subsample_scoring_log.jsonl")
CLEAN_SOURCE = Path("datasets/alpaca_data_cleaned-dutch-clean.jsonl")

ALPACA_6K = Path("datasets/alpaca_high_quality_6k.jsonl")

SYNTHETIC_EXISTING = Path("datasets/synthetic_dutch.jsonl")
# All files in SYNTHETIC_EXTRA_FILES are concatenated onto SYNTHETIC_EXISTING.
# Order matters only for tie-breaking when trimming to TARGET_SYNTHETIC_TOTAL.
SYNTHETIC_EXTRA_FILES = [
    Path("datasets/synthetic_dutch_extra5500.jsonl"),
    Path("datasets/synthetic_dutch_extra3100.jsonl"),
]
SYNTHETIC_7500 = Path("datasets/synthetic_dutch_7500.jsonl")
TARGET_SYNTHETIC_TOTAL = 7500

TRAIN_OUT = Path("datasets/alpaca_train_v2.jsonl")
TEST_OUT = Path("datasets/alpaca_test_v2.jsonl")

# --- target sizes -----------------------------------------------------------
ALPACA_TOTAL = 6000
ALPACA_TEST = 1000  # so train = 5000
SYNTH_TOTAL = 7500
SYNTH_TEST = 500    # so train = 7000
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Step 1: re-select top 6000 alpaca rows from existing scoring log
# ---------------------------------------------------------------------------
def select_top_alpaca(n: int = ALPACA_TOTAL) -> pd.DataFrame:
    log_rows = [json.loads(l) for l in SCORING_LOG.open()]
    log_df = pd.DataFrame(log_rows)
    scored = log_df[log_df["llm_total"].notna()].copy()
    logger.info(
        "Scoring log: %d total rows, %d with llm_total",
        len(log_df),
        len(scored),
    )
    if len(scored) < n:
        raise RuntimeError(
            f"Only {len(scored)} rows have llm_total but {n} requested"
        )

    # Sort by LLM total desc, with heuristic total as tiebreaker
    scored = scored.sort_values(
        by=["llm_total", "total"], ascending=[False, False]
    )
    top_ids = scored.head(n)["id"].tolist()
    logger.info(
        "Top %d cutoff: llm_total=%s total=%.3f",
        n,
        scored.iloc[n - 1]["llm_total"],
        scored.iloc[n - 1]["total"],
    )

    # Materialize full rows from clean source
    source = pd.read_json(CLEAN_SOURCE, lines=True)
    source_by_id = source.set_index("id")
    missing = [i for i in top_ids if i not in source_by_id.index]
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected IDs missing from clean source "
            f"(first 5: {missing[:5]})"
        )

    selected = source_by_id.loc[top_ids].reset_index()
    # Keep ID order = LLM-score order (so re-index later is deterministic)
    return selected


def write_alpaca_6k(df: pd.DataFrame) -> None:
    ALPACA_6K.parent.mkdir(parents=True, exist_ok=True)
    # Preserve a clean schema (id, instruction, input, output) — matches v1 file
    cols = [c for c in ["id", "instruction", "input", "output"] if c in df.columns]
    df[cols].to_json(ALPACA_6K, orient="records", lines=True, force_ascii=False)
    logger.info("Wrote %d rows -> %s", len(df), ALPACA_6K)


# ---------------------------------------------------------------------------
# Step 2: combine existing 2000 + new 5500 synthetic -> 7500
# ---------------------------------------------------------------------------
def combine_synthetic() -> pd.DataFrame:
    if not SYNTHETIC_EXISTING.exists():
        raise FileNotFoundError(SYNTHETIC_EXISTING)
    missing = [p for p in SYNTHETIC_EXTRA_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing synthetic extra files: "
            + ", ".join(str(p) for p in missing)
            + ". Generate them first with scripts/data/generate_synthetic_data.py "
            "(use --no-upload)."
        )

    keep_cols = ["instruction", "input", "output"]

    frames: list[pd.DataFrame] = []
    existing = pd.read_json(SYNTHETIC_EXISTING, lines=True)
    frames.append(existing[[c for c in keep_cols if c in existing.columns]])
    sizes = {str(SYNTHETIC_EXISTING): len(existing)}
    for path in SYNTHETIC_EXTRA_FILES:
        df = pd.read_json(path, lines=True)
        sizes[str(path)] = len(df)
        frames.append(df[[c for c in keep_cols if c in df.columns]])

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Synthetic inputs: %s -> combined=%d",
        ", ".join(f"{k}={v}" for k, v in sizes.items()),
        len(combined),
    )

    if len(combined) < TARGET_SYNTHETIC_TOTAL:
        raise ValueError(
            f"Only {len(combined)} synthetic rows available; need "
            f"{TARGET_SYNTHETIC_TOTAL}. Generate more before continuing."
        )
    if len(combined) > TARGET_SYNTHETIC_TOTAL:
        logger.info(
            "Trimming combined synthetic from %d to %d rows",
            len(combined),
            TARGET_SYNTHETIC_TOTAL,
        )
        combined = combined.iloc[:TARGET_SYNTHETIC_TOTAL].reset_index(drop=True)

    combined.insert(0, "id", range(1, len(combined) + 1))

    SYNTHETIC_7500.parent.mkdir(parents=True, exist_ok=True)
    combined.to_json(SYNTHETIC_7500, orient="records", lines=True, force_ascii=False)
    logger.info("Wrote %d rows -> %s", len(combined), SYNTHETIC_7500)
    return combined


# ---------------------------------------------------------------------------
# Step 3: split + merge into v2 train/test
# ---------------------------------------------------------------------------
def split_exact(df: pd.DataFrame, n_test: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproducibly split off exactly `n_test` rows for the test set."""
    test = df.sample(n=n_test, random_state=seed)
    train = df.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_train_test(alpaca: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpaca = merge_instruction_into_input(alpaca)
    synthetic = merge_instruction_into_input(synthetic)

    a_train, a_test = split_exact(alpaca, ALPACA_TEST, RANDOM_STATE)
    s_train, s_test = split_exact(synthetic, SYNTH_TEST, RANDOM_STATE)

    logger.info(
        "Alpaca split: train=%d test=%d | Synthetic split: train=%d test=%d",
        len(a_train),
        len(a_test),
        len(s_train),
        len(s_test),
    )

    train = pd.concat([a_train, s_train], ignore_index=True)
    test = pd.concat([a_test, s_test], ignore_index=True)

    # Re-index IDs across the combined file
    train["id"] = range(1, len(train) + 1)
    test["id"] = range(1, len(test) + 1)

    # Consistent column order
    cols = ["id", "instruction", "input", "output", "prompt"]
    train = train[[c for c in cols if c in train.columns]]
    test = test[[c for c in cols if c in test.columns]]

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    train.to_json(TRAIN_OUT, orient="records", lines=True, force_ascii=False)
    test.to_json(TEST_OUT, orient="records", lines=True, force_ascii=False)
    logger.info("Wrote train (%d rows) -> %s", len(train), TRAIN_OUT)
    logger.info("Wrote test  (%d rows) -> %s", len(test), TEST_OUT)
    return train, test


# ---------------------------------------------------------------------------
# Step 4: upload
# ---------------------------------------------------------------------------
def upload(paths: list[Path]) -> None:
    storage_account = os.environ["STORAGE_ACCOUNT"]
    container_name = os.environ["CONTAINER_NAME"]
    for p in paths:
        try:
            url = upload_file_to_blob(
                storage_account=storage_account,
                container_name=container_name,
                blob_name=p.name,
                local_path=str(p),
            )
            logger.info("Uploaded -> %s", url)
        except Exception as e:
            logger.error("Failed to upload %s: %s", p.name, e)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-select",
        action="store_true",
        help=f"Skip step 1 if {ALPACA_6K} already exists",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to blob storage",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Step 1: alpaca high-quality 6k
    if args.skip_select and ALPACA_6K.exists():
        logger.info("Step 1: reusing existing %s", ALPACA_6K)
        alpaca = pd.read_json(ALPACA_6K, lines=True)
    else:
        logger.info("Step 1: selecting top %d alpaca rows from scoring log", ALPACA_TOTAL)
        alpaca = select_top_alpaca(ALPACA_TOTAL)
        write_alpaca_6k(alpaca)

    # Step 2: synthetic 7500
    logger.info("Step 2: combining synthetic datasets -> %d rows", SYNTH_TOTAL)
    synthetic = combine_synthetic()

    # Step 3: split + merge
    logger.info("Step 3: building train/test splits")
    build_train_test(alpaca, synthetic)

    # Step 4: upload
    if args.no_upload:
        logger.info("Skipping blob upload (--no-upload)")
    else:
        logger.info("Uploading new files to blob storage")
        upload([ALPACA_6K, SYNTHETIC_7500, TRAIN_OUT, TEST_OUT])


if __name__ == "__main__":
    main()
