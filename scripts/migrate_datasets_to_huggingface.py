"""Backup the v1 and v2 Dutch Alpaca train/test sets to HuggingFace Hub.

Creates a single private dataset repo with two configs:
  - v1: original train/test split used for the first round of fine-tuning
  - v2: expanded split (5K alpaca high-quality + 7K synthetic + balanced test)

Both configs expose `train` and `test` splits. Load via:
  load_dataset("<user>/dutch-alpaca-instruct", "v2", split="train")
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo


REPO_NAME = "dutch-alpaca-instruct"

# (config, split, source file)
LAYOUT = [
    ("v1", "train", "datasets/alpaca_train.jsonl"),
    ("v1", "test", "datasets/alpaca_test.jsonl"),
    ("v2", "train", "datasets/alpaca_train_v2.jsonl"),
    ("v2", "test", "datasets/alpaca_test_v2.jsonl"),
]


README_TEMPLATE = """---
license: apache-2.0
language:
  - nl
task_categories:
  - text-generation
tags:
  - instruction-tuning
  - alpaca
  - dutch
configs:
  - config_name: v1
    data_files:
      - split: train
        path: v1/train.jsonl
      - split: test
        path: v1/test.jsonl
  - config_name: v2
    default: true
    data_files:
      - split: train
        path: v2/train.jsonl
      - split: test
        path: v2/test.jsonl
---

# {repo_id}

Dutch instruction-tuning datasets used to fine-tune Mistral-7B LoRA adapters.
Each row follows the Alpaca schema: `instruction`, `input`, `output`.

## Versions

### v1 (original)
Initial split used for early fine-tuning experiments.

### v2 (expanded, default)
Built from the highest-quality 5K cleaned Alpaca-Dutch examples plus 7K
synthetically generated Dutch examples. Test set: 1000 from Alpaca + 500
synthetic. No row overlap or duplicates between train and test.

## Usage

```python
from datasets import load_dataset

# v2 (default)
ds = load_dataset("{repo_id}", split="train")

# v1 explicitly
ds_v1 = load_dataset("{repo_id}", "v1", split="train")
```
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-user", default=os.environ.get("HF_USER", "SilkeP33"))
    p.add_argument("--public", action="store_true", help="Default is private.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(".env")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token and not args.dry_run:
        print("ERROR: HF_TOKEN not set in env or .env", file=sys.stderr)
        return 2

    repo_id = f"{args.hf_user}/{REPO_NAME}"
    workspace_root = Path(__file__).resolve().parent.parent

    # Validate all source files exist.
    missing = [src for _, _, src in LAYOUT if not (workspace_root / src).is_file()]
    if missing:
        print(f"ERROR: missing source files: {missing}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="hf_datasets_") as stage:
        stage_dir = Path(stage)

        # Copy files into the expected v1/, v2/ layout.
        for config, split, src in LAYOUT:
            dst = stage_dir / config / f"{split}.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(workspace_root / src, dst)
            size_mb = dst.stat().st_size / 1e6
            print(f"  staged {config}/{split}.jsonl  ({size_mb:.1f} MB)  <- {src}")

        # Write README.
        (stage_dir / "README.md").write_text(
            README_TEMPLATE.format(repo_id=repo_id), encoding="utf-8"
        )

        if args.dry_run:
            print(f"\n[dry-run] would create + upload to {repo_id}")
            print(f"[dry-run] staged contents at: {stage_dir}")
            for p in sorted(stage_dir.rglob("*")):
                if p.is_file():
                    print(f"    {p.relative_to(stage_dir)}")
            return 0

        print(f"\nCreating dataset repo {repo_id} (private={not args.public}) ...")
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=not args.public,
            exist_ok=True,
            token=hf_token,
        )

        print(f"Uploading to https://huggingface.co/datasets/{repo_id} ...")
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=str(stage_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Initial upload: v1 + v2 Dutch Alpaca splits",
        )
        print(f"✓ done: https://huggingface.co/datasets/{repo_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
