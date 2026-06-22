"""Migrate trained LoRA model artifacts from Azure Blob Storage to HuggingFace Hub.

For each run under `finetuning-output/` in the AzureML blob store, this script:
  1. Downloads the entire run directory (final adapter + trainer checkpoint) to a
     local staging dir.
  2. Writes a short README.md describing the run.
  3. Creates a private HF repo `<hf_user>/dutch-mistral-<run_name>` (if missing).
  4. Uploads the staged folder to that repo via `huggingface_hub.upload_folder`.
  5. Optionally deletes the local staging dir.

Auth:
  - Azure: DefaultAzureCredential (managed identity on AzureML compute).
  - HuggingFace: reads `HF_TOKEN` env var (or falls back to cached cli login).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

# Make `finetuning` package importable when running this script directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetuning.blob_storage import download_blob_directory, list_blob_prefixes  # noqa: E402

# AzureML default blob container (model artifacts live here under finetuning-output/).
AML_CONTAINER = "azureml-blobstore-4c704101-7a51-4680-bcf8-f13966bf69b4"
BLOB_PREFIX = "finetuning-output/"

# Known runs (also discoverable via list_blob_prefixes, but pinned for safety).
DEFAULT_RUNS = [
    "mistral_r16_a16_e1_b16_w30",
    "mistral_r16_a16_e1_b16_w60",
    "mistral_r32_a64_e2_b16_w30_pack_cos",
]

BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"


README_TEMPLATE = """---
base_model: {base_model}
library_name: peft
tags:
  - lora
  - mistral
  - dutch
  - instruction-tuning
  - alpaca
license: apache-2.0
---

# {run_name}

LoRA adapter for `{base_model}`, fine-tuned on a Dutch instruction-tuning
dataset (cleaned Alpaca-Dutch + synthetic Dutch examples).

## Contents

- `mistral_lora_run/` — final LoRA adapter (load with `peft.PeftModel.from_pretrained`).
- `trainer/checkpoint-*/` — full HF Trainer checkpoint including optimizer
  state (`optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`).
  Use this to resume training; not needed for inference.

## Usage (inference)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

adapter_dir = snapshot_download(
    repo_id="{repo_id}",
    allow_patterns=["mistral_lora_run/*"],
)

base = AutoModelForCausalLM.from_pretrained("{base_model}", device_map="auto")
tok = AutoTokenizer.from_pretrained(f"{{adapter_dir}}/mistral_lora_run")
model = PeftModel.from_pretrained(base, f"{{adapter_dir}}/mistral_lora_run")
```

## Training config (decoded from the run name)

Run name `{run_name}` encodes the hyperparameters used:

- `r` / `a`  — LoRA rank and alpha.
- `e`        — number of epochs.
- `b`        — effective batch size.
- `w`        — wandb run identifier / variant tag.
- `pack_cos` — packing + cosine LR schedule (when present).

Migrated from Azure Blob Storage prior to Azure subscription expiry.
"""


def write_readme(local_dir: Path, run_name: str, repo_id: str) -> None:
    readme_path = local_dir / "README.md"
    readme_path.write_text(
        README_TEMPLATE.format(
            base_model=BASE_MODEL,
            run_name=run_name,
            repo_id=repo_id,
        ),
        encoding="utf-8",
    )


def migrate_run(
    api: HfApi,
    storage_account: str,
    run_name: str,
    hf_user: str,
    stage_root: Path,
    private: bool,
    keep_staged: bool,
    dry_run: bool,
) -> str:
    repo_id = f"{hf_user}/dutch-mistral-{run_name.replace('_', '-')}"
    local_dir = stage_root / run_name
    blob_prefix = f"{BLOB_PREFIX}{run_name}/"

    print(f"\n=== {run_name} ===")
    print(f"  blob:  {AML_CONTAINER}/{blob_prefix}")
    print(f"  stage: {local_dir}")
    print(f"  repo:  {repo_id} (private={private})")

    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"  [skip download] staging dir already populated")
    else:
        print("  Downloading from blob...")
        download_blob_directory(
            storage_account=storage_account,
            container_name=AML_CONTAINER,
            blob_prefix=blob_prefix,
            local_dir=str(local_dir),
        )

    write_readme(local_dir, run_name, repo_id)

    file_count = sum(1 for _ in local_dir.rglob("*") if _.is_file())
    total_bytes = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file())
    print(f"  Staged {file_count} files, {total_bytes / 1e9:.2f} GB")

    if dry_run:
        print("  [dry-run] skipping HF repo create + upload")
        return repo_id

    create_repo(
        repo_id=repo_id,
        private=private,
        repo_type="model",
        exist_ok=True,
    )
    print(f"  Uploading to https://huggingface.co/{repo_id} ...")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Initial upload: {run_name} migrated from Azure Blob",
    )
    print(f"  ✓ uploaded https://huggingface.co/{repo_id}")

    if not keep_staged:
        shutil.rmtree(local_dir)
        print(f"  Removed staging dir {local_dir}")

    return repo_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hf-user",
        default=os.environ.get("HF_USER", "SilkeP33"),
        help="HuggingFace username (target namespace). Default: SilkeP33",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Run names to upload. Default: all known runs.",
    )
    p.add_argument(
        "--stage-dir",
        default="/tmp/hf_migration",
        help="Local staging directory for downloads.",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Create public repos. Default is private.",
    )
    p.add_argument(
        "--keep-staged",
        action="store_true",
        help="Keep local staging dir after upload (default: delete).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Download + write README but do NOT create or upload to HF.",
    )
    p.add_argument(
        "--list-blobs",
        action="store_true",
        help="List discoverable runs in the blob store and exit.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    load_dotenv(".env")
    storage_account = os.environ.get("STORAGE_ACCOUNT")
    if not storage_account:
        print("ERROR: STORAGE_ACCOUNT not set (check .env)", file=sys.stderr)
        return 2

    if args.list_blobs:
        print(f"Listing {AML_CONTAINER}/{BLOB_PREFIX} ...")
        for p in list_blob_prefixes(storage_account, AML_CONTAINER, BLOB_PREFIX):
            print(f"  {p}")
        return 0

    # HF auth check.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token and not args.dry_run:
        print(
            "ERROR: HF_TOKEN env var not set. Either:\n"
            "  export HF_TOKEN=hf_xxx   (token from https://huggingface.co/settings/tokens, needs write scope)\n"
            "  -- or --\n"
            "  huggingface-cli login    (writes a cached token)\n",
            file=sys.stderr,
        )
        return 2

    api = HfApi(token=hf_token) if hf_token else HfApi()

    runs = args.runs or DEFAULT_RUNS
    stage_root = Path(args.stage_dir)
    stage_root.mkdir(parents=True, exist_ok=True)

    print(f"Migrating {len(runs)} run(s) to HF user '{args.hf_user}'")
    print(f"  private={not args.public}  dry_run={args.dry_run}  keep_staged={args.keep_staged}")

    uploaded: list[str] = []
    failed: list[tuple[str, str]] = []
    for run in runs:
        try:
            repo_id = migrate_run(
                api=api,
                storage_account=storage_account,
                run_name=run,
                hf_user=args.hf_user,
                stage_root=stage_root,
                private=not args.public,
                keep_staged=args.keep_staged,
                dry_run=args.dry_run,
            )
            uploaded.append(repo_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ FAILED {run}: {exc!r}", file=sys.stderr)
            failed.append((run, repr(exc)))

    print("\n=== Summary ===")
    for repo_id in uploaded:
        print(f"  ✓ {repo_id}")
    for run, exc in failed:
        print(f"  ✗ {run}: {exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
