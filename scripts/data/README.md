# Data Processing Scripts

Scripts for preparing and processing Dutch instruction-following datasets for fine-tuning.

## Pipeline Overview

Run the scripts in this order:

```
1. clean_dataset.py        Clean raw translated data
2. split_dataset.py        Split into train/test sets
3. generate_synthetic_data.py   Generate additional synthetic data (optional)
4. merge_synthetic_data.py      Merge synthetic data into train/test (optional)
5. format_dataset.py       Format train set with chat template for training
```

## Scripts

### 1. `clean_dataset.py`

Cleans the raw `alpaca_data_cleaned-dutch.jsonl` dataset by removing:

- Duplicate (instruction + input) pairs
- English / untranslated rows (detected via function-word heuristics)
- Translation-to-English tasks
- Coding examples (code blocks, programming keywords)
- Math / calculation questions

**Input:** `datasets/alpaca_data_cleaned-dutch.jsonl` (hardcoded)
**Output:** `datasets/alpaca_data_cleaned-dutch-clean.jsonl` (hardcoded)

```bash
python scripts/data/clean_dataset.py
```

No parameters — input/output paths are defined as constants in the script. Prints a detailed summary of what was removed and why.

---

### 2. `split_dataset.py`

Splits a cleaned JSONL dataset into train and test sets. Merges the `instruction` and `input` columns into a single `prompt` column before splitting.

```bash
python scripts/data/split_dataset.py --data datasets/alpaca_data_cleaned-dutch-clean.jsonl
```

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--data` | Yes | — | Path to the input JSONL file |
| `--train-out` | No | `datasets/alpaca_train.jsonl` | Output path for the train split |
| `--test-out` | No | `datasets/alpaca_test.jsonl` | Output path for the test split |
| `--train-frac` | No | `0.8` | Fraction of data for training (0–1) |
| `--random-state` | No | `42` | Random seed for reproducible splits |

---

### 3. `generate_synthetic_data.py`

Generates synthetic Dutch instruction-following data using Azure OpenAI. Calls the API with async batching for throughput, sampling from a variety of topics (science, culture, economics, daily life, politics) and task types (explanation, creative writing, summary, argumentation, etc.) with configurable output lengths.

Requires the following environment variables (loaded from `.env` in the project root):

- `ENDPOINT` — Azure OpenAI endpoint URL
- `DEPLOYMENT` — Model deployment name
- `STORAGE_ACCOUNT` — Azure Storage account name (for upload)
- `CONTAINER_NAME` — Blob container name (for upload)

```bash
# Generate 5000 examples (default)
python scripts/data/generate_synthetic_data.py

# Generate 100 examples with lower concurrency, skip upload
python scripts/data/generate_synthetic_data.py --num-examples 100 --concurrency 5 --no-upload

# Preview a sample prompt without making API calls
python scripts/data/generate_synthetic_data.py --dry-run
```

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--num-examples` | No | `5000` | Number of examples to generate |
| `--output` | No | `datasets/synthetic_dutch.jsonl` | Output JSONL path |
| `--concurrency` | No | `10` | Maximum concurrent API calls |
| `--dry-run` | No | off | Print a sample prompt without making API calls |
| `--no-upload` | No | off | Skip uploading the result to blob storage |

---

### 4. `merge_synthetic_data.py`

Merges the synthetic dataset into the existing train/test splits. Splits the synthetic data (default 70/30), appends each portion to the corresponding set, re-indexes all IDs, and optionally uploads the merged files to Azure Blob Storage.

Requires the same `STORAGE_ACCOUNT` and `CONTAINER_NAME` environment variables as the generation script (unless `--no-upload` is used).

```bash
# Merge with defaults and upload
python scripts/data/merge_synthetic_data.py

# Merge without uploading
python scripts/data/merge_synthetic_data.py --no-upload

# Use a custom synthetic file and train/test split ratio
python scripts/data/merge_synthetic_data.py --synthetic datasets/my_synthetic.jsonl --train-frac 0.8
```

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--synthetic` | No | `datasets/synthetic_dutch.jsonl` | Path to the synthetic JSONL file |
| `--train` | No | `datasets/alpaca_train.jsonl` | Path to the existing train set |
| `--test` | No | `datasets/alpaca_test.jsonl` | Path to the existing test set |
| `--train-frac` | No | `0.7` | Fraction of synthetic data to add to train |
| `--random-state` | No | `42` | Random seed for reproducible split |
| `--no-upload` | No | off | Skip uploading to blob storage |

---

### 5. `format_dataset.py`

Applies a chat template to the train JSONL file and saves it as a HuggingFace Dataset on disk. Uses the tokenizer specified in the QLoRA config to format each (prompt, output) pair into the model's expected chat format.

```bash
python scripts/data/format_dataset.py --data datasets/alpaca_train.jsonl --config configs/qlora_config.json
```

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--data` | Yes | — | Path to the train JSONL file (output of `split_dataset.py`) |
| `--config` | Yes | — | Path to the QLoRA config JSON (used to resolve the tokenizer/model name) |
| `--output` | No | `datasets/alpaca_train_formatted` | Output directory for the formatted HuggingFace dataset |


## Full Example

```bash
# 1. Clean the raw dataset
python scripts/data/clean_dataset.py

# 2. Split into train/test
python scripts/data/split_dataset.py --data datasets/alpaca_data_cleaned-dutch-clean.jsonl

# 3. (Optional) Generate synthetic data
python scripts/data/generate_synthetic_data.py --num-examples 5000

# 4. (Optional) Merge synthetic data into train/test
python scripts/data/merge_synthetic_data.py

# 5. Format for training
python scripts/data/format_dataset.py --data datasets/alpaca_train.jsonl --config configs/qlora_config.json
```
