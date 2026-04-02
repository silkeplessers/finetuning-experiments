# Inference Scripts

Scripts for running model inference on the test dataset.

## `run_inference.py`

Runs inference on the test set using the baseline model, the fine-tuned model, or both. For the fine-tuned model, LoRA adapters are downloaded from Azure Blob Storage automatically. Results are uploaded to blob storage as JSONL.

### Prerequisites

- Azure credentials (via `DefaultAzureCredential`) for blob storage access
- A GPU with enough memory to load the model in 4-bit
- The QLoRA config JSON (specifies model name, adapter location, etc.)

### Usage

```bash
# Run both baseline and finetuned inference
python scripts/inference/run_inference.py \
    --config configs/qlora_config.json \
    --test-data datasets/alpaca_test.jsonl

# Baseline only
python scripts/inference/run_inference.py \
    --config configs/qlora_config.json \
    --test-data datasets/alpaca_test.jsonl \
    --mode baseline

# Finetuned only, limit to 50 samples
python scripts/inference/run_inference.py \
    --config configs/qlora_config.json \
    --test-data datasets/alpaca_test.jsonl \
    --mode finetuned \
    --max-samples 50

# Custom batch size and token limit
python scripts/inference/run_inference.py \
    --config configs/qlora_config.json \
    --test-data datasets/alpaca_test.jsonl \
    --batch-size 4 \
    --max-new-tokens 512
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--config` | Yes | — | Path to `qlora_config.json` |
| `--test-data` | Yes | — | Path to the test JSONL file (e.g. `datasets/alpaca_test.jsonl`) |
| `--mode` | No | `both` | Which model(s) to run: `baseline`, `finetuned`, or `both` |
| `--max-new-tokens` | No | `1000` | Maximum tokens to generate per example |
| `--batch-size` | No | `8` | Batch size for inference |
| `--max-samples` | No | all | Limit test set to N samples (useful for quick tests) |
| `--storage-account` | No | `llmaml5615532443` | Azure Storage account name |
| `--results-container` | No | `inference-results` | Blob container for storing inference results |
| `--adapter-container` | No | `finetuning-output` | Blob container where LoRA adapters are stored |

### Config sections used

| Section | Purpose |
|---|---|
| `model.name` | Base model to load |
| `model.max_seq_length` | Maximum sequence length |
| `model.load_in_4bit` | Whether to load in 4-bit quantisation |
| `wandb.run_name` | Used to locate the adapter directory in blob storage |
| `training.final_model_subdir` | Subdirectory within the run folder containing the adapter |

### How it works

1. **Baseline mode:** Loads the base model directly, runs inference on all test prompts, uploads results to `<results-container>/baseline/inference_results.jsonl`.
2. **Finetuned mode:** Downloads LoRA adapters from `<adapter-container>/<run_name>/<final_model_subdir>/`, merges them with the base model, runs inference, uploads results to `<results-container>/<run_name>/inference_results.jsonl`.
3. **Both mode:** Runs baseline first, frees GPU memory, then runs finetuned.

### Output format

Each result JSONL contains rows with:

| Field | Description |
|---|---|
| `input` | The prompt sent to the model |
| `expected_output` | The reference answer from the test set |
| `predicted_output` | The model's generated response |
| `model` | `"baseline"` or the W&B run name |
