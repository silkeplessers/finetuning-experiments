# Inference Scripts

Scripts for running model inference on the test dataset, both locally and as Azure ML jobs.

## `run_inference.py`

Runs inference on the test set using the baseline model, the fine-tuned model, or both. For the fine-tuned model, LoRA adapters are downloaded from Azure Blob Storage automatically. Results are saved as JSONL to the specified output directory.

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
| `--output-dir` | No | from config | Directory to save inference results |
| `--mode` | No | `both` | Which model(s) to run: `baseline`, `finetuned`, or `both` |
| `--max-new-tokens` | No | `1000` | Maximum tokens to generate per example |
| `--batch-size` | No | `8` | Batch size for inference |
| `--max-samples` | No | all | Limit test set to N samples (useful for quick tests) |
| `--storage-account` | No | `llmaml5615532443` | Azure Storage account name |
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

1. **Baseline mode:** Loads the base model directly, runs inference on all test prompts, saves results to `<output-dir>/baseline_results.jsonl`.
2. **Finetuned mode:** Downloads LoRA adapters from `<adapter-container>/<run_name>/<final_model_subdir>/`, merges them with the base model, runs inference, saves results to `<output-dir>/<run_name>_results.jsonl`.
3. **Both mode:** Runs baseline first, frees GPU memory, then runs finetuned.

---

## `submit_azureml_inference_job.py`

Submits an inference job to an Azure ML compute cluster. The job runs `run_inference.py` remotely using a dedicated inference environment (`azureml/conda_inference.yml`), separate from the training environment.

### Prerequisites

- Azure ML SDK v2 (`azure-ai-ml`)
- Azure credentials configured locally
- The QLoRA config JSON with an `azureml` section

### Usage

```bash
# Submit with defaults (both models)
python scripts/inference/submit_azureml_inference_job.py \
    --config configs/qlora_config.json

# Baseline only with batch size 16
python scripts/inference/submit_azureml_inference_job.py \
    --config configs/qlora_config.json \
    --mode baseline \
    --batch-size 16

# Finetuned only, limit to 200 samples
python scripts/inference/submit_azureml_inference_job.py \
    --config configs/qlora_config.json \
    --mode finetuned \
    --max-samples 200
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--config` | No | `configs/qlora_config.json` | Path to qlora config JSON |
| `--test-data` | No | `datasets/alpaca_test.jsonl` | Path to test JSONL file |
| `--mode` | No | `both` | Which model(s) to run: `baseline`, `finetuned`, or `both` |
| `--max-new-tokens` | No | `2048` | Max tokens to generate per example |
| `--batch-size` | No | `8` | Batch size for inference |
| `--max-samples` | No | all | Limit test set to N samples |

### Config sections used (`azureml`)

| Key | Default | Description |
|---|---|---|
| `compute` | — (required) | Azure ML compute target |
| `experiment_name` | `dutch-mistral-qlora` | Experiment name in Azure ML |
| `inference_environment_name` | `qlora-inference-env` | Name for the inference environment |
| `inference_environment_version` | `1` | Version of the inference environment |
| `inference_results_uri` | `azureml://datastores/workspaceblobstore/paths/inference-results/` | Output URI for results |
| `timeout_minutes` | `1440` | Job timeout in minutes |
| `stream_logs` | `true` | Whether to stream logs after submission |

### Environment

The inference job uses a **separate conda environment** defined in `azureml/conda_inference.yml`. This environment includes only the packages needed for inference (no `trl` or `wandb`), plus `azure-identity` and `azure-storage-blob` for downloading LoRA adapters from blob storage.

---

## Output format

Each result JSONL contains rows with:

| Field | Description |
|---|---|
| `input` | The prompt sent to the model |
| `expected_output` | The reference answer from the test set |
| `predicted_output` | The model's generated response |
| `model` | `"baseline"` or the W&B run name |
