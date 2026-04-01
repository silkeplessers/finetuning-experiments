# Training Scripts

Scripts for fine-tuning and submitting training jobs to Azure ML.

## Scripts

### 1. `finetune_qlora.py`

Runs QLoRA fine-tuning using [Unsloth](https://github.com/unslothai/unsloth) and the HuggingFace `SFTTrainer`. Loads a pre-quantised model, applies LoRA adapters, trains on a pre-formatted HuggingFace Dataset, and saves the final adapter weights + tokenizer.

Experiment tracking is done via Weights & Biases (configured in the config JSON).

```bash
# Minimal — uses paths from the config
python scripts/training-scripts/finetune_qlora.py --config configs/qlora_config.json

# Override dataset and output directory
python scripts/training-scripts/finetune_qlora.py \
    --config configs/qlora_config.json \
    --train-data datasets/alpaca_train_formatted \
    --model-output-dir outputs/
```

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--config` | Yes | — | Path to `qlora_config.json` |
| `--train-data` | No | Value from `config.data.train_data_path` | Path to a pre-formatted HuggingFace Dataset folder (output of `format_dataset.py`) |
| `--model-output-dir` | No | Derived from `config.training.output_dir` | Root directory where the run folder, trainer checkpoints, and final model are saved |

**Config sections used:**

| Section | Purpose |
|---|---|
| `model` | Base model name, max sequence length, 4-bit quantisation flag |
| `lora` | LoRA rank (`r`), alpha, dropout, target modules, gradient checkpointing |
| `training.sft_config` | All SFTTrainer/TrainingArguments settings (batch size, learning rate, scheduler, etc.) |
| `training.output_dir` | Default output directory for the run |
| `training.final_model_subdir` | Subdirectory name for the saved adapter + tokenizer |
| `wandb` | W&B project name, run name, enabled flag |

**Output structure:**

```
<output_dir>/<run_name>/
├── trainer/            # Checkpoints saved during training
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── ...
└── <final_model_subdir>/   # Final LoRA adapter + tokenizer
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files...
```

**Precision:** If `bf16`/`fp16` are not set in the config, the script auto-detects GPU capability (bf16 on Ampere+, fp16 otherwise).

---

### 2. `submit_azureml_job.py`

Submits the `finetune_qlora.py` training job to an Azure ML compute cluster. Packages the `scripts/` directory as the code artifact, uploads the config and training data as inputs, and mounts a blob store for output.

Requires:
- Azure credentials (via `DefaultAzureCredential`)
- A `WANDB_API_KEY` environment variable (or the name configured in `config.azureml.wandb_api_key_env_var`) if W&B is enabled
- An `.env` file in the project root (optional, for loading env vars)

```bash
python scripts/training-scripts/submit_azureml_job.py --config configs/qlora_config.json
```

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--config` | No | `configs/qlora_config.json` | Path to the QLoRA config JSON |

**Config section: `azureml`**

| Key | Default | Description |
|---|---|---|
| `compute` | — | Name of the Azure ML compute target (required) |
| `experiment_name` | `dutch-mistral-qlora` | Azure ML experiment name |
| `display_name` | `qlora-finetune` | Display name for the job |
| `description` | `QLoRA finetuning job` | Job description |
| `output_uri` | `azureml://datastores/workspaceblobstore/paths/finetuning-output/` | Blob store path for model output |
| `environment_name` | `qlora-unsloth-env` | Name of the Azure ML environment |
| `environment_version` | `1` | Environment version |
| `instance_count` | `1` | Number of compute instances |
| `timeout_minutes` | `1440` | Job timeout in minutes |
| `stream_logs` | `true` | Stream logs to the terminal after submission |
| `wandb_api_key_env_var` | `WANDB_API_KEY` | Name of the env var holding the W&B API key |
| `subscription_id` | auto | Azure subscription (optional; falls back to `MLClient.from_config`) |
| `resource_group` | auto | Azure resource group (optional) |
| `workspace_name` | auto | Azure ML workspace (optional) |

The Conda environment is defined in `azureml/conda.yml`.
