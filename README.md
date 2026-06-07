# Dutch Language Finetuning Experiments

Finetuning Mistral 7B for Dutch language instruction-following using QLoRA on Azure ML.

## Goal

Improve a small language model's (SLM) ability to understand and generate natural Dutch text by:
1. Cleaning and preparing a Dutch instruction-following dataset (translated Alpaca)
2. Augmenting it with synthetic Dutch data generated via Azure OpenAI
3. Finetuning Mistral 7B Instruct v0.3 using QLoRA (4-bit quantization + LoRA adapters)
4. Evaluating the finetuned model against the base model using LLM-as-judge

## Project Structure

```
finetuning-experiments/
├── configs/
│   └── qlora_config.json              # Model, LoRA, training, and Azure ML hyperparameters
├── datasets/
│   ├── alpaca_data_cleaned-dutch.jsonl        # Original Dutch Alpaca dataset (~51K rows)
│   ├── alpaca_data_cleaned-dutch-clean.jsonl  # Cleaned (no code/math/english/duplicates)
│   ├── alpaca_train.jsonl             # Training split (+ synthetic data)
│   ├── alpaca_test.jsonl              # Test split (+ synthetic data)
│   └── synthetic_dutch.jsonl          # Synthetic Dutch data from Azure OpenAI
├── finetuning/                        # Shared Python modules
│   ├── blob_storage.py                # Azure Blob upload/download helpers
│   ├── config.py                      # Config loader
│   ├── data.py                        # Dataset loading, preprocessing, splitting
│   ├── eval_visualization.py          # Per-judge charts + agreement heatmap
│   ├── evaluation.py                  # Judge calls, aggregation, MLflow, persistence
│   ├── formatting.py                  # Chat template formatting helpers
│   ├── inference.py                   # Inference utilities
│   ├── judge_prompts.py               # System prompts for dual-judge evaluation
│   ├── model.py                       # Model loading (base + finetuned)
│   ├── prompts.py                     # System prompt and chat template builders
│   └── schemas.py                     # Pydantic models for structured output
├── scripts/
│   ├── data/                          # Data pipeline scripts
│   │   ├── clean_dataset.py
│   │   ├── split_dataset.py
│   │   ├── generate_synthetic_data.py
│   │   ├── merge_synthetic_data.py
│   │   └── format_dataset.py
│   ├── training/
│   │   ├── finetune_qlora.py          # QLoRA finetuning with Unsloth
│   │   └── submit_azureml_job.py      # Submit training job to Azure ML
│   ├── evaluation/
│   │   ├── run_evaluation.py
│   │   └── dashboard.py
│   └── inference/
│       ├── run_inference.py
│       └── submit_azureml_inference_job.py
├── azureml/                           # Azure ML environment definitions
│   ├── conda.yml
│   └── conda_inference.yml
├── infra/                             # Bicep templates to recreate the Azure environment
│   ├── main.bicep
│   └── README.md
├── outputs/                           # Local training/inference outputs
├── .env                               # Secrets — NOT committed
├── .env_example                       # Template
└── requirements.txt
```

---

## Azure resources you need to provision

This project assumes a specific Azure footprint. Recreating it in a new subscription
requires the following resources. All of them are provisioned by [infra/main.bicep](infra/main.bicep)
— see [infra/README.md](infra/README.md) for one-command deployment.

| Resource | Purpose | Notes |
|---|---|---|
| **Resource group** | Container for everything | Single RG keeps cleanup simple. |
| **Azure ML workspace** | Job orchestration, MLflow tracking, datastores | Auto-creates Key Vault, App Insights, Container Registry. |
| **Storage account** (StorageV2, HNS off) | Datasets + experiment outputs | The workspace also auto-creates one; you can reuse it. |
| ↳ Container `azureml-blobstore-<workspaceId>` | Default `workspaceblobstore` datastore | Auto-created by the workspace. Holds `finetuning-output/`, `inference-results/`, `eval-results/`. |
| ↳ Container `finetuning-training-data` | User-uploaded datasets | Created manually or by Bicep. |
| **GPU compute cluster** | Training + inference | One `Standard_NC24ads_A100_v4` node (or `Standard_NC6s_v3` for cheaper experiments). Min 0 / max 1 so it scales to zero when idle. |
| **User-assigned managed identity** (optional but recommended) | Blob access from training jobs without API keys | Referenced by `azureml.managed_identity_client_id` in `configs/qlora_config.json`. |
| **Azure OpenAI account** | Synthetic data generation + LLM-as-judge | Needs `gpt-5.3-chat` (data gen + judge 1) and `grok-4-1-fast-reasoning` (judge 2), or substitute equivalents. |
| ↳ Model deployments | `JUDGE_LLM_1`, `JUDGE_LLM_2`, `DEPLOYMENT` | Names referenced from `.env`. |
| **W&B account** | Training metrics | External (not Azure). Free tier is fine. |

### Required RBAC role assignments

| Identity | Role | Scope | Reason |
|---|---|---|---|
| Your user | `Storage Blob Data Contributor` | Storage account | Local scripts use `DefaultAzureCredential`. |
| Compute cluster system identity (or UAMI above) | `Storage Blob Data Contributor` | Storage account | Training/inference jobs read datasets, write outputs. |
| Your user | `Cognitive Services OpenAI User` | Azure OpenAI account | Synthetic data generation + judge calls. |

---

## Storage layout (blob containers + prefixes)

The code uses **two containers in one storage account**. If you recreate the project,
keep these names (or update the references listed below).

### Container 1: `azureml-blobstore-<workspaceId>` (default `workspaceblobstore`)

Auto-created by the AzureML workspace. Used for AzureML-managed inputs/outputs.

| Prefix | Written by | Read by | Contents |
|---|---|---|---|
| `finetuning-output/` | `scripts/training/submit_azureml_job.py` | `scripts/inference/run_inference.py` | Final merged model + LoRA adapter + tokenizer. ~14 GB merged, ~150 MB adapter only. |
| `inference-results/` | `scripts/inference/submit_azureml_inference_job.py` | `scripts/evaluation/run_evaluation.py` | `{model_label}_results.jsonl` files. |
| `eval-results/` | `scripts/evaluation/run_evaluation.py` | `scripts/evaluation/dashboard.py` | `{model_label}/row_scores.jsonl`, `aggregate.json`, `pairwise.jsonl`, `charts/*.png`. |

### Container 2: `finetuning-training-data` (user-managed)

Created manually (or via Bicep). Used by the data-pipeline scripts.

| Prefix | Written by | Contents |
|---|---|---|
| (root) | `scripts/data/split_dataset.py`, `merge_synthetic_data.py`, `generate_synthetic_data.py`, `subsample_alpaca.py` | `alpaca_train.jsonl`, `alpaca_test.jsonl`, `synthetic_dutch.jsonl`, subsample logs. |

### Hard-coded references to be aware of when migrating

When you stand the project up in a new subscription, search-and-replace these so
they point at your new resources:

- `scripts/inference/run_inference.py` — `STORAGE_ACCOUNT`, `ADAPTER_CONTAINER` constants.
- `scripts/evaluation/run_evaluation.py` — `CONTAINER` constant (the `azureml-blobstore-...` GUID).
- `scripts/evaluation/dashboard.py` — `STORAGE_ACCOUNT`, `EVAL_CONTAINER` constants.
- `configs/qlora_config.json` — `azureml.managed_identity_client_id` (UAMI client id).
- `.env` — `STORAGE_ACCOUNT`, `CONTAINER_NAME`, `ENDPOINT`, `DEPLOYMENT`, `JUDGE_LLM_1`, `JUDGE_LLM_2`.

---

## Setup

### 1. Local Python environment

```bash
conda create -n finetune_env python=3.12 -y
conda activate finetune_env
pip install -r requirements.txt
```

Training is GPU-only (Unsloth + bitsandbytes). Local inference also requires a CUDA GPU.

### 2. Sign in to Azure

```bash
az login
az account set --subscription <your-subscription-id>
```

### 3. Provision Azure resources

Either:

- **Automated:** `cd infra && ./deploy.sh` (see [infra/README.md](infra/README.md)), or
- **Manual:** create the resources listed in the table above via the portal/CLI.

### 4. Create the user container and assign RBAC

```bash
# After Bicep deployment (or do this manually)
az storage container create \
  --account-name <storage-account> \
  --name finetuning-training-data \
  --auth-mode login

# Give yourself blob data access
az role assignment create \
  --assignee $(az ad signed-in-user show --query id -o tsv) \
  --role "Storage Blob Data Contributor" \
  --scope $(az storage account show -n <storage-account> --query id -o tsv)
```

### 5. Create `.env`

```bash
cp .env_example .env
# Then edit .env with your endpoints, deployments, storage account, tokens.
```

Required variables:

```dotenv
HF_TOKEN=<your-huggingface-token>           # for gated/private models, optional otherwise
WANDB_API_KEY=<your-wandb-key>              # only if wandb.enabled=true in config
ENDPOINT=https://<aoai-resource>.openai.azure.com/openai/v1/
DEPLOYMENT=<aoai-chat-deployment>           # used by synthetic data generation
JUDGE_LLM_1=<aoai-judge-1-deployment>
JUDGE_LLM_2=<aoai-judge-2-deployment>
STORAGE_ACCOUNT=<your-storage-account>
CONTAINER_NAME=finetuning-training-data
```

### 6. Patch the config

Edit [configs/qlora_config.json](configs/qlora_config.json) and update the
`azureml` block with your `compute` name and (optionally) `managed_identity_client_id`.

---

## Data Pipeline

Run the scripts in order (from the project root):

```bash
# 1. Clean the original dataset (removes duplicates, english, code, math)
python scripts/data/clean_dataset.py

# 2. Split into train/test (80/20)
python scripts/data/split_dataset.py --data datasets/alpaca_data_cleaned-dutch-clean.jsonl

# 3. Generate synthetic Dutch data via Azure OpenAI
python scripts/data/generate_synthetic_data.py --num-examples 5000 --concurrency 50

# 4. Merge synthetic data into train/test (70/30 split)
python scripts/data/merge_synthetic_data.py

# 5. Format for Mistral chat template
python scripts/data/format_dataset.py --data datasets/alpaca_train.jsonl --config configs/qlora_config.json
python scripts/data/format_dataset.py --data datasets/alpaca_test.jsonl  --config configs/qlora_config.json --output datasets/alpaca_test_formatted
```

## Training

```bash
# Local training (requires CUDA GPU + ~20GB VRAM)
python scripts/training/finetune_qlora.py --config configs/qlora_config.json

# Submit to Azure ML
python scripts/training/submit_azureml_job.py --config configs/qlora_config.json
```

## Inference

```bash
python scripts/inference/run_inference.py --config configs/qlora_config.json
```

## Evaluation

Dual-judge evaluation with structured outputs, parallel execution, baseline caching, and inter-judge agreement (Cohen's Kappa). Results are persisted to blob storage and logged to MLflow.

```bash
# Evaluate baseline (scores are cached for reuse)
python scripts/evaluation/run_evaluation.py \
    --config configs/qlora_config.json \
    --model-label baseline

# Evaluate finetuned model (runs pairwise vs cached baseline)
python scripts/evaluation/run_evaluation.py \
    --config configs/qlora_config.json \
    --model-label mistral_r16_a16_e1_b16_w30

# Streamlit dashboard for cross-experiment comparison
streamlit run scripts/evaluation/dashboard.py
```

See [scripts/evaluation/README.md](scripts/evaluation/README.md) for full details.

## Model

- **Base model:** `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- **Method:** QLoRA (rank=16, alpha=16, 4-bit NF4 quantization)
- **Max sequence length:** 2048 tokens
- **Tracking:** Weights & Biases (`dutch-mistral` project)

## Dataset

The training data combines:
- **Cleaned Alpaca Dutch** (~47K rows) — translated Stanford Alpaca with code, math, English, and duplicates removed
- **Synthetic Dutch** — generated via Azure OpenAI with varied topics, task types, and response lengths to supplement the short Alpaca outputs

---

## Migrating to a new Azure subscription

Your current subscription will be closed, so plan to move artefacts out **before** it
expires. The recipe below assumes the destination subscription does not exist yet.

### Phase A — Export everything (do this now)

The total exportable footprint is roughly:

| Artefact | Approx size | Where it lives today |
|---|---|---|
| Source code | < 5 MB | This git repo |
| Cleaned + synthetic datasets | ~80 MB | `finetuning-training-data` container + `datasets/` locally |
| LoRA adapter (one run) | ~150 MB | `finetuning-output/<run>/adapter_*` in `azureml-blobstore-...` |
| Merged 7B model (fp16) | ~14 GB per run | `finetuning-output/<run>/` in `azureml-blobstore-...` |
| Inference results JSONL | < 50 MB total | `inference-results/` |
| Evaluation results (row_scores, charts, aggregates) | < 200 MB total | `eval-results/` |

**Recommendation:** export everything **except** the merged 14 GB models. Keep only the
LoRA adapters — you can re-merge later in 30 seconds with the base model. If you
absolutely need a merged copy, pick one (your best run) and offload that.

#### 1. Push the code

```bash
git remote add backup git@github.com:<you>/finetuning-experiments-backup.git
git push backup --all && git push backup --tags
```

#### 2. Dump blob containers locally

Install [azcopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10), then:

```bash
mkdir -p backup/{datasets,outputs,inference,eval}

# Datasets
azcopy login
azcopy copy "https://llmaml5615532443.blob.core.windows.net/finetuning-training-data/*" \
  ./backup/datasets/ --recursive=true

# Eval + inference results
azcopy copy "https://llmaml5615532443.blob.core.windows.net/azureml-blobstore-4c704101-7a51-4680-bcf8-f13966bf69b4/eval-results/*" \
  ./backup/eval/ --recursive=true
azcopy copy "https://llmaml5615532443.blob.core.windows.net/azureml-blobstore-4c704101-7a51-4680-bcf8-f13966bf69b4/inference-results/*" \
  ./backup/inference/ --recursive=true

# Only adapter files from each run (NOT the merged model)
azcopy copy "https://llmaml5615532443.blob.core.windows.net/azureml-blobstore-4c704101-7a51-4680-bcf8-f13966bf69b4/finetuning-output/*" \
  ./backup/outputs/ --recursive=true \
  --include-pattern "adapter_*;tokenizer*;special_tokens*;chat_template*;README.md;qlora_config.json"
```

#### 3. Park the artefacts somewhere durable

Pick one — listed cheapest-first:

| Destination | Pros | Cons |
|---|---|---|
| **Hugging Face Hub** (private repo, model + dataset) | Free up to 1 TB; built for ML artefacts; resumable LFS push | Adapters/datasets only — don't push secrets or PII |
| **GitHub LFS** | Same place as the code | LFS has a 1 GB free quota; pay-as-you-go above |
| **Google Drive / OneDrive** | Free 15 GB / 5 GB | Manual; no programmatic access |
| **Local NAS / external SSD** | Zero cost, fully under your control | Single point of failure |
| **Azure Blob in another existing subscription** | One-shot `azcopy copy` server-side | Requires the subscription |

For the **adapter + datasets path (~250 MB total)** any of the above works. A merged 14 GB
model fits on Google Drive free but is awkward; Hugging Face Hub (private repo with LFS)
is the cleanest option.

#### 4. Verify checksums before deleting anything

```bash
find backup -type f -exec sha256sum {} \; > backup/MANIFEST.sha256
```

### Phase B — Re-provision (when the new subscription arrives)

```bash
cd infra
az login
az account set --subscription <new-subscription-id>
./deploy.sh  # see infra/README.md
```

This creates the workspace, storage account, both containers, the compute cluster,
the user-assigned managed identity, and the OpenAI account + deployments.

### Phase C — Reload artefacts + reconfigure

1. Restore data + outputs:
   ```bash
   azcopy copy ./backup/datasets/* \
     "https://<new-storage>.blob.core.windows.net/finetuning-training-data/" --recursive=true
   azcopy copy ./backup/outputs/* \
     "https://<new-storage>.blob.core.windows.net/azureml-blobstore-<new-id>/finetuning-output/" --recursive=true
   # (and similarly for eval/, inference/)
   ```
2. Update the hard-coded references listed in [Hard-coded references](#hard-coded-references-to-be-aware-of-when-migrating).
3. Update `.env` with the new endpoints, deployment names, storage account, and UAMI client id.
4. Smoke-test:
   ```bash
   python scripts/inference/run_inference.py --config configs/qlora_config.json
   streamlit run scripts/evaluation/dashboard.py
   ```
