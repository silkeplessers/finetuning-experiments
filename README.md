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
│   └── qlora_config.json              # Model, LoRA, and training hyperparameters
├── datasets/
│   ├── alpaca_data_cleaned-dutch.jsonl # Original Dutch Alpaca dataset (51K rows)
│   ├── alpaca_data_cleaned-dutch-clean.jsonl  # Cleaned version (no code/math/english)
│   ├── alpaca_train.jsonl             # Training split (+ synthetic data)
│   ├── alpaca_test.jsonl              # Test split (+ synthetic data)
│   └── synthetic_dutch.jsonl          # Synthetic Dutch data from Azure OpenAI
├── finetuning/                        # Shared Python modules
│   ├── blob_storage.py                # Azure Blob upload/download helpers
│   ├── config.py                      # Config loader
│   ├── data.py                        # Dataset loading and preprocessing
│   ├── inference.py                   # Inference utilities
│   ├── model.py                       # Model loading (base + finetuned)
│   └── prompts.py                     # System prompt and chat template builders
├── scripts/
│   ├── data-processing/               # Data pipeline scripts
│   │   ├── clean_dataset.py           # Remove duplicates, english, code, math
│   │   ├── split_dataset.py           # Random train/test split
│   │   ├── generate_synthetic_data.py # Generate Dutch data via Azure OpenAI
│   │   ├── merge_synthetic_data.py    # Append synthetic data to train/test
│   │   ├── format_dataset.py          # Format for Mistral chat template
│   │   └── formatting.py             # Formatting helpers
│   ├── training-scripts/
│   │   ├── finetune_qlora.py          # QLoRA finetuning with Unsloth
│   │   └── submit_azureml_job.py      # Submit training job to Azure ML
│   ├── evaluation-scripts/
│   │   └── run_evaluation.py          # Evaluate finetuned vs base model
│   └── inference-scripts/
│       └── run_inference.py           # Run inference on finetuned model
├── outputs/                           # Training checkpoints and merged models
├── azureml/                           # Azure ML environment config
├── .env                               # Azure endpoints, storage config (not committed)
├── requirements.txt
└── data_quality.ipynb                 # Dataset quality analysis notebook
```

## Setup

```bash
conda activate finetune_env
pip install -r requirements.txt
```

Create a `.env` file with:

```
ENDPOINT=https://<your-resource>.openai.azure.com/openai/v1/
DEPLOYMENT=<your-deployment-name>
STORAGE_ACCOUNT=<your-storage-account>
CONTAINER_NAME=<your-container-name>
```

## Data Pipeline

Run the scripts in order (from the project root):

```bash
# 1. Clean the original dataset (removes duplicates, english, code, math)
python scripts/data-processing/clean_dataset.py

# 2. Split into train/test (80/20)
python scripts/data-processing/split_dataset.py --data datasets/alpaca_data_cleaned-dutch-clean.jsonl

# 3. Generate synthetic Dutch data via Azure OpenAI
python scripts/data-processing/generate_synthetic_data.py --num-examples 5000 --concurrency 50

# 4. Merge synthetic data into train/test (70/30 split)
python scripts/data-processing/merge_synthetic_data.py

# 5. Format for Mistral chat template
python scripts/data-processing/format_dataset.py --data datasets/alpaca_train.jsonl --config configs/qlora_config.json
python scripts/data-processing/format_dataset.py --data datasets/alpaca_test.jsonl --config configs/qlora_config.json --output datasets/alpaca_test_formatted
```

## Training

```bash
# Local training
python scripts/training-scripts/finetune_qlora.py --config configs/qlora_config.json

# Submit to Azure ML
python scripts/training-scripts/submit_azureml_job.py --config configs/qlora_config.json
```

## Inference & Evaluation

```bash
python scripts/inference-scripts/run_inference.py --config configs/qlora_config.json
python scripts/evaluation-scripts/run_evaluation.py --config configs/qlora_config.json
```

## Model

- **Base model:** `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- **Method:** QLoRA (rank=16, alpha=16, 4-bit NF4 quantization)
- **Max sequence length:** 2048 tokens
- **Tracking:** Weights & Biases (`dutch-mistral` project)

## Dataset

The training data combines:
- **Cleaned Alpaca Dutch** (~47K rows) — translated Stanford Alpaca with code, math, English, and duplicates removed
- **Synthetic Dutch** — generated via Azure OpenAI with varied topics, task types, and response lengths to supplement the short Alpaca outputs