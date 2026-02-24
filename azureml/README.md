# Azure ML QLoRA finetuning

This folder contains Azure ML runtime assets for submitting your QLoRA finetuning job.

## Files

- `conda.yml`: Azure ML environment definition.

## Submit job

From repository root:

```bash
python scripts/submit_azureml_job.py --config configs/qlora_config.json
```

## W&B authentication on Azure ML

When `wandb.enabled` is true, the submit script reads `WANDB_API_KEY` from your local environment.

You can either:

- export it in your shell before submit, or
- store it in a local `.env` file (auto-loaded by the submit script).

Example `.env` entry:

```bash
WANDB_API_KEY=your_wandb_token
```

## Model output persistence

The job writes the final model + tokenizer to `outputs.model_output` in Azure ML. This output is persisted by Azure ML and remains available after compute nodes are deallocated.
