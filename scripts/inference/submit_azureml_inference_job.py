"""Submit an inference job to an Azure ML compute cluster.

Usage:
    python scripts/inference/submit_azureml_inference_job.py --config configs/qlora_config.json

    # Override defaults via CLI
    python scripts/inference/submit_azureml_inference_job.py \
        --config configs/qlora_config.json \
        --mode finetuned \
        --batch-size 16 \
        --max-samples 200
"""

import argparse
import json
import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_ml_client(azure_cfg: dict) -> MLClient:
    subscription_id = azure_cfg.get("subscription_id")
    resource_group = azure_cfg.get("resource_group")
    workspace_name = azure_cfg.get("workspace_name")

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

    if subscription_id and resource_group and workspace_name:
        return MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )

    return MLClient.from_config(credential=credential)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit inference job to Azure ML"
    )
    parser.add_argument(
        "--config",
        default="configs/qlora_config.json",
        help="Path to qlora config JSON",
    )
    parser.add_argument(
        "--test-data",
        default="datasets/alpaca_test.jsonl",
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "finetuned", "both"],
        default="both",
        help="Which model(s) to run inference with (default: both)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate per example",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit test set to N samples (default: use all)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    repo_root = (
        config_path.parent.parent.resolve()
        if config_path.parent.name == "configs"
        else Path.cwd().resolve()
    )

    load_dotenv(repo_root / ".env", override=False)
    load_dotenv(override=False)

    cfg = load_config(str(config_path))
    azure_cfg = cfg["azureml"]

    test_data_path = Path(args.test_data)
    if not test_data_path.is_absolute():
        test_data_path = (repo_root / test_data_path).resolve()

    # The inference script needs the finetuning package, so code_dir is repo root.
    code_dir = repo_root.resolve()

    conda_file = (repo_root / "azureml" / "conda_inference.yml").resolve()
    environment = Environment(
        name=azure_cfg.get("inference_environment_name", "qlora-inference-env"),
        version=azure_cfg.get("inference_environment_version", "1"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file=str(conda_file),
    )

    print(f"Using registered environment: {environment}")
    output_uri = azure_cfg.get(
        "inference_results_uri",
        "azureml://datastores/workspaceblobstore/paths/inference-results/",
    )
    
    # Build the command string
    cmd_parts = [
        "python scripts/inference/run_inference.py",
        "--config ${{inputs.qlora_config}}",
        "--test-data ${{inputs.test_data}}",
        "--output-dir ${{outputs.model_output}}",
        f"--mode {args.mode}",
        f"--max-new-tokens {args.max_new_tokens}",
        f"--batch-size {args.batch_size}",
    ]
    if args.max_samples is not None:
        cmd_parts.append(f"--max-samples {args.max_samples}")

    run_name = cfg.get("wandb", {}).get("run_name", "inference")
    display_name = f"inference-{args.mode}-{run_name}"

    job = command(
        code=str(code_dir),
        command=" ".join(cmd_parts),
        inputs={
            "qlora_config": Input(type="uri_file", path=str(config_path)),
            "test_data": Input(type="uri_file", path=str(test_data_path)),
        },
        outputs={
            "model_output": Output(
                type="uri_folder", mode="rw_mount", path=output_uri
            ),
        },
        environment=environment,
        compute=azure_cfg["compute"],
        experiment_name=azure_cfg.get("experiment_name", "dutch-mistral-qlora-inference"),
        display_name=display_name,
        description=f"Inference job ({args.mode}) for {run_name}",
        instance_count=1,
        timeout=int(azure_cfg.get("timeout_minutes", 1440)) * 60,
    )

    ml_client = build_ml_client(azure_cfg)
    submitted_job = ml_client.jobs.create_or_update(job)

    print(f"Submitted Azure ML inference job: {submitted_job.name}")
    print(f"Studio URL: {submitted_job.studio_url}")
    print(f"Output path: {output_uri}")

    if azure_cfg.get("stream_logs", True):
        ml_client.jobs.stream(submitted_job.name)


if __name__ == "__main__":
    main()
