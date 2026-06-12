"""Submit evaluation job to Azure ML for comparing baseline and finetuned models.

Wraps the evaluation pipeline to run as an Azure ML job. Supports optional baseline
evaluation or uses cached baseline scores for faster turnaround.

Usage:
    python scripts/evaluation/submit_eval_job.py \
        --model-label mistral_r16_a16_e1_b16_w30 \
        --baseline-eval false

    python scripts/evaluation/submit_eval_job.py \
        --model-label baseline \
        --baseline-eval true \
        --test-size 10
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
    with open(config_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


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
        description="Submit evaluation job to Azure ML (baseline vs finetuned)"
    )
    parser.add_argument(
        "--model-label",
        type=str,
        required=True,
        help="Model label to evaluate (e.g., 'baseline' or the wandb run_name)",
    )
    parser.add_argument(
        "--baseline-eval",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Whether to run baseline evaluation (true) or use cached baseline scores (false)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Optional: evaluate only N samples for testing. Skips blob save and MLflow.",
    )
    parser.add_argument(
        "--config",
        default="azureml/eval_job_config.json",
        help="Path to evaluation job config JSON",
    )
    parser.add_argument(
        "--qlora-config",
        default="configs/qlora_config.json",
        help="Path to qlora config JSON (for judge endpoints and settings)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    repo_root = (
        config_path.parent.parent.resolve()
        if config_path.parent.name == "azureml"
        else Path.cwd().resolve()
    )

    load_dotenv(repo_root / ".env", override=False)
    load_dotenv(override=False)

    cfg = load_config(str(config_path))
    azure_cfg = cfg["azureml"]
    eval_cfg = cfg.get("evaluation", {})

    # Build job display name with model label and optional baseline flag
    baseline_flag = args.baseline_eval.lower() == "true"
    baseline_suffix = "-with-baseline" if baseline_flag else "-cached-baseline"
    display_name = f"eval-{args.model_label}{baseline_suffix}"

    code_dir = repo_root.resolve()
    conda_file = (repo_root / "azureml" / "conda_eval.yml").resolve()

    environment = Environment(
        name=azure_cfg.get("environment_name", "qlora-eval-env"),
        version=azure_cfg.get("environment_version", "1"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file=str(conda_file),
    )

    # Build command with CLI arguments
    # Note: entire repo_root is code_dir, so we use relative paths
    command_args = (
        "python scripts/evaluation/job_eval.py "
        "--model-label ${{inputs.model_label}} "
        "--baseline-eval ${{inputs.baseline_eval}} "
        "--qlora-config configs/qlora_config.json"
    )

    # Add optional test-size if provided
    if args.test_size is not None:
        command_args += f" --test-size {args.test_size}"

    # Add optional max-workers from config
    if "max_workers" in eval_cfg:
        command_args += f" --max-workers {eval_cfg['max_workers']}"

    job = command(
        code=str(code_dir),
        command=command_args,
        inputs={
            "model_label": args.model_label,
            "baseline_eval": str(args.baseline_eval),
        },
        outputs={
            "eval_output": Output(
                type="uri_folder",
                mode="rw_mount",
                path=azure_cfg.get(
                    "output_uri",
                    "azureml://datastores/workspaceblobstore/paths/eval-results/",
                ),
            ),
        },
        environment=environment,
        compute=azure_cfg["compute"],
        experiment_name=azure_cfg.get("experiment_name", "dutch-mistral"),
        display_name=display_name,
        description=azure_cfg.get("description", "Evaluation job"),
        instance_count=int(azure_cfg.get("instance_count", 1)),
        timeout=int(azure_cfg.get("timeout_minutes", 480)) * 60,
    )

    ml_client = build_ml_client(azure_cfg)
    submitted_job = ml_client.jobs.create_or_update(job)

    print(f"✓ Submitted evaluation job: {submitted_job.name}")
    print(f"  Model label: {args.model_label}")
    print(f"  Baseline eval: {baseline_flag}")
    if args.test_size is not None:
        print(f"  Test size: {args.test_size}")
    print(f"  Studio URL: {submitted_job.studio_url}")

    if azure_cfg.get("stream_logs", True):
        ml_client.jobs.stream(submitted_job.name)


if __name__ == "__main__":
    main()
