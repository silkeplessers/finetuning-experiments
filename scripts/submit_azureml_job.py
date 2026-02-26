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
    parser = argparse.ArgumentParser(description="Submit QLoRA fine-tuning job to Azure ML")
    parser.add_argument("--config", default="configs/qlora_config.json", help="Path to qlora config JSON")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    repo_root = config_path.parent.parent.resolve() if config_path.parent.name == "configs" else Path.cwd().resolve()

    load_dotenv(repo_root / ".env", override=False)
    load_dotenv(override=False)

    cfg = load_config(str(config_path))
    azure_cfg = cfg["azureml"]
    data_cfg = cfg["data"]
    wandb_cfg = cfg.get("wandb", {})
    output_uri = azure_cfg.get(
        "output_uri",
        "azureml://datastores/workspaceblobstore/paths/finetuning-output/",
    )

    train_data_path = Path(data_cfg["train_data_path"])
    if not train_data_path.is_absolute():
        train_data_path = (repo_root / train_data_path).resolve()

    code_dir = (repo_root / "scripts").resolve()
    conda_file = (repo_root / "azureml" / "conda.yml").resolve()

    environment = Environment(
        name=azure_cfg.get("environment_name", "qlora-unsloth-env"),
        version=azure_cfg.get("environment_version", "1"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file=str(conda_file),
    )

    environment_variables = {}
    wandb_api_key_env_var = azure_cfg.get("wandb_api_key_env_var", "WANDB_API_KEY")
    wandb_api_key_value = os.getenv(wandb_api_key_env_var)

    if wandb_cfg.get("enabled", True):
        if not wandb_api_key_value:
            raise RuntimeError(
                f"W&B is enabled, but env var '{wandb_api_key_env_var}' is not set. "
                f"Set it before submitting or add it to a local .env file, for example: {wandb_api_key_env_var}=<your_wandb_token>"
            )
        environment_variables["WANDB_API_KEY"] = wandb_api_key_value

    job = command(
        code=str(code_dir),
        command=(
            "python finetune_qlora.py "
            "--config ${{inputs.qlora_config}} "
            "--train-data ${{inputs.train_data}} "
            "--model-output-dir ${{outputs.model_output}}"
        ),
        inputs={
            "qlora_config": Input(type="uri_file", path=str(config_path)),
            "train_data": Input(type="uri_file", path=str(train_data_path)),
        },
        outputs={
            "model_output": Output(type="uri_folder", mode="rw_mount", path=output_uri),
        },
        environment=environment,
        compute=azure_cfg["compute"],
        experiment_name=azure_cfg.get("experiment_name", "dutch-mistral-qlora"),
        display_name=azure_cfg.get("display_name", "qlora-finetune"),
        description=azure_cfg.get("description", "QLoRA finetuning job"),
        instance_count=int(azure_cfg.get("instance_count", 1)),
        timeout=int(azure_cfg.get("timeout_minutes", 1440)),
        environment_variables=environment_variables,
    )

    ml_client = build_ml_client(azure_cfg)
    submitted_job = ml_client.jobs.create_or_update(job)

    print(f"Submitted Azure ML job: {submitted_job.name}")
    print(f"Studio URL: {submitted_job.studio_url}")
    print(f"Model output path: {output_uri}")

    if azure_cfg.get("stream_logs", True):
        ml_client.jobs.stream(submitted_job.name)


if __name__ == "__main__":
    main()
