import argparse
import json
import os
import re
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import wandb
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from formatting import format_prompt_batch


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def load_train_dataset(config: dict, train_data_override: str | None = None) -> Dataset:
    data_cfg = config["data"]
    train_data_path = train_data_override or data_cfg["train_data_path"]

    train_dataframe = pd.read_json(train_data_path, orient="records", lines=True)
    return Dataset.from_pandas(train_dataframe)


def init_wandb(config: dict, peft_config) -> wandb.sdk.wandb_run.Run | None:
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", True):
        return None

    return wandb.init(
        project=wandb_cfg["project"],
        name=wandb_cfg["run_name"],
        config=peft_config,
    )


def sanitize_path_component(value: str) -> str:
    cleaned_value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned_value or "run"


def resolve_run_name(config: dict, run: wandb.sdk.wandb_run.Run | None) -> str:
    if run is not None and run.name:
        return sanitize_path_component(run.name)

    wandb_cfg = config.get("wandb", {})
    configured_run_name = wandb_cfg.get("run_name", "run")
    return sanitize_path_component(configured_run_name)


def build_sft_config(config: dict, output_dir: str) -> SFTConfig:
    sft_cfg = config["training"]["sft_config"].copy()
    sft_cfg["output_dir"] = output_dir

    if "bf16" not in sft_cfg and "fp16" not in sft_cfg:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            supports_bf16 = major >= 8
            sft_cfg["bf16"] = supports_bf16
            sft_cfg["fp16"] = not supports_bf16
        else:
            sft_cfg["bf16"] = False
            sft_cfg["fp16"] = False

    return SFTConfig(**sft_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a QLoRA model from JSON config")
    parser.add_argument("--config", required=True, help="Path to qlora_config.json")
    parser.add_argument(
        "--train-data",
        required=False,
        default=None,
        help="Optional path override for training .jsonl",
    )
    parser.add_argument(
        "--model-output-dir",
        required=False,
        default=None,
        help="Optional output directory where final model/tokenizer will be saved",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    # format dataset for training
    dataset = load_train_dataset(config, args.train_data)
    data_cfg = config["data"]
    eos_token = tokenizer.eos_token
    formatted_dataset = dataset.map(
        partial(
            format_prompt_batch,
            chat_template=data_cfg["chat_template"],
            input_column=data_cfg["input_column"],
            output_column=data_cfg["output_column"],
            eos_token=eos_token,
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # initialize model and tokenizer
    model_cfg = config["model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
    )
    # apply peft/qlora to model
    lora_cfg = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
        use_rslora=lora_cfg["use_rslora"],
        loftq_config=lora_cfg["loftq_config"],
    )
    # intialize wandb experiment tracking
    run = init_wandb(config, model.peft_config)
    
    # determine output directories for trainer and final model
    run_dir_name = resolve_run_name(config, run)
    
    training_cfg = config["training"]
    configured_output_dir = Path(training_cfg.get("output_dir", "outputs/run"))

    if args.model_output_dir:
        output_root_dir = Path(args.model_output_dir)
    else:
        output_root_dir = configured_output_dir.parent if configured_output_dir.parent != Path("") else configured_output_dir

    run_dir = output_root_dir / run_dir_name
    trainer_output_dir = run_dir / "trainer"
    sft_config = build_sft_config(config, str(trainer_output_dir))
    print(f"Precision settings -> bf16: {sft_config.bf16}, fp16: {sft_config.fp16}")

    # initialize trainer and start training
    trainer = SFTTrainer(
        model=model,
        processing_class = tokenizer,
        train_dataset=formatted_dataset,
        #dataset_text_field="text",
        #max_seq_length=model_cfg["max_seq_length"],
        #packing=sft_config.packing,
        args=sft_config,
    )

    trainer.train()

    final_model_dir = run_dir / training_cfg["final_model_subdir"]
    # save final model and tokenizer to output directory
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    if run is not None:
        run.finish()

    print(f"Run directory: {run_dir}")
    print(f"Trainer outputs saved to: {trainer_output_dir}")
    print(f"Final model saved to: {final_model_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
