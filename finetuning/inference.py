import logging

import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

from finetuning.prompts import build_inference_prompt

logger = logging.getLogger(__name__)


def run_inference(
    model,
    tokenizer,
    inputs: list[str],
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> list[str]:
    """Run batched inference on a list of input texts and return generated answers."""
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = [build_inference_prompt(tokenizer, text) for text in inputs]
    predictions = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Running inference"):
        batch_prompts = prompts[i : i + batch_size]
        tokens = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True,
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                temperature=0.1,
                use_cache=True,
            )

        input_len = tokens["input_ids"].shape[-1]
        for output in outputs:
            generated = output[input_len:]
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            predictions.append(answer)

    return predictions
