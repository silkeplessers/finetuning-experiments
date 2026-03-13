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
) -> list[str]:
    """Run inference on a list of input texts and return generated answers."""
    FastLanguageModel.for_inference(model)

    predictions = []
    for input_text in tqdm(inputs, desc="Running inference"):
        prompt = build_inference_prompt(input_text)
        tokens = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        input_len = tokens["input_ids"].shape[-1]
        generated = outputs[0][input_len:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        predictions.append(answer)

    return predictions
