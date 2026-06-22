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
    max_new_tokens: int = 512,
    batch_size: int = 8,
    system_prompt: str | None = None,
    max_seq_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    seed: int | None = 3407,
) -> list[str]:
    """Run batched inference on a list of input texts and return generated answers.

    Uses sampling (temperature, top_p) plus a light repetition penalty to avoid
    the greedy-decoding loops that fine-tuned Mistral-7B exhibits on Alpaca-style
    SFT data. A fixed seed keeps runs reproducible.
    """
    FastLanguageModel.for_inference(model)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = [
        build_inference_prompt(tokenizer, text, system_prompt=system_prompt)
        for text in inputs
    ]
    max_input_length = max_seq_length - max_new_tokens

    # Warn on any prompts that will be truncated
    for idx, prompt in enumerate(prompts):
        full_len = len(tokenizer.encode(prompt, add_special_tokens=True))
        if full_len > max_input_length:
            logger.warning(
                "Prompt %d will be truncated from %d to %d tokens",
                idx,
                full_len,
                max_input_length,
            )

    # Sort by length for efficient batching (less padding waste)
    sorted_indices = sorted(range(len(prompts)), key=lambda k: len(prompts[k]))
    sorted_prompts = [prompts[k] for k in sorted_indices]
    sorted_predictions: list[str | None] = [None] * len(prompts)

    for i in tqdm(range(0, len(sorted_prompts), batch_size), desc="Running inference"):
        batch_indices = sorted_indices[i : i + batch_size]
        batch_prompts = sorted_prompts[i : i + batch_size]
        tokens = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        input_len = tokens["input_ids"].shape[-1]
        for j, output in enumerate(outputs):
            generated = output[input_len:]
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            sorted_predictions[batch_indices[j]] = answer

    return sorted_predictions
