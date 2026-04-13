def _build_user_content(user_input: str) -> str:
    return user_input


def _strip_bos(text: str, tokenizer) -> str:
    """Strip leading BOS token inserted by apply_chat_template to avoid double-BOS
    when the tokenizer adds its own via add_special_tokens=True."""
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        return text[len(tokenizer.bos_token):]
    return text


def build_training_text(tokenizer, user_input: str, assistant_output: str) -> str:
    """Build a full training example in Mistral chat format."""
    messages = [{"role": "user", "content": _build_user_content(user_input)}]
    prompt_part = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_part = _strip_bos(prompt_part, tokenizer)
    return prompt_part + assistant_output + tokenizer.eos_token


def build_inference_prompt(tokenizer, user_input: str, system_prompt: str | None = None) -> str:
    """Build an inference prompt in Mistral chat format (no assistant output)."""
    content = f"{system_prompt}\n\n{user_input}" if system_prompt else _build_user_content(user_input)
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return _strip_bos(text, tokenizer)
