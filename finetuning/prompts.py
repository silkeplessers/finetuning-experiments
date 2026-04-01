SYSTEM_PROMPT = (
    "Beantwoord de volgende vraag zo goed mogelijk. "
    "Soms wordt er extra input meegegeven die je moet gebruiken bij het beantwoorden."
)


def _build_user_content(user_input: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{user_input}"


def build_training_text(tokenizer, user_input: str, assistant_output: str) -> str:
    """Build a full training example in Mistral chat format."""
    messages = [{"role": "user", "content": _build_user_content(user_input)}]
    prompt_part = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return prompt_part + assistant_output + tokenizer.eos_token


def build_inference_prompt(tokenizer, user_input: str) -> str:
    """Build an inference prompt in Mistral chat format (no assistant output)."""
    messages = [{"role": "user", "content": _build_user_content(user_input)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
