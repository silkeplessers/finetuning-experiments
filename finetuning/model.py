from unsloth import FastLanguageModel


def load_base_model(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def load_finetuned_model(
    adapter_path: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """Load a finetuned model directly via Unsloth — it resolves the base model from the adapter config."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer
