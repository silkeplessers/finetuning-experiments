from unsloth import FastLanguageModel

def load_model(model_name: str, max_sequence_length: int = 2048, load_in_4bit: bool = True):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_sequence_length,
        load_in_4bit = load_in_4bit,  #
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    return model, tokenizer