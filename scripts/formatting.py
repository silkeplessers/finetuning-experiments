from finetuning.prompts import build_training_text


def format_prompt_batch(batch, tokenizer, input_column, output_column):
    inputs = batch[input_column]
    outputs = batch[output_column]
    texts = []
    for user_input, assistant_output in zip(inputs, outputs):
        text = build_training_text(tokenizer, user_input, assistant_output)
        texts.append(text)
    return {"text": texts}
