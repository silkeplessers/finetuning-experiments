def format_prompt_batch(batch, chat_template, input_column, output_column, eos_token):
    inputs = batch[input_column]
    outputs = batch[output_column]
    texts = []
    for user_input, assistant_output in zip(inputs, outputs):
        text = chat_template.format(INPUT=user_input, OUTPUT=assistant_output) + eos_token
        texts.append(text)
    return {"text": texts}
