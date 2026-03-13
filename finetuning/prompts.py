CHAT_TEMPLATE = """\
Beantwoord de volgende vraag zo goed mogelijk.

### Vraag:
{INPUT}

### Antwoord:
{OUTPUT}
"""

INFERENCE_TEMPLATE = """\
Beantwoord de volgende vraag zo goed mogelijk.

### Vraag:
{INPUT}

### Antwoord:
"""


def build_inference_prompt(input_text: str) -> str:
    return INFERENCE_TEMPLATE.format(INPUT=input_text)
