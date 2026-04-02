import logging
import math
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset

logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def merge_instruction_into_input(
    df: pd.DataFrame,
    instruction_col: str = "instruction",
    input_col: str = "input",
) -> pd.DataFrame:
    """Concatenate the instruction and input columns into a single input column."""
    df = df.copy()
    df["prompt"] = (
        df[instruction_col].fillna("").str.strip()
        + "\ninput: "
        + df[input_col].fillna("").str.strip()
    ).str.strip()
    return df


def split_train_test(
    data: pd.DataFrame,
    train_frac: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.sample(frac=train_frac, random_state=random_state).reset_index(drop=True)
    test = data.drop(train.index).reset_index(drop=True)
    return train, test


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df)


# ---------------------------------------------------------------------------
# Model-based language quality scoring
# ---------------------------------------------------------------------------
_fasttext_model = None
_ppl_model = None
_ppl_tokenizer = None

FASTTEXT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lid.176.bin"
DUTCH_GPT2_MODEL = "GroNLP/gpt2-small-dutch"


def _get_fasttext_model():
    global _fasttext_model
    if _fasttext_model is None:
        import fasttext
        _fasttext_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
        logger.info("Loaded fastText LID model from %s", FASTTEXT_MODEL_PATH)
    return _fasttext_model


def _get_perplexity_model():
    global _ppl_model, _ppl_tokenizer
    if _ppl_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _ppl_tokenizer = AutoTokenizer.from_pretrained(DUTCH_GPT2_MODEL)
        _ppl_model = AutoModelForCausalLM.from_pretrained(DUTCH_GPT2_MODEL)
        if torch.cuda.is_available():
            _ppl_model = _ppl_model.cuda()
        _ppl_model.eval()
        logger.info("Loaded %s for perplexity scoring", DUTCH_GPT2_MODEL)
    return _ppl_model, _ppl_tokenizer


def dutch_confidence(text: str) -> float:
    """Return fastText confidence that the text is Dutch (0-1)."""
    model = _get_fasttext_model()
    clean = text.replace("\n", " ").strip()
    if not clean:
        return 0.0
    # Use the internal C API to avoid NumPy 2.x incompatibility in fasttext's predict()
    results = model.f.predict(clean, 1, 0.0, "")
    if not results:
        return 0.0
    conf, label = results[0]
    lang = label.replace("__label__", "")
    return conf if lang == "nl" else 0.0


def perplexity(text: str, max_length: int = 512) -> float:
    """Compute perplexity of text using the Dutch GPT-2 model. Lower = more fluent."""
    model, tokenizer = _get_perplexity_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return math.exp(loss.item())


def perplexity_batch(texts: list[str], batch_size: int = 64, max_length: int = 512) -> list[float]:
    """Compute perplexity for a list of texts in batches on GPU. Much faster than one-by-one."""
    model, tokenizer = _get_perplexity_model()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    n_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info("Computing perplexity for %d texts in %d batches (batch_size=%d) on %s",
                len(texts), n_batches, batch_size, device)

    all_ppls = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Create labels: ignore padding tokens by setting them to -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # outputs.loss is the mean over all non-ignored tokens in the batch.
            # We need per-example loss, so compute it manually.
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            # (batch, seq_len)
            token_losses = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())

            # Mask out padding and compute per-example mean loss
            mask = (shift_labels != -100).float()
            per_example_loss = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        for loss_val in per_example_loss:
            all_ppls.append(math.exp(loss_val.item()))

        if (i // batch_size + 1) % 50 == 0 or i + batch_size >= len(texts):
            logger.info("  perplexity progress: %d/%d texts", min(i + batch_size, len(texts)), len(texts))

    return all_ppls


def _ppl_to_score(ppl: float) -> float:
    if ppl < 50:
        return 0.35
    elif ppl < 100:
        return 0.25
    elif ppl < 200:
        return 0.15
    elif ppl < 400:
        return 0.05
    return 0.0


def _lang_to_score(nl_conf_avg: float) -> float:
    if nl_conf_avg > 0.9:
        return 0.30
    elif nl_conf_avg > 0.7:
        return 0.20
    elif nl_conf_avg > 0.5:
        return 0.10
    return 0.0


def _length_to_score(word_count: int) -> float:
    if word_count < 10:
        return 0.0
    elif word_count < 20:
        return 0.05
    elif word_count < 30:
        return 0.10
    elif word_count <= 150:
        return 0.20
    elif word_count <= 250:
        return 0.15
    return 0.05


def _completeness_to_score(output: str) -> float:
    stripped = output.rstrip()
    if stripped and stripped[-1] in ".!?\"')":
        return 0.15
    elif stripped and stripped[-1] in ":;,":
        return 0.05
    return 0.0


def heuristic_quality_score(row: dict) -> dict:
    """Score a row using fastText language ID, Dutch GPT-2 perplexity, and structure checks.

    Returns a dict with component scores and the total (0-1).
    """
    output = row["output"]
    instruction = row["instruction"]
    word_count = len(output.split())

    nl_conf_output = dutch_confidence(output)
    nl_conf_instr = dutch_confidence(instruction)
    nl_conf_avg = (nl_conf_output * 0.7) + (nl_conf_instr * 0.3)
    lang_score = _lang_to_score(nl_conf_avg)

    ppl = perplexity(output)
    ppl_score = _ppl_to_score(ppl)

    length_score = _length_to_score(word_count)
    completeness_score = _completeness_to_score(output)

    total = lang_score + ppl_score + length_score + completeness_score

    return {
        "total": round(total, 2),
        "nl_confidence_output": round(nl_conf_output, 3),
        "nl_confidence_instruction": round(nl_conf_instr, 3),
        "nl_confidence_avg": round(nl_conf_avg, 3),
        "perplexity": round(ppl, 1),
        "word_count": word_count,
        "lang_score": lang_score,
        "ppl_score": ppl_score,
        "length_score": length_score,
        "completeness_score": completeness_score,
    }


def heuristic_quality_score_batch(
    rows: list[dict], batch_size: int = 64
) -> list[dict]:
    """Score rows in bulk. Batches perplexity on GPU; fastText runs on CPU per-row.

    Returns a list of score dicts (same format as heuristic_quality_score).
    """
    # 1. fastText language confidence (CPU, very fast — already batch-like)
    nl_conf_outputs = [dutch_confidence(r["output"]) for r in rows]
    nl_conf_instrs = [dutch_confidence(r["instruction"]) for r in rows]

    # 2. Batch perplexity on GPU
    output_texts = [r["output"] for r in rows]
    ppls = perplexity_batch(output_texts, batch_size=batch_size)

    # 3. Combine scores
    results = []
    for i, row in enumerate(rows):
        word_count = len(row["output"].split())
        nl_conf_avg = (nl_conf_outputs[i] * 0.7) + (nl_conf_instrs[i] * 0.3)
        lang_score = _lang_to_score(nl_conf_avg)
        ppl_score = _ppl_to_score(ppls[i])
        length_score = _length_to_score(word_count)
        completeness_score = _completeness_to_score(row["output"])

        total = lang_score + ppl_score + length_score + completeness_score

        results.append({
            "total": round(total, 2),
            "nl_confidence_output": round(nl_conf_outputs[i], 3),
            "nl_confidence_instruction": round(nl_conf_instrs[i], 3),
            "nl_confidence_avg": round(nl_conf_avg, 3),
            "perplexity": round(ppls[i], 1),
            "word_count": word_count,
            "lang_score": lang_score,
            "ppl_score": ppl_score,
            "length_score": length_score,
            "completeness_score": completeness_score,
        })

    return results
