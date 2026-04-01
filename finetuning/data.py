import pandas as pd
from datasets import Dataset


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


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df)
