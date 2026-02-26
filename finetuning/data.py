import pandas as pd
from datasets import Dataset

def to_hf_dataset(df):
    return Dataset.from_pandas(df)
