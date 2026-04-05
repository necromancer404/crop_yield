import numpy as np
import pandas as pd


def fill_missing_numeric(df: pd.DataFrame, columns: list[str], strategy: str = "median") -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            continue
        col = pd.to_numeric(out[c], errors="coerce")
        if strategy == "median":
            fill = col.median()
        elif strategy == "mean":
            fill = col.mean()
        else:
            fill = 0.0
        if pd.isna(fill):
            fill = 0.0
        out[c] = col.fillna(fill)
    return out


def simple_impute_frame(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    return fill_missing_numeric(df, numeric_cols, strategy="median")
