"""CDAS data prep. Builds scale scores, the anxiety target, and Model A/B tables."""
import numpy as np
import pandas as pd

DATA_PATH = r"D:\chanchal project\AI_Literacy_questionnaire_data_.csv"

A_ITEMS = ["A1", "A2", "A3", "A4", "A5"]
SCALES_NOZERO = {
    "L": ["L1", "L2", "L3", "L4", "L5"],
    "RE": ["RE1", "RE2", "RE3", "RE4", "RE5", "RE6"],
    "R": ["R1", "R2", "R3", "R4", "R5", "R6"],
    "SG": ["SG1", "SG2", "SG3", "SG4", "SG5"],
    "CM": ["CM1", "CM2", "CM3", "CM4"],
}
SCALES_ZERO = {  # 0 = N/A code -> NaN
    "IM": ["IM1", "IM2", "IM3", "IM4"],
    "S": ["S1", "S2", "S3", "S4", "S5"],
    "C": ["C1", "C2", "C3", "C4", "C5"],
    "BI": ["BI1", "BI2", "BI3", "BI4", "BI5"],
}
CAT = ["Gender", "Country", "Study_programme"]
NUM_A = ["L", "RE", "R", "SG", "CM", "Age", "Grade", "AI_hours"]
NUM_B = NUM_A + ["C", "IM", "S", "BI"]


def _find(cols, start):
    """Find the real column whose name starts with `start` (case-insensitive)."""
    for c in cols:
        if c.strip().lower().startswith(start.lower()):
            return c
    raise KeyError(f"No column starting with '{start}'. Columns: {list(cols)[:12]}...")


def load():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        _find(df.columns, "Age"): "Age",
        _find(df.columns, "Grade"): "Grade",
        _find(df.columns, "Gender"): "Gender",
        _find(df.columns, "Country"): "Country",
        _find(df.columns, "Study programme"): "Study_programme",
        _find(df.columns, "How many hours"): "AI_hours",
    })

    # Target: mean of A1..A5. A items are NEVER used as features.
    df["anxiety"] = df[A_ITEMS].mean(axis=1)

    # Scales with no 0 codes
    for name, items in SCALES_NOZERO.items():
        df[name] = df[items].mean(axis=1)

    # Scales where 0 = N/A: score = mean of non-zero items, NaN if all zero
    for name, items in SCALES_ZERO.items():
        df[name] = df[items].replace(0, np.nan).mean(axis=1)

    for c in CAT:
        df[c] = df[c].astype(str)
    return df


def model_a(df):
    """All rows. No structural-zero scales."""
    cols = NUM_A + CAT + ["anxiety", "user_id"]
    return df[cols].dropna().reset_index(drop=True)


def model_b(df):
    """Rows where C, IM, S, BI all exist. C is a PROXY for self-doubt (inverse)."""
    cols = NUM_B + CAT + ["anxiety", "user_id"]
    return df[cols].dropna().reset_index(drop=True)


if __name__ == "__main__":
    df = load()
    A, B = model_a(df), model_b(df)
    print("Rows loaded:", len(df))
    print("Model A rows:", len(A), "| Model B rows:", len(B))
    print("\nAnxiety target summary:")
    print(df["anxiety"].describe().round(3))
    print("\nScale score ranges:")
    print(df[list(SCALES_NOZERO) + list(SCALES_ZERO)].describe().loc[["count", "min", "max", "mean"]].round(2))
    print("\nCategorical levels:")
    for c in CAT:
        print(f"  {c}: {df[c].nunique()} levels, top: {df[c].value_counts().head(3).to_dict()}")
    print("\nCorrelation of each scale with anxiety:")
    print(df[list(SCALES_NOZERO) + list(SCALES_ZERO) + ["Age", "Grade", "AI_hours"]]
          .corrwith(df["anxiety"]).round(3).sort_values())