"""Dataset loading for the XAI fault-localization benchmark.

Every loader returns (X, y): a pandas DataFrame of features with human-readable
column names, and a 0/1 pandas Series for the label. Categorical columns stay as
strings here; encoding for the models happens in features.py so that encoded
columns can be traced back to their original column.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]

GERMAN_COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_rate", "personal_status",
    "other_parties", "residence_since", "property", "age", "other_installment",
    "housing", "existing_credits", "job", "num_dependents", "telephone",
    "foreign_worker", "credit_risk",
]


def load_adult() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "adult.csv", header=None, names=ADULT_COLUMNS,
                     skipinitialspace=True, na_values="?")
    df = df.dropna().reset_index(drop=True)
    y = (df["income"] == ">50K").astype(int)
    # fnlwgt is a survey sampling weight, not an attribute of the person.
    # education is redundant with education_num.
    X = df.drop(columns=["income", "fnlwgt", "education"])
    return X, y


def load_german() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "german.csv", header=None, names=GERMAN_COLUMNS)
    y = (df["credit_risk"] == 2).astype(int)  # 1 = bad credit risk
    X = df.drop(columns=["credit_risk"])
    return X, y


def load_compas() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "compas.csv")
    # Standard ProPublica filtering for the two-year recidivism analysis.
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].reset_index(drop=True)
    y = df["two_year_recid"].astype(int)
    # decile_score / score_text are COMPAS's own prediction of the label, so
    # including them would be an accidental leak. age_cat is derived from age.
    keep = ["sex", "age", "race", "juv_fel_count", "juv_misd_count",
            "juv_other_count", "priors_count", "c_charge_degree"]
    X = df[keep].copy()
    return X, y


LOADERS = {"adult": load_adult, "german": load_german, "compas": load_compas}


def load_dataset(name: str, n_max: int | None = None, seed: int = 0):
    """Load a dataset by name. If n_max is set and the data is larger, take a
    seeded random subsample (keeps experiments affordable on small machines)."""
    X, y = LOADERS[name]()
    if n_max is not None and len(X) > n_max:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X), size=n_max, replace=False))
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    return X, y
