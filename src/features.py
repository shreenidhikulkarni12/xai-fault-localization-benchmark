"""Feature encoding that remembers which original column each encoded column came from.

Models need numbers, so categorical columns are one-hot encoded and numeric columns
are standardized. Explanations are produced per encoded column, but the "culprit"
of an injected fault is an original column, so we keep a map from encoded column
to source column and can aggregate attributions back to original columns.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_encoder(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("str", "string", "category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        verbose_feature_names_out=False,
    )


def encoded_to_source(encoder: ColumnTransformer, X: pd.DataFrame) -> list[str]:
    """For each encoded column, the name of the original column it came from."""
    num_cols = list(encoder.transformers_[0][2])
    cat_cols = list(encoder.transformers_[1][2])
    sources = list(num_cols)
    ohe = encoder.named_transformers_["cat"]
    for col, cats in zip(cat_cols, ohe.categories_):
        sources.extend([col] * len(cats))
    return sources


def aggregate_to_source(attr: np.ndarray, sources: list[str]) -> dict[str, float]:
    """Importance of each ORIGINAL column: sum the SIGNED attributions of its encoded
    columns, then take the absolute value. (SHAP values are additive, so signs must
    be kept when grouping; taking |.| first would inflate one-hot categorical columns.)"""
    total: dict[str, float] = {}
    for a, s in zip(attr, sources):
        total[s] = total.get(s, 0.0) + float(a)
    return {k: abs(v) for k, v in total.items()}
