"""Build one experimental setup: dataset + fault + severity + model + seed.

A setup returns the trained model, the encoded data, the mapping from encoded
columns back to original columns, the culprit column, and the indices of
mispredicted test rows (the instances we will later explain).
"""
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from baseline import make_model, N_MAX
from data import load_dataset
from faults import inject_leakage, inject_spurious, inject_subgroup_noise
from features import encoded_to_source, make_encoder

FAULTS = {
    "spurious": (inject_spurious, {"mild": 0.70, "severe": 0.95}),
    "subgroup": (inject_subgroup_noise, {"mild": 0.40, "severe": 0.70}),
    "leakage": (inject_leakage, {"mild": 0.50, "severe": 1.00}),
}
FAULT_IDX = {"spurious": 1, "subgroup": 2, "leakage": 3}


@dataclass
class Setup:
    dataset: str
    model_name: str
    fault: str
    severity: str
    seed: int
    model: object
    encoder: object
    sources: list          # source (original) column for each encoded column
    Ztr: np.ndarray        # encoded training data
    Zte: np.ndarray        # encoded test data
    yte: np.ndarray
    culprit: str           # original column an ideal explanation should point to
    wrong_idx: np.ndarray  # fault-induced errors: faulty model wrong AND clean model right
    test_acc: float
    n_wrong_all: int       # all mispredicted test rows of the faulty model
    clean_acc: float


def build_setup(dataset, model_name, fault, severity, seed):
    X, y = load_dataset(dataset, n_max=N_MAX, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    # Clean twin: same split, same model type and seed, no fault. Used only to decide
    # which errors the fault caused (faulty model wrong, clean model right).
    enc0 = make_encoder(Xtr)
    clean = make_model(model_name, seed).fit(enc0.fit_transform(Xtr), ytr)
    clean_pred = clean.predict(enc0.transform(Xte))
    inject, levels = FAULTS[fault]
    rng = np.random.default_rng(1000 * FAULT_IDX[fault] + seed)
    Xtr, ytr, Xte, culprit = inject(Xtr, ytr, Xte, rng, levels[severity])
    enc = make_encoder(Xtr)
    Ztr, Zte = enc.fit_transform(Xtr), enc.transform(Xte)
    model = make_model(model_name, seed).fit(Ztr, ytr)
    pred = model.predict(Zte)
    yte_arr = yte.to_numpy()
    induced = (pred != yte_arr) & (clean_pred == yte_arr)
    return Setup(
        dataset=dataset, model_name=model_name, fault=fault, severity=severity, seed=seed,
        model=model, encoder=enc, sources=encoded_to_source(enc, Xtr),
        Ztr=Ztr, Zte=Zte, yte=yte_arr, culprit=culprit,
        wrong_idx=np.flatnonzero(induced), test_acc=float((pred == yte_arr).mean()),
        n_wrong_all=int((pred != yte_arr).sum()), clean_acc=float((clean_pred == yte_arr).mean()),
    )


def sample_wrong(setup: Setup, n: int, seed: int = 0) -> np.ndarray:
    """A seeded random sample of up to n fault-induced errors (all of them if fewer than n)."""
    rng = np.random.default_rng(777 + seed)
    k = min(n, len(setup.wrong_idx))
    return np.sort(rng.choice(setup.wrong_idx, size=k, replace=False))
