"""Pilot check for the spurious-feature fault: does it actually hurt test accuracy?

Run: python src/pilot_spurious.py
For each dataset/model/severity, compare clean vs faulty test accuracy over a few seeds.
If the faulty model's test accuracy does not drop, the fault is too weak to be useful.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split

from baseline import make_model, N_MAX
from data import load_dataset
from faults import inject_spurious
from features import make_encoder

SEEDS = [0, 1, 2]


def run(ds, model_name, seed, agreement=None):
    X, y = load_dataset(ds, n_max=N_MAX, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    if agreement is not None:
        rng = np.random.default_rng(1000 + seed)
        Xtr, ytr, Xte, _ = inject_spurious(Xtr, ytr, Xte, rng, agreement)
    enc = make_encoder(Xtr)
    Ztr, Zte = enc.fit_transform(Xtr), enc.transform(Xte)
    model = make_model(model_name, seed).fit(Ztr, ytr)
    return model.score(Ztr, ytr), model.score(Zte, yte), int((model.predict(Zte) != yte).sum())


if __name__ == "__main__":
    print(f"{'dataset':8s} {'model':7s} {'setting':9s} {'train_acc':>9s} {'test_acc':>9s} {'n_errors':>9s}")
    for ds in ["adult", "german", "compas"]:
        for m in ["logreg", "gbt"]:
            for label, agr in [("clean", None), ("mild .70", 0.70), ("severe .95", 0.95)]:
                res = np.mean([run(ds, m, s, agr) for s in SEEDS], axis=0)
                print(f"{ds:8s} {m:7s} {label:9s} {res[0]:9.3f} {res[1]:9.3f} {res[2]:9.0f}")
