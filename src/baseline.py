"""Clean (no fault) baseline: train each model type on each dataset, report test accuracy.

Run: python src/baseline.py
Purpose: confirm the pipeline works and give a reference accuracy for later
comparison with the faulty models.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from data import load_dataset
from features import make_encoder

N_MAX = 10000  # subsample large datasets to keep the full grid affordable
SEED = 0


def make_model(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "gbt":
        return HistGradientBoostingClassifier(random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=seed)
    raise ValueError(name)


if __name__ == "__main__":
    print(f"{'dataset':8s} {'model':7s} {'n':>6s} {'pos%':>6s} {'train_acc':>9s} {'test_acc':>9s} {'secs':>6s}")
    for ds in ["adult", "german", "compas"]:
        X, y = load_dataset(ds, n_max=N_MAX, seed=SEED)
        # Split BEFORE any fitting or fault injection; stratify keeps class balance.
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        enc = make_encoder(Xtr)
        Ztr = enc.fit_transform(Xtr)  # fit encoder on training data only
        Zte = enc.transform(Xte)
        for m in ["logreg", "gbt", "mlp"]:
            t0 = time.time()
            model = make_model(m, SEED).fit(Ztr, ytr)
            print(f"{ds:8s} {m:7s} {len(X):6d} {100*y.mean():6.1f} "
                  f"{model.score(Ztr, ytr):9.3f} {model.score(Zte, yte):9.3f} {time.time()-t0:6.1f}")
