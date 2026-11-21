"""Pilot check for the subgroup-label-noise fault.

Run: python src/pilot_subgroup.py
Reports overall test accuracy and test accuracy inside the age<30 subgroup,
clean vs mild (40%) vs severe (70%) majority-label flips.
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
from faults import in_subgroup, inject_subgroup_noise
from features import make_encoder

SEEDS = [0, 1, 2]


def run(ds, model_name, seed, flip=None):
    X, y = load_dataset(ds, n_max=N_MAX, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    if flip is not None:
        rng = np.random.default_rng(2000 + seed)
        Xtr, ytr, Xte, _ = inject_subgroup_noise(Xtr, ytr, Xte, rng, flip)
    enc = make_encoder(Xtr)
    Ztr, Zte = enc.fit_transform(Xtr), enc.transform(Xte)
    model = make_model(model_name, seed).fit(Ztr, ytr)
    pred = model.predict(Zte)
    sub = in_subgroup(Xte)
    return ((pred == yte).mean(), (pred[sub] == yte.to_numpy()[sub]).mean(),
            (pred[~sub] == yte.to_numpy()[~sub]).mean(), sub.sum(), int((pred != yte).sum()))


if __name__ == "__main__":
    print(f"{'dataset':8s} {'model':7s} {'setting':10s} {'all_acc':>8s} {'sub_acc':>8s} {'rest_acc':>8s} {'n_sub':>6s} {'errors':>7s}")
    for ds in ["adult", "german", "compas"]:
        for m in ["logreg", "gbt"]:
            for label, fl in [("clean", None), ("mild .40", 0.40), ("severe .70", 0.70)]:
                r = np.mean([run(ds, m, s, fl) for s in SEEDS], axis=0)
                print(f"{ds:8s} {m:7s} {label:10s} {r[0]:8.3f} {r[1]:8.3f} {r[2]:8.3f} {r[3]:6.0f} {r[4]:7.0f}")
