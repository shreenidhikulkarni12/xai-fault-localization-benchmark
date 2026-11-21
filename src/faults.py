"""Fault injection. Each function returns modified copies of the train and test
features/labels plus the name of the 'culprit': the original column an ideal
explanation should point to.

All functions take a numpy Generator so a seed fully determines the result.
Faults are applied AFTER the train/test split, so test data is never used to
build the fault in training.
"""
import numpy as np
import pandas as pd

SPURIOUS_COL = "spurious_feature"
LEAK_COL = "leaked_feature"
SUBGROUP_COL = "age"
SUBGROUP_MAX_AGE = 30  # subgroup = rows with age < 30


def inject_spurious(Xtr, ytr, Xte, rng, agreement):
    """Add a binary column that equals the label on `agreement` fraction of
    TRAINING rows (random otherwise) but is independent random noise in TEST.

    agreement: 0.70 (mild) or 0.95 (severe).
    Culprit: the added column.
    """
    Xtr, Xte = Xtr.copy(), Xte.copy()
    n_tr, n_te = len(Xtr), len(Xte)
    follow = rng.random(n_tr) < agreement
    rand_bits = rng.integers(0, 2, size=n_tr)
    Xtr[SPURIOUS_COL] = np.where(follow, ytr.to_numpy(), rand_bits).astype(int)
    Xte[SPURIOUS_COL] = rng.integers(0, 2, size=n_te)
    return Xtr, ytr, Xte, SPURIOUS_COL


def in_subgroup(X):
    """Boolean mask of rows in the noisy subgroup (age under 30)."""
    return (X[SUBGROUP_COL] < SUBGROUP_MAX_AGE).to_numpy()


def inject_subgroup_noise(Xtr, ytr, Xte, rng, flip_rate):
    """Systematically corrupt TRAINING labels for rows with age < 30: flip the
    subgroup's majority label to the minority label at rate `flip_rate`.
    Test labels are left clean, so test errors reveal what the model learned wrongly.

    Why the majority label: random two-way flips below 50% barely change what a
    model predicts (the majority still wins). Flipping the majority label shifts
    the subgroup's training label distribution, so the model learns wrong
    patterns for that group.

    flip_rate: 0.40 (mild) or 0.70 (severe).
    Culprit: the 'age' column, which defines the corrupted subgroup.
    """
    Xtr, Xte, ytr = Xtr.copy(), Xte.copy(), ytr.copy()
    sub = in_subgroup(Xtr)
    majority = int(round(ytr[sub].mean()))
    flip = sub & (ytr.to_numpy() == majority) & (rng.random(len(Xtr)) < flip_rate)
    ytr[flip] = 1 - majority
    return Xtr, ytr, Xte, SUBGROUP_COL


def inject_leakage(Xtr, ytr, Xte, rng, coverage):
    """Add a column computed EXACTLY from the label (a perfect cheat sheet) on a
    `coverage` fraction of TRAINING rows. On the remaining training rows, and on
    every TEST row, it is an independent random bit, so it carries no information
    at test time (the feature is not available when the model is actually used).

    A small jitter is added so the column looks like a numeric score, not a copy
    of the label.

    coverage: 0.50 (mild) or 1.00 (severe).
    Culprit: the added column.
    """
    Xtr, Xte = Xtr.copy(), Xte.copy()
    n_tr, n_te = len(Xtr), len(Xte)
    leaked = rng.random(n_tr) < coverage
    rand_bits = rng.integers(0, 2, size=n_tr)
    base = np.where(leaked, ytr.to_numpy(), rand_bits).astype(float)
    Xtr[LEAK_COL] = base + rng.normal(0, 0.05, size=n_tr)
    Xte[LEAK_COL] = rng.integers(0, 2, size=n_te).astype(float) + rng.normal(0, 0.05, size=n_te)
    return Xtr, ytr, Xte, LEAK_COL
