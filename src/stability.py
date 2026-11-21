"""Explanation correctness and stability for one experimental setup.

For each sampled mispredicted instance and each explainer (SHAP, LIME) we record:
  - correctness: rank of the culprit column in the explanation (top-1, top-3, reciprocal rank)
  - seed stability: the explanation is regenerated K times with different random seeds
    (same input); stability = mean pairwise Jaccard similarity of the top-k column sets
  - perturbation stability: the explanation is regenerated on K slightly perturbed copies of
    the input; stability = mean Jaccard between the original top-k set and each perturbed one
  - Spearman rank correlation of the per-column importance vectors, for the same two settings
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from explain import (culprit_rank, lime_attributions, make_lime_explainer, make_shap_explainer,
                     rank_sources, shap_attributions)
from features import aggregate_to_source
from pipeline import Setup, sample_wrong

TOP_K = 3
N_REPEATS = 10
COLUMNS = ["dataset", "model", "fault", "severity", "seed", "explainer", "instance", "culprit",
           "culprit_rank", "top1", "top3", "rr", "stab_seed", "stab_perturb", "rho_seed",
           "rho_perturb", "conf", "attr_total", "attr_top1_share", "test_acc", "clean_acc",
           "n_induced", "n_wrong_all"]
PERTURB_SD = 0.10  # noise std, in standardized units, added to numeric columns only


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def importance_vector(attr, sources, order):
    agg = aggregate_to_source(attr, sources)
    return np.array([agg[c] for c in order])


def safe_spearman(u, v):
    if np.ptp(u) == 0 or np.ptp(v) == 0:
        return np.nan
    return spearmanr(u, v)[0]


def explain_all(setup, name, Z, seed, shap_pair=None):
    """Attributions for rows of Z with one explainer at one random seed."""
    if name == "shap":
        pair = shap_pair if shap_pair is not None else make_shap_explainer(setup, seed=seed)
        return shap_attributions(setup, pair, Z, seed=seed)
    return lime_attributions(setup, make_lime_explainer(setup, seed=seed), Z)


def run_setup(setup: Setup, n_instances=50, n_repeats=N_REPEATS, top_k=TOP_K, perturb_sd=PERTURB_SD):
    rows = sample_wrong(setup, n_instances, seed=0)
    if len(rows) == 0:  # no fault-induced errors in this setup: record that, explain nothing
        return pd.DataFrame(columns=COLUMNS)
    Z0 = setup.Zte[rows]
    order = sorted(set(setup.sources))
    n_num = len(setup.encoder.transformers_[0][2])  # numeric columns come first after encoding
    rng = np.random.default_rng(5000 + setup.seed)
    # K perturbed copies of the instances (same copies reused for both explainers)
    perturbed = []
    for _ in range(n_repeats):
        Zp = Z0.copy()
        Zp[:, :n_num] += rng.normal(0, perturb_sd, size=(len(Z0), n_num))
        perturbed.append(Zp)

    out = []
    # RQ3 baselines: the model's confidence in its (wrong) prediction, per instance
    conf = setup.model.predict_proba(Z0).max(axis=1) if len(Z0) else np.array([])
    for name in ["shap", "lime"]:
        # seed repeats on the unchanged input
        base_pair = make_shap_explainer(setup, seed=0) if name == "shap" else None
        seed_attrs = []
        for sd in range(n_repeats):
            pair = base_pair if (name == "shap" and sd == 0) else None
            seed_attrs.append(explain_all(setup, name, Z0, sd, shap_pair=pair))
        # perturbation repeats: seed-0 explainer on perturbed inputs
        pert_attrs = [explain_all(setup, name, Zp, 0, shap_pair=base_pair) for Zp in perturbed]

        # RQ3 baselines from the base explanation: total attribution and how concentrated it is
        aggs = [np.array(list(aggregate_to_source(a, setup.sources).values())) for a in seed_attrs[0]]
        agg_total = np.array([v.sum() for v in aggs])
        agg_share = np.array([v.max() / v.sum() if v.sum() > 0 else np.nan for v in aggs])
        for i in range(len(Z0)):
            ranks_seed = [rank_sources(a[i], setup.sources) for a in seed_attrs]
            ranks_pert = [rank_sources(a[i], setup.sources) for a in pert_attrs]
            base_rank = ranks_seed[0]
            r = culprit_rank(base_rank, setup.culprit)
            vec_seed = [importance_vector(a[i], setup.sources, order) for a in seed_attrs]
            vec_pert = [importance_vector(a[i], setup.sources, order) for a in pert_attrs]
            out.append(dict(
                dataset=setup.dataset, model=setup.model_name, fault=setup.fault,
                severity=setup.severity, seed=setup.seed, explainer=name,
                instance=int(rows[i]), culprit=setup.culprit, culprit_rank=r,
                top1=int(r == 1), top3=int(r <= 3), rr=1.0 / r,
                stab_seed=np.mean([jaccard(x[:top_k], y[:top_k]) for x, y in combinations(ranks_seed, 2)]),
                stab_perturb=np.mean([jaccard(base_rank[:top_k], x[:top_k]) for x in ranks_pert]),
                rho_seed=np.nanmean([safe_spearman(vec_seed[0], v) for v in vec_seed[1:]]),
                rho_perturb=np.nanmean([safe_spearman(vec_seed[0], v) for v in vec_pert]),
                conf=float(conf[i]), attr_total=float(agg_total[i]), attr_top1_share=float(agg_share[i]),
                test_acc=setup.test_acc, clean_acc=setup.clean_acc,
                n_induced=len(setup.wrong_idx), n_wrong_all=setup.n_wrong_all,
            ))
    return pd.DataFrame(out, columns=COLUMNS)
