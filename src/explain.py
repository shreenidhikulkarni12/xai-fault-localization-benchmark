"""Generate SHAP and LIME explanations and turn them into rankings of ORIGINAL columns.

Each explanation is a vector of per-encoded-column attributions. We sum absolute
attribution over encoded columns that share an original column (so a one-hot
'occupation' counts as one feature), then rank original columns by that total.
"""
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

from features import aggregate_to_source


def rank_sources(attr, sources):
    """Original column names ordered from most to least important."""
    agg = aggregate_to_source(attr, sources)
    return [k for k, _ in sorted(agg.items(), key=lambda kv: -kv[1])]


def make_shap_explainer(setup, n_background=30, seed=0):
    """TreeSHAP for gradient boosting where supported, otherwise KernelSHAP."""
    rng = np.random.default_rng(seed)
    bg = setup.Ztr[rng.choice(len(setup.Ztr), size=n_background, replace=False)]
    f = lambda z: setup.model.predict_proba(z)[:, 1]
    if setup.model_name == "gbt":
        try:
            return "tree", shap.TreeExplainer(setup.model)
        except Exception:
            pass
    return "kernel", shap.KernelExplainer(f, bg)


def shap_attributions(setup, explainer_pair, Z, seed=0, nsamples=500):
    """SHAP attributions for each row of the encoded matrix Z."""
    kind, expl = explainer_pair
    np.random.seed(seed)  # KernelSHAP samples coalitions with numpy's global RNG
    if kind == "tree":
        vals = expl.shap_values(Z)
    else:
        vals = expl.shap_values(Z, nsamples=nsamples, silent=True)
    vals = np.asarray(vals)
    if vals.ndim == 3:  # (n, d, classes) -> positive class
        vals = vals[:, :, 1]
    return vals


def make_lime_explainer(setup, seed=0):
    return LimeTabularExplainer(
        setup.Ztr, mode="classification", discretize_continuous=False,
        feature_names=[f"f{i}" for i in range(setup.Ztr.shape[1])],
        random_state=seed,
    )


def lime_attributions(setup, lime_expl, Z, num_samples=1000):
    """LIME attributions for each row of the encoded matrix Z."""
    d = setup.Ztr.shape[1]
    out = np.zeros((len(Z), d))
    for i in range(len(Z)):
        exp = lime_expl.explain_instance(
            Z[i], setup.model.predict_proba, num_features=d,
            num_samples=num_samples, labels=(1,),
        )
        for idx, w in exp.local_exp[1]:
            out[i, idx] = w
    return out


def culprit_rank(ranking, culprit):
    """1-based rank of the culprit in the ranking (len+1 if missing)."""
    return ranking.index(culprit) + 1 if culprit in ranking else len(ranking) + 1
