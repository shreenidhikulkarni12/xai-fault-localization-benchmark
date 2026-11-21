"""Pilot for the explanation step on ONE setup: how fast is it, and does it find the culprit?

Run: python src/pilot_explain.py [dataset] [model] [fault]
Defaults: adult logreg spurious.
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np

from explain import (culprit_rank, lime_attributions, make_lime_explainer, make_shap_explainer,
                     rank_sources, shap_attributions)
from pipeline import build_setup, sample_wrong

N_INSTANCES = 20

if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "adult"
    model = sys.argv[2] if len(sys.argv) > 2 else "logreg"
    fault = sys.argv[3] if len(sys.argv) > 3 else "spurious"
    print(f"{ds} / {model} / {fault}, {N_INSTANCES} mispredicted instances, seed 0")
    print(f"{'severity':8s} {'expl':5s} {'method':7s} {'top1':>6s} {'top3':>6s} {'MRR':>6s} {'secs':>7s}")
    for sev in ["mild", "severe"]:
        s = build_setup(ds, model, fault, sev, seed=0)
        rows = sample_wrong(s, N_INSTANCES, seed=0)
        for name in ["shap", "lime"]:
            t0 = time.time()
            if name == "shap":
                pair = make_shap_explainer(s, seed=0)
                attr, method = shap_attributions(s, pair, s.Zte[rows], seed=0), pair[0]
            else:
                attr, method = lime_attributions(s, make_lime_explainer(s, seed=0), s.Zte[rows]), "lime"
            ranks = np.array([culprit_rank(rank_sources(a, s.sources), s.culprit) for a in attr])
            print(f"{sev:8s} {name:5s} {method:7s} {np.mean(ranks == 1):6.2f} "
                  f"{np.mean(ranks <= 3):6.2f} {np.mean(1 / ranks):6.2f} {time.time() - t0:7.1f}")
