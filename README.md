# Can Developers Trust the Explanation? A Benchmark of Fault Localization and Stability in SHAP and LIME

This repo has the code, data, results, and figure scripts for my paper **"Can Developers Trust the Explanation? A Benchmark of Fault Localization and Stability in SHAP and LIME"** (Foundations of Software Engineering, Clemson University, Fall 2026).

Author: Shrinidhi Kulkarni ([@shreenidhikulkarni12](https://github.com/shreenidhikulkarni12))

## Overview

Developers often use SHAP and LIME to debug machine-learning models, but it is hard to know whether an explanation is pointing at the actual problem. This benchmark tests that using faults with a known cause, and also checks whether an explanation's stability tells you how much to trust it.

For each setup we inject a fault with a known culprit column into the training data, train a model, find the test errors the fault caused, explain them with SHAP and LIME, and record where the culprit ranks.

**Research questions**
- **RQ1:** How accurately do SHAP and LIME localize the culprit, by fault type, severity, and model?
- **RQ2:** How stable are explanations across seeds and input perturbations, and does stability predict correctness?
- **RQ3:** Does a stability flag identify unreliable explanations better than simpler signals?

## Experimental design

The grid has **540 setups**: 3 datasets x 3 models x 3 faults x 2 severities x 10 seeds.

| Factor | Levels |
|---|---|
| Datasets | Adult Income (seeded 10,000-row subsample, 12 features), German Credit (1,000 rows, 20 features), COMPAS (6,172 rows after standard filtering, 8 features) |
| Models | `logreg` (logistic regression), `gbt` (histogram gradient boosting), `mlp` (neural network, hidden layers 32 and 16) |
| Faults | `spurious`, `subgroup`, `leakage` |
| Severities | `mild`, `severe` |
| Seeds | 0-9 (control the train/test split, fault injection, and model initialization) |

| Fault | What is injected | Mild | Severe | Culprit |
|---|---|---|---|---|
| `spurious` | Binary column equal to the label on a fraction of training rows; random at test time | 70% | 95% | `spurious_feature` |
| `subgroup` | Majority-label flips for training rows with age < 30; test labels stay clean | 40% | 70% | `age` |
| `leakage` | Numeric column computed from the label on a fraction of training rows; random at test time | 50% | 100% | `leaked_feature` |

**Ground truth (clean twin).** To know which errors the fault caused, each setup also trains a second model with the same split, model type, and seed but no fault. An instance is explained only if the faulty model gets it wrong and the clean twin gets it right. Up to 50 such instances are sampled per setup, giving 26,216 explained instances per explainer (52,432 rows in total).

**Explainers.** SHAP uses TreeExplainer for `gbt` and KernelExplainer (30 background rows, 500 samples) for the other models. LIME uses the tabular explainer (no discretization, 1,000 samples, positive class, all features). One-hot columns are merged back to their source column by summing signed attributions, then taking the absolute value; source columns are then ranked.

**Metrics.**
- Correctness: top-1, top-3, and reciprocal rank of the culprit.
- Seed stability: mean pairwise Jaccard similarity of top-3 sets across 10 explainer seeds.
- Perturbation stability: Jaccard similarity of the top-3 set under Gaussian noise (sd 0.10 on standardized numeric columns, 10 copies).
- RQ3 baselines: model confidence, total attribution mass, and attribution concentration on the top feature.

## Repository layout

```
.
├── data/                 # adult.csv, german.csv, compas.csv
├── src/
│   ├── data.py           # dataset loading and COMPAS filtering
│   ├── features.py       # encoding; maps encoded columns back to source columns
│   ├── faults.py         # the three fault injectors
│   ├── pipeline.py       # builds one setup (model, clean twin, induced errors)
│   ├── explain.py        # SHAP and LIME attributions, culprit ranking
│   ├── stability.py      # correctness and stability metrics for one setup
│   ├── run_grid.py       # runs the grid, one CSV per setup
│   ├── baseline.py       # clean-model accuracy sanity check
│   └── pilot_*.py        # pilot checks that each fault lowers test accuracy
├── results/              # 540 per-setup CSVs, e.g. adult__mlp__leakage__severe__seed9.csv
├── results_v2.zip        # archive of the results
├── make_figures.py       # generates Figures 1-5
├── figures_paper/        # figure outputs (PDF and PNG) and rq3_auroc.csv
├── requirements.txt      # package versions
└── README.md
```

## Result file columns

| Column | Meaning |
|---|---|
| `dataset, model, fault, severity, seed` | Setup identifiers |
| `explainer` | `shap` or `lime` |
| `instance` | Test-set row index |
| `culprit`, `culprit_rank` | Ground-truth culprit column; its rank in the explanation (1 = top) |
| `top1, top3, rr` | Top-1 hit, top-3 hit, reciprocal rank |
| `stab_seed, stab_perturb` | Mean top-3 Jaccard across explainer seeds / input perturbations |
| `rho_seed, rho_perturb` | Spearman rank correlation of importances across seeds / perturbations |
| `conf` | Model confidence in its (wrong) prediction |
| `attr_total, attr_top1_share` | Total attribution mass; share on the top column |
| `test_acc, clean_acc` | Test accuracy of the faulty model and of its clean twin |
| `n_induced, n_wrong_all` | Fault-induced errors in the setup / all test errors |

## Reproducing the results

Install dependencies (Python 3):

```bash
pip install -r requirements.txt
```

The datasets are already in `data/`. Run from the repository root. The grid was run one dataset at a time; each run writes one CSV per setup to `results/` and skips setups that already exist:

```bash
python src/run_grid.py --datasets adult
python src/run_grid.py --datasets german
python src/run_grid.py --datasets compas
```

These three runs produce all 540 setups (50 instances per setup, 10 repeats for the stability measures). `python src/run_grid.py` with no arguments runs the whole grid at once. For a quick test:

```bash
python src/run_grid.py --seeds 0 --instances 20
```

Then generate the figures:

```bash
python make_figures.py
```

The full grid took a few hours on my laptop. If you don't want to rerun it, the CSVs in `results/` are enough to run `make_figures.py` directly. All random choices are seeded.

**Sanity checks (optional).** `python src/baseline.py` reports clean-model accuracy for every dataset and model. `python src/pilot_spurious.py`, `pilot_subgroup.py`, and `pilot_leakage.py` check that each fault lowers test accuracy.

## Data sources

- **Adult Income** and **German Credit:** UCI Machine Learning Repository. In German Credit, label 1 = bad credit risk.
- **COMPAS:** ProPublica recidivism data with the standard two-year filtering. The COMPAS risk scores (`decile_score`, `score_text`) and `age_cat` are dropped to avoid leakage.

## Citation

```
Kulkarni, S. Can Developers Trust the Explanation? A Benchmark of Fault Localization and
Stability in SHAP and LIME. Clemson University, 2026.
```
