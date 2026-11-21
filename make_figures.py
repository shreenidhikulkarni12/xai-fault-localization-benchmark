"""Make the paper figures from the experiment results.

Usage (from the project folder):
    pip install numpy pandas scikit-learn matplotlib seaborn
    python make_figures.py                      # reads ./results, writes ./figures_paper
    python make_figures.py --results path/to/results --out my_figs

Reads every CSV in the results folder (one per experimental setup), merges them, and writes
each figure as a PNG (300 dpi) and a vector PDF. Widths follow the IEEE two-column format:
a single column is ~3.5 in wide, a full page width is ~7.1 in.

Uncertainty: every bar / point shows a 95% bootstrap confidence interval computed by resampling
SETUPS (dataset x model x fault x severity x seed), not individual instances, because the
50 explained instances inside one trained model are not independent of each other.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

# ---- look & feel ---------------------------------------------------------------------------
SHAP_C, LIME_C = "#2a78d6", "#eb6834"          # blue / orange: colorblind-safe pair
PALETTE = {"SHAP": SHAP_C, "LIME": LIME_C}
HUE_ORDER = ["SHAP", "LIME"]
CELLS = [("spurious", "mild"), ("spurious", "severe"), ("leakage", "mild"),
         ("leakage", "severe"), ("subgroup", "mild"), ("subgroup", "severe")]
CELL_LABELS = [f"{f}\n{s}" for f, s in CELLS]
MODEL_NAMES = {"logreg": "Logistic\nregression", "gbt": "Gradient-\nboosted trees", "mlp": "Neural\nnetwork"}
SIGNALS = {  # name -> (label, score where HIGHER = more suspicious)
    "instability_seed": "Instability across seeds",
    "instability_perturb": "Instability across input changes",
    "instability_either": "Instability (either kind)",
    "low_confidence": "Low model confidence",
    "low_attr_total": "Low total attribution",
    "diffuse_attr": "Diffuse attribution",
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0,
              rc={"axes.spines.top": False, "axes.spines.right": False, "grid.linewidth": 0.5,
                  "axes.edgecolor": "#bbbbbb", "legend.frameon": False, "figure.dpi": 120})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---- data ----------------------------------------------------------------------------------
def load(results_dir):
    parts = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        try:
            d = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue                       # a setup with no fault-induced errors writes an empty file
        if len(d):
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["Explainer"] = df["explainer"].str.upper()
    df["setup"] = df[["dataset", "model", "fault", "severity", "seed"]].astype(str).agg("_".join, axis=1)
    df["cell"] = df["fault"] + "\n" + df["severity"]
    print(f"loaded {len(parts)} files, {len(df):,} rows, {df['setup'].nunique()} setups")
    return df


def setup_level(df):
    """One row per setup x explainer: the mean over that setup's explained instances."""
    keys = ["dataset", "model", "fault", "severity", "seed", "Explainer", "cell"]
    return df.groupby(keys)[["top1", "top3", "rr", "stab_seed", "stab_perturb"]].mean().reset_index()


# ---- Figure 1: RQ1, how often is the culprit ranked first? ---------------------------------
def fig_rq1(S, out):
    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    sns.barplot(data=S, x="cell", y="top1", hue="Explainer", order=CELL_LABELS, hue_order=HUE_ORDER,
                palette=PALETTE, errorbar=("ci", 95), n_boot=2000, seed=0, capsize=0.12,
                saturation=1, err_kws={"linewidth": 0.9, "color": "#222222"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Culprit ranked first")
    ax.set_ylim(0, 1.05)
    ax.legend(title="", ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, out, "fig1_rq1_localization")


# ---- Figure 2: RQ1 detail, dataset x fault heatmap (shows where the tools differ) ---------
def fig_heatmap(S, out):
    fig = plt.figure(figsize=(7.1, 2.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.08)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    cax = fig.add_subplot(gs[2])
    short = [f"{f[:4]}.\n{s}" for f, s in CELLS]            # spur. / leak. / subg.  +  mild / severe
    for i, (ax, ex) in enumerate(zip(axes, HUE_ORDER)):
        t = (S[S.Explainer == ex].pivot_table(index="dataset", columns="cell", values="top1", aggfunc="mean")
             .reindex(columns=CELL_LABELS, index=["adult", "compas", "german"]))
        t.columns = short
        sns.heatmap(t, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues", linewidths=0.8,
                    linecolor="white", annot_kws={"size": 7.5}, ax=ax, cbar=(i == 1), cbar_ax=cax if i == 1 else None,
                    cbar_kws={"label": "Culprit ranked first"} if i == 1 else None)
        ax.set_title(ex, fontsize=9, loc="left")
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=0)
        if i == 1: ax.set_yticks([])
    save(fig, out, "fig2_dataset_heatmap")


# ---- Figure 3: stability by model ----------------------------------------------------------
def fig_stability(S, out):
    L = S.melt(id_vars=["model", "Explainer"], value_vars=["stab_seed", "stab_perturb"],
               var_name="kind", value_name="stability")
    L["kind"] = L["kind"].map({"stab_seed": "Across random seeds", "stab_perturb": "Across small input changes"})
    g = sns.catplot(data=L, x="model", y="stability", hue="Explainer", col="kind", kind="bar",
                    order=["logreg", "gbt", "mlp"], hue_order=HUE_ORDER, palette=PALETTE,
                    errorbar=("ci", 95), n_boot=2000, seed=0, capsize=0.1, err_kws={"linewidth": 0.9}, saturation=1,
                    height=2.5, aspect=1.4, legend_out=False)
    for ax, t in zip(g.axes.flat, ["Across random seeds", "Across small input changes"]):
        ax.set_title("", loc="center")          # clear seaborn's default centered "kind = ..." title
        ax.set_title(t, fontsize=9, loc="left")
    g.set_axis_labels("", "Stability (top-3 Jaccard)")
    g.set_xticklabels([MODEL_NAMES[m] for m in ["logreg", "gbt", "mlp"]])
    g.set(ylim=(0, 1.08))
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.07), ncol=2, title=None)
    save(g.figure, out, "fig3_stability_by_model")


# ---- Figure 4: RQ2, does the culprit get found more often when the explanation is stable? --
def fig_rq2(df, out):
    bins, labels = [0, 0.6, 0.8, 0.999, 1.01], ["< 0.6", "0.6-0.8", "0.8-<1", "1.0"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), sharey=True)
    for ax, (col, title) in zip(axes, [("stab_seed", "Stability across seeds"),
                                       ("stab_perturb", "Stability across input changes")]):
        d = df.copy()
        d["band"] = pd.cut(d[col], bins, labels=labels, right=False)
        n = d.groupby(["Explainer", "band"], observed=True).size()
        keep = n[n >= 200].index                                  # skip bands with too few instances
        d = d[[(e, b) in keep for e, b in zip(d.Explainer, d.band)]]
        P = d.groupby(["setup", "Explainer", "band"], observed=True)["top1"].mean().reset_index()
        P["band"] = P["band"].astype(str)
        sns.pointplot(data=P, x="band", y="top1", hue="Explainer", order=labels, hue_order=HUE_ORDER,
                      palette=PALETTE, errorbar=("ci", 95), n_boot=1000, seed=0, markersize=4,
                      linewidth=1.6, err_kws={"linewidth": 1.0}, ax=ax)
        ax.set_title(title, fontsize=9, loc="left"); ax.set_xlabel("Stability band"); ax.set_ylim(0, 1.02)
        ax.get_legend().remove() if ax is axes[0] else ax.legend(title="", loc="lower right")
    axes[0].set_ylabel("Culprit ranked first")
    fig.tight_layout()
    save(fig, out, "fig4_rq2_stability_vs_correctness")


# ---- Figure 5: RQ3, how well does each warning signal flag a wrong explanation? ------------
def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) and np.ptp(s) > 0 else np.nan


def fig_rq3(df, out, B=300):
    rng = np.random.default_rng(1)
    d = df.copy()
    d["instability_seed"] = 1 - d["stab_seed"]
    d["instability_perturb"] = 1 - d["stab_perturb"]
    d["instability_either"] = 1 - np.minimum(d["stab_seed"], d["stab_perturb"])
    d["low_confidence"] = 1 - d["conf"]
    d["low_attr_total"] = -d["attr_total"]
    d["diffuse_attr"] = -d["attr_top1_share"]
    rows = []
    for ex in HUE_ORDER:
        g = d[d.Explainer == ex].reset_index(drop=True)
        y = (1 - g["top1"]).to_numpy()                            # 1 = explanation missed the culprit
        groups = [v.to_numpy() for _, v in g.groupby("setup").groups.items()]
        for sig in SIGNALS:
            s = g[sig].to_numpy()
            boots = []
            for _ in range(B):                                    # resample SETUPS, not instances
                ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                boots.append(auc(y[ix], s[ix]))
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            rows.append(dict(Explainer=ex, signal=sig, auroc=auc(y, s), lo=lo, hi=hi))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "rq3_auroc.csv"), index=False)

    order = list(SIGNALS)[::-1]
    fig, ax = plt.subplots(figsize=(7.1, 3.3)); h = 0.36
    for j, ex in enumerate(HUE_ORDER):
        r = R[R.Explainer == ex].set_index("signal").loc[order]
        y = np.arange(len(order)) + (j - 0.5) * (h + 0.03)
        ax.barh(y, r.auroc - 0.5, h, left=0.5, color=PALETTE[ex], label=ex, zorder=3)
        ax.errorbar(r.auroc, y, xerr=[r.auroc - r.lo, r.hi - r.auroc], fmt="none", ecolor="#222222",
                    elinewidth=0.9, capsize=2, zorder=4)
    ax.axvline(0.5, color="#555555", lw=1, zorder=5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([SIGNALS[k] for k in order])
    ax.set_xlim(0.3, 0.9); ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUROC for flagging explanations that miss the culprit (0.5 = chance)")
    ax.legend(title="", ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    save(fig, out, "fig5_rq3_warning_signals")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures_paper")
    ap.add_argument("--boot", type=int, default=300, help="bootstrap repeats for the RQ3 AUROC intervals")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.results)
    S = setup_level(df)
    fig_rq1(S, a.out)
    fig_heatmap(S, a.out)
    fig_stability(S, a.out)
    fig_rq2(df, a.out)
    fig_rq3(df, a.out, B=a.boot)
