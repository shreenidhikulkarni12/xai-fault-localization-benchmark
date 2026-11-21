"""Run the experiment grid, one CSV per configuration in results/. Re-running skips finished ones.

Usage examples:
  python src/run_grid.py --seeds 0 --instances 20              (small test)
  python src/run_grid.py --datasets adult --models logreg gbt  (a slice)
  python src/run_grid.py                                       (full grid, 10 seeds, 50 instances)
"""
import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from pipeline import FAULTS, build_setup
from stability import run_setup

RESULTS = Path(__file__).resolve().parent.parent / "results"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["adult", "german", "compas"])
    ap.add_argument("--models", nargs="+", default=["logreg", "gbt", "mlp"])
    ap.add_argument("--faults", nargs="+", default=list(FAULTS))
    ap.add_argument("--severities", nargs="+", default=["mild", "severe"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--instances", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    combos = list(itertools.product(a.datasets, a.models, a.faults, a.severities, a.seeds))
    print(f"{len(combos)} configurations")
    t_start = time.time()
    for n, (ds, m, f, sev, seed) in enumerate(combos, 1):
        path = RESULTS / f"{ds}__{m}__{f}__{sev}__seed{seed}.csv"
        if path.exists():
            continue
        t0 = time.time()
        try:
            df = run_setup(build_setup(ds, m, f, sev, seed), n_instances=a.instances, n_repeats=a.repeats)
        except Exception as e:  # keep going; record the failure for the Threats to Validity log
            print(f"[{n}/{len(combos)}] FAILED {path.stem}: {e!r}")
            continue
        df.to_csv(path, index=False)
        print(f"[{n}/{len(combos)}] {path.stem}  {len(df)} rows  {time.time() - t0:.1f}s  "
              f"(elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)
