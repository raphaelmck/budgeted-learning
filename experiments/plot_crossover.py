from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

RUNS_PATH = "results/runs_v1.jsonl"
OUT_PATH = "figures/fig1_crossover_acc_vs_compute.png"

BUDGETS = [200_000_000, 800_000_000, 3_000_000_000]
WIDTHS = [0.5, 1.0, 2.0]


def load_latest_runs(path: str):
    """
    Deduplicate by (budget, width, seed) keeping the latest timestamp.
    """
    latest = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (int(r["budget_param_steps"]), float(r["width_mult"]), int(r["seed"]))
            ts = float(r.get("timestamp", 0.0))
            if key not in latest or ts > float(latest[key].get("timestamp", 0.0)):
                latest[key] = r
    return list(latest.values())


def main():
    runs = load_latest_runs(RUNS_PATH)

    # (width, budget) -> list(acc), and store effective compute per group
    accs = defaultdict(list)
    eff = {}

    for r in runs:
        b = int(r["budget_param_steps"])
        w = float(r["width_mult"])
        if b not in BUDGETS or w not in WIDTHS:
            continue
        acc = float(r["test_acc"])
        beff = int(r.get("effective_budget_param_steps", int(r["params"]) * int(r["steps"])))
        accs[(w, b)].append(acc)
        eff[(w, b)] = beff

    os.makedirs("figures", exist_ok=True)
    plt.figure()

    for w in WIDTHS:
        xs, ys, es = [], [], []
        for b in BUDGETS:
            vals = accs.get((w, b), [])
            if not vals:
                continue
            xs.append(eff[(w, b)])
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)))
        # sort by x
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        es = np.array(es)[order]
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"width_mult={w}")

    plt.xscale("log")
    plt.xlabel("Effective training compute (params × steps)")
    plt.ylabel("Test accuracy (mean ± std over seeds)")
    plt.title("Compute-constrained training shows a size–compute crossover")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=220)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
