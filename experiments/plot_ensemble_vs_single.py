from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

RUNS_PATH = "results/runs_v1.jsonl"
ENS_PATH = "results/ensembles_v1.jsonl"
OUT_PATH = "figures/fig2_ensemble_vs_single_8e8.png"

BUDGET = 800_000_000

SINGLE_WIDTHS = [0.5, 1.0, 2.0]
ENS_WIDTH = 0.5
ENS_K = 3


def load_latest_by_key(path: str, key_fn):
    latest = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = key_fn(r)
            ts = float(r.get("timestamp", 0.0))
            if key not in latest or ts > float(latest[key].get("timestamp", 0.0)):
                latest[key] = r
    return list(latest.values())


def main():
    runs = load_latest_by_key(
        RUNS_PATH,
        lambda r: (int(r["budget_param_steps"]), float(r["width_mult"]), int(r["seed"]))
    )
    ens = load_latest_by_key(
        ENS_PATH,
        lambda r: (int(r["budget_total_param_steps"]), float(r["width_mult"]), int(r["k"]), int(r["seed_base"]))
    )

    groups = []

    # singles
    for w in SINGLE_WIDTHS:
        vals = [float(r["test_acc"]) for r in runs
                if int(r["budget_param_steps"]) == BUDGET and float(r["width_mult"]) == w]
        groups.append((f"single w={w}", vals))

    # ensemble
    vals = [float(r["ensemble_test_acc"]) for r in ens
            if int(r["budget_total_param_steps"]) == BUDGET and float(r["width_mult"]) == ENS_WIDTH and int(r["k"]) == ENS_K]
    groups.append((f"ens w={ENS_WIDTH} k={ENS_K}", vals))

    labels = [g[0] for g in groups]
    means = np.array([np.mean(g[1]) for g in groups], dtype=float)
    stds  = np.array([np.std(g[1]) for g in groups], dtype=float)
    mins  = np.array([np.min(g[1]) for g in groups], dtype=float)

    os.makedirs("figures", exist_ok=True)
    plt.figure()

    x = np.arange(len(groups))
    plt.errorbar(x, means, yerr=stds, fmt="o", capsize=4)
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("Test accuracy (mean ± std over seeds)")
    plt.title(f"Ensembles dominate single models at fixed compute (budget={BUDGET:,})")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=220)
    print(f"saved {OUT_PATH}")

    print("\nWorst-case (min over seeds) at budget=8e8:")
    for lab, m in zip(labels, mins):
        print(f"{lab:<16} min={m:.4f}")


if __name__ == "__main__":
    main()
