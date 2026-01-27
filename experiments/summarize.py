from __future__ import annotations

import json
from collections import defaultdict
import numpy as np

RUNS_PATH = "results/runs_v1.jsonl"
ENS_PATH = "results/ensembles_v1.jsonl"

BUDGET = 800_000_000

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

def summarize(vals):
    vals = np.array(vals, dtype=float)
    return dict(mean=float(vals.mean()), std=float(vals.std()), min=float(vals.min()), n=int(vals.size))

def main():
    # Dedup single runs by (budget,width,seed)
    runs = load_latest_by_key(
        RUNS_PATH,
        lambda r: (int(r["budget_param_steps"]), float(r["width_mult"]), int(r["seed"]))
    )
    # Dedup ensembles by (budget,width,k,seed_base)
    ens = load_latest_by_key(
        ENS_PATH,
        lambda r: (int(r["budget_total_param_steps"]), float(r["width_mult"]), int(r["k"]), int(r["seed_base"]))
    )

    # Collect singles at BUDGET
    single = defaultdict(list)  # width -> accs
    for r in runs:
        if int(r["budget_param_steps"]) != BUDGET:
            continue
        single[float(r["width_mult"])].append(float(r["test_acc"]))

    # Collect ensembles at BUDGET (only k=3 for now)
    ens_accs = defaultdict(list)  # (width,k) -> ensemble accs
    for r in ens:
        if int(r["budget_total_param_steps"]) != BUDGET:
            continue
        key = (float(r["width_mult"]), int(r["k"]))
        ens_accs[key].append(float(r["ensemble_test_acc"]))

    print(f"\n=== Summary at budget={BUDGET:,} (test_acc) ===")
    for w in sorted(single):
        s = summarize(single[w])
        print(f"single w={w:<4}  mean={s['mean']:.4f}  std={s['std']:.4f}  min={s['min']:.4f}  n={s['n']}")

    for (w,k) in sorted(ens_accs):
        s = summarize(ens_accs[(w,k)])
        print(f"ens   w={w:<4} k={k:<2} mean={s['mean']:.4f}  std={s['std']:.4f}  min={s['min']:.4f}  n={s['n']}")

if __name__ == "__main__":
    main()
