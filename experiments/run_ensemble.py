from __future__ import annotations

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.budget import count_params, steps_for_budget
from src.data import cifar10_loaders
from src.models import TinyCNN


@dataclass
class EnsembleResult:
    timestamp: float
    seed_base: int
    member_seeds: List[int]

    width_mult: float
    k: int

    budget_total_param_steps: int
    budget_per_member_param_steps: int

    params: int
    steps_per_member: int
    effective_compute_per_member: int
    effective_compute_total: int

    batch_size: int
    lr: float
    device: str

    member_test_accs: List[float]
    member_test_losses: List[float]

    ensemble_test_acc: float
    ensemble_test_loss: float

    seconds: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_member(
    *,
    width_mult: float,
    seed: int,
    budget_param_steps: int,
    train_loader,
    device: torch.device,
    lr: float,
    max_steps: int = 12000,
) -> Tuple[nn.Module, int, int]:
    set_seed(seed)
    model = TinyCNN(width_mult=width_mult).to(device)

    params = count_params(model)
    steps = steps_for_budget(budget_param_steps, params, min_steps=200, max_steps=max_steps)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()

    train_iter = iter(train_loader)
    model.train()

    pbar = tqdm(range(steps), desc=f"member seed={seed} w={width_mult} steps={steps}", leave=False)
    for _ in pbar:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()

    return model, params, steps


@torch.no_grad()
def eval_single(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_ensemble(models: List[nn.Module], loader, device: torch.device) -> Tuple[float, float]:
    for m in models:
        m.eval()

    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits_sum = None
        for m in models:
            logits = m(x)
            logits_sum = logits if logits_sum is None else (logits_sum + logits)

        logits_avg = logits_sum / len(models)
        loss = ce(logits_avg, y)

        total_loss += float(loss.item()) * x.size(0)
        pred = logits_avg.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

    return total_loss / total, correct / total


def run_ensemble(
    *,
    width_mult: float,
    k: int,
    budget_total: int,
    seed_base: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    prefer_mps: bool,
    log_path: str,
    max_steps: int,
) -> EnsembleResult:
    t0 = time.time()
    device = get_device(prefer_mps=prefer_mps)

    # Use a single dataset/dataloader for all members for consistency.
    train_loader, test_loader = cifar10_loaders(batch_size=batch_size, num_workers=num_workers)

    # Split compute across members
    budget_per = budget_total // k

    # deterministic member seeds
    member_seeds = [seed_base * 1000 + i for i in range(k)]

    models: List[nn.Module] = []
    member_accs: List[float] = []
    member_losses: List[float] = []

    params_ref = None
    steps_ref = None

    for s in member_seeds:
        m, params, steps = train_one_member(
            width_mult=width_mult,
            seed=s,
            budget_param_steps=budget_per,
            train_loader=train_loader,
            device=device,
            lr=lr,
            max_steps=max_steps,
        )
        if params_ref is None:
            params_ref = params
        if steps_ref is None:
            steps_ref = steps

        loss_i, acc_i = eval_single(m, test_loader, device=device)
        member_losses.append(float(loss_i))
        member_accs.append(float(acc_i))
        models.append(m)

    ens_loss, ens_acc = eval_ensemble(models, test_loader, device=device)

    params = int(params_ref)
    steps_per_member = int(steps_ref)
    effective_per = int(params * steps_per_member)
    effective_total = int(effective_per * k)

    res = EnsembleResult(
        timestamp=time.time(),
        seed_base=seed_base,
        member_seeds=member_seeds,
        width_mult=float(width_mult),
        k=int(k),
        budget_total_param_steps=int(budget_total),
        budget_per_member_param_steps=int(budget_per),
        params=int(params),
        steps_per_member=int(steps_per_member),
        effective_compute_per_member=int(effective_per),
        effective_compute_total=int(effective_total),
        batch_size=int(batch_size),
        lr=float(lr),
        device=str(device),
        member_test_accs=member_accs,
        member_test_losses=member_losses,
        ensemble_test_acc=float(ens_acc),
        ensemble_test_loss=float(ens_loss),
        seconds=float(time.time() - t0),
    )

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(res)) + "\n")

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-mult", type=float, required=True)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--budget", type=int, required=True, help="Total budget (param-steps) for the whole ensemble")
    ap.add_argument("--seed", type=int, default=0, help="Base seed (members use seed*1000+i)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-mps", action="store_true")
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--log-path", type=str, default="results/ensembles_v1.jsonl")
    args = ap.parse_args()

    res = run_ensemble(
        width_mult=args.width_mult,
        k=args.k,
        budget_total=args.budget,
        seed_base=args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        prefer_mps=not args.no_mps,
        log_path=args.log_path,
        max_steps=args.max_steps,
    )

    print("\n=== Ensemble summary ===")
    print(f"device: {res.device}")
    print(f"width_mult: {res.width_mult}   k={res.k}")
    print(f"budget_total: {res.budget_total_param_steps:,}")
    print(f"budget_per_member: {res.budget_per_member_param_steps:,}")
    print(f"params: {res.params:,}")
    print(f"steps_per_member: {res.steps_per_member:,}")
    print(f"effective_compute_total: {res.effective_compute_total:,}")
    print(f"member_accs: {[round(a,4) for a in res.member_test_accs]}")
    print(f"ensemble_acc: {res.ensemble_test_acc:.4f}")
    print(f"ensemble_loss: {res.ensemble_test_loss:.4f}")
    print(f"seconds: {res.seconds:.1f}")
    print(f"logged to: {args.log_path}")


if __name__ == "__main__":
    main()
