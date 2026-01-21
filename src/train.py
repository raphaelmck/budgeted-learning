from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.budget import count_params, steps_for_budget
from src.data import cifar10_loaders
from src.models import TinyCNN

@dataclass
class RunResult:
	timestamp: float
	seed: int
	width_mult: float
	budget_param_steps: int
	params: int
	steps: int
	batch_size: int
	lr: float
	device: str
	train_loss_last: float
	test_acc: float
	test_loss: float
	seconds: float

def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

def get_device(prefer_mps: bool = True) -> torch.device:
	if prefer_mps and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")

@torch.no_grad()
def eval_model(model: nn.Module, loader, device: torch.device):
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

def train_one_run(
		seed: int,
		width_mult: float,
		budget_param_steps: int,
		batch_size: int,
		lr: float,
		num_workers: int,
		prefer_mps: bool,
		log_path: str
) -> RunResult:
	t0 = time.time()
	set_seed(seed)

	device = get_device(prefer_mps=prefer_mps)

	train_loader, test_loader = cifar10_loaders(batch_size=batch_size, num_workers=num_workers)

	model = TinyCNN(width_mult=width_mult).to(device)
	params = count_params(model)
	steps = steps_for_budget(budget_param_steps, params, min_steps=200)

	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
	ce = nn.CrossEntropyLoss()

	train_iter = iter(train_loader)

	model.train()
	last_loss = float("nan")

	pbar = tqdm(range(steps), desc=f"train w={width_mult} P={params} T={steps}", leave=False)
	for _ in pbar:
		try:
			x, y = next(train_iter)
		except StopIteration:
			train_iter = iter(train_loader)
			x, y = next(train_iter)

		x, y = x.to(device), y.to(device)

		optimizer.zero_grad(set_to_none=True)
		logits = model(x)
		loss = ce(logits, y)
		loss.backward()
		optimizer.step()

		last_loss = float(loss.item())
		pbar.set_postfix(loss=f"{last_loss:.4f}")
	
	test_loss, test_acc = eval_model(model, test_loader, device=device)

	seconds = time.time() - t0
	# see types if conversion is necessary
	result = RunResult(
		timestamp=time.time(),
		seed=seed,
		width_mult=width_mult,
		budget_param_steps=int(budget_param_steps),
		params=int(params),
		steps=int(steps),
		batch_size=int(batch_size),
		lr=float(lr),
		device=str(device),
		train_loss_last=float(last_loss),
		test_acc=float(test_acc),
		test_loss=float(test_loss),
		seconds=float(seconds)
	)

