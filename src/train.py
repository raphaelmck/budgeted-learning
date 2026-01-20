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

