from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
	def __init__(self, width_mult: float = 1.0, num_classes: int = 10):
		super().__init__()
		base = 32
		c = max(8, int(base * width_mult))

		self.conv1 = nn.Conv2d(3, c, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(c, 2*c, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(2*c, 4*c, kernel_size=3, padding=1)

		self.pool = nn.MaxPool2d(2)
		self.fc = nn.Linear(4*c, num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool(x)

		x = F.relu(self.conv3(x))
		x = self.pool(x)

		x = x.mean(dim=(2, 3))
		return self.fc(x)


