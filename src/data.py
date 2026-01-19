from __future__ import annotations

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def cifar10_loaders(batch_size: int = 128, num_workers: int = 2):
	tfm = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(
			(0.4914, 0.4822, 0.4465),
			(0.2470, 0.2435, 0.2616),
		),
	])

	train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
	test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=tfm)

	train_loader = DataLoader(
		train_ds, batch_size=batch_size, shuffle=True,
		num_workers=num_workers, persisten_workers=False
	)
	test_loader = DataLoader(
		test_ds, batch_size=batch_size, shuffle=False,
		num_workers=num_workers, persistent_workers=False
	)
	return train_loader, test_loader
