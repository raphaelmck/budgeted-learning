from __future__ import annotations

def count_params(model) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def steps_for_budget(budget_param_steps: int, params: int, min_steps: int = 200) -> int:
	steps = budget_param_steps // max(params, 1)
	return max(steps, min_steps)
