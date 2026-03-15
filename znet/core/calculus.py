from collections.abc import Mapping
from typing import Any

import torch

# UNTESTED


def _to_differentiable_tensor(value: Any) -> torch.Tensor:
	"""Convert a value into a detached tensor that tracks gradients."""
	if isinstance(value, torch.Tensor):
		tensor = value.clone().detach()
	else:
		tensor = torch.tensor(value, dtype=torch.float32)

	if not torch.is_floating_point(tensor):
		tensor = tensor.to(torch.float32)

	return tensor.requires_grad_(True)


def compute_derivatives(
	root_node,
	inputs,
	n: int = 2,
	temperature: float = 293.15,
	allow_unused: bool = False,
):
	"""
	Apply inputs to a root node and extract derivatives up to order ``n``.

	By default this returns first and second derivatives. Higher-order
	derivatives are computed recursively as repeated derivatives with respect
	to each input variable.

	Args:
		root_node: Callable module/node with signature ``root_node(inputs, temperature=...)``.
		inputs: Either a mapping of named inputs or a single tensor input.
		n (int): Maximum derivative order to compute. Must be >= 1.
		temperature (float): Temperature passed to the node forward call.
		allow_unused (bool): Forwarded to ``torch.autograd.grad``.

	Returns:
		dict: {
			"output": model output tensor,
			"inputs": differentiable inputs,
			"derivatives": {order: derivatives_for_order}
		}

		For mapping inputs, each derivative order is a ``dict[str, Tensor|None]``.
		For tensor input, each derivative order is a ``Tensor|None``.
	"""
	if not isinstance(n, int) or n < 1:
		raise ValueError("n must be an integer >= 1")

	if isinstance(inputs, Mapping):
		differentiable_inputs = {k: _to_differentiable_tensor(v) for k, v in inputs.items()}
		output = root_node(differentiable_inputs, temperature=temperature)
		base = output.sum()

		input_keys = list(differentiable_inputs.keys())
		input_tensors = [differentiable_inputs[k] for k in input_keys]

		first_grads = torch.autograd.grad(
			outputs=base,
			inputs=input_tensors,
			create_graph=n > 1,
			retain_graph=n > 1,
			allow_unused=allow_unused,
		)

		derivatives = {1: {k: g for k, g in zip(input_keys, first_grads)}}
		current = derivatives[1]

		for order in range(2, n + 1):
			next_derivatives = {}
			for index, key in enumerate(input_keys):
				prev_grad = current[key]
				if prev_grad is None:
					next_derivatives[key] = None
					continue

				keep_graph = order < n or index < (len(input_keys) - 1)
				next_derivatives[key] = torch.autograd.grad(
					outputs=prev_grad.sum(),
					inputs=differentiable_inputs[key],
					create_graph=order < n,
					retain_graph=keep_graph,
					allow_unused=allow_unused,
				)[0]

			derivatives[order] = next_derivatives
			current = next_derivatives

		return {
			"output": output,
			"inputs": differentiable_inputs,
			"derivatives": derivatives,
		}

	differentiable_inputs = _to_differentiable_tensor(inputs)
	output = root_node(differentiable_inputs, temperature=temperature)
	current = torch.autograd.grad(
		outputs=output.sum(),
		inputs=differentiable_inputs,
		create_graph=n > 1,
		retain_graph=n > 1,
		allow_unused=allow_unused,
	)[0]

	derivatives: dict[int, Any] = {1: current}
	for order in range(2, n + 1):
		if current is None:
			derivatives[order] = None
			continue

		current = torch.autograd.grad(
			outputs=current.sum(),
			inputs=differentiable_inputs,
			create_graph=order < n,
			retain_graph=order < n,
			allow_unused=allow_unused,
		)[0]
		derivatives[order] = current

	return {
		"output": output,
		"inputs": differentiable_inputs,
		"derivatives": derivatives,
	}
