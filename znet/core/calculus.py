# from collections.abc import Mapping
# from typing import Any
import torch
from torch.func import jacfwd, jacrev, vmap, grad_and_value

def value_and_derivatives(f, inputs, n=1):
	def _build_n_derivative_engine(f, n=1):
		"""
		Builds a pure function that calculates f(x) and its first n derivatives.
		Uses optimal AD stacking: grad for 1st derivative, jacfwd for higher orders.
		"""
		
		# 1. Define a scalar function for a SINGLE sample vector x.
		# Most nodes expect a batch dimension, so we temporarily add one.
		def scalar_fcn(x):
			out = f(x.unsqueeze(0))
			if out.numel() != 1:
				raise RuntimeError(
					"value_and_derivatives expects f to produce one scalar per sample. "
					f"Got output shape {tuple(out.shape)} for a single sample."
				)
			return out.reshape(())

		def base_fcn(x):
			# Get gradient and value
			g, v = grad_and_value(scalar_fcn)(x)
			# unsqueeze v here (scalar) so vmap produces (batch, 1) directly
			# Primary output: g (so jacfwd differentiates it into a Hessian)
			# Auxiliary output: (v, g) (so we don't lose them)
			return g, (v.unsqueeze(-1), g)
		
		if n==1:
			return base_fcn

		current_fn = base_fcn
		for i in range(2, n + 1):
			# We must isolate the loop variable scope in Python
			def make_next_fn(prev_fn):
				def next_fn(x):
					# jacfwd differentiates the primary output, passes aux through!
					j, aux = jacfwd(prev_fn, has_aux=True)(x)
					
					# Append the new n-th derivative to the aux tuple
					return j, aux + (j,)
				return next_fn
				
			current_fn = make_next_fn(current_fn)

		return current_fn


	if not torch.is_tensor(inputs):
		raise TypeError("inputs must be a torch.Tensor")
	if inputs.ndim < 2:
		raise ValueError("inputs must have shape (..., n_features)")

	batch_shape = inputs.shape[:-1]
	n_features = inputs.shape[-1]
	flat_inputs = inputs.reshape(-1, n_features)

	n_deriv_engine = vmap(_build_n_derivative_engine(f, n=n))
	_, all_terms = n_deriv_engine(flat_inputs)

	def _restore_batch_dims(tensor):
		return tensor.reshape(*batch_shape, *tensor.shape[1:])

	return tuple(_restore_batch_dims(term) for term in all_terms)

# UNTESTED


# ** Compute first derivative, value, and hessian as one. Gives complete information. 
# def legendre_transform_direct():
# 	#** can use JVP for legendre transform but this eliminates the dual variable recovery. Maybe useful for 
# 	#maybe useful for recovery of convex region since transform correlated to specific dual variable?
# 	pass

# def _to_differentiable_tensor(value: Any) -> torch.Tensor:
# 	"""Convert a value into a detached tensor that tracks gradients."""
# 	if isinstance(value, torch.Tensor):
# 		tensor = value.clone().detach()
# 	else:
# 		tensor = torch.tensor(value, dtype=torch.float32)

# 	if not torch.is_floating_point(tensor):
# 		tensor = tensor.to(torch.float32)

# 	return tensor.requires_grad_(True)


# def compute_derivatives(
# 	root_node,
# 	inputs,
# 	n: int = 2,
# 	allow_unused: bool = False,
# ):
# 	"""
# 	Apply inputs to a root node and extract derivatives up to order ``n``.

# 	By default this returns first and second derivatives. Higher-order
# 	derivatives are computed recursively as repeated derivatives with respect
# 	to each input variable.

# 	Args:
# 		root_node: Callable module/node with signature ``root_node(inputs, temperature=...)``.
# 		inputs: Either a mapping of named inputs or a single tensor input.
# 		n (int): Maximum derivative order to compute. Must be >= 1.
# 		temperature (float): Temperature passed to the node forward call.
# 		allow_unused (bool): Forwarded to ``torch.autograd.grad``.

# 	Returns:
# 		dict: {
# 			"output": model output tensor,
# 			"inputs": differentiable inputs,
# 			"derivatives": {order: derivatives_for_order}
# 		}

# 		For mapping inputs, each derivative order is a ``dict[str, Tensor|None]``.
# 		For tensor input, each derivative order is a ``Tensor|None``.
# 	"""
# 	if not isinstance(n, int) or n < 1:
# 		raise ValueError("n must be an integer >= 1")

# 	if isinstance(inputs, Mapping):
# 		differentiable_inputs = {k: _to_differentiable_tensor(v) for k, v in inputs.items()}
# 		output = root_node(differentiable_inputs)
# 		base = output.sum()

# 		input_keys = list(differentiable_inputs.keys())
# 		input_tensors = [differentiable_inputs[k] for k in input_keys]

# 		first_grads = torch.autograd.grad(
# 			outputs=base,
# 			inputs=input_tensors,
# 			create_graph=n > 1,
# 			retain_graph=n > 1,
# 			allow_unused=allow_unused,
# 		)

# 		derivatives = {1: {k: g for k, g in zip(input_keys, first_grads)}}
# 		current = derivatives[1]

# 		for order in range(2, n + 1):
# 			next_derivatives = {}
# 			for index, key in enumerate(input_keys):
# 				prev_grad = current[key]
# 				if prev_grad is None:
# 					next_derivatives[key] = None
# 					continue

# 				keep_graph = order < n or index < (len(input_keys) - 1)
# 				next_derivatives[key] = torch.autograd.grad(
# 					outputs=prev_grad.sum(),
# 					inputs=differentiable_inputs[key],
# 					create_graph=order < n,
# 					retain_graph=keep_graph,
# 					allow_unused=allow_unused,
# 				)[0]

# 			derivatives[order] = next_derivatives
# 			current = next_derivatives

# 		return {
# 			"output": output,
# 			"inputs": differentiable_inputs,
# 			"derivatives": derivatives,
# 		}

# 	differentiable_inputs = _to_differentiable_tensor(inputs)
# 	output = root_node(differentiable_inputs)
# 	current = torch.autograd.grad(
# 		outputs=output.sum(),
# 		inputs=differentiable_inputs,
# 		create_graph=n > 1,
# 		retain_graph=n > 1,
# 		allow_unused=allow_unused,
# 	)[0]

# 	derivatives: dict[int, Any] = {1: current}
# 	for order in range(2, n + 1):
# 		if current is None:
# 			derivatives[order] = None
# 			continue

# 		current = torch.autograd.grad(
# 			outputs=current.sum(),
# 			inputs=differentiable_inputs,
# 			create_graph=order < n,
# 			retain_graph=order < n,
# 			allow_unused=allow_unused,
# 		)[0]
# 		derivatives[order] = current

# 	return {
# 		"output": output,
# 		"inputs": differentiable_inputs,
# 		"derivatives": derivatives,
# 	}
