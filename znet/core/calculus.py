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