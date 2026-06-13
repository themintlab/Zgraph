# import torch
# from torch.func import jacfwd, jacrev, vmap, grad_and_value
import torch
from torch.func import jacfwd, jacrev, vmap, grad_and_value

# def value_and_derivatives(f, inputs, n=1):
# 	def _build_n_derivative_engine(f, n=1):
# 		"""
def value_and_derivatives(f, inputs, n=1):
	def _build_n_derivative_engine(f, n=1):
		"""
# 		Builds a pure function that calculates f(x) and its first n derivatives.
# 		Uses optimal AD stacking: grad for 1st derivative, jacfwd for higher orders.
# 		"""
		
# 		# 1. Define a scalar function for a SINGLE sample vector x.
# 		# Most nodes expect a batch dimension, so we temporarily add one.
		# 1. Define a scalar function for a SINGLE sample vector x.
		# Most nodes expect a batch dimension, so we temporarily add one.
# 		def scalar_fcn(x):
# 			out = f(x.unsqueeze(0))
# 			if out.numel() != 1:
# 				raise RuntimeError(
# 					"value_and_derivatives expects f to produce one scalar per sample. "
# 					f"Got output shape {tuple(out.shape)} for a single sample."
# 				)
			out = f(x.unsqueeze(0))
			if out.numel() != 1:
				raise RuntimeError(
					"value_and_derivatives expects f to produce one scalar per sample. "
					f"Got output shape {tuple(out.shape)} for a single sample."
				)
# 			return out.reshape(())

# 		def base_fcn(x):
# 			# Get gradient and value
# 			g, v = grad_and_value(scalar_fcn)(x)
# 			# unsqueeze v here (scalar) so vmap produces (batch, 1) directly
# 			# Primary output: g (so jacfwd differentiates it into a Hessian)
# 			# Auxiliary output: (v, g) (so we don't lose them)
		def base_fcn(x):
			# Get gradient and value
			g, v = grad_and_value(scalar_fcn)(x)
			# unsqueeze v here (scalar) so vmap produces (batch, 1) directly
			# Primary output: g (so jacfwd differentiates it into a Hessian)
			# Auxiliary output: (v, g) (so we don't lose them)
# 			return g, (v.unsqueeze(-1), g)
		
# 		if n==1:
# 			return base_fcn
		if n==1:
			return base_fcn

# 		current_fn = base_fcn
# 		for i in range(2, n + 1):
# 			# We must isolate the loop variable scope in Python
# 			def make_next_fn(prev_fn):
# 				def next_fn(x):
# 					# jacfwd differentiates the primary output, passes aux through!
# 					j, aux = jacfwd(prev_fn, has_aux=True)(x)
		current_fn = base_fcn
		for i in range(2, n + 1):
			# We must isolate the loop variable scope in Python
			def make_next_fn(prev_fn):
				def next_fn(x):
					# jacfwd differentiates the primary output, passes aux through!
					j, aux = jacfwd(prev_fn, has_aux=True)(x)
					
# 					# Append the new n-th derivative to the aux tuple
					# Append the new n-th derivative to the aux tuple
# 					return j, aux + (j,)
# 				return next_fn
				return next_fn
				
# 			current_fn = make_next_fn(current_fn)
			current_fn = make_next_fn(current_fn)

# 		return current_fn
		return current_fn


# 	if not torch.is_tensor(inputs):
# 		raise TypeError("inputs must be a torch.Tensor")
# 	if inputs.ndim < 2:
# 		raise ValueError("inputs must have shape (..., n_features)")
	if not torch.is_tensor(inputs):
		raise TypeError("inputs must be a torch.Tensor")
	if inputs.ndim < 2:
		raise ValueError("inputs must have shape (..., n_features)")

# 	batch_shape = inputs.shape[:-1]
# 	n_features = inputs.shape[-1]
# 	flat_inputs = inputs.reshape(-1, n_features)
	batch_shape = inputs.shape[:-1]
	n_features = inputs.shape[-1]
	flat_inputs = inputs.reshape(-1, n_features)

# 	n_deriv_engine = vmap(_build_n_derivative_engine(f, n=n))
# 	_, all_terms = n_deriv_engine(flat_inputs)
	n_deriv_engine = vmap(_build_n_derivative_engine(f, n=n))
	_, all_terms = n_deriv_engine(flat_inputs)

# 	def _restore_batch_dims(tensor):
# 		return tensor.reshape(*batch_shape, *tensor.shape[1:])
	def _restore_batch_dims(tensor):
		return tensor.reshape(*batch_shape, *tensor.shape[1:])

# 	return tuple(_restore_batch_dims(term) for term in all_terms)
	return tuple(_restore_batch_dims(term) for term in all_terms)

# # UNTESTED

def partial_legendre_transform(f, inputs, active_indices):
	"""
	Computes a partial Legendre transform of function f evaluated at inputs.
	
	Args:
		f (callable): The function (e.g., PhaseA module) to transform.
		inputs (torch.Tensor): Batched input tensor of shape (..., n_features).
		active_indices (list[int] or torch.Tensor): The channel indices 
			corresponding to the variables being transformed.
			
	Returns:
		tuple:
			- conjugate_vars (torch.Tensor): The computed conjugate variables (partial grads).
			- transformed_val (torch.Tensor): The new Legendre-transformed energy.
	"""
	if not torch.is_tensor(inputs):
		raise TypeError("inputs must be a torch.Tensor")
		
	batch_shape = inputs.shape[:-1]
	n_features = inputs.shape[-1]
	flat_inputs = inputs.reshape(-1, n_features)

# # ** Compute first derivative, value, and hessian as one. Gives complete information. 
# # def legendre_transform_direct():
# # 	#** can use JVP for legendre transform but this eliminates the dual variable recovery. Maybe useful for 
# # 	#maybe useful for recovery of convex region since transform correlated to specific dual variable?
# # 	pass
	def scalar_fcn(x):
		out = f(x.unsqueeze(0))
		if out.numel() != 1:
			raise RuntimeError("Function must return a scalar per sample.")
		return out.reshape(())

	# 1. Setup the vmap engine
	gv_fn = grad_and_value(scalar_fcn)
	batched_gv_fn = vmap(gv_fn)

	# 2. Compute full gradient and value (Highly optimized by torch.compile)
	full_grad, val = batched_gv_fn(flat_inputs)

	# 3. Extract partial gradients (conjugate variables)
	conjugate_vars = full_grad[..., active_indices]
	
	# 4. Extract active input variables
	active_x = flat_inputs[..., active_indices]

	# 5. Compute Legendre transform: L = F(x) - sum(x_i * conjugate_i)
	transform_term = (active_x * conjugate_vars).sum(dim=-1)
	transformed_val = val - transform_term

	# Restore original batch dimensions
	conjugate_vars = conjugate_vars.reshape(*batch_shape, -1)
	transformed_val = transformed_val.reshape(*batch_shape, 1)

	return conjugate_vars, transformed_val