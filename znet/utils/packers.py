from collections.abc import Mapping, Sequence
from typing import Any

import torch


def build_input_tensor(
	state_values: Mapping[str, Any],
	state_order: Sequence[str] | None = None,
	*,
	dtype: torch.dtype | None = None,
	device: torch.device | str | None = None,
	mode: str = "cartesian",
	indexing: str = "ij",
	flatten: bool = False,
	requires_grad: bool = False,
):
	"""
	Build a graph input tensor from a {state: values} mapping.

	By default, each state's values are treated as a 1D axis in a Cartesian
	product grid. Optionally, mode="matched" zips states by index instead.
	The final tensor's last dimension is the state/channel axis expected by
	SourceNode indexing.

	Args:
		state_values: Mapping of state name -> 1D values.
		state_order: Optional explicit order of states in the final channel axis.
			If omitted, insertion order from ``state_values`` is used.
		dtype: Optional output dtype. If omitted, inferred from input tensors.
		device: Optional output device. If omitted, inferred from input tensors.
		mode: "cartesian" for full permutations, "matched" for index-wise pairing.
		indexing: Meshgrid indexing mode, usually "ij".
		flatten: If True, returns shape (N_points, N_states).
			Otherwise returns (*state_sizes, N_states) in cartesian mode.
			In matched mode, output is already (N_points, N_states).
		requires_grad: If True, output tensor requires grad.

	Returns:
		tuple[torch.Tensor, list[str]]:
			- Packed input tensor for the graph.
			- The resolved state order used for channels.
	"""
	if not isinstance(state_values, Mapping) or not state_values:
		raise ValueError("state_values must be a non-empty mapping of state -> values")

	if mode not in {"grid", "matched"}:
		raise ValueError("mode must be either 'grid' or 'matched'")

	if state_order is None:
		ordered_keys = list(state_values.keys())
	else:
		ordered_keys = list(state_order)
		missing = [k for k in ordered_keys if k not in state_values]
		if missing:
			raise KeyError(f"state_order contains keys not found in state_values: {missing}")

	vectors = []
	for key in ordered_keys:
		value = state_values[key]
		if isinstance(value, torch.Tensor):
			vec = value
		else:
			vec = torch.as_tensor(value)

		if vec.ndim == 0:
			vec = vec.unsqueeze(0)
		elif vec.ndim != 1:
			vec = vec.reshape(-1)

		vectors.append(vec)

	inferred_device = device
	if inferred_device is None:
		for vec in vectors:
			if vec.device.type != "cpu":
				inferred_device = vec.device
				break

	inferred_dtype = dtype
	if inferred_dtype is None:
		inferred_dtype = torch.promote_types(
			vectors[0].dtype if vectors else torch.float32,
			torch.float32,
		)
		for vec in vectors[1:]:
			inferred_dtype = torch.promote_types(inferred_dtype, vec.dtype)

	vectors = [vec.to(device=inferred_device, dtype=inferred_dtype) for vec in vectors]

	if mode == "matched":
		lengths = [vec.numel() for vec in vectors]
		if len(set(lengths)) != 1:
			raise ValueError(
				"matched mode requires all state vectors to have the same length. "
				f"Got lengths: {lengths}"
			)

		packed = torch.stack(vectors, dim=-1)
		if requires_grad:
			packed.requires_grad_(True)
		return packed, ordered_keys

	grid_views = torch.meshgrid(*vectors, indexing=indexing)
	out_shape = tuple(vec.numel() for vec in vectors) + (len(vectors),)
	packed = torch.empty(out_shape, dtype=inferred_dtype, device=inferred_device)

	# Copy each broadcasted meshgrid view into its state/channel slice.
	for idx, grid in enumerate(grid_views):
		packed[..., idx] = grid

	if flatten:
		packed = packed.reshape(-1, len(vectors))

	if requires_grad:
		packed.requires_grad_(True)

	return packed, ordered_keys
