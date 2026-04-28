from collections.abc import Sequence


def _iter_graph_modules(root_node):
	"""Yield graph modules for traversal across nn.Module and custom trees."""
	modules_fn = getattr(root_node, "modules", None)
	if callable(modules_fn):
		yield from modules_fn()
		return

	# Fallback for non-nn.Module graphs: DFS over common child containers.
	stack = [root_node]
	seen = set()

	while stack:
		node = stack.pop()
		node_id = id(node)
		if node_id in seen:
			continue
		seen.add(node_id)
		yield node

		sub_nodes = getattr(node, "sub_nodes", None)
		if sub_nodes is not None:
			stack.extend(reversed(list(sub_nodes)))


def build_bus_registry(ordered_keys: Sequence[str]) -> dict[str, int]:
	"""Create the state->channel index mapping used by node binding."""
	if not isinstance(ordered_keys, Sequence) or not ordered_keys:
		raise ValueError("ordered_keys must be a non-empty sequence of state names")

	keys = list(ordered_keys)
	if len(set(keys)) != len(keys):
		raise ValueError("ordered_keys contains duplicate state names")

	return {name: idx for idx, name in enumerate(keys)}


def bind_graph_to_bus(root_node, ordered_keys: Sequence[str]) -> dict[str, int]:
	"""
	Bind all graph nodes that expose ``bind_to_bus`` using an ordered key bus.

	This utility is graph-level and does not require a ``ZNet`` wrapper.

	Args:
		root_node: Root graph node/module.
		ordered_keys: State ordering returned by build_input_tensor.

	Returns:
		dict[str, int]: Registry mapping state name -> channel index.
	"""
	registry = build_bus_registry(ordered_keys)

	for module in _iter_graph_modules(root_node):
		bind_fn = getattr(module, "bind_to_bus", None)
		if callable(bind_fn):
			bind_fn(registry)

	return registry
