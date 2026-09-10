import equinox as eqx

def save_zgraph(node: eqx.Module, path: str):
    """
    Saves a ZGraph module to disk using equinox serialization.
    """
    eqx.tree_serialise_leaves(path, node)

def load_zgraph(node_blueprint: eqx.Module, path: str):
    """
    Loads weights into a pure blueprint module.
    Remember to apply transforms (vmap, compile) dynamically after loading.

    Args:
        node_blueprint: Uncompiled module instance with matching architecture.
        path: Path to a checkpoint created by ``save_zgraph``.
    """
    return eqx.tree_deserialise_leaves(path, node_blueprint)

# Backward-compatibility aliases
save_znet = save_zgraph
load_znet = load_zgraph
