import torch
import torch.nn as nn

def save_zgraph(node: nn.Module, path: str):
    """
    Saves the state_dict of an uncompiled ZGraph module safely.
    """
    torch.save(node.state_dict(), path)

def load_zgraph(node_blueprint: nn.Module, path: str, map_location: str | torch.device = "cpu"):
    """
    Loads weights into a pure blueprint module.
    Remember to apply transforms (vmap, compile) dynamically after loading.

    Args:
        node_blueprint: Uncompiled module instance with matching architecture.
        path: Path to a checkpoint created by ``save_zgraph``.
        map_location: Device mapping for ``torch.load``. Defaults to ``"cpu"``
            to keep loads portable across machines without GPU availability.
    """
    state_dict = torch.load(path, map_location=map_location)
    node_blueprint.load_state_dict(state_dict)
    return node_blueprint

# Backward-compatibility aliases
save_znet = save_zgraph
load_znet = load_zgraph
