import torch
import torch.nn as nn

def save_zgraph(node: nn.Module, path: str):
    """
    Saves the state_dict of an uncompiled ZGraph module safely.
    """
    torch.save(node.state_dict(), path)

def load_zgraph(node_blueprint: nn.Module, path: str):
    """
    Loads weights into a pure blueprint module.
    Remember to apply transforms (vmap, compile) dynamically after loading.
    """
    node_blueprint.load_state_dict(torch.load(path))
    return node_blueprint

# Backward-compatibility aliases
save_znet = save_zgraph
load_znet = load_zgraph
