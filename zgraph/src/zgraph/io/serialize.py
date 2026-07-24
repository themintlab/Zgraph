import torch
import torch.nn as nn

def save_znet(node: nn.Module, path: str):
    """
    Saves the state_dict of an uncompiled ZGraph module safely.
    """
    torch.save(node.state_dict(), path)

def load_znet(node_blueprint: nn.Module, path: str):
    """
    Loads weights into a pure blueprint module.
    Remember to apply transforms (vmap, compile) dynamically after loading.
    """
    node_blueprint.load_state_dict(torch.load(path))
    return node_blueprint
