from zgraph import load as zgraph_load
from zgraph.core.base import ZGraphNode
from thermograph.config import LIBRARY_DIR

def load_library(model_name: str) -> ZGraphNode:
    """
    Loads a thermograph reference model from the configured centralized library directory.
    Example: load_library('unary/GHSERCU')
    """
    return zgraph_load(model_name, library_dir=LIBRARY_DIR)
