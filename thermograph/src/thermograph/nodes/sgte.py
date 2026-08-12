# Placeholder for SGTE Node

class SGTENode:
    """
    Placeholder for the SGTE polynomial thermodynamic model.
    Will eventually compile into a zgraph execution node.
    """
    def __init__(self, *args, **kwargs):
        pass

    def compile_zgraph_engine(self):
        """
        Compiles the SGTE parameters into a purely numerical zgraph FactorNode.
        """
        raise NotImplementedError("SGTENode is currently a placeholder.")
