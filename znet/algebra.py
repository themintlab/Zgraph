
# znet/algebra.py
#import copy
import torch

class ThermoAlgebra:
    """Mixin that enables 'Thermo-Algebra' DSL syntax."""

    def __or__(self, other):
        """ 
        Binary operator for 'or' : |
        Implies stacking of nodes
        """
        from .nodes.geometry import StackNode 

        left = list(self.sub_nodes) if isinstance(self, StackNode) else [self]
        right = list(other.sub_nodes) if isinstance(other, StackNode) else [other]
        return StackNode(left + right)
    
    def __add__(self, other):
        """
        Binary operator for 'addition' : +
        Implies mixing of nodes
        """
        from .nodes.geometry import MixingNode

        left = list(self.sub_nodes) if (isinstance(self, MixingNode) and self.enthalpy is None) else [self]
        right = list(other.sub_nodes) if (isinstance(other, MixingNode) and other.enthalpy is None) else [other]
        return MixingNode(left + right)

    def __matmul__(self, enthalpy):
        """
        Binary operator for matrix multiply : @
        Applies an enthalpy node to a mixing expression.

        Non-mixing nodes are promoted to a degenerate single-node MixingNode,
        so expressions like SourceNode(...) @ enthalpy are valid.
        """
        from .nodes.geometry import MixingNode

        if not callable(enthalpy):
            raise TypeError("Enthalpy must be a callable node/module with forward(inputs, temperature).")

        base = self if isinstance(self, MixingNode) else MixingNode([self])

        # Build a new node to preserve functional-style DSL behavior.
        return MixingNode(list(base.sub_nodes), enthalpy=enthalpy, scale=base.scale)