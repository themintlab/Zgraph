
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
        from .nodes.geometry import OrNode 

        left = list(self.sub_nodes) if isinstance(self, OrNode) else [self]
        right = list(other.sub_nodes) if isinstance(other, OrNode) else [other]
        return OrNode(left + right)
    
    def __add__(self, other):
        """
        Binary operator for 'addition' : +
        Implies mixing of nodes
        """
        from .nodes.geometry import AndNode

        left = list(self.sub_nodes) if (isinstance(self, AndNode) and self.enthalpy is None) else [self]
        right = list(other.sub_nodes) if (isinstance(other, AndNode) and other.enthalpy is None) else [other]
        return AndNode(left + right)

    def __matmul__(self, enthalpy):
        """
        Binary operator for matrix multiply : @
        Applies an enthalpy node to a mixing expression.

        Non-mixing nodes are promoted to a degenerate single-node MixingNode,
        so expressions like SourceNode(...) @ enthalpy are valid.
        """
        from .nodes.geometry import AndNode

        if not callable(enthalpy):
            raise TypeError("Enthalpy must be a callable node/module with forward(inputs, temperature).")

        base = self if isinstance(self, AndNode) else AndNode([self])

        # Build a new node to preserve functional-style DSL behavior.
        return AndNode(list(base.sub_nodes), enthalpy=enthalpy)


class _CollapseBracket:
    """Bracket helper so DSL can use _[expr] -> CollapseNode(expr)."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def __getitem__(self, expr):
        from .nodes.geometry import CollapseNode

        return CollapseNode(expr, scale=self.scale)


# Usage: _[muA | muB]
_ = _CollapseBracket()

# Usage: __[muA | muB] for sharper collapse.
__ = _CollapseBracket(scale=1e-6)