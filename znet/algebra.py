
# znet/algebra.py

class _GraphAlgebra:
    """Mixin that enables compositional DSL syntax for graph nodes."""

    def __or__(self, other):
        """ 
        Binary operator for 'or' : |
        Stacks/combines nodes into a vector of outputs
        """
        from .nodes.geometry import OrNode 

        left = list(self.sub_nodes) if isinstance(self, OrNode) else [self]
        right = list(other.sub_nodes) if isinstance(other, OrNode) else [other]
        return OrNode(left + right)
    
    def __add__(self, other):
        """
        Binary operator for 'addition' : +
        Combines nodes with optional interaction term
        """
        from .nodes.geometry import AndNode

        left = list(self.sub_nodes) if (isinstance(self, AndNode) and self.enthalpy is None) else [self]
        right = list(other.sub_nodes) if (isinstance(other, AndNode) and other.enthalpy is None) else [other]
        return AndNode(left + right)

    def __matmul__(self, interaction):
        """
        Binary operator for matrix multiply : @
        Applies an interaction term to a composite node.

        Non-composite nodes are promoted to a single-node composite,
        so expressions like SourceNode(...) @ interaction_fn are valid.
        """
        from .nodes.geometry import AndNode

        if not callable(interaction):
            raise TypeError("Interaction must be a callable node/module.")

        base = self if isinstance(self, AndNode) else AndNode([self])

        # Build a new node to preserve functional-style DSL behavior.
        return AndNode(list(base.sub_nodes), enthalpy=interaction)


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