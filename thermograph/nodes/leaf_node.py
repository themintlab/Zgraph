import znet.core as zn

class LeafNode:
    """
    The metadata wrapper for thermodynamic equations. 
    Manages naming and state variables.
    """
    def __init__(self, name, equation_name, required_vars, energy_function, signal_indicies=None, **kwargs):
        self.name = name
        self.equation_name = equation_name
        self.required_vars = required_vars
        
        # Instantiate the pure zNet engine
        self.engine = zn.LeafNode(energy_function, signal_indicies, **kwargs)
        
    def __call__(self, input_signals):
        return self.engine(input_signals)

    def link_to_environment(self, global_env_map):
        """Translates variable strings into zNet integer indices."""
        try:
            indices = [global_env_map[var] for var in self.required_vars]
            self.engine.bind_indices(indices)
        except KeyError as e:
            raise ValueError(f"Module '{self.name}' requires variable {e}, "
                             f"but it is missing from the environment map.")
            
    def get_parameters(self):
        """Introspection without touching PyTorch internals."""
        return {k: v.item() for k, v in self.engine.theta.items()}

    def __repr__(self):
        return f"Leaf node({self.name} | Vars: {self.required_vars})"