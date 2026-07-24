import zgraph.core as zn

### NB: UNTESTED! ##

class PhaseModel:
    """
    The user-facing Phase object. Handles hot-swapping and compilation.
    """
    def __init__(self, name, M_matrix, microstate_labels, column_labels, 
                 global_env_map, subgraphs_dict):
        self.name = name
        self.M_matrix = M_matrix
        self.microstate_labels = microstate_labels
        self.column_labels = column_labels
        self.global_env_map = global_env_map
        self.subgraphs_dict = subgraphs_dict
        
        # Assemble the zNet engine
        self._compile_engine()

    def _compile_engine(self):
        """The internal Linker. Rebuilds the pure math graph safely."""
        ordered_engines = []
        for label in self.column_labels:
            module = self.subgraphs_dict[label]
            # Late Binding: Tell the module where its inputs are
            module.link_to_environment(self.global_env_map)
            ordered_engines.append(module.engine)
            
        self.engine = zn.FactorNode(self.M_matrix, ordered_engines)

    def swap_physics(self, column_label, new_physics_module):
        """Dynamically hot-swaps physics and recompiles the math engine."""
        if column_label not in self.column_labels:
            raise ValueError(f"Column '{column_label}' does not exist in the Phase geometry.")
            
        print(f"Swapping [{column_label}] to {new_physics_module.equation_name}...")
        self.subgraphs_dict[column_label] = new_physics_module
        self._compile_engine()

    def calculate_energy(self, local_signals):
        """Executes the zNet forward pass."""
        return self.engine(local_signals)

    # --- Introspection Methods ---

    def get_configuration_matrix(self) -> pd.DataFrame:
        """Returns the M Matrix as a Pandas DataFrame."""
        return pd.DataFrame(
            self.engine.M.detach().cpu().numpy(), 
            index=self.microstate_labels, 
            columns=self.column_labels
        )
        
    def get_physics_routing(self):
        """Prints the active physics map."""
        print(f"--- Phase: {self.name} Physics Map ---")
        for i, col in enumerate(self.column_labels):
            mod = self.subgraphs_dict[col]
            print(f"Col {i}: [{col}] ---> {mod.equation_name} (Params: {mod.get_parameters()})")