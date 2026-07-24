# Thermograph Architecture & Design Philosophy

`thermograph` is the Application and UI layer built on top of the `zgraph`. It acts as the bridge between human-readable thermodynamic concepts (Elements, Microstates, Enthalpies, Phases) and the strict, tensor-only requirements of the `zgraph` math engine.

`thermograph` knows physics, but does not do math. Rather, it constructructs `zgraph` objects for calculation. 

## 1. The Separation of Concerns
`thermograph` classes (e.g., `PhaseModel`, `PhysicsModule`) are standard Python classes, that contain '.engine' attributes which are zgraph components.  `thermograph` **never** executes mathematical loops over microstates, rather delegating all heavy lifting to its internal `zgraph` engine instances.

## 2. The Linker / Compiler Pattern
`thermograph` acts as a compiler, mapping human-readable requirements of its physics modules to the input channels (indicies). Indices are then bound through 'zgraph' methods. 

## 3. Safe Physics Hot-Swapping
`thermograph` provides the user-facing API for dynamically changing thermodynamic models (e.g., swapping a Regular Solution model for a Redlich-Kister polynomial).

Because `zgraph` graph structures are immutable (to preserve `torch.compile` compatibility), `thermograph` handles hot-swapping through **Envelope Replacement**. 

## 4. Introspection is Offline
All methods that return Pandas DataFrames or human-readable mappings (e.g., `get_configuration_matrix()`) must safely detach data from the PyTorch computation graph using `.detach().cpu().numpy()`. This ensures that data scientists can interrogate the thermodynamic states visually without accidentally breaking the Autograd backpropagation tree.