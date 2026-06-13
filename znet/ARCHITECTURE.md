# ZNet Architecture & Design Philosophy

`znet` is an auto-differentiable linear algebra library designed to perform hyper-dimensional LogSumExp contractions and microstate energy calculations using pytorch.

To ensure speed and portability across platforms and hardware architectures all code within this package MUST adhere to the following strict architectural directives. This will ensure robust compilation via torch.script (old) or torch.compile (modern). 

`znet` is purely mathematical and has no knowledge of the application. All labelling of variables, modules, graphs, etc, are relgated to the application layers (e.g. Thermograph) 


## 1. Pure Math, No Python Objects
**No standard Python objects (strings, lists of strings, dictionaries) or Python control flow (`if` statements based on string matching) may exist inside `znet` `nn.Module` classes.**

### The Reason: `torch.compile` and `torch.jit.script`
ZNet is designed to be compiled directly into optimized C++ and CUDA kernels using PyTorch's JIT compilers. TorchScript requires strict, static typing. If a module contains a dictionary of strings or a Pandas DataFrame, the compiler will crash because it cannot translate these into low-level math kernels. All metadata and naming must be handled by the application layer (`thermograph`) *before* data enters ZNet.

## 2. Structural Immutability
`znet` routing nodes (e.g., `FactorNode`) are structurally immutable after creation. You may NOT include methods that mutate the underlying `nn.ModuleList` or swap subgraphs in-place (e.g., `self.subgraphs[2] = new_model`).

### The Reason: Graph Integrity
In PyTorch, the mathematical operations graph must remain static. Swapping a simple polynomial equation for a neural network mid-execution changes the fundamental plumbing of the math. Doing this in-place silently corrupts PyTorch's Autograd (gradient tracking) engine and instantly invalidates any fused C++ kernels. If the physics must change, the application layer must throw away the lightweight `FactorNode` envelope and instantiate a new one.


## . The `**kwargs` Parameter Standard
All `LeafEngine` instances must register trainable parameters dynamically using `**kwargs` and `nn.ParameterDict`. This ensures that ZNet can optimize arbitrary equations without needing to hardcode specific variable shapes into the engine block.

** May be subject to change as application layer matures**

## . Tensor-Only Communication
All inputs and outputs between `znet` modules must be `torch.Tensor` types. No custom classes, tuples of mixed types, or optional arguments are permitted in the `forward` signature.

### The Reason: Kernel Fusion
`torch.compile` performs best when it can trace a continuous stream of tensor operations. Passing non-tensor objects forces the compiler to "break" the graph and return to the Python interpreter (a "graph break"), which significantly degrades performance and prevents the fusion of multiple operations into a single GPU kernel.
