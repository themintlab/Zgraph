# Zgraph Monorepo

[![Unit Tests](https://github.com/themintlab/Zgraph/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/themintlab/Zgraph/actions/workflows/unit-tests.yml)

Welcome to the `Zgraph` monorepo. This repository contains a suite of tools for physics-free tensor graph execution and their domain-specific application wrappers.

## 📦 Packages

- **[`zgraph`](./zgraph):** The core physics-free tensor-graph math engine. It provides high-performance, differentiable LogSumExp contractions and factor graph evaluation natively in PyTorch 2.0 (`vmap`, `torch.compile`).
- **[`thermograph`](./thermograph):** The thermodynamic application layer built on top of `zgraph`. It maps human-readable CALPHAD and statistical mechanics concepts (elements, phases) down to the unlabeled math engine.

### Future Expansions
The architecture is designed to support arbitrary domain graphs in the future (e.g., `trafficgraph`, `economicgraph`) as independent packages that leverage the same underlying `zgraph` math engine.

## 🏗️ Architecture Philosophy
Each package strictly isolates concerns:
- `zgraph` knows purely about tensors, indices, and math.
- Domain packages (like `thermograph`) know about physical models, units, and nomenclature.

For detailed design principles and directives for contributing, please see the `ARCHITECTURE.md` file within each package's source directory.
