# Zgraph Examples

This directory contains Jupyter notebooks demonstrating the usage of `zgraph` for thermodynamic equilibrium calculations, Legendre transformations, and soft-tropical tensor computations.

All notebooks include one-click Google Colab badges to run interactively in the cloud with zero setup.

---

## 📓 Available Examples

### 1. 2D Binary Phase Equilibrium (`binary.ipynb`)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/Zgraph/blob/main/zgraph/examples/binary.ipynb)

Demonstrates 2D binary phase equilibrium construction using `zgraph` factor graphs, automatic Legendre transformation, and Plotly visualization of Grand Potential and Free Energy curves.

---

### 2. 3D Phase Surface & Extruded Dimension (`extruded_dimension.ipynb`)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/Zgraph/blob/main/zgraph/examples/extruded_dimension.ipynb)

Extends binary phase equilibrium into 3D parameter spaces over phase fraction ($p_a$) grids, rendering 3D surface plots of equilibrium energy landscapes using Plotly 3D.

---

### 3. Stoichiometric Phase Equilibrium (`stoichiometric.ipynb`)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/Zgraph/blob/main/zgraph/examples/stoichiometric.ipynb)

Calculates stoichiometric phase equilibrium curves and transformed Legendre free energy representations.

---

## ⚡ Running Locally

To run these notebooks locally:

```bash
pip install -e ./zgraph
pip install jupyter plotly
jupyter notebook
```
