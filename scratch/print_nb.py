import json

with open("thermograph/examples/bayesian_si_training.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "H =" in source or "np.random.normal" in source or "theta_samples" in source:
            print(f"Cell {i}:")
            print(source)
            print("-" * 40)

