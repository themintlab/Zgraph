import json

with open("thermograph/examples/einstein_oscillator.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            if "jnp.array([t])" in line:
                line = line.replace("jnp.array([t])", "jnp.atleast_1d(t)")
            new_source.append(line)
        cell["source"] = new_source

with open("thermograph/examples/einstein_oscillator.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

