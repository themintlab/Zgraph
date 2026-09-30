import json

with open("thermograph/examples/einstein_oscillator.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            if "fcns_compiled[0](jnp.atleast_1d(t))[0]" in line:
                line = line.replace("fcns_compiled[0](jnp.atleast_1d(t))[0]", "fcns_compiled[0](jnp.atleast_1d(t))")
            if "fcns_compiled[1](jnp.atleast_1d(t))[0]" in line:
                line = line.replace("fcns_compiled[1](jnp.atleast_1d(t))[0]", "fcns_compiled[1](jnp.atleast_1d(t))")
            
            # also replace if it used array([t])
            if "fcns_compiled[0](jnp.array([t]))[0]" in line:
                line = line.replace("fcns_compiled[0](jnp.array([t]))[0]", "fcns_compiled[0](jnp.atleast_1d(t))")
            if "fcns_compiled[1](jnp.array([t]))[0]" in line:
                line = line.replace("fcns_compiled[1](jnp.array([t]))[0]", "fcns_compiled[1](jnp.atleast_1d(t))")
                
            new_source.append(line)
        cell["source"] = new_source

with open("thermograph/examples/einstein_oscillator.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

