import json

with open("thermograph/examples/einstein_oscillator.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            if "fcns_compiled = graph_to_function" in line:
                continue # remove this line entirely
            
            if "fcns_compiled[0](jnp.atleast_1d(t))" in line:
                line = line.replace("fcns_compiled[0](jnp.atleast_1d(t))", "cu_phase(jnp.atleast_1d(t))")
            if "fcns_compiled[1](jnp.atleast_1d(t))" in line:
                line = line.replace("fcns_compiled[1](jnp.atleast_1d(t))", "al_phase(jnp.atleast_1d(t))")
                
            new_source.append(line)
        cell["source"] = new_source

with open("thermograph/examples/einstein_oscillator.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

