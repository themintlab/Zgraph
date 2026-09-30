import json

with open("thermograph/examples/bayesian_si_training.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        
        # Fix sys.path for VSCode
        if "root_dir = os.path.abspath" in source_str and "sys.path.insert" in source_str:
            new_source = []
            for line in cell["source"]:
                if "root_dir =" in line:
                    new_source.append("root_dir = os.getcwd() if not os.getcwd().endswith('examples') else os.path.abspath('../../')\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source
            
        # Fix BFGS joint initialization and scaling
        elif "joint_loss_fn" in source_str:
            new_source = []
            for line in cell["source"]:
                if "joint_init =" in line or "joint_init_scaled =" in line:
                    new_source.append("# Give BFGS a much better starting point so it doesn't instantly explode the gradient\n")
                    new_source.append("joint_init_scaled = jnp.array([-2.8, 5.0])  # Scaled by 10000 and 100\n")
                elif "print(f\"Optimization Success" in line:
                    new_source.append(line)
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open("thermograph/examples/bayesian_si_training.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

