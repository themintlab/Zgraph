import json

with open("thermograph/examples/bayesian_si_training.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        if "joint_loss_fn" in source_str and "E_0, theta_E = params[0], params[1]" in source_str:
            new_source = []
            for line in cell["source"]:
                if "E_0, theta_E = params[0], params[1]" in line:
                    new_source.append("    # Scale parameters for BFGS stability\n")
                    new_source.append("    E_0 = params[0] * 10000.0\n")
                    new_source.append("    theta_E = params[1] * 100.0\n")
                elif "joint_init =" in line:
                    new_source.append("joint_init_scaled = jnp.array([0.0, 3.0])\n")
                elif "x0=joint_init" in line:
                    new_source.append("    x0=joint_init_scaled,\n")
                elif "E_0_opt, theta_E_opt =" in line:
                    new_source.append("E_0_opt, theta_E_opt = res_joint.x[0] * 10000.0, res_joint.x[1] * 100.0\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source
            
        elif "H_joint =" in source_str:
            new_source = []
            for line in cell["source"]:
                if "H_joint =" in line:
                    new_source.append("# We must apply the chain rule to the Hessian to get the unscaled Covariance matrix.\n")
                    new_source.append("H_scaled = jax.hessian(joint_loss_fn)(res_joint.x)\n")
                    new_source.append("scale_matrix = jnp.array([[10000.0, 0], [0, 100.0]])\n")
                    new_source.append("H_unscaled = jnp.matmul(jnp.linalg.inv(scale_matrix), jnp.matmul(H_scaled, jnp.linalg.inv(scale_matrix)))\n")
                    new_source.append("Cov_matrix = jnp.linalg.inv(H_unscaled)\n")
                elif "Cov_matrix =" in line:
                    continue
                elif "np.random.multivariate_normal" in line:
                    new_source.append("joint_samples = np.random.multivariate_normal(jnp.array([E_0_opt, theta_E_opt]), Cov_matrix, size=500)\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open("thermograph/examples/bayesian_si_training.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

