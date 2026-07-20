import os
import glob
import json

examples_dir = r"c:\Users\wellandm\Code\Znet\znet\examples"
notebooks = glob.glob(os.path.join(examples_dir, "*.ipynb"))

new_imports = [
    "from znet.core import FactorNode, SignalNode, ConstantNode, LeafNode\n",
    "from znet.transforms import finalize, legendre_transform\n"
]

for nb_path in notebooks:
    print(f"Processing {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    modified = False
    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") == "code":
            new_source = []
            for line in cell.get("source", []):
                if "from znet import *" in line:
                    leading_spaces = line[:len(line) - len(line.lstrip())]
                    newline = "\n" if line.endswith("\n") else ""
                    
                    for i, imp in enumerate(new_imports):
                        out_line = leading_spaces + imp.strip()
                        if i < len(new_imports) - 1 or newline:
                            out_line += "\n"
                        new_source.append(out_line)
                    modified = True
                else:
                    new_source.append(line)
            cell["source"] = new_source
            
    if modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1)
            f.write("\n")
        print(f"Updated {nb_path}")
    else:
        print(f"No changes for {nb_path}")

print("Update complete!")
