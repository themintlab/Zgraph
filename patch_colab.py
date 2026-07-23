import nbformat
import glob
import os

repo_url = "themintlab/Znet"
branch = "main"
examples_dir = r"c:\Users\wellandm\Code\Znet\znet\examples"
notebooks = glob.glob(os.path.join(examples_dir, "*.ipynb"))

install_code = """# Install Znet if running in Google Colab
import sys
if 'google.colab' in sys.modules:
    import getpass
    import subprocess
    
    print("This repository is private. Please enter a GitHub Personal Access Token (PAT).")
    print("You can generate one at https://github.com/settings/tokens (needs 'repo' scope).")
    pat = getpass.getpass('GitHub PAT: ')
    
    repo_url = f"https://{pat}@github.com/themintlab/Znet.git"
    print("Installing Znet...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"git+{repo_url}"], check=True)
    print("Successfully installed Znet!")"""

for nb_path in notebooks:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    filename = os.path.basename(nb_path)
    colab_url = f"https://colab.research.google.com/github/{repo_url}/blob/{branch}/znet/examples/{filename}"
    badge_md = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})"
    
    # Remove any existing Colab install cells
    nb.cells = [cell for cell in nb.cells if "github.com/themintlab/Znet" not in cell.source and "Open In Colab" not in cell.source]
    
    cells_to_add = []
    
    md_cell = nbformat.v4.new_markdown_cell(source=badge_md)
    cells_to_add.append(md_cell)
        
    code_cell = nbformat.v4.new_code_cell(source=install_code)
    cells_to_add.append(code_cell)
        
    nb.cells = cells_to_add + nb.cells
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Updated {filename}")
