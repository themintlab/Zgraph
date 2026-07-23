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
    !pip install -q git+https://github.com/themintlab/Znet.git"""

for nb_path in notebooks:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    filename = os.path.basename(nb_path)
    colab_url = f"https://colab.research.google.com/github/{repo_url}/blob/{branch}/znet/examples/{filename}"
    badge_md = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})"
    
    # Check if badge already exists
    has_badge = any(badge_md in cell.source for cell in nb.cells if cell.cell_type == 'markdown')
    has_install = any("https://github.com/themintlab/Znet.git" in cell.source for cell in nb.cells if cell.cell_type == 'code')
    
    cells_to_add = []
    
    if not has_badge:
        md_cell = nbformat.v4.new_markdown_cell(source=badge_md)
        cells_to_add.append(md_cell)
        
    if not has_install:
        code_cell = nbformat.v4.new_code_cell(source=install_code)
        cells_to_add.append(code_cell)
        
    if cells_to_add:
        nb.cells = cells_to_add + nb.cells
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Updated {filename}")
    else:
        print(f"{filename} already updated")
