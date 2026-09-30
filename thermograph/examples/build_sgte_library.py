import os
import sys

# Setup paths to ensure we can load zgraph and thermograph
examples_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(examples_dir, '..', '..'))
sys.path.insert(0, os.path.join(root_dir, 'zgraph', 'src'))
sys.path.insert(0, os.path.join(root_dir, 'thermograph', 'src'))
sys.path.insert(0, root_dir)

from thermograph.nodes.sgte import SGTENode
from zgraph import save

def build_library():
    lib_dir = os.path.join(examples_dir, 'library')
    os.makedirs(lib_dir, exist_ok=True)
    
    # We will import all the datasets we have
    from external_data.si_ge_sgte import GHSERSI, GLIQSI, GHSERGE, GLIQGE
    from external_data.cu_au_sgte import GHSERCU, GLIQCU, GHSERAU, GLIQAU
    from external_data.cu_ni_sgte import GHSERNI, GLIQNI
    
    datasets = {
        'GHSERSI': GHSERSI,
        'GLIQSI': GLIQSI,
        'GHSERGE': GHSERGE,
        'GLIQGE': GLIQGE,
        'GHSERCU': GHSERCU,
        'GLIQCU': GLIQCU,
        'GHSERAU': GHSERAU,
        'GLIQAU': GLIQAU,
        'GHSERNI': GHSERNI,
        'GLIQNI': GLIQNI
    }
    
    for name, data in datasets.items():
        node = SGTENode(data, T_index=0)
        save(node, os.path.join(lib_dir, f'{name}.zg'))
        print(f"Serialized {name}.zg")

if __name__ == "__main__":
    build_library()

