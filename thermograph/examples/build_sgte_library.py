import os
import sys

# Setup paths to ensure we can load zgraph and thermograph
examples_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(examples_dir, '..', '..'))
sys.path.insert(0, os.path.join(root_dir, 'zgraph', 'src'))
sys.path.insert(0, os.path.join(root_dir, 'thermograph', 'src'))
sys.path.insert(0, root_dir)

from thermograph.io.tdb import extract_sgte_library

def build_library():
    tdb_path = os.path.join(examples_dir, 'unary.tdb')
    lib_dir = os.path.join(examples_dir, 'library')
    
    # We only extract the Si and Ge phases as an example
    phases_to_extract = ['GHSERSI', 'GLIQSI', 'GHSERGE', 'GLIQGE']
    
    print(f"Extracting phases from {tdb_path}...")
    extract_sgte_library(tdb_path, lib_dir, phases=phases_to_extract)

if __name__ == "__main__":
    build_library()


