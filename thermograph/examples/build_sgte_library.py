import os
from thermograph.io.tdb import extract_sgte_library
from thermograph.config import LIBRARY_DIR

def build_library():
    tdb_path = os.path.join(os.path.dirname(LIBRARY_DIR), 'unary.tdb')
    
    # Extract all phases from the unary database
    output_dir = os.path.join(LIBRARY_DIR, 'unary')
    print(f"Extracting all phases from {tdb_path} to {output_dir}...")
    extract_sgte_library(tdb_path, output_dir)

if __name__ == "__main__":
    build_library()



