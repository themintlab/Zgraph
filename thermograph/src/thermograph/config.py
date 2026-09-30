import os

# Root directory of the thermograph package
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Centralized data paths
LIBRARY_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, '..', '..', 'examples', 'library'))

# Create the library directory if it doesn't exist
os.makedirs(LIBRARY_DIR, exist_ok=True)
