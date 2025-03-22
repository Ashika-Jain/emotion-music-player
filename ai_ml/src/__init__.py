# This file allows the directory to be recognized as a Python module.
# This file makes the directory a Python package
import os
import sys

# Add this directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
