import os
import sys

# Ensure src directory is on python search path for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
