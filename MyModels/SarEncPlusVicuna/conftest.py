"""
conftest.py — pytest configuration for SarEncPlusVicuna project.

Adds the project root to sys.path so that `from model.xxx import ...`
works from any test file without needing to install the package.
"""
import sys
import os

# Ensure project root is always on the path
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
