"""
Ensure this strategy's utils package is imported instead of another strategy's.

Each strategy ships its own top-level `utils` package, so a repo-wide pytest
run would otherwise reuse whichever `utils` was imported first.
"""

import os
import sys

STRATEGY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for name in list(sys.modules):
    if name == 'utils' or name.startswith('utils.'):
        del sys.modules[name]

while STRATEGY_DIR in sys.path:
    sys.path.remove(STRATEGY_DIR)
sys.path.insert(0, STRATEGY_DIR)
