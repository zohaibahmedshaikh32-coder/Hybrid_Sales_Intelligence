"""
conftest.py — ensures the project root is on sys.path when pytest runs.

Without this, `import src.config` fails because pytest executes test
modules from `tests/` and Python doesn't automatically add the project
root to the module search path.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))