"""
Shared pytest configuration for all eval tiers. Ensures the project root
is importable regardless of where pytest is invoked from.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
