"""
Shared pytest configuration for all eval tiers. Ensures the project root
is importable regardless of where pytest is invoked from.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from metrics import collector

# Hook to run after all tests have completed. Prints a summary of
# collected evaluation metrics.
def pytest_sessionfinish(session, exitstatus):
    """Hook to run after all tests have completed. Prints a summary of
    collected evaluation metrics."""
    print("\n\n=== Evaluation Metrics Summary ===")
    summary = collector.summary()
    for name, s in summary.items():
        print(
            f"{name:<22} | mean={s['mean']:.4f} | std={s['std']:.4f} "
            f"| min={s['min']:.2f} | max={s['max']:.2f} | samples={s['count']}"
        )
