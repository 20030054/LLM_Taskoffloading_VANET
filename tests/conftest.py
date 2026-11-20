import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("llm", "analysis", "experiments", "scenarios"):
    sys.path.insert(0, str(ROOT / p))
