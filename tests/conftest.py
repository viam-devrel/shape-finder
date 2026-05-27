import sys
from pathlib import Path

# Allow `from models.shape_detection import ...` in tests, matching how
# main.py imports modules at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
