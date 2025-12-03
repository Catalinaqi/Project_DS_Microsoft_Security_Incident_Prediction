# src/crispml/common/output/path_utils.py

from pathlib import Path

def get_output_dir(phase_name: str) -> Path:
    """
    Returns output directory <root>/out/<phase_name> and ensures existence.
    """
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "out" / phase_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
