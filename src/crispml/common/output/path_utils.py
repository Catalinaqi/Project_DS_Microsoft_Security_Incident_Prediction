# src/crispml/common/output/path_utils.py

from pathlib import Path
from src.crispml.config.enums.enums_config import PhaseName
def get_output_dir(phase_name: PhaseName) -> Path:
    """
    Build and return the output directory for a given CRISP-ML phase.

    Flow:
    1. Convert the enum value `phase_name` to its name in lowercase
       (e.g. PhaseName.PHASE2_DATA_UNDERSTANDING -> "phase2_data_understanding").
    2. Resolve the project root by taking the current file path (`__file__`)
       and moving two levels up in the directory tree.
    3. Append "out" and the lowercase phase name to this root path to form
       the final output directory: <project_root>/out/<phase_name>.
    4. Create the directory (and any missing parent directories) if it does
       not already exist.
    5. Return the resulting `Path` object so other code can use it to save
       files for that phase.
    """

    phase_str: str = phase_name.name.lower()
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "out" / phase_str
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
