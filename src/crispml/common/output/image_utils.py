# src/crispml/common/output/image_utils.py

from pathlib import Path
import matplotlib.pyplot as plt

from .path_utils import get_output_dir
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)

def save_figure(fig: plt.Figure, filename: str, phase_name: str, dpi: int = 150):
    out_dir = get_output_dir(phase_name)
    if not filename.lower().endswith(".png"):
        filename += ".png"

    path = out_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("[image_utils] Figure saved: %s", path)
    return path
