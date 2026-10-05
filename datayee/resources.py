"""Locate files bundled with a source checkout or an installed wheel."""

from pathlib import Path
import sys


def asset_path(relative_path):
    """Return the absolute path for a model or other DataYee asset."""
    relative = Path(relative_path)
    candidates = [
        Path(__file__).resolve().parents[1] / relative,
        Path(sys.prefix) / "DataYee" / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
