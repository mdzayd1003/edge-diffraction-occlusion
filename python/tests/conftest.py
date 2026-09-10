import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffraction.geometry import Point, Wedge  # noqa: E402


@pytest.fixture()
def thin_screen():
    return Wedge(open_angle=2 * math.pi, height=3.0)


@pytest.fixture()
def source():
    return Point(-5.0, 0.0, 1.5)


@pytest.fixture()
def receiver():
    return Point(10.0, 0.0, 1.5)


@pytest.fixture()
def shadow_boundary_receiver():
    """Exactly on the line from the source over the edge at (0, ·, 3)."""
    return Point(10.0, 0.0, 1.5 + (3.0 - 1.5) * (10.0 + 5.0) / 5.0)


@pytest.fixture()
def frequencies():
    return np.array([125.0, 500.0, 2000.0, 8000.0])


@pytest.fixture(scope="session")
def matlab_reference():
    path = ROOT.parent / "data" / "matlab_reference_study.csv"
    import csv

    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    return {
        (float(r["height_m"]), float(r["receiver_distance_m"]), float(r["wedge_angle_deg"])): [
            float(v) for k, v in r.items() if k.startswith("il_")
        ]
        for r in rows
    }
