"""
terrain_analysis.py - Slope, obstacle density, traversability and risk.
Shared by the ground-truth generator AND by the knowledge maps (so both use identical formulas).
"""
import numpy as np
from scipy.ndimage import uniform_filter
from .energy import MAX_SLOPE_DEG

CLASS_NAMES = ["Plain", "Rough", "Steep", "Impassable"]


def compute_slope(z, cell_m):
    """Slope angle in degrees from a DEM using finite differences."""
    gy, gx = np.gradient(z, cell_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def obstacle_density(obstacle, size=5):
    """Fraction of boulder cells in a size x size window."""
    return uniform_filter(obstacle.astype(float), size=size, mode="nearest")


def traversability(slope, rough, dens, blocked):
    """0 = impassable, 1 = perfectly easy."""
    sn = np.clip(slope / MAX_SLOPE_DEG, 0, 1)
    t = (1 - sn ** 1.5) * (1 - 0.6 * rough) * (1 - np.clip(dens * 2.5, 0, 0.9))
    return np.where(blocked, 0.0, np.clip(t, 0, 1))


def risk(slope, rough, dens, blocked):
    """0 = safe, 1 = extremely hazardous."""
    sn = np.clip(slope / MAX_SLOPE_DEG, 0, 1)
    r = np.clip(0.55 * sn + 0.30 * rough + 1.2 * dens, 0, 1)
    return np.where(blocked, 1.0, r)


def classify(slope, rough, blocked):
    """0 plain, 1 rough, 2 steep, 3 impassable."""
    c = np.zeros(slope.shape, dtype=int)
    c[rough > 0.5] = 1
    c[slope > 15] = 2
    c[blocked] = 3
    return c


def analyze(z, rough, obstacle, cell_m):
    slope = compute_slope(z, cell_m)
    dens = obstacle_density(obstacle)
    blocked = obstacle | (slope > MAX_SLOPE_DEG)
    return {
        "slope_deg": slope, "obstacle_dens": dens, "blocked": blocked,
        "traversability": traversability(slope, rough, dens, blocked),
        "risk": risk(slope, rough, dens, blocked),
        "terrain_class": classify(slope, rough, blocked),
    }
