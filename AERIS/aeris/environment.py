"""
environment.py - Procedural planetary terrain (the GROUND TRUTH).

A digital elevation model (DEM) is built from layered smoothed noise (hills + valleys),
bowl-and-rim craters, fine roughness (rocks), and boulder fields. Everything is
reproducible from (seed, planet parameters).
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, label
from . import terrain_analysis as ta


@dataclass
class Terrain:
    seed: int
    n: int
    cell_m: float
    elevation: np.ndarray
    roughness: np.ndarray
    obstacle: np.ndarray
    slope_deg: np.ndarray
    obstacle_dens: np.ndarray
    blocked: np.ndarray
    traversability: np.ndarray
    risk: np.ndarray
    terrain_class: np.ndarray
    start: tuple
    goal: tuple
    attempt: int = 0


def _norm(a):
    a = a - a.mean()
    return a / (a.std() + 1e-9)


def _build(planet, seed, attempt, n, cell_m, start, goal, smooth=0.0):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(attempt)]))
    z = np.zeros((n, n))
    for sigma, amp in ((n / 7, 1.0), (n / 14, 0.55), (n / 28, 0.28), (1.5, 0.10)):
        z += amp * _norm(gaussian_filter(rng.normal(size=(n, n)), sigma, mode="reflect"))
    z *= planet.hill_amp_m * planet.slope_severity * 0.5

    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(int(planet.n_craters)):                       # bowl + raised rim
        cy, cx = rng.uniform(0.08 * n, 0.92 * n, 2)
        R = rng.uniform(3.0, 8.0)
        d = np.hypot(yy - cy, xx - cx)
        depth = 0.25 * R * cell_m * planet.slope_severity
        z -= np.where(d < R, depth * (1 - (d / R) ** 2), 0.0)
        z += 0.25 * depth * np.exp(-(((d - R) / (0.22 * R)) ** 2))

    if smooth > 0:
        z = gaussian_filter(z, smooth)

    for (pr, pc) in (start, goal):                               # flat landing / target zones
        d = np.hypot(yy - pr, xx - pc)
        w = np.exp(-(d / 2.5) ** 2)
        z = z * (1 - w) + z[pr, pc] * w

    slope0 = ta.compute_slope(z, cell_m)
    sn = np.clip(slope0 / 25.0, 0, 1)
    fine = _norm(gaussian_filter(rng.normal(size=(n, n)), 1.2))
    patch = _norm(gaussian_filter(rng.normal(size=(n, n)), 4.0))
    rough = np.clip(planet.terrain_roughness * (0.55 + 0.25 * fine + 0.35 * patch) + 0.25 * sn, 0, 1)

    dens = planet.obstacle_density
    blob = _norm(gaussian_filter(rng.normal(size=(n, n)), 2.0))
    obstacle = np.zeros((n, n), dtype=bool)
    if dens > 0:
        thr = np.quantile(blob, 1 - 0.6 * dens)
        obstacle |= blob > thr
        weight = np.clip(0.6 + rough, 0.1, 2.0)
        obstacle |= rng.random((n, n)) < 0.4 * dens * weight
    for (pr, pc) in (start, goal):
        obstacle[np.hypot(yy - pr, xx - pc) <= 2.6] = False
    return z, rough, obstacle


def generate_terrain(planet, seed=7, n=64, cell_m=10.0, start=None, goal=None):
    """Build a terrain where start and goal are connected through passable cells."""
    start = start or (int(0.08 * n), int(0.08 * n))
    goal = goal or (int(0.90 * n), int(0.88 * n))
    smooth = 0.0
    for attempt in range(60):
        if attempt >= 40:                       # last resort: progressively smooth the DEM
            smooth = 0.6 * (attempt - 39)
        z, rough, obst = _build(planet, seed, attempt, n, cell_m, start, goal, smooth)
        a = ta.analyze(z, rough, obst, cell_m)
        lab, _ = label(~a["blocked"])           # 4-connectivity
        ok = lab[start] > 0 and lab[start] == lab[goal] and a["blocked"].mean() < 0.45
        if ok:
            return Terrain(seed=int(seed), n=n, cell_m=cell_m, elevation=z, roughness=rough,
                           obstacle=obst, slope_deg=a["slope_deg"], obstacle_dens=a["obstacle_dens"],
                           blocked=a["blocked"], traversability=a["traversability"], risk=a["risk"],
                           terrain_class=a["terrain_class"], start=start, goal=goal, attempt=attempt)
    raise RuntimeError("Could not generate a connected terrain; lower slope/obstacle settings.")
