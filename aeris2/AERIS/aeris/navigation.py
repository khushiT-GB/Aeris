"""
navigation.py - A* path planning on a KNOWLEDGE map (never on hidden ground truth).

Cost of entering a cell = distance * multiplier.
  * KNOWN cell   : multiplier grows with slope, roughness and risk. Known hazards are blocked.
  * UNKNOWN cell : conservative fixed penalty (UNKNOWN_MULT) + obstacle-density hint.
    -> unknown terrain is passable (otherwise no plan could exist) but is "expensive", so the
       planner prefers terrain that has been observed to be benign.
"""
import heapq
import math
import numpy as np
from .energy import MAX_SLOPE_DEG, rover_segment

UNKNOWN_MULT = 2.6
W_SLOPE, W_ROUGH, W_RISK = 2.0, 1.0, 2.0
# For ENERGY ESTIMATES over unknown cells the rover uses planet-wide PRIOR statistics
# (as if from orbital imagery), inflated by a safety factor. It never sees the actual unknown cells.
PRIOR_SAFETY = 1.25
SQ2 = math.sqrt(2.0)


def cost_grid(kn, mode="balanced", unknown_mult=UNKNOWN_MULT):
    """Returns (multiplier grid, blocked grid) built ONLY from the knowledge map."""
    sn = np.clip(np.nan_to_num(kn.slope, nan=0.0) / MAX_SLOPE_DEG, 0, 1)
    rough = np.nan_to_num(kn.rough, nan=0.0)
    dens = kn.dens_est()
    risk = np.clip(0.55 * sn + 0.30 * rough + 1.2 * dens, 0, 1)
    ws, wr, wk = (1.0, 0.8, 0.0) if mode == "economy" else (W_SLOPE, W_ROUGH, W_RISK)
    mult = 1.0 + ws * sn + wr * rough + wk * risk
    mult = np.where(kn.known, mult, unknown_mult + 1.5 * dens)
    return mult, kn.blocked()


def astar(mult, blocked, start, goal):
    """8-connected A*. Diagonals may not squeeze between two blocked cells. Returns list of (r,c)."""
    n, m = mult.shape
    ml = mult.tolist()
    bl = blocked.tolist()
    nbrs = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2))
    gr, gc = goal

    def h(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return (dr + dc) + (SQ2 - 2.0) * min(dr, dc)

    best = {start: 0.0}
    parent = {}
    closed = set()
    heap = [(h(*start), 0.0, start[0], start[1])]
    while heap:
        _, g, r, c = heapq.heappop(heap)
        if (r, c) in closed:
            continue
        closed.add((r, c))
        if (r, c) == goal:
            path = [(r, c)]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return path[::-1]
        for dr, dc, w in nbrs:
            rr, cc = r + dr, c + dc
            if rr < 0 or cc < 0 or rr >= n or cc >= m or bl[rr][cc] or (rr, cc) in closed:
                continue
            if dr and dc and (bl[r + dr][c] or bl[r][c + dc]):
                continue
            ng = g + w * 0.5 * (ml[r][c] + ml[rr][cc])
            if ng < best.get((rr, cc), 1e18):
                best[(rr, cc)] = ng
                parent[(rr, cc)] = (r, c)
                heapq.heappush(heap, (ng + h(rr, cc), ng, rr, cc))
    return None


def path_length_m(path, cell_m):
    return sum(cell_m * (SQ2 if (a[0] != b[0] and a[1] != b[1]) else 1.0)
               for a, b in zip(path[:-1], path[1:]))


def estimate_path_energy(path, kn, planet, cell_m):
    """
    PLANNER-SIDE estimate (uses knowledge only). Unknown cells get conservative assumptions.
    Returns (total_wh incl. electronics, time_s).
    """
    total, tsum = 0.0, 0.0
    for a, b in zip(path[:-1], path[1:]):
        dist = cell_m * (SQ2 if (a[0] != b[0] and a[1] != b[1]) else 1.0)
        pr_slope = min(20.0, 7.0 * planet.slope_severity * PRIOR_SAFETY)
        pr_rough = min(1.0, planet.terrain_roughness * PRIOR_SAFETY)
        if kn.known[a] and kn.known[b]:
            dz = kn.elev[b] - kn.elev[a]
        else:
            dz = 0.3 * math.tan(math.radians(pr_slope)) * dist
        slope = kn.slope[b] if kn.known[b] else pr_slope
        rough = kn.rough[b] if kn.known[b] else pr_rough
        d, e, t, _ = rover_segment(planet, dist, float(dz), float(rough), float(slope))
        total += d + e
        tsum += t
    return total, tsum
