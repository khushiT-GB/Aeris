"""
knowledge.py - A partial map of the world held by one agent.

AERIS keeps FOUR separate maps (see mission_controller):
  1. ground truth      (environment.Terrain - the simulator's secret)
  2. rover knowledge   (KnowledgeMap owned by the rover)
  3. scout observations(KnowledgeMap owned by the scout)
  4. shared knowledge  (KnowledgeMap of packets that were actually delivered)
"""
import numpy as np
from .energy import MAX_SLOPE_DEG
from . import terrain_analysis as ta


class KnowledgeMap:
    def __init__(self, shape):
        self.shape = shape
        self.known = np.zeros(shape, dtype=bool)
        self.elev = np.full(shape, np.nan)
        self.slope = np.full(shape, np.nan)
        self.rough = np.full(shape, np.nan)
        self.obstacle = np.zeros(shape, dtype=bool)

    def update(self, rows, cols, elev, slope, rough, obstacle):
        self.known[rows, cols] = True
        self.elev[rows, cols] = elev
        self.slope[rows, cols] = slope
        self.rough[rows, cols] = rough
        self.obstacle[rows, cols] = obstacle

    @property
    def n_known(self):
        return int(self.known.sum())

    def blocked(self):
        s = np.nan_to_num(self.slope, nan=0.0)
        return self.known & (self.obstacle | (s > MAX_SLOPE_DEG))

    def dens_est(self):
        return ta.obstacle_density(self.obstacle & self.known)

    def derived(self):
        """(slope, rough, trav, risk) with NaN where unknown."""
        s = np.nan_to_num(self.slope, nan=0.0)
        r = np.nan_to_num(self.rough, nan=0.0)
        d = self.dens_est()
        b = self.blocked()
        trav = ta.traversability(s, r, d, b)
        rk = ta.risk(s, r, d, b)
        nanmask = ~self.known
        return (np.where(nanmask, np.nan, self.slope), np.where(nanmask, np.nan, self.rough),
                np.where(nanmask, np.nan, trav), np.where(nanmask, np.nan, rk))
