"""
scout.py - The aerial scout (rotorcraft) or, where flight is impossible, the orbital recon platform.

Sensor model (simplified, NOT image reconstruction): every tick the scout observes all cells inside
a circular footprint. Observations of the DEM are noisy: elevation/slope/roughness get Gaussian
noise; boulders are detected with probability p_detect (can be MISSED) plus rare false alarms.
"""
import math
from functools import lru_cache
import numpy as np
from . import energy
from .communication import Packet
from .knowledge import KnowledgeMap
from .energy import MAX_SLOPE_DEG

BATCH_INTERVAL_S = 6.0
MAX_BATCH_CELLS = 120
BYTES_PER_CELL = 12


@lru_cache(maxsize=64)
def disk_offsets(radius):
    r = int(math.ceil(radius))
    ii, jj = np.mgrid[-r:r + 1, -r:r + 1]
    m = ii ** 2 + jj ** 2 <= radius ** 2
    return ii[m], jj[m]


def disk_cells(r0, c0, radius, n):
    di, dj = disk_offsets(round(float(radius), 2))
    rr, cc = r0 + di, c0 + dj
    ok = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
    return rr[ok], cc[ok]


class Scout:
    def __init__(self, terrain, planet, kind, radius, rng, log, capacity_wh=None):
        self.t, self.planet, self.kind, self.radius, self.rng, self.log = terrain, planet, kind, radius, rng, log
        n = terrain.n
        self.n = n
        self.base = (float(terrain.start[0]), float(terrain.start[1]))
        self.pos = list(self.base)
        self.capacity = capacity_wh if capacity_wh is not None else planet.scout_battery_wh
        self.battery_wh = self.capacity
        self.energy_used = 0.0
        self.status = "GROUNDED"
        self.obs = KnowledgeMap((n, n))                       # SCOUT OBSERVATIONS map
        self.surveyed = np.zeros((n, n), dtype=bool)
        self.pending = []
        self.last_packet_t = 0.0
        self.pid = 0
        self.trail = [tuple(self.pos)]
        self.target = None
        self.last_sector = None
        self.timer = 0.0
        self.noise = 1.0 if kind == "aerial" else 2.0
        self.p_detect = 0.93 if kind == "aerial" else 0.85
        self.distance_flown_m = 0.0
        if kind == "aerial":
            self.p_hover = energy.hover_power_w(planet)
            self.p_fwd = self.p_hover * energy.FWD_FACTOR
            self.speed = energy.SCOUT_SPEED_MS / terrain.cell_m           # cells / s
        else:
            self.p_hover = self.p_fwd = energy.ORBITAL_PAYLOAD_W * planet.scout_energy_mult
            self.speed = energy.ORBITAL_SPEED_MS / terrain.cell_m
            self.passes = self._plan_passes()
            self.pass_idx = -1
            self.altitude_m = energy.ORBITAL_ALTITUDE_M

    # ------------------------------------------------------------- helpers
    @property
    def active(self):
        return self.status in ("TAKEOFF", "SURVEYING", "RETURNING", "LANDING", "GROUNDED")

    def sector(self, r, c):
        k = max(1, self.n // 8)
        return f"Sector {int(r) // 8 * k + int(c) // 8 + 1:02d}"

    def _consume(self, power_w, dt):
        e = power_w * dt / 3600.0
        self.battery_wh -= e
        self.energy_used += e

    def _plan_passes(self):
        s, g = np.array(self.base), np.array(self.t.goal, float)
        u = (g - s) / (np.linalg.norm(g - s) + 1e-9)
        nrm = np.array([-u[1], u[0]])
        w = 1.7 * self.radius
        out = []
        for off in (0.0, w, -w):
            a, b = s + nrm * off - u * 2, g + nrm * off + u * 2
            out.append((np.clip(a, 0, self.n - 1), np.clip(b, 0, self.n - 1)))
        return out

    def _move_toward(self, tgt, dt):
        dr, dc = tgt[0] - self.pos[0], tgt[1] - self.pos[1]
        d = math.hypot(dr, dc)
        step = self.speed * dt
        if d <= step:
            self.pos[0], self.pos[1] = tgt
            moved, reached = d, True
        else:
            self.pos[0] += dr / d * step
            self.pos[1] += dc / d * step
            moved, reached = step, False
        self.distance_flown_m += moved * self.t.cell_m
        return reached

    def _dist_base_m(self):
        return math.hypot(self.pos[0] - self.base[0], self.pos[1] - self.base[1]) * self.t.cell_m

    # ------------------------------------------------------------- sensing
    def survey(self):
        r0, c0 = int(round(self.pos[0])), int(round(self.pos[1]))
        rr, cc = disk_cells(r0, c0, self.radius, self.n)
        m = ~self.surveyed[rr, cc]
        rr, cc = rr[m], cc[m]
        k = len(rr)
        if k == 0:
            return
        tr, rng, ns = self.t, self.rng, self.noise
        elev = tr.elevation[rr, cc] + rng.normal(0, 0.25 * ns, k)
        slope = np.clip(tr.slope_deg[rr, cc] + rng.normal(0, 1.2 * ns, k), 0, 90)
        rough = np.clip(tr.roughness[rr, cc] + rng.normal(0, 0.04 * ns, k), 0, 1)
        obst = (tr.obstacle[rr, cc] & (rng.random(k) < self.p_detect)) | (rng.random(k) < 0.005)
        self.obs.update(rr, cc, elev, slope, rough, obst)
        self.surveyed[rr, cc] = True
        self.pending.append((rr, cc, elev, slope, rough, obst))

    def make_packet(self, t, force=False):
        if not self.pending:
            return None
        n_pending = sum(len(p[0]) for p in self.pending)
        if not (force or t - self.last_packet_t >= BATCH_INTERVAL_S or n_pending >= MAX_BATCH_CELLS):
            return None
        rr, cc, el, sl, ro, ob = (np.concatenate([p[i] for p in self.pending]) for i in range(6))
        self.pending = []
        self.last_packet_t = t
        self.pid += 1
        haz = ob | (sl > MAX_SLOPE_DEG)
        nh = int(haz.sum())
        kind = "HAZARD_ALERT" if nh >= 4 else "TERRAIN_OBS"
        sec = self.sector(self.pos[0], self.pos[1])
        self.log("SCOUT", "Terrain observation generated.")
        if nh >= 4:
            what = "Boulder field" if ob.sum() >= (sl > MAX_SLOPE_DEG).sum() else "High slope region"
            self.log("SCOUT", f"Hazard detected - {what} ({nh} cells, {sec}).")
        return Packet(self.pid, kind, "SCOUT", "ROVER", t, rr, cc, el, sl, ro, ob,
                      BYTES_PER_CELL * len(rr) + 16, sec, nh)

    # ------------------------------------------------------------- behaviour
    def step(self, dt, t, rover):
        if self.status in ("LANDED", "DEPLETED", "COMPLETE"):
            return
        if self.kind == "aerial":
            self._step_aerial(dt, rover)
        else:
            self._step_orbital(dt)
        self.trail.append((self.pos[0], self.pos[1]))
        if self.battery_wh <= 0:
            self.battery_wh = 0.0
            self.status = "DEPLETED"
            self.log("SCOUT", "Battery depleted - forced landing.")

    def _pick_target(self, rover):
        path = rover.path[rover.path_idx:] if rover.path else []
        for i, (r, c) in enumerate(path):
            if not self.surveyed[r, c]:
                j = min(len(path) - 1, i + int(0.6 * self.radius))
                return (float(path[j][0]), float(path[j][1]))
        return None

    def _step_aerial(self, dt, rover):
        if self.status == "GROUNDED":
            self.status, self.timer = "TAKEOFF", energy.TAKEOFF_S
            self.log("SCOUT", "Deploying - rotor spin-up and takeoff.")
        if self.status == "TAKEOFF":
            self._consume(self.p_hover, dt)
            self.timer -= dt
            if self.timer <= 0:
                self.status = "SURVEYING"
            self.survey()
            return
        if self.status == "LANDING":
            self._consume(self.p_hover, dt)
            self.timer -= dt
            if self.timer <= 0:
                self.status = "LANDED"
                self.log("SCOUT", f"Landed safely. Battery {100 * self.battery_wh / self.capacity:.0f}%.")
            return
        if self.status == "SURVEYING":
            if self.target is None:
                self.target = self._pick_target(rover)
                if self.target is None:
                    self.status = "RETURNING"
                    self.log("SCOUT", "Planned corridor fully surveyed - returning to base.")
                else:
                    sec = self.sector(*self.target)
                    if sec != self.last_sector:
                        self.last_sector = sec
                        self.log("SCOUT", f"Survey initiated - {sec}.")
            if self.target is not None and self._move_toward(self.target, dt):
                self.target = None
            self._consume(self.p_fwd, dt)
            self.survey()
            need = (self._dist_base_m() / energy.SCOUT_SPEED_MS * self.p_fwd * 1.2
                    + energy.LAND_S * self.p_hover) / 3600.0 + 0.03 * self.capacity
            if self.status == "SURVEYING" and self.battery_wh <= need:
                self.status = "RETURNING"
                self.target = None
                self.log("SCOUT", "Battery reserve reached - returning to base.")
            return
        if self.status == "RETURNING":
            if self._move_toward(self.base, dt):
                self.status, self.timer = "LANDING", energy.LAND_S
            self._consume(self.p_fwd, dt)
            self.survey()

    def _step_orbital(self, dt):
        if self.status == "GROUNDED":
            self.status = "SURVEYING"
            self.log("SCOUT", "Orbital recon platform ONLINE (no atmosphere - drone flight impossible).")
        if self.status != "SURVEYING":
            return
        if self.target is None:
            self.pass_idx += 1
            if self.pass_idx >= len(self.passes):
                self.status = "COMPLETE"
                self.log("SCOUT", "All recon passes complete.")
                return
            a, b = self.passes[self.pass_idx]
            self.pos = [float(a[0]), float(a[1])]
            self.target = (float(b[0]), float(b[1]))
            e = energy.ORBITAL_PASS_OVERHEAD_WH * self.planet.scout_energy_mult
            self.battery_wh -= e
            self.energy_used += e
            self.log("SCOUT", f"Survey initiated - recon pass {self.pass_idx + 1}/{len(self.passes)}.")
        if self._move_toward(self.target, dt):
            self.target = None
        self._consume(self.p_fwd, dt)
        self.survey()
