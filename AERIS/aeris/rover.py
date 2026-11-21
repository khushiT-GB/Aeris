"""
rover.py - The ground rover: knowledge map, planning, driving physics, battery-aware decisions.

Honesty note: the rover reads ground truth ONLY for (a) its own local sensing radius and
(b) the physics of the cell it is actually driving through. Route PLANNING uses its knowledge map.
"""
import math
import numpy as np
from . import energy
from .energy import MAX_SLOPE_DEG
from .knowledge import KnowledgeMap
from .navigation import cost_grid, astar, estimate_path_energy, path_length_m, SQ2
from .scout import disk_cells
from .terrain_analysis import CLASS_NAMES

TERMINAL = {"ARRIVED", "RETURNED", "ABORTED", "STRANDED", "NO_ROUTE"}


class Rover:
    def __init__(self, terrain, planet, battery_wh, log=None, sense_radius=4, reserve_frac=0.10):
        self.t, self.planet = terrain, planet
        self.log = log or (lambda tag, msg: None)
        self.sense_radius = sense_radius
        self.kn = KnowledgeMap((terrain.n, terrain.n))          # ROVER KNOWLEDGE map
        self.own = np.zeros((terrain.n, terrain.n), dtype=bool)
        self.cell, self.base, self.goal = terrain.start, terrain.start, terrain.goal
        self.pos = [float(self.cell[0]), float(self.cell[1])]
        self.capacity = battery_wh
        self.battery_wh = battery_wh
        self.reserve_wh = reserve_frac * battery_wh
        self.drive_wh = self.elec_wh = 0.0
        self.distance_m = self.time_s = 0.0
        self.risk_m = 0.0
        self.high_risk_cells = 0
        self.path, self.path_idx = [], 1
        self.initial_path = []
        self.trail = [self.cell]
        self.status = "IDLE"
        self.terrain_label = CLASS_NAMES[int(terrain.terrain_class[self.cell])]
        self.speed_ms = 0.0
        self.seg, self.seg_prog = None, 0.0
        self.needs_replan = False
        self.cells_since_replan = 0
        self.replans = 0
        self.hold = False
        self.returning = False
        self.economy = False
        self.low_battery = False
        self.projected_wh = None
        self.more_info_expected = lambda: False
        self.initial_len_m = None

    # ------------------------------------------------------------------ properties
    @property
    def energy_used(self):
        return self.drive_wh + self.elec_wh

    @property
    def battery_pct(self):
        return 100.0 * self.battery_wh / self.capacity

    @property
    def done(self):
        return self.status in TERMINAL

    def remaining_len_m(self):
        if not self.path or self.path_idx >= len(self.path):
            return 0.0
        rest = [tuple(map(int, self.cell))] + self.path[self.path_idx:]
        return path_length_m(rest, self.t.cell_m)

    def progress(self):
        if self.status == "ARRIVED":
            return 1.0
        if not self.initial_len_m:
            return 0.0
        return float(np.clip(1.0 - self.remaining_len_m() / self.initial_len_m, 0.0, 1.0))

    def est_range_m(self):
        if self.distance_m > 30:
            per_m = self.energy_used / self.distance_m
        else:
            d, e, tt, _ = energy.rover_segment(self.planet, 10.0, 0.0, 0.3, 5.0)
            per_m = (d + e) / 10.0
        return max(0.0, self.battery_wh / max(per_m, 1e-9))

    def return_cost_wh(self):
        if len(self.trail) < 2:
            return 0.0
        return estimate_path_energy(self.trail[::-1], self.kn, self.planet, self.t.cell_m)[0]

    def status_text(self):
        if self.status == "DRIVING" and self.speed_ms and self.speed_ms < 0.6 * energy.V_NOM_MS:
            return "DRIVING - SLOWED (difficult terrain)"
        return {"IDLE": "STANDBY", "READY": "READY", "HOLDING": "HOLDING - waiting for recon",
                "DRIVING": "DRIVING", "ARRIVED": "ARRIVED AT DESTINATION", "RETURNING": "RETURNING TO BASE",
                "RETURNED": "RETURNED TO BASE (mission aborted)", "ABORTED": "MISSION ABORTED",
                "STRANDED": "STRANDED - BATTERY EMPTY", "NO_ROUTE": "NO SAFE ROUTE"}.get(self.status, self.status)

    # ------------------------------------------------------------------ knowledge
    def sense(self):
        rr, cc = disk_cells(self.cell[0], self.cell[1], self.sense_radius, self.t.n)
        tr = self.t
        self.kn.update(rr, cc, tr.elevation[rr, cc], tr.slope_deg[rr, cc], tr.roughness[rr, cc], tr.obstacle[rr, cc])
        self.own[rr, cc] = True

    def receive(self, p):
        m = ~self.own[p.rows, p.cols]            # rover's own measurements take priority
        if m.any():
            self.kn.update(p.rows[m], p.cols[m], p.elev[m], p.slope[m], p.rough[m], p.obstacle[m])
        self.needs_replan = True
        self.log("ROVER", f"Updated terrain knowledge (+{int(m.sum())} cells from packet #{p.pid}).")

    # ------------------------------------------------------------------ planning
    def replan(self, reason="periodic"):
        self.needs_replan = False
        self.cells_since_replan = 0
        goal = self.base if self.returning else self.goal
        mode = "economy" if (self.returning or self.economy) else "balanced"
        old_ahead = self.path[self.path_idx:self.path_idx + 15]
        if reason != "periodic":
            self.log("ROVER", "Recalculating route.")
        mult, blocked = cost_grid(self.kn, mode)
        path = astar(mult, blocked, self.cell, goal)
        if path is None:
            self.status = "NO_ROUTE"
            self.log("ROVER", "No safe route exists on current knowledge - holding position.")
            return
        self.path, self.path_idx, self.seg, self.seg_prog = path, 1, None, 0.0
        self.replans += 1
        if not self.initial_path:
            self.initial_path = list(path)
            self.initial_len_m = path_length_m(path, self.t.cell_m)
        if old_ahead and path[1:16] != old_ahead and reason != "periodic":
            self.log("ROVER", "Route changed - detouring around newly mapped hazard.")
        if not self.returning:
            self._energy_policy()

    def _energy_policy(self):
        cm = self.t.cell_m
        est, _ = estimate_path_energy(self.path, self.kn, self.planet, cm)
        self.projected_wh = self.battery_wh - est
        if self.projected_wh >= self.reserve_wh:
            self.low_battery = False
            if self.status == "HOLDING":
                self.status = "READY"
                self.log("ROVER", "Energy margin now sufficient - proceeding.")
            return
        mult, blocked = cost_grid(self.kn, "economy")
        alt = astar(mult, blocked, self.cell, self.goal)
        if alt:
            est2, _ = estimate_path_energy(alt, self.kn, self.planet, cm)
            if self.battery_wh - est2 >= self.reserve_wh:
                self.path, self.path_idx, self.seg, self.seg_prog = alt, 1, None, 0.0
                self.projected_wh = self.battery_wh - est2
                self.economy = True
                self.log("ROVER", "Battery margin tight - switching to energy-saving route.")
                return
        at_base = len(self.trail) <= 1
        ret = self.return_cost_wh()
        if at_base:
            if self.more_info_expected():
                self.status = "HOLDING"
                self.log("ROVER", "Energy margin insufficient on current knowledge - waiting for recon.")
            else:
                self.status = "ABORTED"
                self.log("ROVER", "Mission ABORTED: cannot reach destination and keep reserve (conservative estimate).")
            return
        if self.battery_wh >= 1.15 * ret:
            self.returning = True
            self.status = "RETURNING"
            self.log("ROVER", "Cannot safely complete route - mission RECONSIDERED, returning to base.")
            self.replan("return")
        else:
            self.low_battery = True
            self.log("ROVER", "LOW BATTERY - reserve will be violated and no safe return exists; committing to destination.")

    # ------------------------------------------------------------------ physics
    def _consume(self, wh, kind):
        self.battery_wh -= wh
        if kind == "drive":
            self.drive_wh += wh
        else:
            self.elec_wh += wh
        if self.battery_wh <= 0:
            self.battery_wh = 0.0
            self.status = "STRANDED"
            self.speed_ms = 0.0
            self.log("ROVER", "Battery EMPTY - rover stranded.")

    def _make_seg(self, a, b):
        tr = self.t
        diag = a[0] != b[0] and a[1] != b[1]
        dist = tr.cell_m * (SQ2 if diag else 1.0)
        dz = float(tr.elevation[b] - tr.elevation[a])
        rough = float(0.5 * (tr.roughness[a] + tr.roughness[b]))
        slope = float(max(tr.slope_deg[a], tr.slope_deg[b]))
        d, e, tt, v = energy.rover_segment(self.planet, dist, dz, rough, slope)
        return {"a": a, "b": b, "dist": dist, "drive": d, "elec": e, "t": tt, "v": v,
                "risk": float(0.5 * (tr.risk[a] + tr.risk[b]))}

    def step(self, dt):
        if self.done:
            return
        if self.status == "IDLE":
            self.sense()
            self.replan("initial plan" if False else "periodic")
            if self.done:
                return
            if self.status == "IDLE":
                self.status = "READY"
        left = dt
        guard = 0
        while left > 1e-9 and not self.done and guard < 50:
            guard += 1
            if self.seg is None and self.needs_replan:
                self.replan("new terrain data")
                continue
            if self.status == "HOLDING" and not self.more_info_expected():
                self.needs_replan = True       # no more help coming - re-decide
                if self.seg is None:
                    self.replan("no more recon")
                    continue
            if self.hold or self.status == "HOLDING":
                self._consume(energy.idle_wh(self.planet, left), "elec")
                self.time_s += left
                self.speed_ms = 0.0
                return
            if self.seg is None:
                if self.path_idx >= len(self.path):
                    self.status = "RETURNED" if self.returning else "ARRIVED"
                    self.log("ROVER", "Destination reached." if not self.returning else "Back at base.")
                    return
                tgt = self.path[self.path_idx]
                if self.t.blocked[tgt]:                 # safety guard: never enter an impassable cell
                    self.kn.update(np.array([tgt[0]]), np.array([tgt[1]]), self.t.elevation[tgt],
                                   self.t.slope_deg[tgt], self.t.roughness[tgt], np.array([True]))
                    self.own[tgt] = True
                    self.replan("obstacle ahead")
                    continue
                self.seg, self.seg_prog = self._make_seg(self.cell, tgt), 0.0
                if self.status in ("READY", "IDLE"):
                    self.status = "DRIVING"
            seg = self.seg
            use = min(left, seg["t"] * (1 - self.seg_prog))
            frac = use / seg["t"]
            self._consume(seg["drive"] * frac, "drive")
            self._consume(seg["elec"] * frac, "elec")
            self.distance_m += seg["dist"] * frac
            self.risk_m += seg["risk"] * seg["dist"] * frac
            self.time_s += use
            left -= use
            self.seg_prog += frac
            self.speed_ms = seg["v"]
            a, b = seg["a"], seg["b"]
            self.pos = [a[0] + (b[0] - a[0]) * self.seg_prog, a[1] + (b[1] - a[1]) * self.seg_prog]
            if self.done:
                return
            if self.seg_prog >= 1 - 1e-9:
                self.cell = b
                self.pos = [float(b[0]), float(b[1])]
                self.trail.append(b)
                if seg["risk"] >= 0.5:
                    self.high_risk_cells += 1
                self.terrain_label = CLASS_NAMES[int(self.t.terrain_class[b])]
                self.sense()
                self.path_idx += 1
                self.seg = None
                self.cells_since_replan += 1
                ahead = self.path[self.path_idx:self.path_idx + 10]
                blk = self.kn.blocked()
                if self.cells_since_replan >= 4 or any(blk[c] for c in ahead):
                    self.needs_replan = True
                if self.cell == (self.base if self.returning else self.goal):
                    self.status = "RETURNED" if self.returning else "ARRIVED"
                    self.speed_ms = 0.0
                    self.log("ROVER", "Back at base." if self.returning else "Destination reached.")
                    return
