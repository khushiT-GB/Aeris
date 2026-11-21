"""
mission_controller.py - Owns terrain, rover, scout, comm link, event log and the 4 knowledge maps.
Advance with step() (1 tick = cfg.dt simulated seconds). Streamlit just calls step() repeatedly.
"""
from dataclasses import dataclass, field
import numpy as np
from . import energy
from .environment import generate_terrain
from .rover import Rover
from .scout import Scout
from .communication import CommLink
from .planetary_models import Planet, describe_scout
from . import metrics


@dataclass
class MissionConfig:
    planet: Planet
    seed: int = 7
    grid_n: int = 64
    cell_m: float = 10.0
    goal: tuple = None                 # (row, col) or None for default
    survey_radius: float = 6.0         # cells
    comm_failures: bool = False
    packet_loss: float = 0.05
    delay_s: float = 1.5
    bitrate_bps: float = 4000.0
    comm_range_m: float = 900.0
    sense_radius: int = 4
    reserve_frac: float = 0.10
    recon_wait_s: float = 45.0
    dt: float = 1.0
    max_time_s: float = 8000.0


def simulate_rover_only(terrain, cfg):
    """Baseline: same rover, same terrain, NO scout. Returns a summary dict (incl. trail)."""
    r = Rover(terrain, cfg.planet, cfg.planet.rover_battery_wh, None, cfg.sense_radius, cfg.reserve_frac)
    t = 0.0
    while not r.done and t < cfg.max_time_s:
        r.step(cfg.dt)
        t += cfg.dt
    s = metrics.rover_summary(r)
    s["trail"] = list(r.trail)
    s["initial_path"] = list(r.initial_path)
    s["replans"] = r.replans
    return s


class Mission:
    def __init__(self, cfg: MissionConfig, terrain=None, baseline=None):
        self.cfg = cfg
        self.planet = cfg.planet
        self.terrain = terrain or generate_terrain(cfg.planet, cfg.seed, cfg.grid_n, cfg.cell_m, goal=cfg.goal)
        self.baseline = baseline or simulate_rover_only(self.terrain, cfg)
        self.reset()

    # ---------------------------------------------------------------- setup
    def reset(self):
        cfg, tr = self.cfg, self.terrain
        self.t = 0.0
        self.events = []
        self.history = []
        self.scout_info = describe_scout(self.planet)
        self.kind = self.scout_info["kind"]
        self.rng_scout = np.random.default_rng(np.random.SeedSequence([cfg.seed, 11]))
        self.rng_comm = np.random.default_rng(np.random.SeedSequence([cfg.seed, 23]))
        log = self.log
        self.rover = Rover(tr, self.planet, self.planet.rover_battery_wh, log, cfg.sense_radius, cfg.reserve_frac)
        self.scout = Scout(tr, self.planet, self.kind, cfg.survey_radius, self.rng_scout, log)
        self.comm = CommLink((tr.n, tr.n), self.planet.comm_quality, self.rng_comm, log,
                             range_m=cfg.comm_range_m, bitrate_bps=cfg.bitrate_bps,
                             base_delay_s=cfg.delay_s, loss_base=cfg.packet_loss,
                             failures=cfg.comm_failures)
        self.rover.hold = True
        self.rover.more_info_expected = lambda: self.scout.active or self.comm.busy
        self.rover.sense()
        self.rover.replan("periodic")                       # initial plan from local knowledge only
        if self.rover.status == "IDLE":
            self.rover.status = "READY"
        self.log("MISSION", f"{self.planet.name} mission initialised - seed {cfg.seed}.")
        self.log("MISSION", f"Scout: {self.scout_info['name']}. {self.scout_info['why']}")
        self.log("ROVER", "Initial route planned from local sensing only; unknown terrain penalised.")
        self.released = False
        self.done = False

    def log(self, tag, msg):
        self.events.append((self.t, tag, msg))
        if len(self.events) > 600:
            del self.events[:100]

    def update_live(self):
        """Apply live-adjustable settings (radius, comm) without resetting."""
        c = self.cfg
        self.scout.radius = c.survey_radius
        self.comm.failures, self.comm.loss_base = c.comm_failures, c.packet_loss
        self.comm.base_delay, self.comm.bitrate, self.comm.range_m = c.delay_s, c.bitrate_bps, c.comm_range_m
        if self.kind == "orbital":
            self.scout.passes = self.scout._plan_passes() if self.scout.pass_idx < 0 else self.scout.passes

    # ---------------------------------------------------------------- stepping
    def _scout_rover_dist_m(self):
        d = np.hypot(self.scout.pos[0] - self.rover.pos[0], self.scout.pos[1] - self.rover.pos[1]) * self.terrain.cell_m
        if self.kind == "orbital":
            d = float(np.hypot(d, self.scout.altitude_m))
        return float(d)

    def step(self):
        if self.done:
            return
        dt, t = self.cfg.dt, self.t
        self.scout.step(dt, t, self.rover)
        pk = self.scout.make_packet(t, force=self.scout.status in ("LANDED", "COMPLETE", "DEPLETED"))
        if pk:
            self.comm.enqueue(pk)
        self.comm.link_in_use = self.scout.active or self.comm.busy
        for p in self.comm.step(t, dt, self._scout_rover_dist_m()):
            self.rover.receive(p)
        if not self.released and (self.comm.received > 0 or t >= self.cfg.recon_wait_s or
                                  (not self.scout.active and not self.comm.busy)):
            self.released = True
            self.rover.hold = False
            self.log("ROVER", "Recon data available - beginning traverse." if self.comm.received > 0
                     else "No recon data yet - proceeding on local knowledge.")
        self.rover.step(dt)
        self.t += dt
        if int(self.t) % 5 == 0:
            self.history.append({"t": self.t, "rover_pct": self.rover.battery_pct,
                                 "scout_pct": 100.0 * self.scout.battery_wh / self.scout.capacity,
                                 "rover_wh": self.rover.energy_used, "scout_wh": self.scout.energy_used})
        if self.rover.done or self.t >= self.cfg.max_time_s:
            self.done = True
            self.log("MISSION", f"Mission ended: {self.rover.status_text()}.")

    def run(self, max_steps=None):
        n = 0
        while not self.done and (max_steps is None or n < max_steps):
            self.step()
            n += 1

    # ---------------------------------------------------------------- results
    def aeris_metrics(self, project=True):
        return metrics.rover_summary(self.rover, project=project)

    def comparison(self):
        return metrics.comparison(self.baseline, self.aeris_metrics(), self.scout.energy_used)

    def maps(self):
        """The four separate maps."""
        return {"truth": self.terrain, "rover": self.rover.kn, "scout": self.scout.obs, "shared": self.comm.shared}
