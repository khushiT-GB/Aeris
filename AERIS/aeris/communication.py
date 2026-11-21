"""
communication.py - Simulated scout -> rover radio link (store-and-forward).

Modelled (all simplified):
  * observation packets with a size in bytes
  * serial link: one packet in flight at a time; transmit time = bits / (bitrate * quality) + delay
  * signal quality falls with distance:  q = comm_quality * (1 - (d/range)^2)   (optional)
  * random packet loss (base + extra when quality is low); lost packets are retried up to 2 times
  * random link OUTAGES (optional). During an outage nothing is delivered; packets wait in queue.
"""
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from .knowledge import KnowledgeMap


@dataclass
class Packet:
    pid: int
    kind: str                       # TERRAIN_OBS or HAZARD_ALERT
    sender: str
    receiver: str
    created_t: float
    rows: np.ndarray
    cols: np.ndarray
    elev: np.ndarray
    slope: np.ndarray
    rough: np.ndarray
    obstacle: np.ndarray
    size_bytes: int
    sector: str = ""
    hazard_cells: int = 0
    status: str = "QUEUED"
    tx_start: float = None
    tx_end: float = None
    quality: float = None
    retries: int = 0


class CommLink:
    MAX_RETRY = 2

    def __init__(self, shape, comm_quality, rng, log, range_m=900.0, bitrate_bps=4000.0,
                 base_delay_s=1.5, loss_base=0.05, degradation=True, failures=False,
                 outage_rate=1 / 50.0):
        self.rng, self.log = rng, log
        self.comm_quality, self.range_m, self.bitrate = comm_quality, range_m, bitrate_bps
        self.base_delay, self.loss_base = base_delay_s, loss_base
        self.degradation, self.failures, self.outage_rate = degradation, failures, outage_rate
        self.queue = deque()
        self.inflight = None
        self.outage_until = -1.0
        self.shared = KnowledgeMap(shape)          # SHARED KNOWLEDGE map (delivered data only)
        self.recent = deque(maxlen=40)
        self.sent = self.received = self.lost = self.dropped = self.obs_tx = 0
        self.quality_now = comm_quality
        self.link_up = True
        self._was_up = True
        self.link_in_use = True          # mission sets this; outages only START while the link is needed

    # --- helpers
    def quality(self, dist_m):
        q = self.comm_quality
        if self.degradation:
            q *= max(0.0, 1.0 - (dist_m / self.range_m) ** 2)
        return float(q)

    def enqueue(self, p: Packet):
        p.status = "QUEUED"
        if p.kind == "HAZARD_ALERT":
            self.queue.appendleft(p)                  # hazards jump the queue
        else:
            self.queue.append(p)
        self.recent.append(p)

    @property
    def busy(self):
        return bool(self.queue) or self.inflight is not None

    def _outage(self, t, dt):
        if self.failures and self.link_in_use and t >= self.outage_until and self.rng.random() < self.outage_rate * dt:
            self.outage_until = t + self.rng.uniform(25.0, 60.0)
        up = t >= self.outage_until
        if up != self._was_up:
            self.log("COMM", "Link restored - resuming transmission." if up
                     else "Link LOST - packets will be held in queue.")
            self._was_up = up
        return up

    # --- main step
    def step(self, t, dt, dist_m):
        up = self._outage(t, dt)
        q = self.quality(dist_m)
        self.link_up = up and q >= 0.05
        self.quality_now = q if self.link_up else 0.0
        delivered = []
        p = self.inflight
        if p is not None and t >= p.tx_end:
            self.inflight = None
            if not self.link_up:
                p.status = "INTERRUPTED"
                self.queue.appendleft(p)
                self.log("COMM", f"Packet #{p.pid} interrupted by link loss - re-queued.")
            else:
                ploss = min(0.9, self.loss_base + (1.0 - q / max(self.comm_quality, 1e-6)) * 0.3
                            if self.degradation else self.loss_base)
                if self.rng.random() < ploss:
                    self.lost += 1
                    if p.retries < self.MAX_RETRY:
                        p.retries += 1
                        p.status = "RETRY"
                        self.queue.appendleft(p)
                        self.log("COMM", f"Packet #{p.pid} LOST in transit - retry {p.retries}/{self.MAX_RETRY}.")
                    else:
                        p.status = "DROPPED"
                        self.dropped += 1
                        self.log("COMM", f"Packet #{p.pid} DROPPED after {self.MAX_RETRY} retries.")
                else:
                    p.status = "RECEIVED"
                    self.received += 1
                    self.obs_tx += len(p.rows)
                    self.shared.update(p.rows, p.cols, p.elev, p.slope, p.rough, p.obstacle)
                    delivered.append(p)
                    self.log("COMM", f"Packet #{p.pid} received successfully ({len(p.rows)} cells, q={q:.2f}).")
        if self.inflight is None and self.queue and self.link_up:
            p = self.queue.popleft()
            br = self.bitrate * max(q, 0.1)
            tx = p.size_bytes * 8.0 / br + self.base_delay
            p.tx_start, p.tx_end, p.quality, p.status = t, t + tx, q, "TRANSMITTING"
            self.inflight = p
            self.sent += 1
            self.log("COMM", f"Packet #{p.pid} ({p.kind}) transmitted to rover - ETA {tx:.1f}s, q={q:.2f}.")
        return delivered
