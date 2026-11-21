**This is a hackathon prototype, not a flight- or rover-certified system.** Nothing here is validated against real hardware.

| Item | What is real in this code | What is simplified / assumed |
|---|---|---|
| Terrain | Real NumPy/SciPy computation of a procedural DEM, slope (finite differences), roughness, obstacle density, traversability, risk | The terrain is **synthetic**; no real planetary data is used |
| Rover energy | Rolling resistance + climbing (m·g·Δh) + slip + electronics power, in consistent Wh units; gravity enters via m·g | Mass 900 kg, Crr, efficiency, 60 W electronics are **assumed**. Downhill energy is not recovered. Drive speed is **sim-scaled** (1 m/s) so demos finish quickly. No soil mechanics |
| Scout energy | Momentum-theory hover power P = (m·g)^1.5 / (√(2ρA)·FM·η) + avionics; falls with air density, rises with gravity; flight is impossible when required power exceeds the motor limit | Ingenuity-*inspired* numbers, not Ingenuity data. No wind, no tip-speed/Mach limits, no thermal limits |
| Moon / no-air case | Atmospheric flight is **not used**; a short-range orbital recon platform flies pre-planned ground-track passes | The platform's energy is an *allocation* (payload power + per-pass overhead), not an orbit-mechanics model |
| Sensing | Scout/rover observe only a footprint; scout data are noisy and **can miss boulders** (p_detect 93 % / 85 %) | A DEM-based sensor model - NOT image-based 3-D reconstruction |
| Communication | Packet queue, bit-rate, delay, distance-based quality, random loss with retries, random outages, store-and-forward | Not a real RF link budget; no real satellite relay; light-time delay not modelled |
| Planning | A* on the rover's **knowledge** map. Unknown cells are passable but cost 2.6×; known hazards are blocked. Energy estimates over unknown cells use planet-wide prior statistics ×1.25 | The rover-only baseline replans as it senses (radius 4 cells), so it is a fair, adaptive baseline |
| Fairness | Same terrain, start, destination, rover model and battery for both strategies; scout energy is **added** to AERIS total | AERIS rover waits for first recon packet (≤ 45 s), which costs idle energy and time |
| Machine learning | None. All models are deterministic formulas plus seeded random noise | - |

**"Worth it" rule:** recon is worth it if (a) AERIS total system energy ≤ rover-only, or (b) hazard exposure falls ≥ 25 % for ≤ 15 % more system energy, or (c) rover-only fails but AERIS arrives.
AERIS is **not** forced to win: run the multi-seed study to see seeds where it is not worth the scout energy.
