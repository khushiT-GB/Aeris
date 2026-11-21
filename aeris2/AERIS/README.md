# 🛰️ AERIS — Aerial Exploration and Rover Intelligence System

An interactive planetary-mission simulator that asks one question:

> **Can selective aerial reconnaissance improve rover navigation enough to justify the energy the scout uses?**

A scout (helicopter-style rotorcraft, or an orbital recon platform where flight is impossible) surveys terrain ahead of a ground rover and radios observations back. The rover replans using that extra knowledge. AERIS is compared with a rover-only baseline on the **same terrain, start, destination and rover**, and the **scout's energy is added to AERIS's total**. AERIS is **not rigged to win** — try the multi-seed study.

---

## 1. Install and run (Windows, beginner-friendly)

1. Install **Python 3.11 or newer** from https://www.python.org/downloads/ — tick **"Add Python to PATH"** in the installer.
2. Unzip the AERIS folder somewhere (e.g. `C:\AERIS`).
3. **Easiest:** double-click **`run_app.bat`**. The first run creates a virtual environment and installs everything (needs internet, ~2 minutes). Your browser opens at http://localhost:8501.

Manual alternative (Command Prompt inside the AERIS folder):
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Next time just `.venv\Scripts\activate` then `streamlit run app.py`.

**No browser needed to check the engine:**
```
python run_headless.py --planet Mars --seed 4 --log
python run_headless.py --planet Moon --study 10
```
**Run the tests:** `python -m unittest discover -s tests -v`

---

## 2. Demo script (matches the 10 scenes in the brief)

Start with **Mars, seed 4** (sidebar). The big map is the main view.

| Scene | What to do | What to point at |
|---|---|---|
| 1 Initiation | Pick Mars; click **🎲 Regenerate terrain** for new terrain; move the *Destination* sliders | Craters, hills, red = true hazards |
| 2 Rover-only | Look at the orange **dashed** route | Planned from local sensing only; fog = unknown terrain |
| 3 Scout deploys | Press **▶ Start** | Purple diamond flies ahead; dashed circle = sensor footprint |
| 4 Hazard detection | Watch the log | `[SCOUT] Hazard detected - Boulder field…` |
| 5 Communication | Watch the log + *Packets* table | `[COMM] Packet #n transmitted / received`, quality, loss, queue |
| 6 Route update | Log | `[ROVER] Updated terrain knowledge` → `Recalculating route` → `Route changed` |
| 7 Execution | Raise *Simulation speed*; watch KPI cards | Both batteries fall; rover slows on rough/steep terrain |
| 8 Comparison | Scroll to **Performance comparison** (or **⏩ Run to end**) | Distance, time, rover energy, **scout energy**, **total system energy**, hazard exposure, verdict |
| 9 Moon | Pick **Moon** | Scout becomes an **orbital recon platform (NOT a drone)**; panel explains why |
| 10 Night vision | Sidebar → *Visual mode* → **Night vision** (or **Thermal**) | Whole dashboard + map re-themed; routes keep contrast outlines |

Extras: *Enable random link outages* (rover keeps driving on old knowledge, packets queue and flush when the link returns), *Map source* → **Rover knowledge / Scout observations / Shared knowledge** (the four separate maps), **Custom** planet sliders, the **Multi-seed study** expander (shows seeds where recon is *not* worth it).

---

## 3. How it works

```
app.py                      Streamlit mission-control dashboard
aeris/
  planetary_models.py       Earth / Mars / Moon / Custom presets + scout selection
  environment.py            Procedural DEM: hills, valleys, craters, rocks, boulder fields (ground truth)
  terrain_analysis.py       Slope, obstacle density, traversability, risk, terrain class
  knowledge.py              KnowledgeMap: what ONE agent knows (unknown = NaN)
  energy.py                 Rover & scout energy / speed models (documented, assumed)
  scout.py                  Rotorcraft + orbital platform, noisy sensor footprint, packet creation
  communication.py          Queue, bit-rate, delay, distance degradation, loss+retry, outages, shared map
  navigation.py             A* on a knowledge map; unknown cells penalised
  rover.py                  Driving physics, battery-aware decisions, return/abort logic
  mission_controller.py     Owns everything; step() advances 1 tick; builds the rover-only baseline
  metrics.py                Scoring, comparison table, "worth it" rule, multi-seed study
  visuals.py / ui.py        Plotly figures (Day / Night vision / Thermal) and HTML panels
assets/styles.css           Mission-control theme
tests/                      18 unit tests + a UI smoke test
```

**Four separate maps:** ground truth (`Terrain`) · rover knowledge (`Rover.kn`) · scout observations (`Scout.obs`) · shared knowledge (`CommLink.shared`, delivered packets only). The planner receives *only* the rover's knowledge map; a test proves its output is independent of the ground truth.

**Planetary parameters really act on the simulation**

| Parameter | Effect |
|---|---|
| Gravity | Rover rolling (m·g) and climbing (m·g·Δh) energy; scout hover power ∝ g^1.5 |
| Air density | Hover power ∝ 1/√ρ; if required power > motor limit, **no rotorcraft → orbital platform** |
| Roughness | Rolling resistance, drive speed, risk, terrain generation |
| Slope severity | Hill/crater-wall relief → slopes → blocked cells (> 25°), climb cost, speed |
| Obstacle density | Boulder fields → blocked cells and risk |
| Comm quality | Signal quality, bit-rate, loss probability |
| Energy multipliers | Scout / rover power draw |

**Battery matters:** the rover estimates energy-to-go on its knowledge map (unknown cells use planet-wide priors × 1.25). If it cannot reach the goal with a reserve it first tries an energy-saving route, then **waits for recon** (at base), **returns to base**, or aborts. Try the Custom planet with a small rover battery.

**"Worth it" rule:** recon is worth it if (a) AERIS total system energy ≤ rover-only, or (b) hazard exposure falls ≥ 25 % for ≤ 15 % more system energy, or (c) rover-only fails and AERIS arrives. Percentages are hidden when the two strategies end differently (e.g. one turned back).

---

## 4. Honesty: what is real and what is simplified

This is a **hackathon prototype**, not flight- or rover-certified. Full table in `LIMITATIONS.md` (also in the app). In short:

* Terrain is **synthetic**. Energy, speed, sensor and comm models are **simplified and assumed**, not validated.
* Rover drive speed is **scaled to 1 m/s** so a mission takes minutes, not days.
* The Moon platform's energy is an **allocation** (payload power + per-pass overhead), not orbital mechanics.
* The scout sensor reads the DEM with noise and can miss boulders — it is **not** image-based 3-D reconstruction.
* No real RF link budget, satellite relay or light-time delay. No wind, dust storms or thermal limits.
* **No machine learning** anywhere — deterministic formulas plus seeded noise (runs are reproducible per seed).
* Orbital passes are pre-planned lines (not adaptive); the rotorcraft adaptively surveys unexplored cells along the rover's current route.

## 5. Tuning knobs
`aeris/energy.py` (masses, efficiencies, powers) · `aeris/planetary_models.py` (presets, batteries) · `aeris/navigation.py` (`UNKNOWN_MULT`, weights) · `MissionConfig` in `mission_controller.py` (comm range, bit-rate, reserve, recon wait).
