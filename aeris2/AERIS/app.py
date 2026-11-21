"""
AERIS - Aerial Exploration and Rover Intelligence System  |  Mission-control dashboard.
Run with:   streamlit run app.py
"""
import os
import sys
import time
import random
from dataclasses import astuple

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aeris.planetary_models import PRESETS, custom_planet, describe_scout
from aeris.mission_controller import Mission, MissionConfig
from aeris import metrics, ui, visuals
from aeris.visuals import THEMES

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="AERIS Mission Control", page_icon="🛰️", layout="wide")


# ------------------------------------------------------------------ helpers
def show_fig(fig, key):
    """Works on old and new Streamlit versions (use_container_width was replaced by width='stretch')."""
    try:
        ver = tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        ver = (1, 40)
    if ver >= (1, 50):
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def regenerate_cb():
    st.session_state["seed"] = random.randint(1, 9999)
    st.session_state["running"] = False


THEME_VARS = {
    "Day": "",
    "Night vision": ":root{--bg:#010a03;--panel:#04140a;--panel2:#072010;--line:#0f5a24;--text:#b9ffc7;"
                    "--muted:#5fb872;--accent:#39ff6a;--accent2:#b6ff00;--warn:#ffe14d;--crit:#ff5a4d;--good:#39ff6a;}",
    "Thermal": ":root{--bg:#0a0508;--panel:#160a10;--panel2:#220e16;--line:#5a2a1a;--text:#ffd9b8;"
               "--muted:#c08a64;--accent:#ff7a18;--accent2:#ffd000;--warn:#ffd000;--crit:#ff3d71;--good:#b6ff00;}",
}

# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.markdown("## 🛰️ AERIS")
sb.caption("Aerial Exploration & Rover Intelligence System")

planet_name = sb.selectbox("Planet / environment", list(PRESETS.keys()), index=0, key="planet")
base = PRESETS[planet_name]
if planet_name == "Custom":
    sb.markdown("**Custom planet** - every slider feeds the simulation")
    planet = custom_planet(
        gravity=sb.slider("Gravity (m/s²)", 0.5, 15.0, 3.71, 0.01, key="c_g"),
        atmo_density=sb.slider("Atmospheric density (kg/m³)", 0.0, 1.5, 0.02, 0.001, key="c_rho"),
        terrain_roughness=sb.slider("Terrain roughness", 0.0, 1.0, 0.45, 0.01, key="c_r"),
        slope_severity=sb.slider("Slope severity", 0.3, 1.8, 1.0, 0.05, key="c_s"),
        obstacle_density=sb.slider("Obstacle density", 0.0, 0.25, 0.07, 0.01, key="c_o"),
        comm_quality=sb.slider("Communication quality", 0.2, 1.0, 0.85, 0.01, key="c_q"),
        scout_energy_mult=sb.slider("Scout energy consumption ×", 0.5, 3.0, 1.0, 0.05, key="c_se"),
        rover_energy_mult=sb.slider("Rover energy consumption ×", 0.5, 3.0, 1.0, 0.05, key="c_re"),
        rover_battery_wh=sb.slider("Rover battery (Wh)", 100.0, 1500.0, 520.0, 10.0, key="c_rb"),
        scout_battery_wh=sb.slider("Scout battery / allocation (Wh)", 2.0, 20.0, 8.0, 0.5, key="c_sb"),
    )
else:
    planet = base
    sb.markdown(ui.table_html(pd.DataFrame([
        ("Gravity", f"{planet.gravity} m/s²"), ("Air density", f"{planet.atmo_density} kg/m³"),
        ("Pressure", f"{planet.pressure_kpa} kPa"), ("Roughness", planet.terrain_roughness),
        ("Slope severity", planet.slope_severity), ("Obstacles", planet.obstacle_density),
        ("Comm quality", planet.comm_quality), ("Rover battery", f"{planet.rover_battery_wh:.0f} Wh"),
        ("Scout energy", f"{planet.scout_battery_wh} Wh")], columns=["Parameter", "Value"])),
        unsafe_allow_html=True)

sb.markdown("---")
sb.markdown("**Mission**")
if "seed" not in st.session_state:
    st.session_state["seed"] = 4
seed = int(sb.number_input("Random seed", 1, 99999, step=1, key="seed"))
goal_r = sb.slider("Destination row (% of map)", 40, 95, 90, key="goal_r")
goal_c = sb.slider("Destination column (% of map)", 40, 95, 88, key="goal_c")
radius = sb.slider("Scout survey radius (cells = 10 m)", 3.0, 12.0, 6.0, 0.5, key="radius")
speed = sb.select_slider("Simulation speed (ticks per refresh)", options=[1, 2, 5, 10, 20, 50], value=10, key="speed")

sb.markdown("**Communication**")
comm_fail = sb.toggle("Enable random link outages", value=False, key="comm_fail")
loss = sb.slider("Base packet loss", 0.0, 0.5, 0.05, 0.01, key="loss")
delay = sb.slider("Transmission delay (s)", 0.0, 10.0, 1.5, 0.5, key="delay")

sb.markdown("**Display**")
mode = sb.radio("Visual mode", ["Day", "Night vision", "Thermal"], horizontal=True, key="mode")
view = sb.selectbox("Map source", ["Ground truth", "Rover knowledge", "Scout observations", "Shared knowledge"], key="view")
layer = sb.selectbox("Map layer", ["Elevation", "Terrain type", "Slope", "Roughness", "Traversability", "Risk"], key="layer")
show_unknown = sb.checkbox("Fog of unknown terrain (truth view)", True, key="s_unk")
show_haz = sb.checkbox("Show true hazards (truth view)", True, key="s_haz")
show_cov = sb.checkbox("Scout survey coverage", True, key="s_cov")
show_foot = sb.checkbox("Scout sensor footprint", True, key="s_foot")

# ------------------------------------------------------------------ css
with open(os.path.join(HERE, "assets", "styles.css"), encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}{THEME_VARS[mode]}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------ mission state
goal = (int(64 * goal_r / 100) - 1, int(64 * goal_c / 100) - 1)
cfg_key = (astuple(planet), seed, goal)
if st.session_state.get("cfg_key") != cfg_key:
    with st.spinner("Generating terrain and computing the rover-only baseline..."):
        st.session_state["mission"] = Mission(MissionConfig(planet=planet, seed=seed, goal=goal))
    st.session_state["cfg_key"] = cfg_key
    st.session_state["running"] = False
m: Mission = st.session_state["mission"]
m.cfg.survey_radius, m.cfg.comm_failures, m.cfg.packet_loss, m.cfg.delay_s = radius, comm_fail, loss, delay
m.update_live()
st.session_state.setdefault("running", False)

# ------------------------------------------------------------------ banner + controls
sinfo = m.scout_info
st.markdown(
    f'<div class="banner"><div><div class="title">AERIS</div>'
    f'<div class="sub">AERIAL EXPLORATION &amp; ROVER INTELLIGENCE SYSTEM · MISSION CONTROL · SIMULATION</div></div>'
    f'<div>{ui.chip(m.planet.name.upper(), "info")}{ui.chip(sinfo["name"], "warn" if sinfo["kind"] == "orbital" else "good")}'
    f'{ui.chip("SEED " + str(m.cfg.seed))}{ui.chip(mode.upper())}</div></div>', unsafe_allow_html=True)

b1, b2, b3, b4, b5 = st.columns(5)
if b1.button("▶ Start / Resume"):
    if m.done:
        m.reset()
    st.session_state["running"] = True
if b2.button("⏸ Pause"):
    st.session_state["running"] = False
if b3.button("↺ Reset mission"):
    m.reset()
    st.session_state["running"] = False
b4.button("🎲 Regenerate terrain", on_click=regenerate_cb)
if b5.button("⏩ Run to end"):
    with st.spinner("Simulating..."):
        m.run()
    st.session_state["running"] = False

if st.session_state["running"] and not m.done:
    for _ in range(int(speed)):
        m.step()
        if m.done:
            break
if m.done:
    st.session_state["running"] = False
running = st.session_state["running"]

rv, sc, cm = m.rover, m.scout, m.comm
phase = ("COMPLETE" if m.done else "IN PROGRESS" if running else "PAUSED" if m.t > 0 else "READY")

# ------------------------------------------------------------------ mission status
cell = m.terrain.cell_m
rb_tone = "good" if rv.battery_pct > 35 else ("warn" if rv.battery_pct > 15 else "crit")
sp = 100.0 * sc.battery_wh / sc.capacity
cards = [
    ui.kpi("Planet", m.planet.name, f"g={m.planet.gravity:.2f} m/s²", tone="info"),
    ui.kpi("Mission status", phase, rv.status_text(), tone="good" if m.done and rv.status == "ARRIVED" else "info"),
    ui.kpi("Sim time", f"{m.t:,.0f} s", f"{m.t / 60:.1f} min"),
    ui.kpi("Rover battery", f"{rv.battery_wh:,.0f} Wh", f"used {rv.energy_used:,.1f} Wh", pct=rv.battery_pct, tone=rb_tone),
    ui.kpi("Scout battery", f"{sc.battery_wh:.2f} Wh", sc.status + (" (allocation)" if sc.kind == "orbital" else ""), pct=sp),
    ui.kpi("Rover position", f"({rv.pos[0] * cell:.0f}, {rv.pos[1] * cell:.0f}) m", f"cell {rv.cell}"),
    ui.kpi("Scout position", f"({sc.pos[0] * cell:.0f}, {sc.pos[1] * cell:.0f}) m", sinfo["kind"]),
    ui.kpi("Distance travelled", f"{rv.distance_m:,.0f} m", f"speed {rv.speed_ms:.2f} m/s (sim-scaled)"),
    ui.kpi("Completion", f"{rv.progress() * 100:.0f}%", "route progress", pct=rv.progress() * 100),
    ui.kpi("Terrain under rover", rv.terrain_label, f"known cells {rv.kn.n_known}/{m.terrain.n ** 2}"),
]
st.markdown(ui.kpi_grid(cards, 5), unsafe_allow_html=True)

# ------------------------------------------------------------------ map + side panels
col_map, col_side = st.columns([3, 1.25])
with col_map:
    st.markdown("### Terrain map")
    show_fig(visuals.terrain_figure(m, mode, view, layer, show_unknown, show_cov, show_haz, show_foot), "map")
    st.caption("Dashed = rover-only route · solid = AERIS route (dotted = remaining plan) · fog = unknown to the rover · "
               "red = true hazards · cyan = scout coverage. Switch 'Map source' to see what each agent actually knows.")

with col_side:
    st.markdown("### Communication")
    link = "UP" if cm.link_up else ("OUTAGE" if m.t < cm.outage_until else "OUT OF RANGE")
    st.markdown(
        f'<div class="panel">'
        f'{ui.chip("LINK " + link, "good" if link == "UP" else "crit")}{ui.chip(f"Q {cm.quality_now:.2f}")}'
        f'{ui.chip(f"QUEUE {len(cm.queue)}")}<br>'
        f'<span class="muted">sent {cm.sent} · received {cm.received} · lost {cm.lost} · dropped {cm.dropped} · '
        f'cells delivered {cm.obs_tx}</span>'
        f'{ui.log_html(m.events, 70)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel"><h4>Packets</h4>{ui.packet_table(cm.recent)}</div>', unsafe_allow_html=True)

    ret = rv.return_cost_wh()
    low = rv.battery_pct < 20 or rv.low_battery
    warn = '<div class="warnbox">⚠ LOW BATTERY - reserve at risk</div>' if low else ""
    st.markdown(
        f'<div class="panel"><h4>Rover energy</h4>{warn}'
        f'<span class="muted">drive {rv.drive_wh:.1f} Wh · electronics {rv.elec_wh:.1f} Wh<br>'
        f'est. range left {rv.est_range_m():.0f} m<br>return-to-base cost {ret:.1f} Wh · hard reserve {rv.reserve_wh:.0f} Wh<br>'
        f'projected battery at goal: {"-" if rv.projected_wh is None else f"{rv.projected_wh:.0f} Wh"}<br>'
        f'scout flown {sc.distance_flown_m:.0f} m · coverage {100 * sc.surveyed.mean():.0f}% of map</span></div>',
        unsafe_allow_html=True)

st.markdown(f'<div class="panel"><h4>Scout selection - {ui.e(sinfo["name"])}</h4>'
            f'<span class="muted">{ui.e(sinfo["why"])}</span></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ energy + performance
ro = m.baseline
ae = m.aeris_metrics(project=True)
cmp = metrics.comparison(ro, ae, sc.energy_used)
proj = "PROJECTED (mission still running)" if ae["projected"] else "FINAL"

st.markdown("### Energy panel")
e1, e2 = st.columns([1, 1])
with e1:
    tiles = [ui.kpi("Rover-only energy", f"{ro['energy_wh']:.1f} Wh", ro["status"]),
             ui.kpi("AERIS rover energy", f"{ae['energy_wh']:.1f} Wh", ae["status"] + (" · projected" if ae["projected"] else "")),
             ui.kpi("Scout energy", f"{cmp['scout_wh']:.2f} Wh", "included in total", tone="info"),
             ui.kpi("Total AERIS energy", f"{cmp['aeris_total_wh']:.1f} Wh",
                    f"{cmp['system_energy_diff_wh']:+.1f} Wh vs rover-only",
                    tone="good" if cmp["system_energy_diff_wh"] <= 0 else "warn")]
    st.markdown(ui.kpi_grid(tiles, 2), unsafe_allow_html=True)
    show_fig(visuals.battery_figure(m.history, mode), "batt")
with e2:
    show_fig(visuals.energy_figure(ro, ae, cmp, mode), "energy")

st.markdown("### Performance comparison")
p1, p2 = st.columns([1.1, 1])
with p1:
    st.markdown(f'<span class="muted">Status: {proj}. Same terrain, start, destination and rover for both strategies. '
                f'Scores use ground truth; planning never does.</span>', unsafe_allow_html=True)
    st.markdown(ui.table_html(metrics.comparison_table(ro, ae, cmp), "AERIS"), unsafe_allow_html=True)
    ok = cmp["comparable"]
    imp = [ui.kpi("Route improvement (distance)", ui.fmt_pct(cmp["distance_improvement_pct"], ok), "+ = AERIS shorter"),
           ui.kpi("Safety improvement", ui.fmt_pct(cmp["safety_improvement_pct"], ok), "hazard exposure cut"),
           ui.kpi("Rover energy", ui.fmt_pct(cmp["rover_energy_improvement_pct"], ok), "+ = AERIS lower"),
           ui.kpi("System energy (incl. scout)", ui.fmt_pct(cmp["system_energy_improvement_pct"], ok), "+ = AERIS lower")]
    st.markdown(ui.kpi_grid(imp, 4), unsafe_allow_html=True)
    st.markdown(ui.verdict(cmp), unsafe_allow_html=True)
with p2:
    show_fig(visuals.comparison_figure(ro, ae, cmp, mode), "cmp")

# ------------------------------------------------------------------ terrain intelligence
st.markdown("### Terrain intelligence")
src = st.radio("Data source", ["Rover knowledge", "Ground truth"], horizontal=True, key="intel_src")
T = m.terrain
if src == "Ground truth":
    sl, ro_, tv = T.slope_deg, T.roughness, T.traversability
else:
    sl, ro_, tv, _ = rv.kn.derived()
which = st.radio("Layer", ["Slope", "Roughness", "Traversability", "Unknown terrain", "Survey coverage"],
                 horizontal=True, key="intel_layer")
pk = m.planet.colorscale
if which == "Slope":
    fig_i = visuals.small_heat(sl, "Slope (deg)", mode, 0, 30, planet_key=pk)
elif which == "Roughness":
    fig_i = visuals.small_heat(ro_, "Roughness (0-1)", mode, 0, 1, planet_key=pk)
elif which == "Traversability":
    fig_i = visuals.small_heat(tv, "Traversability (0 = blocked, 1 = easy)", mode, 0, 1, planet_key=pk)
elif which == "Unknown terrain":
    fig_i = visuals.small_heat(~rv.kn.known, f"Unknown to rover ({100 * (1 - rv.kn.known.mean()):.0f}% of map)",
                               mode, 0, 1, binary=True)
else:
    fig_i = visuals.small_heat(sc.surveyed, f"Scout survey coverage ({100 * sc.surveyed.mean():.0f}% of map)",
                               mode, 0, 1, binary=True)
show_fig(fig_i, "intel")

# ------------------------------------------------------------------ study + honesty
with st.expander("🔬 Multi-seed study - does reconnaissance always pay off?"):
    n_seeds = st.slider("Number of seeds", 4, 20, 8, key="n_seeds")
    if st.button("Run study on the selected planet"):
        with st.spinner(f"Running {n_seeds} headless missions..."):
            df = metrics.run_study(m.planet, range(1, n_seeds + 1), m.cfg)
        st.session_state["study"] = df
    if st.session_state.get("study") is not None:
        df = st.session_state["study"]
        st.markdown(ui.table_html(df), unsafe_allow_html=True)
        st.markdown(f"**Recon judged worth its energy in {int(df['worth it'].sum())} of {len(df)} missions.**")

with st.expander("📋 Models, assumptions and honest limitations"):
    st.markdown(open(os.path.join(HERE, "LIMITATIONS.md"), encoding="utf-8").read())

# ------------------------------------------------------------------ auto-refresh loop
if running and not m.done:
    time.sleep(0.08)
    st.rerun()
