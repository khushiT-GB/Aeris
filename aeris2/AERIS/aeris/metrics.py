"""
metrics.py - Scoring: rover-only vs AERIS, using GROUND TRUTH (scoring only, never planning).
"""
import numpy as np
import pandas as pd
from . import energy
from .navigation import SQ2


def evaluate_path(terrain, planet, path):
    """Truth-based cost of driving a path (used to project the unfinished part of a route)."""
    d_wh = e_wh = tsum = dist = risk = 0.0
    hi = 0
    for a, b in zip(path[:-1], path[1:]):
        diag = a[0] != b[0] and a[1] != b[1]
        dd = terrain.cell_m * (SQ2 if diag else 1.0)
        dz = float(terrain.elevation[b] - terrain.elevation[a])
        rough = float(0.5 * (terrain.roughness[a] + terrain.roughness[b]))
        slope = float(max(terrain.slope_deg[a], terrain.slope_deg[b]))
        d, e, t, _ = energy.rover_segment(planet, dd, dz, rough, slope)
        rk = float(0.5 * (terrain.risk[a] + terrain.risk[b]))
        d_wh += d
        e_wh += e
        tsum += t
        dist += dd
        risk += rk * dd
        hi += rk >= 0.5
    return {"distance_m": dist, "energy_wh": d_wh + e_wh, "time_s": tsum, "risk_m": risk, "high_risk": hi}


def rover_summary(rover, project=False):
    """Result dict from a Rover. If project=True and unfinished, add the planned remainder."""
    r = {"distance_m": rover.distance_m, "energy_wh": rover.energy_used, "time_s": rover.time_s,
         "risk_m": rover.risk_m, "high_risk": rover.high_risk_cells, "status": rover.status,
         "battery_pct": rover.battery_pct, "projected": False}
    if project and not rover.done and rover.path:
        rest = evaluate_path(rover.t, rover.planet, [rover.cell] + rover.path[rover.path_idx:])
        for k in ("distance_m", "energy_wh", "time_s", "risk_m", "high_risk"):
            r[k] += rest[k]
        r["projected"] = True
    r["feasible"] = (rover.status == "ARRIVED") or (project and not rover.done and
                                                    rover.battery_wh - (r["energy_wh"] - rover.energy_used) > 0)
    return r


def worth_it(ro, ae, scout_wh):
    """
    Documented decision rule. Reconnaissance is 'worth it' when ANY of:
      (a) AERIS total system energy <= rover-only energy
      (b) hazard exposure cut by >= 25 % while total energy rises by <= 15 %
      (c) rover-only FAILED to reach the destination but AERIS succeeded
    """
    total = ae["energy_wh"] + scout_wh
    if (not ro["feasible"]) and ae["feasible"]:
        return True, "Rover-only FAILED to reach the destination; AERIS succeeded."
    if total <= ro["energy_wh"]:
        return True, "Total system energy (rover + scout) is LOWER than rover-only."
    safety = (ro["risk_m"] - ae["risk_m"]) / ro["risk_m"] * 100 if ro["risk_m"] > 0 else 0.0
    pen = (total - ro["energy_wh"]) / max(ro["energy_wh"], 1e-9) * 100
    if safety >= 25 and pen <= 15:
        return True, f"Hazard exposure cut {safety:.0f}% for only +{pen:.0f}% system energy."
    if ro["feasible"] and not ae["feasible"]:
        return False, "AERIS failed to reach the destination while rover-only succeeded."
    return False, f"Scout energy not justified: system energy {pen:+.0f}%, hazard change {safety:+.0f}%."


def comparison(ro, ae, scout_wh):
    total = ae["energy_wh"] + scout_wh
    pct = lambda a, b: (a - b) / a * 100.0 if a else 0.0       # positive = AERIS better (lower)
    worth, why = worth_it(ro, ae, scout_wh)
    comparable = bool(ro["feasible"] and ae["feasible"])   # % changes only meaningful if both arrived
    return {
        "comparable": comparable,
        "scout_wh": scout_wh, "aeris_total_wh": total,
        "system_energy_diff_wh": total - ro["energy_wh"],          # + means AERIS uses MORE
        "distance_improvement_pct": pct(ro["distance_m"], ae["distance_m"]),
        "rover_energy_improvement_pct": pct(ro["energy_wh"], ae["energy_wh"]),
        "system_energy_improvement_pct": pct(ro["energy_wh"], total),
        "time_improvement_pct": pct(ro["time_s"], ae["time_s"]),
        "safety_improvement_pct": pct(ro["risk_m"], ae["risk_m"]),
        "worth_it": worth, "verdict": why,
    }


def comparison_table(ro, ae, cmp):
    def feas(x):
        return "Feasible" if x["feasible"] else f"FAILED ({x['status']})"
    rows = [
        ("Route distance (m)", f"{ro['distance_m']:.0f}", f"{ae['distance_m']:.0f}"),
        ("Rover energy (Wh)", f"{ro['energy_wh']:.1f}", f"{ae['energy_wh']:.1f}"),
        ("Travel time (s)", f"{ro['time_s']:.0f}", f"{ae['time_s']:.0f}"),
        ("Hazard exposure (risk-m)", f"{ro['risk_m']:.0f}", f"{ae['risk_m']:.0f}"),
        ("High-risk cells crossed", f"{ro['high_risk']}", f"{ae['high_risk']}"),
        ("Route feasibility", feas(ro), feas(ae)),
        ("Scout energy (Wh)", "-", f"{cmp['scout_wh']:.2f}"),
        ("TOTAL system energy (Wh)", f"{ro['energy_wh']:.1f}", f"{cmp['aeris_total_wh']:.1f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Rover Only", "AERIS"])


def run_study(planet, seeds, base_cfg=None):
    """Headless multi-seed experiment. Returns a DataFrame (one row per seed)."""
    from .mission_controller import Mission, MissionConfig
    from dataclasses import replace
    rows = []
    for s in seeds:
        cfg = replace(base_cfg, planet=planet, seed=int(s)) if base_cfg else MissionConfig(planet=planet, seed=int(s))
        m = Mission(cfg)
        m.run()
        ro, ae, cmp = m.baseline, m.aeris_metrics(), m.comparison()
        rows.append({"seed": s, "RO status": ro["status"], "AERIS status": ae["status"],
                     "RO energy": round(ro["energy_wh"], 1), "AERIS rover": round(ae["energy_wh"], 1),
                     "scout": round(cmp["scout_wh"], 2), "AERIS total": round(cmp["aeris_total_wh"], 1),
                     "RO risk": round(ro["risk_m"]), "AERIS risk": round(ae["risk_m"]),
                     "dist RO": round(ro["distance_m"]), "dist AERIS": round(ae["distance_m"]),
                     "safety %": round(cmp["safety_improvement_pct"], 1),
                     "sys energy %": round(cmp["system_energy_improvement_pct"], 1),
                     "worth it": cmp["worth_it"]})
    return pd.DataFrame(rows)
