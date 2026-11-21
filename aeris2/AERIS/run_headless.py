"""
Command-line runner - no browser needed. Handy for checking the engine before the demo.

    python run_headless.py                  # one Mars mission, prints the comparison
    python run_headless.py --planet Moon --seed 4
    python run_headless.py --study 10       # multi-seed study on the chosen planet
"""
import argparse
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
from aeris.planetary_models import PRESETS
from aeris.mission_controller import Mission, MissionConfig
from aeris import metrics

ap = argparse.ArgumentParser()
ap.add_argument("--planet", default="Mars", choices=list(PRESETS))
ap.add_argument("--seed", type=int, default=4)
ap.add_argument("--study", type=int, default=0, help="run N seeds and print a table")
ap.add_argument("--log", action="store_true", help="print the mission log")
a = ap.parse_args()
planet = PRESETS[a.planet]

if a.study:
    df = metrics.run_study(planet, range(1, a.study + 1))
    print(df.to_string(index=False))
    print(f"\nRecon worth its energy in {int(df['worth it'].sum())} of {len(df)} missions on {a.planet}.")
else:
    m = Mission(MissionConfig(planet=planet, seed=a.seed))
    print(f"{a.planet} seed {a.seed}: scout = {m.scout_info['name']}\n  {m.scout_info['why']}")
    m.run()
    ro, ae, c = m.baseline, m.aeris_metrics(), m.comparison()
    print(metrics.comparison_table(ro, ae, c).to_string(index=False))
    print(f"\nSystem energy difference: {c['system_energy_diff_wh']:+.1f} Wh (+ = AERIS uses more)")
    print("Worth it:", c["worth_it"], "-", c["verdict"])
    if a.log:
        for t, tag, msg in m.events:
            print(f"{t:6.0f}s [{tag}] {msg}")
