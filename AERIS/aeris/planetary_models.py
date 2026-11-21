"""
planetary_models.py - Planet presets and the scout-selection logic.

Every field below is used by the simulation (terrain generation, energy, comms, scout).
Values are rounded, assumed, hackathon-grade numbers - not mission-grade data.
"""
from dataclasses import dataclass, replace
from . import energy


@dataclass
class Planet:
    name: str
    gravity: float            # m/s^2           -> rover rolling/climbing energy, scout hover power
    atmo_density: float       # kg/m^3 (surface)-> decides whether a rotorcraft can fly at all
    pressure_kpa: float       # informational
    terrain_roughness: float  # 0..1            -> rover energy, rover speed, hazard risk
    slope_severity: float     # relief multiplier -> steepness of hills/crater walls
    obstacle_density: float   # 0..0.3          -> fraction of boulder cells
    comm_quality: float       # 0..1            -> signal quality, loss and bit-rate
    scout_energy_mult: float  # multiplier on scout power
    rover_energy_mult: float  # multiplier on rover power
    n_craters: int
    hill_amp_m: float
    rover_battery_wh: float
    scout_battery_wh: float   # for the Moon this is an ENERGY ALLOCATION, not a physical battery
    colorscale: str           # key used by visuals.py
    blurb: str


EARTH = Planet("Earth", 9.81, 1.225, 101.3, 0.25, 0.9, 0.04, 0.95, 1.0, 1.0, 0, 18.0,
               1000.0, 8.0, "earth",
               "Dense atmosphere, standard gravity. A conventional quad-rotor drone flies easily.")
MARS = Planet("Mars", 3.71, 0.020, 0.6, 0.45, 1.0, 0.07, 0.85, 1.0, 1.0, 8, 18.0,
              520.0, 8.0, "mars",
              "Thin atmosphere (~1 % of Earth). Only a specialised lightweight rotorcraft "
              "(Ingenuity-class) can fly.")
MOON = Planet("Moon", 1.62, 0.0, 0.0, 0.40, 1.1, 0.06, 0.70, 1.0, 1.0, 14, 16.0,
              380.0, 3.5, "moon",
              "No atmosphere - NO helicopter or drone flight. AERIS uses a short-range orbital "
              "reconnaissance platform instead.")
CUSTOM = Planet("Custom", 3.71, 0.020, 0.6, 0.45, 1.0, 0.07, 0.85, 1.0, 1.0, 8, 18.0,
                520.0, 8.0, "mars", "User-defined planet. Every slider feeds the simulation.")

PRESETS = {"Mars": MARS, "Earth": EARTH, "Moon": MOON, "Custom": CUSTOM}


def custom_planet(**kw) -> Planet:
    return replace(CUSTOM, **kw)


def describe_scout(planet: Planet) -> dict:
    """Human-readable explanation of which scout is used and why."""
    kind = energy.scout_kind(planet)
    if kind == "aerial":
        p = energy.hover_power_w(planet)
        name = "Ingenuity-class rotorcraft" if planet.atmo_density < 0.2 else "Quad-rotor drone"
        why = (f"Air density {planet.atmo_density:.3f} kg/m3 is enough for rotor lift. "
               f"Hover power needed ~{p:.0f} W (limit {energy.P_MAX_W:.0f} W).")
    else:
        name = "Orbital recon platform (NOT a drone)"
        if planet.atmo_density <= 1e-6:
            why = "No atmosphere: rotors have no air to push against, so conventional flight is impossible."
        else:
            why = (f"Air density {planet.atmo_density:.4f} kg/m3 is too thin: the rotorcraft would need "
                   f"~{energy.hover_power_w(planet):.0f} W, above its {energy.P_MAX_W:.0f} W limit.")
    return {"kind": kind, "name": name, "why": why}
