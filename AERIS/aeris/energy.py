"""
energy.py - Simplified energy / speed models for the rover and the scout.

ALL of these are SIMPLIFIED, ASSUMED models for a hackathon simulation.
They are NOT validated against real rover or helicopter data.

Units used everywhere:  metres, seconds, kilograms, Watt-hours (Wh), degrees.
"""
import math

# ----------------------------- ROVER (assumed parameters) -----------------------------
ROVER_MASS_KG = 900.0        # assumed rover mass
CRR_BASE = 0.05              # rolling-resistance coefficient on smooth firm ground
CRR_ROUGH = 0.25             # extra rolling resistance per unit roughness (0..1)
DRIVE_EFFICIENCY = 0.55      # battery -> wheel mechanical efficiency
ELEC_ACTIVE_W = 60.0         # electronics + heaters while driving (W)
ELEC_IDLE_W = 25.0           # electronics while stationary (W)
V_NOM_MS = 1.0               # nominal drive speed. SIM-SCALED: real rovers are ~0.04 m/s
MAX_SLOPE_DEG = 25.0         # slopes steeper than this are impassable for the rover


def rover_speed(slope_deg: float, rough: float) -> float:
    """Drive speed (m/s). Slower on steep or rough terrain, never below 15 % nominal."""
    f = 1.0 - 0.55 * rough - 0.8 * min(slope_deg / MAX_SLOPE_DEG, 1.0)
    return V_NOM_MS * max(0.15, min(1.0, f))


def rover_segment(planet, dist_m: float, dz_m: float, rough: float, slope_deg: float):
    """
    Energy/time for driving one segment.

    Model:
      rolling  = Crr * m * g * cos(theta) * d            (Crr grows with roughness)
      climbing = m * g * max(dz, 0)                      (no energy recovered going downhill)
      drive_Wh = (rolling + climbing) / efficiency * slip_factor * planet multiplier / 3600
      elec_Wh  = P_electronics * travel_time / 3600
    Gravity enters through m*g, so heavier-gravity planets cost more to roll and climb.

    Returns (drive_wh, elec_wh, time_s, speed_ms).
    """
    g = planet.gravity
    theta = math.atan2(dz_m, dist_m)
    crr = CRR_BASE + CRR_ROUGH * rough
    rolling = crr * ROVER_MASS_KG * g * math.cos(theta) * dist_m
    climb = ROVER_MASS_KG * g * max(dz_m, 0.0)
    slip = 1.0 + 1.5 * min(slope_deg / MAX_SLOPE_DEG, 1.0) ** 2
    mult = planet.rover_energy_mult
    drive_wh = (rolling + climb) / DRIVE_EFFICIENCY * slip * mult / 3600.0
    v = rover_speed(slope_deg, rough)
    t = dist_m / v
    elec_wh = ELEC_ACTIVE_W * mult * t / 3600.0
    return drive_wh, elec_wh, t, v


def idle_wh(planet, dt: float) -> float:
    return ELEC_IDLE_W * planet.rover_energy_mult * dt / 3600.0


# ----------------------------- SCOUT (assumed parameters) -----------------------------
SCOUT_MASS_KG = 1.8          # Ingenuity-class mass
ROTOR_AREA_M2 = 2.26         # total rotor disk area (two 0.6 m radius rotors)
FIGURE_OF_MERIT = 0.6        # rotor efficiency
MOTOR_ETA = 0.55             # electrical -> shaft efficiency
AVIONICS_W = 15.0            # avionics + camera (W)
P_MAX_W = 700.0              # maximum electrical power the rotorcraft can sustain
FWD_FACTOR = 1.15            # forward-flight power relative to hover
SCOUT_SPEED_MS = 6.0         # cruise speed (m/s)
TAKEOFF_S = 5.0
LAND_S = 5.0

ORBITAL_PAYLOAD_W = 35.0     # imaging + downlink payload power (W)
ORBITAL_PASS_OVERHEAD_WH = 0.6   # extra energy to line up each recon pass (Wh)
ORBITAL_SPEED_MS = 35.0      # ground-track speed of the short-range recon platform
ORBITAL_ALTITUDE_M = 300.0   # altitude used only for the comm-distance calculation


def hover_power_w(planet) -> float:
    """
    Momentum-theory hover power:  P = (m g)^1.5 / ( sqrt(2 rho A) * FM * eta ) + avionics.
    Falls as air density rises, rises with gravity. Returns inf when there is no air.
    """
    rho = planet.atmo_density
    if rho <= 1e-6:
        return float("inf")
    thrust = SCOUT_MASS_KG * planet.gravity
    p = thrust ** 1.5 / (math.sqrt(2.0 * rho * ROTOR_AREA_M2) * FIGURE_OF_MERIT * MOTOR_ETA)
    return (p + AVIONICS_W) * planet.scout_energy_mult


def aerial_possible(planet) -> bool:
    """Rotorcraft can fly only if the hover power needed is within what its motors can deliver."""
    return hover_power_w(planet) <= P_MAX_W * max(1.0, planet.scout_energy_mult)


def scout_kind(planet) -> str:
    return "aerial" if aerial_possible(planet) else "orbital"
