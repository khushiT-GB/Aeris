"""visuals.py - Plotly figure builders (map, energy, comparison, heatmaps). Day / Night vision / Thermal."""
import numpy as np
import plotly.graph_objects as go
from .terrain_analysis import CLASS_NAMES

PALETTES = {
    "mars": [[0, "#2b1a14"], [0.35, "#7a3b24"], [0.65, "#b9673a"], [1, "#e8b98a"]],
    "earth": [[0, "#10331f"], [0.4, "#3f7a3a"], [0.7, "#b8a86a"], [1, "#f2efe6"]],
    "moon": [[0, "#17181c"], [0.5, "#6f7178"], [1, "#d7d8dc"]],
}
NIGHT = [[0, "#000d03"], [0.5, "#0a5a1c"], [1, "#7dff8f"]]

THEMES = {
    "Day": dict(ro="#ff9f1c", ae="#22d3ee", rover="#ffffff", scout="#c4b5fd", start="#34d399", goal="#fb7185",
                haz="rgba(255,40,70,0.50)", fog="rgba(6,9,16,0.62)", cov="rgba(34,211,238,0.22)",
                text="#d7e0ee", bg="#0b0f17", grid="rgba(255,255,255,0.05)"),
    "Night vision": dict(ro="#fff200", ae="#ffffff", rover="#eaffea", scout="#00ffd5", start="#39ff6a",
                         goal="#ff5a4d", haz="rgba(255,70,40,0.60)", fog="rgba(0,6,1,0.68)",
                         cov="rgba(0,255,230,0.20)", text="#b9ffc7", bg="#010a03", grid="rgba(57,255,106,0.08)"),
    "Thermal": dict(ro="#ffffff", ae="#b6ff00", rover="#ffffff", scout="#00e5ff", start="#b6ff00",
                    goal="#00e5ff", haz="rgba(0,229,255,0.50)", fog="rgba(8,2,6,0.65)",
                    cov="rgba(182,255,0,0.20)", text="#ffd9b8", bg="#0a0508", grid="rgba(255,150,60,0.07)"),
}


def _scale(planet_key, mode):
    if mode == "Night vision":
        return NIGHT
    if mode == "Thermal":
        return "Inferno"
    return PALETTES.get(planet_key, PALETTES["mars"])


def _overlay(mask, color):
    return go.Heatmap(z=mask.astype(float), zmin=0, zmax=1, showscale=False, hoverinfo="skip",
                      colorscale=[[0, "rgba(0,0,0,0)"], [1, color]])


def layer_array(source, layer, terrain):
    """Return (array, label, zmin, zmax). source: truth Terrain or KnowledgeMap."""
    if hasattr(source, "elevation"):
        d = {"Elevation": source.elevation, "Slope": source.slope_deg, "Roughness": source.roughness,
             "Traversability": source.traversability, "Risk": source.risk, "Terrain type": source.terrain_class.astype(float)}
        arr = d.get(layer, source.elevation)
    else:
        slope, rough, trav, rk = source.derived()
        d = {"Elevation": source.elev, "Slope": slope, "Roughness": rough, "Traversability": trav, "Risk": rk}
        layer = layer if layer in d else "Elevation"
        arr = d[layer]
    zmin, zmax = {"Slope": (0, 30), "Roughness": (0, 1), "Traversability": (0, 1), "Risk": (0, 1),
                  "Terrain type": (0, 3)}.get(layer, (None, None))
    return arr, layer, zmin, zmax


def terrain_figure(m, mode="Day", source_name="Ground truth", layer="Elevation", show_unknown=True,
                   show_cov=True, show_haz=True, show_foot=True, height=640):
    th = THEMES[mode]
    tr, rv, sc = m.terrain, m.rover, m.scout
    src = {"Ground truth": tr, "Rover knowledge": rv.kn, "Scout observations": sc.obs,
           "Shared knowledge": m.comm.shared}[source_name]
    arr, layer, zmin, zmax = layer_array(src, layer, tr)
    n = tr.n
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=arr, zmin=zmin, zmax=zmax, colorscale=_scale(m.planet.colorscale, mode),
                             hovertemplate="col %{x}, row %{y}<br>" + layer + ": %{z:.2f}<extra></extra>",
                             colorbar=dict(thickness=10, len=0.55, title=dict(text=layer, font=dict(color=th["text"], size=11)),
                                           tickfont=dict(color=th["text"], size=10)), name=layer))
    if source_name == "Ground truth":
        if show_unknown:
            fig.add_trace(_overlay(~rv.kn.known, th["fog"]))
        if show_haz:
            fig.add_trace(_overlay(tr.blocked, th["haz"]))
    if show_cov:
        fig.add_trace(_overlay(sc.surveyed, th["cov"]))

    def route(path, color, name, width, dash=None):
        if path is None or len(path) < 2:
            return
        xs, ys = [c for _, c in path], [r for r, _ in path]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(0,0,0,0.85)", width=width + 3),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash),
                                 name=name, hoverinfo="name"))

    route(m.baseline["trail"], th["ro"], "Rover-only route", 3, "dash")
    route(rv.trail, th["ae"], "AERIS route (driven)", 4)
    if rv.path and rv.path_idx < len(rv.path):
        route([rv.cell] + rv.path[rv.path_idx:], th["ae"], "AERIS planned (remaining)", 2, "dot")
    if len(sc.trail) > 1:
        tt = sc.trail[::3]
        fig.add_trace(go.Scatter(x=[p[1] for p in tt], y=[p[0] for p in tt], mode="lines",
                                 line=dict(color=th["scout"], width=1, dash="dot"), name="Scout flight path",
                                 hoverinfo="skip"))
    if sc.kind == "orbital":
        for a, b in sc.passes:
            fig.add_trace(go.Scatter(x=[a[1], b[1]], y=[a[0], b[0]], mode="lines", showlegend=False,
                                     line=dict(color=th["scout"], width=1, dash="longdash"), hoverinfo="skip", opacity=0.35))
    sr, scn = tr.start, tr.goal
    fig.add_trace(go.Scatter(x=[sr[1]], y=[sr[0]], mode="markers+text", text=["START"], textposition="bottom right",
                             textfont=dict(color=th["text"], size=11), name="Start",
                             marker=dict(symbol="star", size=16, color=th["start"], line=dict(color="#000", width=1.5))))
    fig.add_trace(go.Scatter(x=[scn[1]], y=[scn[0]], mode="markers+text", text=["DESTINATION"], textposition="top left",
                             textfont=dict(color=th["text"], size=11), name="Destination",
                             marker=dict(symbol="x", size=16, color=th["goal"], line=dict(color=th["goal"], width=3))))
    fig.add_trace(go.Scatter(x=[rv.pos[1]], y=[rv.pos[0]], mode="markers+text", text=["ROVER"], textposition="top center",
                             textfont=dict(color=th["text"], size=11), name="Rover",
                             marker=dict(symbol="square", size=14, color=th["rover"], line=dict(color="#000", width=2))))
    sname = "Scout (rotorcraft)" if sc.kind == "aerial" else "Orbital recon platform"
    fig.add_trace(go.Scatter(x=[sc.pos[1]], y=[sc.pos[0]], mode="markers+text",
                             text=["SCOUT" if sc.active else sc.status], textposition="top center",
                             textfont=dict(color=th["scout"], size=11), name=sname,
                             marker=dict(symbol="diamond" if sc.kind == "aerial" else "hexagram", size=15,
                                         color=th["scout"], line=dict(color="#000", width=1.5))))
    if show_foot and sc.status in ("TAKEOFF", "SURVEYING", "RETURNING"):
        r0, c0, rad = sc.pos[0], sc.pos[1], sc.radius
        fig.add_shape(type="circle", x0=c0 - rad, x1=c0 + rad, y0=r0 - rad, y1=r0 + rad,
                      line=dict(color=th["scout"], width=2, dash="dash"), fillcolor=th["cov"])
    ticks = list(range(0, n, 8))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=th["bg"],
        font=dict(color=th["text"]), uirevision="aeris-map",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=11, color=th["text"]),
                    bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(range=[-0.5, n - 0.5], tickvals=ticks, ticktext=[f"{int(t * tr.cell_m)}" for t in ticks],
                     showgrid=False, zeroline=False, color=th["text"], title=dict(text="metres", font=dict(size=10)))
    fig.update_yaxes(range=[n - 0.5, -0.5], tickvals=ticks, ticktext=[f"{int(t * tr.cell_m)}" for t in ticks],
                     showgrid=False, zeroline=False, scaleanchor="x", color=th["text"])
    return fig


def small_heat(arr, title, mode, zmin=None, zmax=None, binary=False, planet_key="mars", height=300):
    th = THEMES[mode]
    scale = ([[0, th["bg"]], [1, th["ae"]]] if binary else _scale(planet_key, mode))
    fig = go.Figure(go.Heatmap(z=arr.astype(float) if binary else arr, zmin=zmin, zmax=zmax, colorscale=scale,
                               colorbar=dict(thickness=8, tickfont=dict(color=th["text"], size=9)),
                               hovertemplate="col %{x}, row %{y}: %{z:.2f}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=5, r=5, t=28, b=5), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor=th["bg"], title=dict(text=title, font=dict(size=12, color=th["text"])),
                      font=dict(color=th["text"]))
    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False, visible=False, autorange="reversed", scaleanchor="x")
    return fig


def energy_figure(ro, ae, cmp, mode="Day", height=300):
    th = THEMES[mode]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Rover energy", x=["Rover-only", "AERIS"], y=[ro["energy_wh"], ae["energy_wh"]],
                         marker_color=th["ro"], text=[f"{ro['energy_wh']:.0f}", f"{ae['energy_wh']:.0f}"], textposition="inside"))
    fig.add_trace(go.Bar(name="Scout energy", x=["Rover-only", "AERIS"], y=[0, cmp["scout_wh"]],
                         marker_color=th["scout"], text=["", f"{cmp['scout_wh']:.1f}"], textposition="outside"))
    fig.update_layout(barmode="stack", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=th["text"]),
                      title=dict(text="System energy (Wh) - scout energy INCLUDED", font=dict(size=13)),
                      legend=dict(orientation="h", y=-0.15), yaxis=dict(gridcolor=th["grid"]))
    return fig


def comparison_figure(ro, ae, cmp, mode="Day", height=300):
    th = THEMES[mode]
    safe = lambda a, b: (a / b * 100.0) if b else 100.0
    cats = ["Distance", "Rover energy", "Travel time", "Hazard exposure", "TOTAL energy"]
    vals = [safe(ae["distance_m"], ro["distance_m"]), safe(ae["energy_wh"], ro["energy_wh"]),
            safe(ae["time_s"], ro["time_s"]), safe(ae["risk_m"], ro["risk_m"]),
            safe(cmp["aeris_total_wh"], ro["energy_wh"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Rover-only (=100)", x=cats, y=[100] * 5, marker_color=th["ro"]))
    fig.add_trace(go.Bar(name="AERIS", x=cats, y=vals, marker_color=th["ae"],
                         text=[f"{v:.0f}" for v in vals], textposition="outside"))
    fig.update_layout(barmode="group", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=th["text"]),
                      title=dict(text="Relative to rover-only (lower is better)", font=dict(size=13)),
                      legend=dict(orientation="h", y=-0.2), yaxis=dict(gridcolor=th["grid"]))
    return fig


def battery_figure(history, mode="Day", height=260):
    th = THEMES[mode]
    fig = go.Figure()
    if history:
        t = [h["t"] for h in history]
        fig.add_trace(go.Scatter(x=t, y=[h["rover_pct"] for h in history], name="Rover battery %",
                                 line=dict(color=th["ae"], width=3)))
        fig.add_trace(go.Scatter(x=t, y=[h["scout_pct"] for h in history], name="Scout battery %",
                                 line=dict(color=th["scout"], width=3)))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=th["text"]),
                      title=dict(text="Battery vs mission time", font=dict(size=13)),
                      xaxis=dict(title="s", gridcolor=th["grid"]), yaxis=dict(range=[0, 105], gridcolor=th["grid"]),
                      legend=dict(orientation="h", y=-0.25))
    return fig
