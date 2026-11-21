"""
Smoke test for app.py WITHOUT Streamlit/Plotly installed: both are replaced by tiny fakes.
It checks that app.py's own logic runs (state handling, stepping, HTML builders, all planets/modes).
It does NOT validate real Streamlit/Plotly API behaviour - run `streamlit run app.py` for that.
"""
import os, sys, types, runpy
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)


class SS(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name):
        return getattr(FAKE, name)


class SmokeRerun(Exception):
    pass


class FakeSt(Ctx):
    __version__ = "1.52.0"
    def __init__(self): self.session_state = SS(); self.sidebar = Ctx(); self.calls = []
    def set_page_config(self, **k): pass
    def markdown(self, s, **k): self.calls.append(("md", len(s)))
    def caption(self, s): pass
    def selectbox(self, label, options, index=0, key=None, **k): return self.session_state.get(key, options[index])
    def radio(self, label, options, horizontal=False, key=None, **k): return self.session_state.get(key, options[0])
    def slider(self, label, mn, mx, value=None, step=None, key=None, **k): return self.session_state.get(key, value)
    def select_slider(self, label, options, value=None, key=None, **k): return self.session_state.get(key, value)
    def number_input(self, label, mn=None, mx=None, value=None, step=None, key=None, **k): return self.session_state[key]
    def toggle(self, label, value=False, key=None, **k): return self.session_state.get(key, value)
    def checkbox(self, label, value=False, key=None, **k): return self.session_state.get(key, value)
    def button(self, label, on_click=None, **k): return self.session_state.get("press:" + label, False)
    def columns(self, spec): return [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    def tabs(self, names): return [Ctx() for _ in names]
    def expander(self, *a, **k): return Ctx()
    def spinner(self, *a, **k): return Ctx()
    def plotly_chart(self, fig, **k): self.calls.append(("fig", k.get("key")))
    def rerun(self): raise SmokeRerun()


class Any:
    def __init__(self, *a, **k): self.a, self.k = a, k
    def __getattr__(self, n): return lambda *a, **k: None


FAKE = FakeSt()
st_mod = types.ModuleType("streamlit")
st_mod.__dict__.update({n: getattr(FAKE, n) for n in dir(FAKE) if not n.startswith("_")})
st_mod.session_state, st_mod.sidebar, st_mod.__version__ = FAKE.session_state, FAKE.sidebar, "1.52.0"
sys.modules["streamlit"] = st_mod
go = types.ModuleType("plotly.graph_objects")
class Figure(Any):
    def __init__(self, *a, **k): self.traces = []
    def add_trace(self, t): self.traces.append(t)
for nm in ("Heatmap", "Scatter", "Bar"):
    setattr(go, nm, type(nm, (Any,), {}))
go.Figure = Figure
pl = types.ModuleType("plotly"); pl.graph_objects = go
sys.modules.update({"plotly": pl, "plotly.graph_objects": go})


def run(state):
    FAKE.session_state.clear(); FAKE.session_state.update(state); FAKE.calls.clear()
    try:
        runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
    except SmokeRerun:
        pass
    return FAKE.session_state


for planet in ("Mars", "Earth", "Moon", "Custom"):
    for mode in ("Day", "Night vision", "Thermal"):
        s = run({"planet": planet, "mode": mode, "running": False})
        m = s["mission"]
        assert m.planet.name == planet
        s2 = dict(s); s2.update({"press:▶ Start / Resume": True})
        # second pass keeps the mission object (same cfg) and steps it
        st = run({**{k: v for k, v in s.items() if k in ("mission", "cfg_key")}, "planet": planet, "mode": mode,
                  "press:▶ Start / Resume": True})
        assert st["mission"].t > 0, "mission did not advance"
        print(f"{planet:6s} {mode:13s} ok  t={st['mission'].t:.0f}s figs={sum(1 for c in FAKE.calls if c[0]=='fig')}")
for view in ("Rover knowledge", "Scout observations", "Shared knowledge"):
    for layer in ("Slope", "Traversability", "Terrain type"):
        run({"view": view, "layer": layer})
st = run({"press:⏩ Run to end": True})
print("run-to-end:", st["mission"].rover.status, "done", st["mission"].done)
print("SMOKE OK")
