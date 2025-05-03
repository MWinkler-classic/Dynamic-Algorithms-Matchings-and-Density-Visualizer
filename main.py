"""
Live Load‑Distribution Visualiser
================================
* Nodes’ fill shade encodes load (0 → white; current max → black).
* Numeric load is now **blue text**, ensuring it remains legible over the varying grey backgrounds.
* Blue dot sits nearer the heavier‑loaded endpoint; loads and dots animate
  together for twenty 1‑second steps.

Render:
```
manim -qm load_distribution_manim.py LoadDistributionScene
```
"""
from __future__ import annotations
from manim import *  # type: ignore
from manim import config as manim_config
# Set video resolution to 1280×720 (720p)
manim_config.pixel_height = 720
manim_config.pixel_width = 1280
import networkx as nx
import numpy as np
from collections import defaultdict
from manim.utils.color import rgb_to_color

# ────────────────────────── Algorithm constants ──────────────────────────
EPSILON = 0.05
D = 20
d = 3

# ────────────────────────── Visual‑run bounds ────────────────────────────
DEMO_NODES = 20
DEMO_EDGES = 80
MAX_VISUAL_STEPS = 30
STEP_SECONDS = 1.0


# ---------------------------- RandomGraphAnalyzer ----------------------------
class RandomGraphAnalyzer:
    """Level‑raising algorithm with per‑edge load recomputation."""

    def __init__(self, n: int = DEMO_NODES, m: int = DEMO_EDGES):
        self.G = nx.gnm_random_graph(n, m)
        self.pos = nx.spring_layout(self.G)
        self.node_levels = {v: 0 for v in self.G.nodes()}
        self.edge_loads: dict[int, dict[int, float]] = defaultdict(dict)
        self.node_loads = {v: self.G.degree(v) / 2 for v in self.G.nodes()}
        for u, v in self.G.edges():
            self.edge_loads[u][v] = 0.5
            self.edge_loads[v][u] = 0.5

    @staticmethod
    def f(delta: int) -> float:
        return max(0.0, min(1.0, EPSILON * round((D - delta) / d)))

    def classify_nodes(self):
        max_load = max(self.node_loads.values())
        return [v for v, l in self.node_loads.items() if l > max_load - EPSILON]

    def iterate_once(self) -> bool:
        dirty = self.classify_nodes()
        if not dirty:
            return False
        for v in dirty:
            self.node_levels[v] += 1
        self.recompute_edge_loads()
        return True

    def recompute_edge_loads(self):
        for u, v in self.G.edges():
            delta = self.node_levels[u] - self.node_levels[v]
            load_to_u = self.f(delta) if delta >= 0 else 1 - self.f(-delta)
            self.edge_loads[u][v] = load_to_u
            self.edge_loads[v][u] = 1 - load_to_u
        self.node_loads = {v: 0.0 for v in self.G.nodes()}
        for u, v in self.G.edges():
            self.node_loads[u] += self.edge_loads[u][v]
            self.node_loads[v] += self.edge_loads[v][u]


# ---------------------------- Manim Scene ------------------------------------
class LoadDistributionScene(Scene):
    def construct(self):
        ana = RandomGraphAnalyzer()

        max_coord = np.max(np.abs(list(ana.pos.values()))) or 1
        scale = 3.5 / max_coord  # tighter layout
        to_vec = lambda p: np.array([p[0] * scale, p[1] * scale, 0])
        node_pos = {str(v): to_vec(p) for v, p in ana.pos.items()}

        global_max_load = ValueTracker(max(ana.node_loads.values()))

        # —Edges and dots—
        dot_trackers = {}
        for u, v in ana.G.edges():
            line = Line(node_pos[str(u)], node_pos[str(v)], stroke_width=1)
            self.add(line)
            dt = ValueTracker(ana.edge_loads[v][u])
            self.add(always_redraw(lambda t=dt, ln=line: Dot(radius=0.05, color=BLUE)
                                     .move_to(ln.point_from_proportion(t.get_value()))))
            dot_trackers[(u, v)] = dt

        # —Nodes—
        load_trackers = {}
        for name, p in node_pos.items():
            vt = ValueTracker(ana.node_loads[int(name)])
            load_trackers[name] = vt

            def make_circle(vt=vt, pos=p, gmax=global_max_load):
                grey = 1 - np.clip(vt.get_value() / (gmax.get_value() or 1), 0, 1)
                return Circle(radius=0.24, stroke_color=BLUE, stroke_width=2,
                               fill_color=rgb_to_color([grey]*3), fill_opacity=1).move_to(pos)
            self.add(always_redraw(make_circle))

            def make_label(vt=vt, pos=p):
                lbl = MathTex(f"{vt.get_value():.2f}", font_size=20)
                lbl.set_color(BLUE)
                return lbl.move_to(pos)
            self.add(always_redraw(make_label))

        # —Animation loop—
        steps = 0
        while steps < MAX_VISUAL_STEPS and ana.iterate_once():
            new_max = max(ana.node_loads.values())
            anims = [dt.animate.set_value(ana.edge_loads[v][u]) for (u, v), dt in dot_trackers.items()] + \
                    [vt.animate.set_value(ana.node_loads[int(n)]) for n, vt in load_trackers.items()] + \
                    [global_max_load.animate.set_value(new_max)]
            self.play(*anims, run_time=STEP_SECONDS,
                      rate_func=rate_functions.ease_in_out_cubic)
            steps += 1

        self.wait(2)


if __name__ == "__main__":
    from manim import tempconfig
    with tempconfig({"quality": "low_quality", "preview": False, "disable_caching": True}):
        LoadDistributionScene().render()
