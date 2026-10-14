"""Рисование сети по схеме ``layout`` из YAML сети: рёбра — ломаные, разбитые на ячейки CTM
(цвет ячейки — плотность), узлы, станции, указатели, датчики.

Встречные рёбра (u→v и v→u) рисуются со смещением вправо по ходу движения, чтобы оба
направления были видны.
"""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

from analysis import style

PAIR_OFFSET_M = 9.0      # смещение встречных рёбер на схеме, м


class NetGeometry:
    """Координаты ячеек сети на схеме."""

    def __init__(self, net, cells):
        lay = net.layout
        if not lay or "nodes" not in lay:
            raise ValueError("у сети нет схемы layout.nodes — рисунок невозможен")
        self.net = net
        self.cells = cells
        self.pos = {k: np.asarray(v, dtype=float) for k, v in lay["nodes"].items()}
        missing = [n.id for n in net.nodes if n.id not in self.pos]
        if missing:
            raise ValueError(f"layout.nodes: нет координат узлов {missing}")
        self.stadium = lay.get("stadium")
        wp = lay.get("waypoints") or {}
        paired = {k for pair in net.bidirectional_pairs() for k in pair}
        self.polylines = []
        for k, e in enumerate(net.edges):
            pts = np.array([self.pos[e.source], *map(np.asarray, wp.get(e.id, [])),
                            self.pos[e.target]], dtype=float)
            if k in paired:
                pts = _offset_right(pts, PAIR_OFFSET_M)
            self.polylines.append(pts)
        # отрезки ячеек: ячейка i ребра — участок ломаной [i/n, (i+1)/n] по длине
        segs = []
        for k, pts in enumerate(self.polylines):
            n = int(cells.n_per_edge[k])
            for i in range(n):
                segs.append(_subpath(pts, i / n, (i + 1) / n))
        self.cell_segments = segs
        self.cell_width = cells.width

    def edge_midpoint(self, k: int) -> tuple[np.ndarray, float]:
        """Середина ребра на схеме и угол направления (градусы)."""
        seg = _subpath(self.polylines[k], 0.45, 0.55)
        d = seg[-1] - seg[0]
        return 0.5 * (seg[0] + seg[-1]), float(np.degrees(np.arctan2(d[1], d[0])))

    def extent(self, pad: float = 60.0):
        pts = np.vstack(self.polylines)
        return (pts[:, 0].min() - pad, pts[:, 0].max() + pad,
                pts[:, 1].min() - pad, pts[:, 1].max() + pad)


def _offset_right(pts: np.ndarray, off: float) -> np.ndarray:
    """Сдвиг ломаной вправо по ходу движения (по нормали к каждому отрезку)."""
    out = pts.copy()
    normals = []
    for a, b in zip(pts[:-1], pts[1:]):
        d = (b - a) / np.linalg.norm(b - a)
        normals.append(np.array([d[1], -d[0]]))
    for i in range(len(pts)):
        ns = [normals[j] for j in (i - 1, i) if 0 <= j < len(normals)]
        n = np.mean(ns, axis=0)
        out[i] = pts[i] + off * n / max(np.linalg.norm(n), 1e-9)
    return out


def _subpath(pts: np.ndarray, f0: float, f1: float) -> np.ndarray:
    """Часть ломаной между долями длины f0 и f1."""
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    L = cum[-1]

    def at(f):
        s = f * L
        j = min(np.searchsorted(cum, s, side="right") - 1, len(seg_len) - 1)
        w = (s - cum[j]) / seg_len[j] if seg_len[j] > 0 else 0.0
        return pts[j] + w * (pts[j + 1] - pts[j])

    inner = [pts[j] for j in range(1, len(pts) - 1) if f0 * L < cum[j] < f1 * L]
    return np.array([at(f0), *inner, at(f1)])


def _cell_lw(width_m: np.ndarray) -> np.ndarray:
    """Толщина линии ячейки, pt: растёт с шириной участка, с ограничением
    (накопители шире 20 м — до 14 pt)."""
    return np.clip(1.2 + 0.55 * width_m, 1.5, np.where(width_m >= 20, 14.0, 9.0))


def draw_network(ax, geo: NetGeometry, density: np.ndarray | None = None, *,
                 vmax: float = 5.4, cmap=None, base_color: str = "#c9c7c1",
                 stadium_label: bool = True):
    """Рёбра сети; при ``density`` (C,) — ячейки цветом плотности. Возвращает LineCollection."""
    if geo.stadium:
        c, w, h = geo.stadium["center"], geo.stadium["width_m"], geo.stadium["height_m"]
        ax.add_patch(Ellipse(c, w, h, facecolor="#efeee9", edgecolor=style.TEXT_SECONDARY,
                             lw=0.8, zorder=1))
        if stadium_label:
            ax.text(c[0], c[1], "Стадион", ha="center", va="center", color=style.TEXT_SECONDARY,
                    fontsize=9, zorder=2)
    lc = LineCollection(geo.cell_segments, linewidths=_cell_lw(geo.cell_width),
                        capstyle="projecting", zorder=3)
    if density is None:
        lc.set_color(base_color)
    else:
        lc.set_cmap(cmap or style.SEQUENTIAL)
        lc.set_norm(Normalize(0.0, vmax))
        lc.set_array(np.asarray(density, dtype=float))
    ax.add_collection(lc)
    # тонкий контур под ячейками — видно и пустые рёбра
    outline = LineCollection(geo.polylines, colors="#b9b7b0",
                             linewidths=[max(lw) + 1.2 for lw in _edge_lws(geo)], zorder=2)
    ax.add_collection(outline)
    x0, x1, y0, y1 = geo.extent()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return lc


def _edge_lws(geo: NetGeometry):
    out = []
    for k in range(len(geo.net.edges)):
        sl = slice(geo.cells.first[k], geo.cells.last[k] + 1)
        out.append(_cell_lw(geo.cell_width[sl]))
    return out


STATION_MARK = {"entrance": "s", "exit": "D"}


def draw_nodes(ax, geo: NetGeometry, sim=None, *, labels: bool = True, signs: bool = True,
               scale: float = 1.0):
    """Стоки (станции, уход из района) и подписи; указатели — треугольники."""
    net = geo.net
    st_names = {}
    if sim is not None:
        for sp in sim.stations.specs:
            # остановка автобусов (направление shuttle в конфиге) подписывается как остановка
            name = (f"Остановка автобусов {sp.id}" if sp.direction == "shuttle"
                    else f"Станция {sp.id}")
            for eid in sp.entrances:
                st_names[eid] = name
    for n in net.nodes:
        p = geo.pos[n.id]
        if n.type == "sink":
            is_ent = n.kind == "entrance"
            ax.plot(*p, STATION_MARK[n.kind], ms=(8 if is_ent else 7) * scale, zorder=6,
                    color=style.TEXT if is_ent else "white", markeredgecolor=style.TEXT,
                    markeredgewidth=1.0)
            if labels:
                name = st_names.get(n.id, "выход из района (пешком)")
                ax.annotate(name, p, xytext=(8, 4), textcoords="offset points", fontsize=8,
                            color=style.TEXT, zorder=7)
        elif n.type == "source" and labels:
            ax.plot(*p, "o", ms=5, color=style.TEXT_SECONDARY, zorder=6)
    if signs and sim is not None:
        # указатели маршрута и смены станции (на одном узле — один значок)
        sign_nodes = list(dict.fromkeys(list(sim.signs.nodes) + list(sim.switch_signs.nodes)))
        for node in sign_nodes:
            ax.plot(*geo.pos[node], "^", ms=10 * scale, color=style.SERIES[3], markeredgecolor=style.TEXT,
                    markeredgewidth=0.8, zorder=7)
