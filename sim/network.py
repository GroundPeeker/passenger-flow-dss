"""Пешеходная сеть: ориентированный граф участков, назначения, маршрутизация,
разбиение рёбер на ячейки CTM.

Узлы трёх типов:
- ``source``   — источник (выход со стадиона / сектора): нет входящих рёбер;
- ``junction`` — развилка / слияние: есть входящие и исходящие рёбра;
- ``sink``     — сток (вход в метро, остановка, уход из района): нет исходящих рёбер.
  Тип стока ``kind`` обязателен: ``entrance`` — вход станции (его можно закрыть;
  перед ним обязательна «площадка» — развилка с ветвью к другим назначениям, чтобы
  людям у закрытого входа было куда уйти), ``exit`` — уход из района.

Назначение (продукт многопродуктовой CTM) — поле ``destination`` стока, по умолчанию
id стока. Несколько стоков с одним назначением — разные входы одной станции.

Маршрутизация — матрица долей ``split`` формы (P, E): ``split[p, e]`` — доля людей
с назначением p в узле-начале ребра e, выбирающих ребро e. Соглашение для всех массивов
по назначениям: ось назначений — первая (в состоянии CTM — сразу после пакетной). По умолчанию — кратчайший
по длине путь к ближайшему стоку назначения (при равных путях — поровну); в YAML
переопределяется полем узла ``split: {назначение: {ребро: доля}}``.

Развороты запрещены: человек, пришедший в узел v по ребру u→v, не может уйти по
ребру v→u (единственное исключение — явный перенос назначения, ``CTM.transfer_destination``).
Достижимость назначения считается по рёбрам с учётом запрета разворотов (``edge_dist``). Граф маршрутов каждого назначения (рёбра с ``split[p, e] > 0``) обязан быть
ациклическим — это проверяется при загрузке и в ``split_with``.

В YAML величины задаются ключами с единицами (см. sim/units.py); внутри — СИ:
м, с, чел, чел/с.
"""

from __future__ import annotations

import heapq
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from sim.units import Section

NODE_TYPES = ("source", "junction", "sink")
SINK_KINDS = ("entrance", "exit")
_TOL = 1e-9


class InfiniteBufferWarning(UserWarning):
    """Бесконечный накопитель стока: допустим только в тестах (очередь не влияет на сеть)."""


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    # Развилка/источник: {назначение: {исходящее ребро: доля}} — переопределение маршрута.
    split: dict[str, dict[str, float]] = field(default_factory=dict)
    # Источник: суммарный поток выпуска, чел/с, и доли по назначениям {назначение: доля}.
    inflow: float = 0.0
    destinations: dict[str, float] = field(default_factory=dict)
    # Сток: тип (entrance / exit), назначение, пропускная способность (чел/с),
    # вместимость накопителя (чел).
    kind: str | None = None
    destination: str | None = None
    capacity: float = math.inf
    buffer: float = 0.0


@dataclass(frozen=True)
class Edge:
    id: str
    source: str
    target: str
    length: float  # м
    width: float   # м
    # Накопитель (широкая зона ожидания: площадь перед входом, эспланада) — одна ячейка
    # с вместимостью площадь × ρ_queue вместо фундаментальной диаграммы (sim/ctm.py).
    zone: bool = False

    @property
    def area(self) -> float:
        return self.length * self.width


@dataclass(frozen=True)
class Cells:
    """Разбиение рёбер на ячейки. Ячейки ребра e — непрерывный диапазон
    ``first[e] .. last[e]`` в порядке движения."""

    edge: np.ndarray    # (C,) индекс ребра ячейки
    length: np.ndarray  # (C,) длина ячейки, м
    width: np.ndarray   # (C,) ширина ячейки, м
    first: np.ndarray   # (E,) индекс первой ячейки ребра
    last: np.ndarray    # (E,) индекс последней ячейки ребра

    @property
    def area(self) -> np.ndarray:
        return self.length * self.width

    @property
    def n_cells(self) -> int:
        return int(self.edge.size)

    @property
    def n_per_edge(self) -> np.ndarray:
        return self.last - self.first + 1


class Network:
    """Ориентированный граф пешеходной сети с назначениями и маршрутизацией."""

    def __init__(self, nodes: list[Node], edges: list[Edge], layout: dict | None = None):
        self.nodes = list(nodes)
        # Схема для рисунков (координаты узлов, изломы рёбер) — в модель не входит.
        self.layout = layout
        self.edges = list(edges)
        self.node_index = {n.id: i for i, n in enumerate(self.nodes)}
        self.edge_index = {e.id: i for i, e in enumerate(self.edges)}
        if len(self.node_index) != len(self.nodes):
            raise ValueError("повторяющиеся id узлов")
        if len(self.edge_index) != len(self.edges):
            raise ValueError("повторяющиеся id рёбер")

        self._in: dict[str, list[int]] = {n.id: [] for n in self.nodes}
        self._out: dict[str, list[int]] = {n.id: [] for n in self.nodes}
        for k, e in enumerate(self.edges):
            for end in (e.source, e.target):
                if end not in self.node_index:
                    raise ValueError(f"ребро {e.id}: неизвестный узел {end}")
            if e.source == e.target:
                raise ValueError(f"ребро {e.id}: петля")
            if not (e.length > 0 and e.width > 0):
                raise ValueError(f"ребро {e.id}: длина и ширина должны быть > 0")
            self._out[e.source].append(k)
            self._in[e.target].append(k)
        for n in self.nodes:
            self._check_node(n)
        self._check_dead_ends()
        self._check_bidirectional()

        sinks = self.nodes_of_type("sink")
        self.destinations: list[str] = list(dict.fromkeys(self.sink_destination(v) for v in sinks))
        self.dest_index = {d: p for p, d in enumerate(self.destinations)}
        self.dist = self._distances()        # (V, P) кратчайшее расстояние до назначения, м
        self.edge_dist = self._edge_distances()  # (P, E) от конца ребра, без разворотов, м
        self._check_reachability()
        self.split = self._default_split()   # (P, E)
        self.validate_split(self.split)
        self.split = self.split_with({n.id: n.split for n in self.nodes if n.split})
        self.source_shares = self._source_shares()  # (P, Ns)

    def _check_node(self, n: Node) -> None:
        if n.type not in NODE_TYPES:
            raise ValueError(f"узел {n.id}: тип {n.type!r} не из {NODE_TYPES}")
        n_in, n_out = len(self._in[n.id]), len(self._out[n.id])
        if n.type == "source" and (n_in > 0 or n_out == 0):
            raise ValueError(f"источник {n.id}: нужны только исходящие рёбра")
        if n.type == "sink" and (n_out > 0 or n_in == 0):
            raise ValueError(f"сток {n.id}: нужны только входящие рёбра")
        if n.type == "junction" and (n_in == 0 or n_out == 0):
            raise ValueError(f"развилка {n.id}: нужны входящие и исходящие рёбра")
        if n.inflow < 0 or n.capacity < 0 or n.buffer < 0:
            raise ValueError(f"узел {n.id}: inflow, capacity, buffer должны быть ≥ 0")
        if n.type == "sink" and n.kind not in SINK_KINDS:
            raise ValueError(f"сток {n.id}: укажите kind из {SINK_KINDS}")
        if n.type != "sink" and n.kind is not None:
            raise ValueError(f"узел {n.id}: kind задаётся только у стоков")
        if n.type == "sink" and math.isinf(n.buffer):
            warnings.warn(f"сток {n.id}: бесконечный накопитель — только для тестов, "
                          f"очередь не распространяется на сеть", InfiniteBufferWarning,
                          stacklevel=3)

    def is_uturn(self, k_in: int, k_out: int) -> bool:
        """Переход с ребра k_in на k_out — разворот (u→v, затем v→u)."""
        a, b = self.edges[k_in], self.edges[k_out]
        return a.target == b.source and b.target == a.source

    def _check_dead_ends(self) -> None:
        for v, node in enumerate(self.nodes):
            if node.type != "junction":
                continue
            for ki in self._in[node.id]:
                if all(self.is_uturn(ki, ko) for ko in self._out[node.id]):
                    raise ValueError(f"узел {node.id}: с ребра {self.edges[ki].id} можно уйти "
                                     f"только разворотом (тупик), развороты запрещены")

    def bidirectional_pairs(self) -> list[tuple[int, int]]:
        """Пары встречных рёбер (u→v, v→u) — два направления одной улицы."""
        return [(a, b) for a in range(len(self.edges)) for b in range(a + 1, len(self.edges))
                if self.is_uturn(a, b)]

    def _check_bidirectional(self) -> None:
        """Временное правило до этапа 10 (общая плотность встречных ячеек): два направления
        улицы — независимые рёбра, каждое с половиной ширины улицы. Значит, встречные
        рёбра должны совпадать по длине и ширине."""
        for a, b in self.bidirectional_pairs():
            ea, eb = self.edges[a], self.edges[b]
            if ea.zone or eb.zone:
                raise ValueError(f"встречные рёбра {ea.id} и {eb.id}: накопитель (zone) не может "
                                 f"быть частью встречной пары")
            if not (math.isclose(ea.length, eb.length) and math.isclose(ea.width, eb.width)):
                raise ValueError(
                    f"встречные рёбра {ea.id} и {eb.id}: длины и ширины должны совпадать "
                    f"(каждое — половина ширины улицы; временное правило до этапа 10)")

    # --- маршрутизация -----------------------------------------------------------

    def sink_destination(self, v: int) -> str:
        n = self.nodes[v]
        return n.destination if n.destination is not None else n.id

    def _distances(self) -> np.ndarray:
        """Дейкстра по обратному графу от стоков каждого назначения."""
        V, P = len(self.nodes), len(self.destinations)
        dist = np.full((V, P), np.inf)
        for p, d in enumerate(self.destinations):
            heap = [(0.0, v) for v in self.nodes_of_type("sink") if self.sink_destination(v) == d]
            while heap:
                dv, v = heapq.heappop(heap)
                if dv >= dist[v, p]:
                    continue
                dist[v, p] = dv
                for k in self._in[self.nodes[v].id]:
                    e = self.edges[k]
                    u = self.node_index[e.source]
                    if dv + e.length < dist[u, p]:
                        heapq.heappush(heap, (dv + e.length, u))
        return dist

    def _edge_distances(self) -> np.ndarray:
        """Кратчайшее расстояние от конца ребра e до назначения p при условии, что
        человек пришёл по e (без разворотов): Дейкстра по графу рёбер."""
        P, E = len(self.destinations), len(self.edges)
        dist = np.full((P, E), np.inf)
        for p, d in enumerate(self.destinations):
            heap = [(0.0, k) for k, e in enumerate(self.edges)
                    if self.nodes[self.node_index[e.target]].type == "sink"
                    and self.sink_destination(self.node_index[e.target]) == d]
            while heap:
                dk, k = heapq.heappop(heap)
                if dk >= dist[p, k]:
                    continue
                dist[p, k] = dk
                nxt = self.edges[k]
                for j in self._in[nxt.source]:
                    if not self.is_uturn(j, k) and dk + nxt.length < dist[p, j]:
                        heapq.heappush(heap, (dk + nxt.length, j))
        return dist

    def edge_reaches(self, k: int, p: int) -> bool:
        """С конца ребра k можно дойти до назначения p без разворотов."""
        return bool(np.isfinite(self.edge_dist[p, k]))

    def _check_reachability(self) -> None:
        for k, e in enumerate(self.edges):
            if not np.any(np.isfinite(self.edge_dist[:, k])):
                raise ValueError(f"ребро {e.id}: с него без разворота не дойти ни до одного стока")
        for v in self.nodes_of_type("sink"):
            node = self.nodes[v]
            if node.kind != "entrance":
                continue
            p = self.dest_index[self.sink_destination(v)]
            for k in self._in[node.id]:
                plaza = self.nodes[self.node_index[self.edges[k].source]]
                alt = [j for j in self._out[plaza.id] if j != k and any(
                    self.edge_reaches(j, q) for q in range(len(self.destinations)) if q != p)]
                if plaza.type != "junction" or not alt:
                    raise ValueError(
                        f"вход {node.id}: перед ним нужна площадка — развилка с ветвью "
                        f"к другим назначениям (ребро {self.edges[k].id} идёт из {plaza.id})")

    def _default_split(self) -> np.ndarray:
        """Кратчайший путь: все люди с назначением p идут по рёбрам, лежащим на
        кратчайшем пути (при равенстве — поровну)."""
        S = np.zeros((len(self.destinations), len(self.edges)))
        for v, node in enumerate(self.nodes):
            outs = self._out[node.id]
            for p in range(len(self.destinations)):
                if not outs or not np.isfinite(self.dist[v, p]):
                    continue
                via = np.array([self.edges[k].length + self.dist[self.node_index[self.edges[k].target], p]
                                for k in outs])
                best = np.flatnonzero(via <= self.dist[v, p] * (1 + 1e-12) + 1e-9)
                for k in np.array(outs)[best]:
                    S[p, k] = 1.0 / best.size
        return S

    def split_with(self, overrides: dict[str, dict[str, dict[str, float]]]) -> np.ndarray:
        """Матрица долей (P, E) с переопределениями {узел: {назначение: {ребро: доля}}}.
        Строки остальных (узел, назначение) берутся из текущей ``self.split``."""
        S = self.split.copy()
        for node_id, by_dest in overrides.items():
            if node_id not in self.node_index:
                raise ValueError(f"split: неизвестный узел {node_id}")
            outs = {self.edges[k].id: k for k in self._out[node_id]}
            for dest, fr in by_dest.items():
                if dest not in self.dest_index:
                    raise ValueError(f"узел {node_id}: неизвестное назначение {dest!r}")
                p = self.dest_index[dest]
                if not np.isfinite(self.dist[self.node_index[node_id], p]):
                    raise ValueError(f"узел {node_id}: назначение {dest} недостижимо")
                if not set(fr) <= set(outs):
                    raise ValueError(f"узел {node_id}: {sorted(set(fr) - set(outs))} "
                                     f"не исходящие рёбра")
                vals = np.array([float(fr.get(e, 0.0)) for e in outs])
                if np.any(vals < 0) or abs(vals.sum() - 1.0) > _TOL:
                    raise ValueError(f"узел {node_id}, {dest}: доли ≥ 0 с суммой 1")
                for e, k in outs.items():
                    if fr.get(e, 0.0) > 0 and not self.edge_reaches(k, p):
                        raise ValueError(f"узел {node_id}: по ребру {e} нельзя дойти до {dest}")
                S[p, list(outs.values())] = vals / vals.sum()
        self.validate_split(S)
        return S

    def routing_cycle(self, S: np.ndarray, p: int) -> list[str] | None:
        """Цикл (список id узлов) в графе маршрутов назначения p или None."""
        succ: dict[int, list[int]] = {}
        for k in np.flatnonzero(S[p] > 0):
            e = self.edges[k]
            succ.setdefault(self.node_index[e.source], []).append(self.node_index[e.target])
        color = dict.fromkeys(range(len(self.nodes)), 0)  # 0 — не посещён, 1 — в стеке, 2 — готов
        for root in range(len(self.nodes)):
            if color[root]:
                continue
            stack = [(root, iter(succ.get(root, [])))]
            path = [root]
            color[root] = 1
            while stack:
                v, it = stack[-1]
                w = next(it, None)
                if w is None:
                    color[v] = 2
                    stack.pop(); path.pop()
                elif color[w] == 1:
                    cyc = path[path.index(w):] + [w]
                    return [self.nodes[x].id for x in cyc]
                elif color[w] == 0:
                    color[w] = 1
                    stack.append((w, iter(succ.get(w, []))))
                    path.append(w)
        return None

    def validate_split(self, S: np.ndarray) -> None:
        """Матрица долей (P, E): доли ≥ 0; в каждом узле, из которого назначение достижимо,
        доли по исходящим рёбрам в сумме 1; доли только на рёбра, ведущие к назначению;
        граф маршрутов каждого назначения ациклический (в частности, без разворотов)."""
        S = np.asarray(S)
        if S.shape != (self.n_dest, len(self.edges)) or np.any(S < 0):
            raise ValueError(f"матрица долей: нужна форма {(self.n_dest, len(self.edges))}, доли ≥ 0")
        for p, dest in enumerate(self.destinations):
            for v, node in enumerate(self.nodes):
                outs = self._out[node.id]
                if not outs:
                    continue
                total = S[p, outs].sum()
                need = 1.0 if np.isfinite(self.dist[v, p]) else 0.0
                if abs(total - need) > _TOL:
                    raise ValueError(f"узел {node.id}, {dest}: сумма долей {total}, нужно {need}")
            bad = [self.edges[k].id for k in np.flatnonzero(S[p] > 0) if not self.edge_reaches(k, p)]
            if bad:
                raise ValueError(f"{dest}: по рёбрам {bad} нельзя дойти до назначения")
            cyc = self.routing_cycle(S, p)
            if cyc:
                raise ValueError(f"{dest}: цикл в графе маршрутов {' → '.join(cyc)}")

    def _source_shares(self) -> np.ndarray:
        srcs = self.nodes_of_type("source")
        X = np.zeros((len(self.destinations), len(srcs)))
        for s, v in enumerate(srcs):
            node = self.nodes[v]
            shares = node.destinations
            if not shares:
                reach = [d for p, d in enumerate(self.destinations) if np.isfinite(self.dist[v, p])]
                if len(reach) != 1:
                    raise ValueError(f"источник {node.id}: задайте доли destinations по {reach}")
                shares = {reach[0]: 1.0}
            for d, x in shares.items():
                if d not in self.dest_index:
                    raise ValueError(f"источник {node.id}: неизвестное назначение {d!r}")
                if x < 0:
                    raise ValueError(f"источник {node.id}: доли должны быть ≥ 0")
                if x > 0 and not np.isfinite(self.dist[v, self.dest_index[d]]):
                    raise ValueError(f"источник {node.id}: назначение {d} недостижимо")
                X[self.dest_index[d], s] = x
            if abs(X[:, s].sum() - 1.0) > _TOL:
                raise ValueError(f"источник {node.id}: доли destinations в сумме ≠ 1")
            X[:, s] /= X[:, s].sum()
        return X

    # --- загрузка ----------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "Network":
        top = Section(d, "сеть")
        nodes = []
        for nd in top.get("nodes"):
            sec = Section(nd, f"узел {nd.get('id')}")
            node_id = str(sec.get("id"))
            node_type = sec.get("type", "junction")
            split = {str(dest): {str(e): float(x) for e, x in fr.items()}
                     for dest, fr in (sec.get("split", None) or {}).items()}
            dests = {str(k): float(v) for k, v in (sec.get("destinations", None) or {}).items()}
            dest = sec.get("destination", None)
            kind = sec.get("kind", None)
            inflow = sec.quantity("inflow", "rate", 0.0)
            capacity = sec.quantity("capacity", "rate", math.inf)
            buffer = sec.quantity("buffer", "count", 0.0)
            sec.finish()
            nodes.append(Node(node_id, node_type, split, inflow, dests, kind,
                              None if dest is None else str(dest),
                              math.inf if capacity is None else capacity,
                              0.0 if buffer is None else buffer))
        edges = []
        for ed in top.get("edges"):
            sec = Section(ed, f"ребро {ed.get('id')}")
            zone = sec.get("zone", False)
            if not isinstance(zone, bool):
                raise ValueError(f"ребро {ed.get('id')}: zone — true или false")
            edges.append(Edge(str(sec.get("id")), str(sec.get("from")), str(sec.get("to")),
                              sec.quantity("length", "length"), sec.quantity("width", "length"),
                              zone))
            sec.finish()
        layout = top.get("layout", None)
        top.finish()
        return cls(nodes, edges, layout)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Network":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    # --- запросы -----------------------------------------------------------------

    @property
    def n_dest(self) -> int:
        return len(self.destinations)

    def in_edges(self, node_id: str) -> list[int]:
        return list(self._in[node_id])

    def out_edges(self, node_id: str) -> list[int]:
        return list(self._out[node_id])

    def nodes_of_type(self, node_type: str) -> list[int]:
        return [i for i, n in enumerate(self.nodes) if n.type == node_type]

    # --- разбиение на ячейки -----------------------------------------------------

    def discretize(self, min_cell_length: float) -> Cells:
        """Разбивает каждое ребро на максимальное число равных ячеек длиной не меньше
        ``min_cell_length`` (условие Куранта: длина ячейки ≥ c_max·dt). Накопитель
        (``zone``) — всегда одна ячейка."""
        n_per = []
        for e in self.edges:
            k = math.floor(e.length / min_cell_length * (1 + 1e-12))
            if e.zone and k >= 1:
                k = 1
            if k < 1:
                raise ValueError(
                    f"ребро {e.id}: длина {e.length} м меньше минимальной длины ячейки "
                    f"{min_cell_length:.2f} м (условие Куранта c_max·dt); "
                    f"уменьшите dt или удлините ребро"
                )
            n_per.append(k)
        n_per = np.array(n_per, dtype=np.int64)
        last = np.cumsum(n_per) - 1
        first = last - n_per + 1
        edge = np.repeat(np.arange(len(self.edges)), n_per)
        lengths = np.array([e.length for e in self.edges])
        widths = np.array([e.width for e in self.edges])
        return Cells(edge=edge, length=(lengths / n_per)[edge], width=widths[edge],
                     first=first, last=last)


def grid_network(rows: int, cols: int, rng: np.random.Generator, *,
                 length_range: tuple[float, float] = (50.0, 200.0),
                 width_range: tuple[float, float] = (3.0, 10.0),
                 inflow: float = 3.0,
                 sink_capacity: float = 2.0,
                 sink_buffer: float = 0.0) -> Network:
    """Синтетическая решётчатая сеть (для тестов и бенчмарка). Величины — в СИ.

    Развилки (i, j) решётки rows×cols, рёбра «вправо» и «вниз»; в каждую развилку
    первой строки входит ребро от своего источника S_j, из каждой развилки последней
    строки выходит ребро в свой сток K_j (отдельное назначение). Число рёбер:
    rows(cols−1) + (rows−1)cols + 2cols (5×5 → 50). Длины, ширины, доли назначений
    у источников и доли маршрутов на развилках случайны (по ``rng``).
    """
    def jid(i, j):
        return f"J{i}_{j}"

    edges = []

    def add_edge(u, v):
        edges.append({"id": f"e{len(edges)}", "from": u, "to": v,
                      "length_m": float(rng.uniform(*length_range)),
                      "width_m": float(rng.uniform(*width_range))})

    for j in range(cols):
        add_edge(f"S{j}", jid(0, j))
    for i in range(rows):
        for j in range(cols):
            if j + 1 < cols:
                add_edge(jid(i, j), jid(i, j + 1))
            if i + 1 < rows:
                add_edge(jid(i, j), jid(i + 1, j))
    for j in range(cols):
        add_edge(jid(rows - 1, j), f"K{j}")

    nodes = []
    for j in range(cols):
        # из столбца j достижимы стоки K_j … K_{cols−1}
        reach = [f"K{m}" for m in range(j, cols)]
        nodes.append({"id": f"S{j}", "type": "source", "inflow_ped_per_s": inflow,
                      "destinations": dict(zip(reach, rng.dirichlet(np.ones(len(reach))).tolist()))})
    nodes += [{"id": jid(i, j), "type": "junction"} for i in range(rows) for j in range(cols)]
    # Стоки решётки — kind: exit (у синтетической решётки нет площадок перед входами).
    nodes += [{"id": f"K{j}", "type": "sink", "kind": "exit", "capacity_ped_per_s": sink_capacity,
               "buffer_ped": sink_buffer} for j in range(cols)]
    d = {"nodes": nodes, "edges": edges}

    # Случайные доли маршрутов там, где к назначению ведут несколько рёбер.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InfiniteBufferWarning)
        net0 = Network.from_dict(d)
    for nd in nodes:
        if nd["type"] != "junction":
            continue
        outs = net0.out_edges(nd["id"])
        split = {}
        for p, dest in enumerate(net0.destinations):
            ok = [net0.edges[k].id for k in outs if net0.edge_reaches(k, p)]
            if len(ok) > 1:
                split[dest] = dict(zip(ok, rng.dirichlet(np.ones(len(ok))).tolist()))
        if split:
            nd["split"] = split
    return Network.from_dict(d)
