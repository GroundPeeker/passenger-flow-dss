"""Векторизованная многопродуктовая пешеходная модель передачи ячеек (CTM).

Продукт — назначение (станция / уход из района, см. sim/network.py). Состояние —
число людей в каждой ячейке по назначениям, n[b, p, c]: так сохранение людей
выполняется точно (поток вычитается из одной ячейки и прибавляется в другую).
Первая ось всех массивов состояния — пакет B независимых прогонов одной сети
(частицы фильтра, сценарии MPC; B = 1 — одиночный прогон), вторая — назначение,
последняя — ячейка (непрерывная в памяти: сдвиги и сборки по ней быстрые).

Шаг dt:
1. Предложение S и приём R ячейки по суммарной плотности ρ = Σ_p n_p / A
   (диаграмма Weidmann): S = W·dt·Q(min(ρ, ρc)), R = W·dt·Q(max(ρ, ρc)),
   S ≤ n (нельзя отдать больше, чем есть), R ≤ (ρ_jam − ρ)·A (нет места).
   Предложение делится по назначениям пропорционально составу ячейки: S_p = S·n_p/n.
2. Внутри ребра поток f = min(S_k, R_{k+1}) с составом ячейки k.
3. Узел. Движение (вход i → выход o, развороты исключены). Человек выбирает движение
   при входе в последнюю ячейку ребра (или при выходе из источника) по долям β
   (матрица маршрутов назначения × ребро, нормированная по разрешённым движениям);
   выбор хранится в ``committed``. Спрос движения D_io,p = (S_i / n_i)·committed_io,p.
   Для выхода o: φ_o = min(1, R_o / Σ_i,p D_io,p) — приём делится пропорционально спросу.
   - без FIFO (по умолчанию): поток движения = φ_o·D_io — выбравшие забитый выход ждут
     его, остальные проходят; спонтанный обход затора — sim/behavior.py (θ, по умолчанию 0);
   - FIFO (флаг node_model.fifo): θ_i = min по выходам с D_io > 0 от φ_o, поток
     движения = θ_i·D_io — затор на любом нужном выходе останавливает весь вход i.
4. Смена назначения (``dest_switch``, указатель «смена станции», sim/behavior.py): доля
   людей слоя p, входящих в последнюю ячейку ребра перед узлом указателя, в момент
   фиксации выбора движения переходит в слой q и выбирает движение по долям q.
   Уже зафиксировавшие выбор не меняются; перенос учитывается в ``transferred``.
5. Источник — виртуальный вход с очередью по назначениям (вышли, но не вошли в сеть).
   Сток — виртуальный выход с пропускной способностью μ и накопителем вместимости
   B_max: приём R = μ·dt + (B_max − b); обслуживание пропорционально составу накопителя.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import brentq, minimize_scalar

from sim.behavior import detour_transfer
from sim.network import Network
from sim.units import Section


# --- фундаментальная диаграмма -------------------------------------------------

@dataclass(frozen=True)
class WeidmannFD:
    """Фундаментальная диаграмма Weidmann (1993), величины в СИ:
    v(ρ) = v0·(1 − exp(−γ·(1/ρ − 1/ρ_jam))),  Q(ρ) = ρ·v(ρ)  [чел/(м·с)]."""

    v0: float = 1.34          # м/с
    rho_jam: float = 5.4      # чел/м²
    gamma: float = 1.913      # 1/м²
    q_cap: float | None = None  # доп. ограничение пропускной способности, чел/(м·с)

    def speed(self, rho):
        rho = np.asarray(rho, dtype=float)
        r = np.maximum(rho, 1e-12)
        with np.errstate(over="ignore"):
            v = self.v0 * (1.0 - np.exp(-self.gamma * (1.0 / r - 1.0 / self.rho_jam)))
        return np.clip(v, 0.0, self.v0)

    def flow(self, rho):
        return np.asarray(rho, dtype=float) * self.speed(rho)

    def flow_derivative(self, rho):
        """dQ/dρ = v(ρ) + ρ·v'(ρ),  ρ·v'(ρ) = −v0·γ/ρ · exp(−γ·(1/ρ − 1/ρ_jam))."""
        r = np.maximum(np.asarray(rho, dtype=float), 1e-12)
        with np.errstate(over="ignore"):
            e = np.exp(-self.gamma * (1.0 / r - 1.0 / self.rho_jam))
        return self.v0 * (1.0 - e) - self.v0 * self.gamma / r * e

    @cached_property
    def rho_crit(self) -> float:
        """Критическая плотность — точка максимума Q(ρ)."""
        res = minimize_scalar(lambda r: -float(self.flow(r)), bounds=(1e-6, self.rho_jam),
                              method="bounded", options={"xatol": 1e-10})
        return float(res.x)

    @cached_property
    def q_max(self) -> float:
        """Пропускная способность на единицу ширины, чел/(м·с)."""
        q = float(self.flow(self.rho_crit))
        return q if self.q_cap is None else min(q, self.q_cap)

    @cached_property
    def backward_wave_speed(self) -> float:
        """Наибольшая скорость обратной волны max |dQ/dρ| на заторной ветви, м/с."""
        rho = np.linspace(self.rho_crit, self.rho_jam, 200_001)
        return float(np.max(-self.flow_derivative(rho)))

    @cached_property
    def max_wave_speed(self) -> float:
        """Наибольшая скорость характеристик для условия Куранта:
        max(v0 — прямая волна при ρ → 0, скорость обратной волны), м/с.
        Ограничение q_cap только срезает диаграмму и скорости не увеличивает."""
        return max(self.v0, self.backward_wave_speed)

    def density_for_flow(self, q: float, branch: str) -> float:
        """Плотность с Q(ρ) = q на свободной (``free``) или заторной (``congested``) ветви."""
        if not 0.0 <= q <= self.q_max:
            raise ValueError(f"поток {q} вне [0, {self.q_max}]")
        lo, hi = (1e-12, self.rho_crit) if branch == "free" else (self.rho_crit, self.rho_jam)
        f = lambda r: float(self.flow(r)) - q
        if f(lo) * f(hi) > 0:  # q = q_max при q_cap < максимума диаграммы
            return self.rho_crit
        return brentq(f, lo, hi, xtol=1e-14)


@dataclass(frozen=True)
class CTMParams:
    dt: float = 5.0  # с
    fd: WeidmannFD = field(default_factory=WeidmannFD)
    fifo: bool = False  # узловая модель FIFO (по умолчанию — без FIFO)
    debug: bool = False  # проверки на каждом шаге (матрица долей, инварианты); в тестах — True
    # Мгновенный разворот при переносе назначения разрешён только на рёбрах длиной
    # ≤ max_uturn_cells · c_max · dt (в минимальных длинах ячейки).
    max_uturn_cells: float = 2.0
    # Плотность ожидания в накопителях (рёбра zone), чел/м²; None — накопители запрещены.
    queue_density: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "CTMParams":
        """Все ключи обязательны; единицы — в суффиксах ключей (sim/units.py)."""
        top = Section(d, "ctm")
        dt = top.quantity("dt", "time")
        fd_sec = Section(top.get("fundamental_diagram"), "fundamental_diagram")
        if fd_sec.get("model") != "weidmann":
            raise ValueError("fundamental_diagram.model: поддерживается только 'weidmann'")
        fd = WeidmannFD(v0=fd_sec.quantity("v0", "speed"),
                        rho_jam=fd_sec.quantity("rho_jam", "density"),
                        gamma=fd_sec.quantity("gamma", "inv_area"),
                        q_cap=fd_sec.quantity("q_max", "specific_flow"))
        fd_sec.finish()
        nm = Section(top.get("node_model"), "node_model")
        fifo = nm.get("fifo")
        if not isinstance(fifo, bool):
            raise ValueError("node_model.fifo: true или false")
        nm.finish()
        debug = top.get("debug")
        if not isinstance(debug, bool):
            raise ValueError("debug: true или false")
        tr = Section(top.get("transfer"), "transfer")
        max_uturn = float(tr.get("max_uturn_cells"))
        if max_uturn <= 0:
            raise ValueError("transfer.max_uturn_cells: должно быть > 0")
        tr.finish()
        # Раздел zones необязателен (исключение из правила «все ключи обязательны»):
        # без него сети с накопителями не принимаются (DECISIONS.md, этап 3а).
        zs = top.get("zones", None)
        queue_density = None
        if zs is not None:
            zsec = Section(zs, "zones")
            queue_density = zsec.quantity("queue_density", "density")
            zsec.finish()
            if not 0 < queue_density <= fd.rho_jam:
                raise ValueError("zones.queue_density: в (0, ρ_jam]")
        top.finish()
        return cls(dt=dt, fd=fd, fifo=fifo, debug=debug, max_uturn_cells=max_uturn,
                   queue_density=queue_density)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CTMParams":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


# --- состояние ------------------------------------------------------------------

@dataclass
class CTMState:
    """Все массивы имеют форму (B, P, ·): пакет, назначение, объект.

    ``committed`` — разбивка людей в последних ячейках рёбер и в очередях источников
    по выбранному движению узла «вход → выход» (выбор делается при входе в последнюю
    ячейку ребра или при выходе из источника); сумма по движениям входа равна
    числу людей в соответствующей ячейке / очереди. Нужна, чтобы люди к забитому
    выходу ждали его, а не перераспределялись (узел без FIFO)."""

    n: np.ndarray             # (B, P, C)  людей в ячейках
    committed: np.ndarray     # (B, P, M)  людей у входов узлов по выбранному движению
    source_queue: np.ndarray  # (B, P, Ns) очередь у источника (вышли, но не вошли в сеть)
    sink_buffer: np.ndarray   # (B, P, Nk) людей в накопителе стока
    generated: np.ndarray     # (B, P, Ns) накоплено выпущено источником
    served: np.ndarray        # (B, P, Nk) накоплено ушло через сток
    transferred: np.ndarray   # (B, P)     накоплено перенесено в слой назначения (Σ_p = 0)
    uturned: np.ndarray       # (B, P, E)  накоплено развернулось с ребра (только при переносе)
    fallback: np.ndarray      # (B, P)     накоплено выбрали движение по запасному правилу
    cell_outflow: np.ndarray  # (B, P, C)  людей вышло из ячейки за последний шаг
    step: int = 0

    @property
    def batch(self) -> int:
        return self.n.shape[0]

    def copy(self) -> "CTMState":
        return CTMState(self.n.copy(), self.committed.copy(), self.source_queue.copy(),
                        self.sink_buffer.copy(), self.generated.copy(), self.served.copy(),
                        self.transferred.copy(), self.uturned.copy(), self.fallback.copy(),
                        self.cell_outflow.copy(), self.step)

    def in_system_by_dest(self) -> np.ndarray:
        """(B, P) людей, вышедших из источников и ещё не ушедших через стоки."""
        return (self.n.sum(axis=-1) + self.source_queue.sum(axis=-1)
                + self.sink_buffer.sum(axis=-1))

    def in_system(self) -> np.ndarray:
        return self.in_system_by_dest().sum(axis=-1)


@dataclass(frozen=True)
class DestSwitchRules:
    """Правила смены назначения на входах узлов (``CTM.dest_switch_rules``): строка r —
    вход ``slot[r]`` (последняя ячейка ``cell[r]`` входящего ребра), перенос p[r] → q[r]."""
    slot: np.ndarray
    cell: np.ndarray
    p: np.ndarray
    q: np.ndarray

    @property
    def size(self) -> int:
        return self.slot.size

    @staticmethod
    def concat(parts: list["DestSwitchRules"]) -> "DestSwitchRules":
        cat = lambda name: np.concatenate([np.zeros(0, dtype=int)]
                                          + [getattr(r, name) for r in parts]).astype(int)
        return DestSwitchRules(cat("slot"), cat("cell"), cat("p"), cat("q"))


# --- модель ---------------------------------------------------------------------

class CTM:
    """Многопродуктовая пешеходная CTM на заданной сети.

    Значения по умолчанию (``source_inflow`` (P, Ns), ``net.split`` (P, E),
    ``sink_capacity`` (Nk,)) — описание сети без пакетной оси. Всё, что передаётся
    в ``step`` и возвращается из него, имеет пакетную ось первой: (B, P, ·) или (B, ·).
    """

    def __init__(self, network: Network, params: CTMParams):
        self.net = network
        self.params = params
        self.dt = params.dt
        self.fd = params.fd
        self.fifo = params.fifo
        self.cells = network.discretize(self.fd.max_wave_speed * self.dt)
        cells = self.cells
        C, E, P = cells.n_cells, len(network.edges), network.n_dest
        self.n_dest = P

        # Рёбра, на которых разрешён мгновенный разворот при переносе назначения.
        # Рёбра «площадка → вход станции» обязаны укладываться в порог: закрытие входа —
        # основной случай, когда людям с такого ребра нужно развернуться.
        self.max_uturn_length = params.max_uturn_cells * self.fd.max_wave_speed * self.dt
        lengths = np.array([e.length for e in network.edges])
        is_zone = np.array([e.zone for e in network.edges], dtype=bool)
        # Накопитель — одна ячейка без внутренней геометрии: разворот из него разрешён
        # при любой длине (люди уже стоят на площади у развилки).
        self.uturn_allowed = (lengths <= self.max_uturn_length * (1 + 1e-12)) | is_zone
        for v in network.nodes_of_type("sink"):
            if network.nodes[v].kind != "entrance":
                continue
            for k in network.in_edges(network.nodes[v].id):
                if not self.uturn_allowed[k]:
                    e = network.edges[k]
                    raise ValueError(
                        f"ребро {e.id} ко входу {e.target}: длина {e.length} м больше порога "
                        f"мгновенного разворота {self.max_uturn_length:.1f} м "
                        f"(max_uturn_cells = {params.max_uturn_cells}); укоротите ребро "
                        f"площадка → вход или вынесите площадку ближе ко входу")

        self.area = cells.area
        self.capacity_cell = self.fd.q_max * cells.width * self.dt  # чел/шаг
        self.width_dt = cells.width * self.dt

        # Накопители: ячейка с вместимостью A·ρ_queue; предложение — свободное движение
        # n·v0·dt/L (≤ пропускной способности), приём — свободное место (≤ пропускной
        # способности); плотность = люди / площадь ≤ ρ_queue.
        self.zone_cells = cells.first[is_zone]
        if self.zone_cells.size and params.queue_density is None:
            raise ValueError("в сети есть накопители (zone), а в параметрах CTM нет "
                             "zones.queue_density")
        qd = params.queue_density if params.queue_density is not None else self.fd.rho_jam
        self.cell_cap = self.fd.rho_jam * self.area                # чел, вместимость ячейки
        self.cell_cap[self.zone_cells] = qd * self.area[self.zone_cells]
        self.zone_cap = self.cell_cap[self.zone_cells]
        self.zone_free = np.minimum(self.fd.v0 * self.dt / cells.length[self.zone_cells], 1.0)

        self.source_nodes = network.nodes_of_type("source")
        self.sink_nodes = network.nodes_of_type("sink")
        src_pos = {v: k for k, v in enumerate(self.source_nodes)}
        snk_pos = {v: k for k, v in enumerate(self.sink_nodes)}
        total = np.array([network.nodes[v].inflow for v in self.source_nodes])
        self.source_inflow = network.source_shares * total  # (P, Ns), чел/с
        self.sink_capacity = np.array([network.nodes[v].capacity for v in self.sink_nodes])
        self.sink_buffer_cap = np.array([network.nodes[v].buffer for v in self.sink_nodes])
        self.sink_dest = np.array([network.dest_index[network.sink_destination(v)]
                                   for v in self.sink_nodes])

        # Внутренние переходы ячейка k → k+1 внутри рёбер.
        is_internal = np.ones(C, dtype=bool)
        is_internal[cells.last] = False
        self.int_up = np.flatnonzero(is_internal)
        self.int_down = self.int_up + 1

        # Узловые слоты. Вход — последняя ячейка входящего ребра или источник; выход —
        # первая ячейка исходящего ребра или сток. Движение — пара (вход, выход) одного
        # узла, кроме разворотов. mov_src — строка матрицы долей: ребро или E (сток, β = 1).
        in_cell, in_src, out_cell, out_snk = [], [], [], []
        in_edge, mov_in, mov_out, mov_src = [], [], [], []
        n_out = 0
        for v, node in enumerate(network.nodes):
            ins = ([("src", src_pos[v], -1)] if node.type == "source" else []) + \
                  [("cell", cells.last[e], e) for e in network.in_edges(node.id)]
            outs = ([("snk", snk_pos[v], E)] if node.type == "sink" else []) + \
                   [("cell", cells.first[e], e) for e in network.out_edges(node.id)]
            o0 = n_out
            for kind, x, e in outs:
                (out_cell if kind == "cell" else out_snk).append((n_out, x))
                n_out += 1
            for kind, x, ei in ins:
                i = len(in_edge)
                (in_cell if kind == "cell" else in_src).append((i, x))
                in_edge.append(ei)
                for o, (_, _, eo) in enumerate(outs):
                    if ei >= 0 and eo < E and network.is_uturn(ei, eo):
                        continue
                    mov_in.append(i); mov_out.append(o0 + o); mov_src.append(eo)
        self.n_in, self.n_out = len(in_edge), n_out
        self.in_cell_slot, self.in_cell = (np.array(a) for a in zip(*in_cell))
        self.out_cell_slot, self.out_cell = (np.array(a) for a in zip(*out_cell))
        self.source_slot = np.array([i for i, _ in sorted(in_src, key=lambda t: t[1])], dtype=int)
        self.sink_slot = np.array([o for o, _ in sorted(out_snk, key=lambda t: t[1])], dtype=int)
        in_edge = np.array(in_edge)
        self.edge_in_slot = np.empty(E, dtype=int)
        self.edge_in_slot[in_edge[in_edge >= 0]] = np.flatnonzero(in_edge >= 0)

        self.mov_in = np.array(mov_in)     # отсортированы по входу
        self.mov_out = np.array(mov_out)
        self.mov_src = np.array(mov_src)
        M = self.mov_in.size
        self.mov_in_starts = _segment_starts(self.mov_in, self.n_in)
        self.A_out = np.zeros((M, n_out))  # суммирование по выходам: (…, M) @ A_out
        self.A_out[np.arange(M), self.mov_out] = 1.0

        # Движения входа, дополненные до K (для обхода затора).
        counts = np.bincount(self.mov_in, minlength=self.n_in)
        K = int(counts.max())
        k_of = np.arange(M) - self.mov_in_starts[self.mov_in]
        self.pad_idx = np.zeros((self.n_in, K), dtype=int)
        self.pad_mask = np.zeros((self.n_in, K), dtype=bool)
        self.pad_idx[self.mov_in, k_of] = np.arange(M)
        self.pad_mask[self.mov_in, k_of] = True

        # reach[p, m]: по выходу движения m можно дойти до назначения p.
        self.reach = np.zeros((P, M), dtype=bool)
        dest_of_sink_slot = dict(zip(self.sink_slot.tolist(), self.sink_dest.tolist()))
        for m, e in enumerate(self.mov_src):
            if e < E:
                self.reach[:, m] = [network.edge_reaches(e, p) for p in range(P)]
            else:
                self.reach[dest_of_sink_slot[self.mov_out[m]], m] = True
        # Запасное правило выбора (нет доли β для своего назначения на этом входе —
        # например, маршрут сменился, пока человек шёл по ребру):
        # 1) привычные доли (матрица сети net.split), нормированные по разрешённым движениям;
        # 2) если и их нет — кратчайшее разрешённое движение к своей станции (поровну при равенстве);
        # 3) если пути к своей станции нет — явный перенос назначения на ближайшую
        #    достижимую станцию (redirect), затем выбор по её долям.
        cost = np.full((P, M), np.inf)
        for m, e in enumerate(self.mov_src):
            if e < E:
                cost[:, m] = network.edges[e].length + network.edge_dist[:, e]
            else:
                cost[dest_of_sink_slot[self.mov_out[m]], m] = 0.0
        raw_h = self._beta_raw(network.split)
        den_h = np.add.reduceat(raw_h, self.mov_in_starts, axis=-1)
        hab = np.zeros((P, M))
        redirect = []
        for i in range(self.n_in):
            ms = np.flatnonzero(self.mov_in == i)
            for p in range(P):
                if den_h[p, i] > 0:
                    hab[p, ms] = raw_h[p, ms] / den_h[p, i]
                elif np.isfinite(cost[p, ms]).any():
                    best = ms[cost[p, ms] <= cost[p, ms].min() * (1 + 1e-12) + 1e-9]
                    hab[p, best] = 1.0 / best.size
                elif i in self.in_cell_slot:
                    q = int(np.argmin(cost[:, ms].min(axis=1)))  # ближайшая достижимая станция
                    redirect.append((p, i, q))
        self.habitual_alloc = hab
        self.src_reachable = np.array([[np.isfinite(cost[p, self.mov_in == i]).any()
                                        for i in self.source_slot] for p in range(P)])
        self.redir_p, self.redir_slot, self.redir_q = (np.array(a, dtype=int).reshape(-1)
                                                       for a in zip(*redirect)) if redirect \
            else (np.zeros(0, dtype=int),) * 3
        self.redir_mask = np.zeros((P, self.n_in), dtype=bool)
        self.redir_mask[self.redir_p, self.redir_slot] = True
        self.default_alloc, self.default_fb = self._alloc(network.split)

    def _beta_raw(self, split: np.ndarray) -> np.ndarray:
        """Доли по движениям (…, P, M) из матрицы долей рёбер (…, P, E), без нормировки."""
        ones = np.ones(split.shape[:-1] + (1,))
        return np.concatenate([split, ones], axis=-1)[..., self.mov_src]

    def _alloc(self, split: np.ndarray):
        """Доли выбора движения на каждом входе (…, P, M): β выхода, нормированные
        по разрешённым движениям входа (развороты исключены); где их нет — запасное
        правило. Второй результат (…, P, n_in) — где применено запасное правило."""
        raw = self._beta_raw(split)
        den = np.add.reduceat(raw, self.mov_in_starts, axis=-1)
        d = den[..., self.mov_in]
        alloc = np.where(d > 0, raw / np.where(d > 0, d, 1.0), self.habitual_alloc)
        return alloc, (den <= 0) | self.redir_mask

    @staticmethod
    def _batched(x, shape, name):
        x = np.asarray(x, dtype=float)
        if x.shape != shape:
            raise ValueError(f"{name}: ожидается форма {shape} (пакет первой осью), а не {x.shape}")
        return x

    # --- интерфейс --------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return self.cells.n_cells

    @property
    def n_mov(self) -> int:
        return self.mov_in.size

    def dest_switch_rules(self, node: str, from_dest: str, to_dest: str) -> DestSwitchRules:
        """Правила смены назначения ``from_dest`` → ``to_dest`` на всех входящих рёбрах
        развилки ``node``. С каждого входа до ``to_dest`` должно быть разрешённое движение
        (без разворота) — иначе ошибка."""
        net = self.net
        p, q = net.dest_index[from_dest], net.dest_index[to_dest]
        if p == q:
            raise ValueError(f"смена назначения {from_dest} → {to_dest}: одно и то же")
        slots = []
        for k in net.in_edges(node):
            i = int(self.edge_in_slot[k])
            if not self.reach[q, self.mov_in == i].any():
                raise ValueError(f"узел {node}: с ребра {net.edges[k].id} до {to_dest} "
                                 f"не дойти без разворота")
            slots.append(i)
        slot = np.array(slots, dtype=int)
        cell = self.in_cell[np.searchsorted(self.in_cell_slot, slot)]
        return DestSwitchRules(slot, cell, np.full(slot.size, p), np.full(slot.size, q))

    def initial_state(self, batch: int = 1) -> CTMState:
        C, Ns, Nk, P = self.n_cells, len(self.source_nodes), len(self.sink_nodes), self.n_dest
        z = lambda k: np.zeros((batch, P, k))
        E = len(self.net.edges)
        return CTMState(z(C), z(self.n_mov), z(Ns), z(Nk), z(Ns), z(Nk),
                        np.zeros((batch, P)), z(E), np.zeros((batch, P)), z(C))

    def density(self, state: CTMState) -> np.ndarray:
        """(B, C) суммарная плотность, чел/м²."""
        return state.n.sum(axis=1) / self.area

    def batch(self, x: np.ndarray, B: int) -> np.ndarray:
        """Значение без пакетной оси, размноженное на пакет из B прогонов."""
        x = np.asarray(x, dtype=float)
        return np.broadcast_to(x, (B,) + x.shape).copy()

    def step(self, state: CTMState, *, inflow=None, split=None, sink_capacity=None,
             detour_rate=None, dest_switch=None) -> CTMState:
        """Один шаг dt. Необязательные переопределения — с пакетной осью первой:
        ``inflow`` (B, P, Ns) — поток источников по назначениям, чел/с;
        ``split`` (B, P, E) — матрица долей маршрутов: эффективные доли с учётом
        рекомендаций и степени следования c; применяется в момент фиксации выбора
        (вход в последнюю ячейку ребра / выход из источника) и не меняет выбор уже
        зафиксировавших. Проверяется ``Network.validate_split`` только в режиме debug;
        ``sink_capacity`` (B, Nk) — пропускная способность стоков, чел/с;
        ``detour_rate`` (B,) — интенсивность обхода затора θ, 1/с (sim/behavior.py);
        ``dest_switch`` — пара (``DestSwitchRules``, доли (B, R)): доля входящих в последнюю
        ячейку ребра людей слоя p, меняющих назначение на q (на одном входе и слое p
        сумма долей ≤ 1)."""
        dt = self.dt
        n = state.n
        B, P = n.shape[0], self.n_dest
        Ns, Nk, E = len(self.source_nodes), len(self.sink_nodes), len(self.net.edges)
        lam = (self.source_inflow if inflow is None
               else self._batched(inflow, (B, P, Ns), "inflow"))
        if np.any((lam > 0) & ~self.src_reachable):
            raise ValueError("inflow: поток в назначение, недостижимое из источника")
        if split is None:
            alloc, fb = self.default_alloc, self.default_fb
        else:
            split = self._batched(split, (B, P, E), "split")
            if self.params.debug:
                for b in range(B):
                    self.net.validate_split(split[b])
            alloc, fb = self._alloc(split)
        mu_dt = (self.sink_capacity if sink_capacity is None
                 else self._batched(sink_capacity, (B, Nk), "sink_capacity")) * dt

        ntot = n.sum(axis=1)                                   # (B, C)
        rho = ntot / self.area
        q = self.fd.flow(rho) * self.width_dt
        free = rho <= self.fd.rho_crit
        S = np.minimum(np.minimum(np.where(free, q, np.inf), self.capacity_cell), ntot)
        R = np.minimum(np.minimum(np.where(free, np.inf, q), self.capacity_cell),
                       (self.fd.rho_jam - rho) * self.area)
        if self.zone_cells.size:
            z = self.zone_cells
            nz = ntot[:, z]
            S[:, z] = np.minimum(nz * self.zone_free, self.capacity_cell[z])
            R[:, z] = np.maximum(np.minimum(self.capacity_cell[z], self.zone_cap - nz), 0.0)

        # Внутренние переходы: доля r = f / n уходит из ячейки, состав сохраняется.
        f_int = np.minimum(S[:, self.int_up], R[:, self.int_down])
        r = np.zeros_like(ntot)
        n_up = ntot[:, self.int_up]
        r[:, self.int_up] = np.divide(f_int, n_up, out=np.zeros_like(f_int), where=n_up > 0)

        # Новые вышедшие из источников фиксируют выбор движения.
        arrivals = np.broadcast_to(lam * dt, state.source_queue.shape)
        S_src = state.source_queue + arrivals
        add_in = np.zeros((B, P, self.n_in))
        add_in[..., self.source_slot] = arrivals
        fallback = state.fallback + (add_in * fb).sum(axis=-1)
        com = state.committed + add_in[..., self.mov_in] * alloc

        # Узел: спрос движения = доля предложения входа × выбравшие движение.
        k_in = np.ones((B, self.n_in))
        nt_last = ntot[:, self.in_cell]
        k_in[:, self.in_cell_slot] = np.divide(S[:, self.in_cell], nt_last,
                                               out=np.zeros_like(nt_last), where=nt_last > 0)
        D_mov_p = com * k_in[:, None, self.mov_in]             # (B, P, M)
        D_mov = D_mov_p.sum(axis=1)
        D_out = D_mov @ self.A_out
        R_out = np.empty((B, self.n_out))
        R_out[:, self.out_cell_slot] = R[:, self.out_cell]
        # ≥ 0: при μ = 0 (закрытый вход) остаток округления в накопителе (~1e-15) иначе
        # даёт отрицательный приём и φ = 0/0 (DECISIONS.md, этап 2а)
        R_out[:, self.sink_slot] = np.maximum(
            mu_dt + (self.sink_buffer_cap - state.sink_buffer.sum(axis=1)), 0.0)
        phi = np.ones_like(D_out)
        np.divide(R_out, D_out, out=phi, where=D_out > R_out)  # φ = min(1, R/D)
        factor = phi[:, self.mov_out]
        if self.fifo:
            factor = np.where(D_mov > 0, factor, 1.0)
            theta = np.minimum.reduceat(factor, self.mov_in_starts, axis=-1)
            factor = theta[:, self.mov_in]
        F = D_mov_p * factor[:, None, :]
        f_in = np.add.reduceat(F, self.mov_in_starts, axis=-1)
        f_out = F @ self.A_out
        com = com - F

        if detour_rate is not None:
            rate = self._batched(detour_rate, (B,), "detour_rate")
            if np.any(rate > 0):
                pi, pm = self.pad_idx, self.pad_mask
                blocked = (phi[:, self.mov_out] < 1.0)[:, pi] & pm
                weight = np.broadcast_to(alloc, com.shape)[..., pi]
                change = detour_transfer(com[..., pi] * pm, blocked, self.reach[:, pi] & pm,
                                         weight, 1.0 - np.exp(-rate * dt))
                com[..., pi[pm]] += change[..., pm]

        # Выход ячейки; вход внутренней ячейки = выход предыдущей (сдвиг на одну ячейку).
        out = n * r[:, None, :]
        out[..., self.in_cell] = f_in[..., self.in_cell_slot]
        inn = np.empty_like(n)
        inn[..., 1:] = out[..., :-1]
        inn[..., self.out_cell] = f_out[..., self.out_cell_slot]
        n_new = n - out
        n_new += inn

        # Вошедшие в последнюю ячейку ребра фиксируют выбор движения. У кого нет пути
        # к своей станции — явный перенос назначения на ближайшую достижимую.
        add_in = np.zeros((B, P, self.n_in))
        add_in[..., self.in_cell_slot] = inn[..., self.in_cell]
        transferred = state.transferred
        if dest_switch is not None:
            rules, frac = dest_switch
            frac = self._batched(frac, (B, rules.size), "dest_switch")
            if np.any((frac < 0) | (frac > 1)):
                raise ValueError("dest_switch: доли в [0, 1]")
            if np.any(frac > 0):
                moved = add_in[:, rules.p, rules.slot] * frac          # (B, R)
                np.add.at(add_in, (slice(None), rules.p, rules.slot), -moved)
                np.add.at(add_in, (slice(None), rules.q, rules.slot), moved)
                np.add.at(n_new, (slice(None), rules.p, rules.cell), -moved)
                np.add.at(n_new, (slice(None), rules.q, rules.cell), moved)
                transferred = transferred.copy()
                np.add.at(transferred, (slice(None), rules.p), -moved)
                np.add.at(transferred, (slice(None), rules.q), moved)
        fallback += (add_in * fb).sum(axis=-1)
        if self.redir_p.size:
            moved = add_in[:, self.redir_p, self.redir_slot]   # (B, R)
            if np.any(moved > 0):
                cells = self.in_cell[np.searchsorted(self.in_cell_slot, self.redir_slot)]
                add_in[:, self.redir_p, self.redir_slot] = 0.0
                np.add.at(add_in, (slice(None), self.redir_q, self.redir_slot), moved)
                np.add.at(n_new, (slice(None), self.redir_p, cells), -moved)
                np.add.at(n_new, (slice(None), self.redir_q, cells), moved)
                transferred = transferred.copy()
                np.add.at(transferred, (slice(None), self.redir_p), -moved)
                np.add.at(transferred, (slice(None), self.redir_q), moved)
        com += add_in[..., self.mov_in] * alloc

        buf = state.sink_buffer + f_out[..., self.sink_slot]
        btot = buf.sum(axis=1)
        frac = np.zeros_like(btot)
        np.divide(np.minimum(btot, mu_dt), btot, out=frac, where=btot > 0)
        served_now = buf * frac[:, None, :]

        new = CTMState(
            n=n_new,
            committed=com,
            source_queue=S_src - f_in[..., self.source_slot],
            sink_buffer=buf - served_now,
            generated=state.generated + arrivals,
            served=state.served + served_now,
            transferred=transferred,
            uturned=state.uturned,
            fallback=fallback,
            cell_outflow=out,
            step=state.step + 1,
        )
        if self.params.debug:
            self.check_invariants(new)
        return new

    def check_invariants(self, s: CTMState, tol: float = 1e-8) -> None:
        """Проверки режима debug: неотрицательность, ρ ≤ ρ_jam, согласованность
        ``committed`` с ячейками/очередями, баланс по назначениям."""
        def fail(msg):
            raise AssertionError(f"CTM debug, шаг {s.step}: {msg}")
        if s.n.min() < -tol or s.committed.min() < -tol or s.source_queue.min() < -tol:
            fail("отрицательное число людей")
        if self.density(s).max() > self.fd.rho_jam * (1 + 1e-9):
            fail("плотность выше ρ_jam")
        if self.zone_cells.size and np.any(
                s.n.sum(axis=1)[:, self.zone_cells] > self.zone_cap * (1 + 1e-9) + tol):
            fail("накопитель переполнен (плотность выше ρ_queue)")
        by_in = np.add.reduceat(s.committed, self.mov_in_starts, axis=-1)
        scale = max(1.0, float(s.generated.sum(axis=(1, 2)).max()))
        if (np.abs(by_in[..., self.in_cell_slot] - s.n[..., self.in_cell]).max() > tol * scale
                or np.abs(by_in[..., self.source_slot] - s.source_queue).max() > tol * scale):
            fail("committed не согласован с последними ячейками / очередями источников")
        lhs = s.generated.sum(axis=-1) + s.transferred
        rhs = s.in_system_by_dest() + s.served.sum(axis=-1)
        if np.abs(lhs - rhs).max() > tol * scale:
            fail("нарушен баланс людей по назначениям")

    def transfer_destination(self, state: CTMState, from_dest: str, to_dest: str,
                             fraction, edges: list[str], split=None) -> CTMState:
        """Явный перенос доли ``fraction`` (скаляр или (B,)) людей слоя ``from_dest``
        в слой ``to_dest`` на рёбрах ``edges`` — например, смена станции при закрытом
        входе. Учитывается в ``state.transferred``.

        - Если с ребра до ``to_dest`` можно дойти без разворота — люди остаются на месте,
          в последней ячейке заново выбирают движение по долям ``to_dest``.
        - Иначе — разворот (единственное место, где он разрешён): люди с ребра u→v
          возвращаются в u и входят в первые ячейки рёбер из u (кроме самого ребра),
          ведущих к ``to_dest``, по долям ``to_dest`` (при их отсутствии — кратчайшим).
          Входит столько, сколько помещается (ρ ≤ ρ_jam); остальные остаются в прежнем
          слое — перенос можно повторить на следующих шагах. Развернувшиеся учитываются
          в ``state.uturned``. Перемещение мгновенное, поэтому разрешено только на рёбрах
          длиной ≤ ``max_uturn_length`` (иначе ошибка).
        """
        net = self.net
        p, q = net.dest_index[from_dest], net.dest_index[to_dest]
        B, P, E = state.batch, self.n_dest, len(net.edges)
        f = np.broadcast_to(np.asarray(fraction, dtype=float), (B,))
        if np.any((f < 0) | (f > 1)):
            raise ValueError("fraction: доля в [0, 1]")
        if not edges:
            raise ValueError("transfer_destination: укажите рёбра")
        split = (np.broadcast_to(net.split, (B, P, E)) if split is None
                 else self._batched(split, (B, P, E), "split"))
        alloc, _ = self._alloc(split)
        new = state.copy()
        first, last = self.cells.first, self.cells.last

        def remove(k, frac_b):
            """Убирает долю frac_b (B,) слоя p с ребра k; возвращает (B,) убранных."""
            cells = np.arange(first[k], last[k] + 1)
            moved = frac_b[:, None] * new.n[:, p, cells]
            new.n[:, p, cells] -= moved
            movs = np.flatnonzero(self.mov_in == self.edge_in_slot[k])
            new.committed[:, p, movs] *= (1.0 - frac_b)[:, None]
            return moved, cells

        def commit(k, amount):
            """Люди слоя q (B,), вошедшие в последнюю ячейку ребра k, выбирают движение."""
            movs = np.flatnonzero(self.mov_in == self.edge_in_slot[k])
            new.committed[:, q, movs] += amount[:, None] * alloc[:, q, movs]

        for eid in edges:
            k = net.edge_index[eid]
            if net.edge_reaches(k, q):
                moved, cells = remove(k, f)
                new.n[:, q, cells] += moved
                commit(k, moved[:, -1])
                total = moved.sum(axis=1)
            else:
                u = net.edges[k].source
                targets = [j for j in net.out_edges(u) if j != k and net.edge_reaches(j, q)]
                if not targets:
                    raise ValueError(f"ребро {eid}: до {to_dest} не дойти даже разворотом")
                if not self.uturn_allowed[k]:
                    raise ValueError(
                        f"ребро {eid}: до {to_dest} только разворотом, а длина ребра "
                        f"{net.edges[k].length} м больше порога мгновенного разворота "
                        f"{self.max_uturn_length:.1f} м")
                w = split[:, q, targets]
                ws = w.sum(axis=1, keepdims=True)
                L = np.array([net.edges[j].length + net.edge_dist[q, j] for j in targets])
                w_short = (L <= L.min() * (1 + 1e-12) + 1e-9).astype(float)
                w = np.where(ws > 0, w / np.where(ws > 0, ws, 1.0), w_short / w_short.sum())
                want = f * new.n[:, p, first[k]:last[k] + 1].sum(axis=1)       # (B,)
                c_t = first[targets]
                room = (self.cell_cap[c_t] - new.n[:, :, c_t].sum(axis=1))
                need = want[:, None] * w
                ratio = np.ones(B)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    lim = np.where(need > 0, np.maximum(room, 0.0) / need, np.inf).min(axis=1)
                ratio = np.minimum(1.0, lim)
                moved, _ = remove(k, f * ratio)
                total = moved.sum(axis=1)
                for t, j in enumerate(targets):
                    add = total * w[:, t]
                    new.n[:, q, first[j]] += add
                    if first[j] == last[j]:
                        commit(j, add)
                new.uturned[:, p, k] += total
            new.transferred[:, p] -= total
            new.transferred[:, q] += total
        if self.params.debug:
            self.check_invariants(new)
        return new

    def run(self, state: CTMState, n_steps: int, *, record: bool = False, **controls):
        """Прогон на ``n_steps`` шагов с постоянными управлениями. Возвращает конечное
        состояние и (если ``record``) историю суммарного числа людей в ячейках (T, B, C)."""
        hist = np.empty((n_steps, state.batch, self.n_cells)) if record else None
        for t in range(n_steps):
            state = self.step(state, **controls)
            if record:
                hist[t] = state.n.sum(axis=1)
        return state, hist


def _segment_starts(seg: np.ndarray, n_seg: int) -> np.ndarray:
    """Начала сегментов в отсортированном массиве номеров; каждый сегмент непуст."""
    if np.any(np.diff(seg) < 0) or not np.array_equal(np.unique(seg), np.arange(n_seg)):
        raise ValueError("у каждого узла должен быть хотя бы один вход и выход")
    return np.searchsorted(seg, np.arange(n_seg))
