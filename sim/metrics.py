"""Метрики прогона (постановка задачи, «Метрики»).

Запись траектории — ``Recorder``: снимок состояния раз в интервал управления Δt
(и в момент t = 0). Метрики эпизода — ``episode_metrics(trajectory, cfg)``, все — массивы
по пакету (B,):

- ``clearance_time`` — время рассасывания: первый момент, после которого в районе
  (на стадионе + на сети + внутри станций) всё время остаётся < ``clearance_fraction``·N
  зрителей; линейная интерполяция между снимками; ``inf`` — не рассосалось за эпизод;
- ``mean_time``, ``p95_time`` — время от выхода со стадиона до ухода из района (посадка
  в поезд / шаттл или уход через сток ``exit``). Среднее — по формуле Литтла
  ∫(A − D)dt / A(T) (не зависит от порядка обслуживания); p95 — по горизонтальному
  расстоянию между накопленными кривыми выхода A(t) и ухода D(t), то есть в приближении
  FIFO по всему району (обгоны между направлениями не учитываются). Не ушедшие к концу
  эпизода (``censored``) учитываются со временем «до конца эпизода» — оценка снизу;
- ``max_density`` — наибольшая плотность ячейки по всем участкам за эпизод;
  ``max_density_edge`` (B, E) — по участкам;
- перегрузки: участок перегружен, если плотность хотя бы одной его ячейки > ρ_crit;
  порог — по типу элемента: улицы и зоны ожидания (накопители) — свой (configs/metrics.yaml).
  ``overload_duration`` — суммарная длительность, участок·с, ``overload_time`` — время,
  когда перегружен хоть один участок, ``overload_any`` — была ли перегрузка,
  ``overload_count`` — число эпизодов перегрузки (связных по интервалам управления
  отрезков) по всем участкам. Доля эпизодов с перегрузкой — среднее ``overload_any``;
- экспозиция ``exposure`` — человеко-секунды в ячейках с ρ > порога своего типа (``exposure_edge``
  (B, E) — по участкам); ``area_above_max`` — наибольшая площадь ячеек с ρ > ρ_crit, м²,
  ``area_above_time`` — её интеграл по времени, м²·с.

Максимум плотности, экспозиция, площадь и длительности перегрузки считаются на каждом шаге
CTM (``Simulator.step``, ``SimState.interval``) и сохраняются агрегатами раз в Δt; число
эпизодов перегрузки — по интервалам Δt (docs/DECISIONS.md, этап 3а). Порог ρ_crit
симулятора и метрик обязан совпадать.

Очередь у входа станции (``queue_outside``) — люди слоя этой станции в связной области
перед входом: от двери вверх по течению, пока ячейки в заторе (ρ > порога); на развилках
вверх по течению — по всем входящим рёбрам (без разворотов). Накопитель (площадь перед
входом) входит в область всегда (люди станции в нём ждут), а область продолжается выше
него, только когда он заполнен (≥ 99 % вместимости). Люди внутри станции —
``station_links`` (по звеньям).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from sim.ctm import CTM, CTMState
from sim.units import Section

ROOT = Path(__file__).resolve().parents[1]
METRICS_CONFIG = ROOT / "configs" / "metrics.yaml"


# --- контроль маршрутизации (этапы 1–2) --------------------------------------------------

def sink_arrivals(state: CTMState) -> np.ndarray:
    """(B, P, Nk) накоплено пришло в сток (обслужено + в накопителе) по назначениям."""
    return state.served + state.sink_buffer


def foreign_arrivals(ctm: CTM, state: CTMState) -> np.ndarray:
    """(B, Nk) накоплено «чужих» прибытий: людей, пришедших в сток с назначением,
    отличным от назначения стока. При корректной маршрутизации — 0; смена станции
    допускается только явным переносом между слоями (``CTM.transfer_destination``)."""
    foreign = np.arange(ctm.n_dest)[:, None] != ctm.sink_dest[None, :]  # (P, Nk)
    return (sink_arrivals(state) * foreign).sum(axis=1)


def uturns(state: CTMState) -> np.ndarray:
    """(B,) накоплено развернувшихся. Разворот возможен только при явном переносе
    назначения (``CTM.transfer_destination``) — вне его счётчик не растёт."""
    return state.uturned.sum(axis=(1, 2))


def fallback_mass(ctm: CTM, state: CTMState) -> np.ndarray:
    """(B,) накоплено людей, выбравших движение по запасному правилу (нет доли для
    своего назначения → привычные доли / кратчайшая ветвь / перенос на ближайшую
    достижимую станцию). В штатных сценариях — 0."""
    return state.fallback.sum(axis=1)


# --- параметры метрик ----------------------------------------------------------------------

@dataclass(frozen=True)
class MetricsConfig:
    congestion_street: float     # порог затора улиц (ячейки CTM), чел/м²
    congestion_zone: float       # порог затора зон ожидания (накопители), чел/м²
    rho_crit_street: float       # порог опасной плотности улиц (движущаяся толпа), чел/м²
    rho_crit_zone: float         # порог опасной плотности зон (стоячая толпа), чел/м²
    zone_full_fraction: float    # зона «заполнена» при n ≥ доли вместимости
    clearance_fraction: float    # доля зрителей, ниже которой район считается свободным
    queue_density: float | None  # порог затора для очереди у входа; None — ρc диаграммы

    @property
    def rho_crit(self) -> tuple[float, float]:
        """Порог опасной плотности (улицы, зоны ожидания) — ограничение MPC и перегрузка."""
        return self.rho_crit_street, self.rho_crit_zone

    @property
    def congestion(self) -> tuple[float, float]:
        """Порог затора (улицы, зоны ожидания)."""
        return self.congestion_street, self.congestion_zone

    @classmethod
    def from_dict(cls, d: dict) -> "MetricsConfig":
        sec = Section(d, "metrics")
        cfg = cls(sec.quantity("congestion_street", "density"),
                  sec.quantity("congestion_zone", "density"),
                  sec.quantity("rho_crit_street", "density"),
                  sec.quantity("rho_crit_zone", "density"),
                  float(sec.get("zone_full_fraction")),
                  float(sec.get("clearance_fraction")), sec.quantity("queue_density", "density"))
        sec.finish()
        if not (0 < cfg.congestion_street < cfg.rho_crit_street
                and 0 < cfg.congestion_zone < cfg.rho_crit_zone):
            raise ValueError("metrics: 0 < порог затора < порога опасной плотности")
        if not (0 < cfg.clearance_fraction < 1 and 0 < cfg.zone_full_fraction <= 1):
            raise ValueError("metrics: clearance_fraction в (0, 1), zone_full_fraction в (0, 1]")
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path = METRICS_CONFIG) -> "MetricsConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


# --- траектория ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """Снимки эпизода раз в Δt, включая t = 0. Ось 0 — время (T), ось 1 — пакет (B)."""
    t: np.ndarray               # (T,) с от окончания матча
    edge_max: np.ndarray        # (T, B, E) наибольшая плотность ячейки участка за интервал
                                #           (по шагам CTM), чел/м²
    edge_mean: np.ndarray       # (T, B, E) средняя плотность участка в снимке (люди / площадь)
    exposure: np.ndarray        # (T, B, E) человеко-секунды при ρ > ρ_крит за интервал
    over_time: np.ndarray       # (T, B, E) время перегрузки (ρ > ρ_крит) участка, с
    any_over_time: np.ndarray   # (T, B) время, когда перегружен хоть один участок, с
    area_above: np.ndarray      # (T, B) наибольшая площадь ячеек с ρ > ρ_крит за интервал, м²
    area_above_time: np.ndarray  # (T, B) ∫ площади ячеек с ρ > ρ_крит за интервал, м²·с
    exposure_congestion: np.ndarray  # (T, B, E) человеко-секунды при ρ > порога затора
    area_congestion: np.ndarray  # (T, B) наибольшая площадь ячеек выше порога затора, м²
    zone_full_time: np.ndarray  # (T, B, Z) время заполнения зоны ≥ zone_full_fraction, с
    in_stadium: np.ndarray      # (T, B) ещё на стадионе
    at_gates: np.ndarray        # (T, B) вышли, но не вошли в сеть (очередь источника)
    on_streets: np.ndarray      # (T, B) в ячейках сети и в накопителях стоков
    in_stations: np.ndarray     # (T, B) внутри станций
    released: np.ndarray        # (T, B) накоплено вышло со стадиона
    left: np.ndarray            # (T, B) накоплено ушло из района (уехало + ушло через exit)
    departed: np.ndarray        # (T, B, S) накоплено уехало по станциям
    station_links: np.ndarray   # (T, B, L) людей в звеньях станций
    queue_outside: np.ndarray   # (T, B, S) очередь перед входом станции
    total: np.ndarray           # (B,) всего зрителей (начальный пул стадиона)
    cell_density: np.ndarray | None = None  # (T, B, C) — если записывались ячейки

    @property
    def on_network(self) -> np.ndarray:
        """(T, B) в районе вне стадиона и станций: у выходов + на улицах."""
        return self.at_gates + self.on_streets

    @property
    def in_area(self) -> np.ndarray:
        """(T, B) ещё в районе: на стадионе + на сети + внутри станций."""
        return self.in_stadium + self.on_network + self.in_stations

    def save(self, path: str | Path, **extra) -> None:
        arrays = {f.name: getattr(self, f.name) for f in fields(self)
                  if getattr(self, f.name) is not None}
        np.savez_compressed(path, **arrays, **extra)

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        with np.load(path) as z:
            return cls(**{f.name: (z[f.name] if f.name in z else None) for f in fields(cls)})

    def select(self, b) -> "Trajectory":
        """Подпакет по индексам/маске эпизодов."""
        out = {}
        for f in fields(self):
            x = getattr(self, f.name)
            if x is None or f.name == "t":
                out[f.name] = x
            elif f.name == "total":
                out[f.name] = x[b]
            else:
                out[f.name] = x[:, b]
        return Trajectory(**out)

    @staticmethod
    def concat(parts: list["Trajectory"]) -> "Trajectory":
        """Объединение по оси пакета (одинаковые моменты времени)."""
        for p in parts[1:]:
            if not np.array_equal(p.t, parts[0].t):
                raise ValueError("Trajectory.concat: разные моменты времени")
        out = {}
        for f in fields(Trajectory):
            xs = [getattr(p, f.name) for p in parts]
            if f.name == "t":
                out[f.name] = xs[0]
            elif any(x is None for x in xs):
                out[f.name] = None
            else:
                out[f.name] = np.concatenate(xs, axis=0 if f.name == "total" else 1)
        return Trajectory(**out)


class Recorder:
    """Запись снимков эпизода симулятора (``sim.simulator.Simulator``).

        rec = Recorder(sim, cfg)
        rec.record(state)              # t = 0
        for k in ...: state, obs = sim.step(state, a, rng); rec.record(state)
        traj = rec.result()
    """

    def __init__(self, sim, cfg: MetricsConfig | None = None, *, cells: bool = False):
        self.sim = sim
        self.cfg = cfg or MetricsConfig.from_yaml()
        if self.cfg != sim.metrics_cfg:
            raise ValueError("пороги метрик не совпадают с порогами симулятора: создайте "
                             "Simulator(metrics=…) с тем же MetricsConfig")
        self.cells = cells
        ctm, net = sim.ctm, sim.net
        E = len(net.edges)
        self.edge_of_cell = ctm.cells.edge
        self.edge_starts = ctm.cells.first
        self.inv_edge_area = 1.0 / np.bincount(self.edge_of_cell, weights=ctm.area, minlength=E)
        self.queue_density = (ctm.fd.rho_crit if self.cfg.queue_density is None
                              else self.cfg.queue_density)
        self.queue_trees = [self._upstream_tree(eid) for eid in sim.stations.ent_ids]
        self.is_zone = np.array([e.zone for e in net.edges], dtype=bool)
        self.cell_cap = ctm.cell_cap
        self.station_ent = [[e for e, s in enumerate(sim.stations.ent_station) if s == st]
                            for st in range(sim.stations.n_stations)]
        self.snaps: list[dict] = []
        self.total = None

    def _upstream_tree(self, entrance: str) -> list[tuple[int, int]]:
        """Рёбра вверх по течению от входа в порядке обхода в ширину: (ребро, родитель),
        у рёбер входа родитель −1. Развороты (u→v после v→u) исключены, каждое ребро —
        один раз."""
        net = self.sim.net
        order = [(k, -1) for k in net.in_edges(entrance)]
        seen = {k for k, _ in order}
        i = 0
        while i < len(order):
            k = order[i][0]
            for j in net.in_edges(net.edges[k].source):
                if j not in seen and not net.is_uturn(j, k):
                    seen.add(j)
                    order.append((j, k))
            i += 1
        return order

    def _queue_outside(self, n: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """(B, S) людей слоя станции в связной заторной области перед её входами."""
        st = self.sim.stations
        cells = self.sim.ctm.cells
        cong = rho > self.queue_density                       # (B, C)
        B = rho.shape[0]
        out = np.zeros((B, st.n_stations))
        for e, tree in enumerate(self.queue_trees):
            s = st.ent_station[e]
            p = st.dest[s]
            full = {}                                          # ребро → (B,) затор на всём ребре
            for k, parent in tree:
                active = np.ones(B, dtype=bool) if parent < 0 else full[parent]
                c = np.arange(cells.first[k], cells.last[k] + 1)
                if self.is_zone[k]:      # накопитель: люди станции в нём ждут
                    out[:, s] += n[:, p, c[0]] * active
                    full[k] = active & (n[:, :, c[0]].sum(axis=1) >= 0.99 * self.cell_cap[c[0]])
                    continue
                run = np.cumprod(cong[:, c[::-1]], axis=1)[:, ::-1].astype(bool)  # затор до конца ребра
                region = run & active[:, None]
                out[:, s] += (n[:, p, c] * region).sum(axis=1)
                full[k] = region[:, 0]
        return out

    def record(self, state) -> None:
        sim = self.sim
        cs = state.ctm
        n_tot = cs.n.sum(axis=1)                                # (B, C)
        rho = n_tot / sim.ctm.area
        B, E = rho.shape[0], len(sim.net.edges)
        edge_max = np.full((B, E), -np.inf)
        np.maximum.at(edge_max, (slice(None), self.edge_of_cell), rho)
        edge_mean = np.add.reduceat(n_tot, self.edge_starts, axis=1) * self.inv_edge_area
        bal = sim.balance(state)
        if self.total is None:
            self.total = bal["in_stadium"] + bal["released"]
        iv = state.interval
        if state.t > 0 and iv is None:
            raise ValueError("Recorder: в состоянии нет мониторинга интервала (SimState.interval)")
        first_snap = not self.snaps
        snap = {
            "t": state.t,
            # t = 0 — снимок; далее — максимум по шагам CTM за интервал
            "edge_max": edge_max if first_snap else np.maximum(iv.edge_max, edge_max),
            "edge_mean": edge_mean,
            "exposure": np.zeros((B, E)) if first_snap else iv.exposure.copy(),
            "over_time": np.zeros((B, E)) if first_snap else iv.over_time.copy(),
            "any_over_time": np.zeros(B) if first_snap else iv.any_time.copy(),
            "area_above": np.zeros(B) if first_snap else iv.area_max.copy(),
            "area_above_time": np.zeros(B) if first_snap else iv.area_time.copy(),
            "exposure_congestion": np.zeros((B, E)) if first_snap else iv.exposure_cong.copy(),
            "area_congestion": np.zeros(B) if first_snap else iv.area_cong_max.copy(),
            "zone_full_time": (np.zeros((B, sim.ctm.zone_cells.size)) if first_snap
                               else iv.zone_full_time.copy()),
            "in_stadium": bal["in_stadium"],
            "at_gates": cs.source_queue.sum(axis=(1, 2)),
            "on_streets": n_tot.sum(axis=1) + cs.sink_buffer.sum(axis=(1, 2)),
            "in_stations": bal["in_stations"],
            "released": bal["released"],
            "left": bal["departed"] + bal["exited"],
            "departed": state.stations.departed.copy(),
            "station_links": state.stations.x.copy(),
            "queue_outside": self._queue_outside(cs.n, rho),
        }
        if self.cells:
            snap["cell_density"] = rho.astype(np.float32)
        self.snaps.append(snap)

    def result(self) -> Trajectory:
        if not self.snaps:
            raise ValueError("Recorder: нет снимков")
        keys = self.snaps[0].keys()
        st = {k: np.stack([s[k] for s in self.snaps]) for k in keys}
        return Trajectory(total=np.asarray(self.total, dtype=float), **st)


def run_episode(sim, params, rng: np.random.Generator, n_steps: int, policy=None,
                cfg: MetricsConfig | None = None, cells: bool = False) -> Trajectory:
    """Прогон эпизода с записью. ``policy(k, state) -> Actions | None`` (None — без
    управления)."""
    rec = Recorder(sim, cfg, cells=cells)
    state = sim.reset(params)
    rec.record(state)
    for k in range(n_steps):
        state, _ = sim.step(state, policy(k, state) if policy else None, rng)
        rec.record(state)
    return rec.result()


# --- метрики эпизода ---------------------------------------------------------------------

def clearance_time(traj: Trajectory, fraction: float) -> np.ndarray:
    """(B,) время рассасывания, с; inf — в конце эпизода в районе ≥ fraction·N."""
    x = traj.in_area                                     # (T, B)
    thr = fraction * traj.total                          # (B,)
    above = x >= thr[None, :]
    out = np.full(x.shape[1], np.inf)
    for b in range(x.shape[1]):
        idx = np.flatnonzero(above[:, b])
        if idx.size == 0:
            out[b] = traj.t[0]
        elif idx[-1] < len(traj.t) - 1:
            k = idx[-1]                                  # последний снимок выше порога
            x0, x1 = x[k, b], x[k + 1, b]
            w = (x0 - thr[b]) / (x0 - x1) if x0 > x1 else 0.0
            out[b] = traj.t[k] + w * (traj.t[k + 1] - traj.t[k])
    return out


def travel_times(traj: Trajectory, quantiles=(0.95,), n_levels: int = 2000):
    """Время в районе по накопленным кривым A (вышли со стадиона) и D (ушли из района).
    Возвращает (среднее (B,), {q: квантиль (B,)}, не ушло к концу (B,)).
    Среднее — по Литтлу; квантили — горизонтальное расстояние между кривыми (FIFO).
    Не ушедшие к концу эпизода получают время до конца эпизода (оценка снизу)."""
    t, A, D = traj.t, traj.released, traj.left
    T_end = t[-1]
    B = A.shape[1]
    mean = np.full(B, np.nan)
    qs = {q: np.full(B, np.nan) for q in quantiles}
    censored = A[-1] - D[-1]
    for b in range(B):
        a, d = A[:, b], D[:, b]
        if a[-1] <= 0:
            continue
        mean[b] = np.trapezoid(a - d, t) / a[-1]
        # Уровни — середины равных долей вышедших; момент уровня на кривой —
        # первое достижение (кривые неубывающие, плато не мешают).
        y = (np.arange(n_levels) + 0.5) / n_levels * a[-1]
        ta = _first_crossing(t, np.maximum.accumulate(a), y)
        td = _first_crossing(t, np.maximum.accumulate(d), y)
        td = np.where(y > d[-1], T_end, td)
        dur = np.maximum(td - ta, 0.0)
        for q in quantiles:
            qs[q][b] = np.quantile(dur, q)
    return mean, qs, censored


def _first_crossing(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Первый момент, когда неубывающая кусочно-линейная x(t) достигает уровней y."""
    k = np.searchsorted(x, y, side="left")              # x[k-1] < y ≤ x[k]
    k = np.clip(k, 1, len(t) - 1)
    x0, x1 = x[k - 1], x[k]
    w = np.where(x1 > x0, (y - x0) / np.where(x1 > x0, x1 - x0, 1.0), 1.0)
    out = t[k - 1] + np.clip(w, 0.0, 1.0) * (t[k] - t[k - 1])
    return np.where(y <= x[0], t[0], out)               # достигнут уже в первом снимке


def overloads(traj: Trajectory) -> dict[str, np.ndarray]:
    """Перегрузки (ρ > ρ_крит), экспозиция опасной плотности и затора, заполнение зон —
    по агрегатам интервалов (посчитаны на шагах CTM)."""
    zf = traj.zone_full_time                              # (T, B, Z)
    first_k = np.where(zf.any(axis=0), np.argmax(zf > 0, axis=0), -1)   # (B, Z)
    # момент первого заполнения — начало интервала, в котором оно случилось (точность Δt)
    t_start = np.concatenate([[traj.t[0]], traj.t[:-1]])
    zone_first = np.where(first_k >= 0, t_start[np.maximum(first_k, 0)], np.inf)
    over = traj.over_time > 0                             # (T, B, E) перегрузка в интервале
    starts = over & ~np.concatenate([np.zeros_like(over[:1]), over[:-1]], axis=0)
    return {
        "overload_count": starts.sum(axis=(0, 2)).astype(float),
        "overload_duration": traj.over_time.sum(axis=(0, 2)),
        "overload_time": traj.any_over_time.sum(axis=0),
        "overload_any": over.any(axis=(0, 2)),
        "overload_duration_edge": traj.over_time.sum(axis=0),           # (B, E)
        "exposure": traj.exposure.sum(axis=(0, 2)),
        "exposure_edge": traj.exposure.sum(axis=0),                      # (B, E)
        "area_above_max": traj.area_above.max(axis=0),
        "area_above_time": traj.area_above_time.sum(axis=0),
        "congestion_exposure": traj.exposure_congestion.sum(axis=(0, 2)),
        "congestion_exposure_edge": traj.exposure_congestion.sum(axis=0),  # (B, E)
        "congestion_area_max": traj.area_congestion.max(axis=0),
        "zone_full_time": zf.sum(axis=0),                                # (B, Z)
        "zone_full_first": zone_first,                                   # (B, Z)
    }


def episode_metrics(traj: Trajectory, cfg: MetricsConfig | None = None) -> dict[str, np.ndarray]:
    """Метрики эпизода по пакету (см. описание модуля). Времена — в секундах."""
    cfg = cfg or MetricsConfig.from_yaml()
    mean, qs, censored = travel_times(traj)
    out = {
        "total_ped": traj.total,
        "clearance_time": clearance_time(traj, cfg.clearance_fraction),
        "mean_time": mean,
        "p95_time": qs[0.95],
        "censored_ped": censored,
        "max_density": traj.edge_max.max(axis=(0, 2)),
        "max_density_edge": traj.edge_max.max(axis=0),
        "max_queue_outside": traj.queue_outside.max(axis=0),       # (B, S)
    }
    out.update(overloads(traj))
    return out


# Главные метрики безопасности при сравнении методов — экспозиция и площадь выше порога;
# максимум плотности — вспомогательная (на улицах стоячая очередь сжимается до ρ_jam
# диаграммы Weidmann, пики интерпретируются качественно; DECISIONS.md, этап 3б).
SCALAR_METRICS = ("total_ped", "clearance_time", "mean_time", "p95_time", "censored_ped",
                  "exposure", "area_above_max", "overload_count", "overload_duration",
                  "overload_time", "overload_any", "congestion_exposure",
                  "congestion_area_max", "max_density")


def scalar_table(m: dict[str, np.ndarray], zone_names: list[str] | None = None):
    """pandas.DataFrame: эпизод × скалярные метрики (времена — в минутах); при
    ``zone_names`` — время заполнения каждой зоны ≥ 95 %, мин."""
    import pandas as pd
    cols = {}
    for k in SCALAR_METRICS:
        v = np.asarray(m[k], dtype=float)
        if k in ("clearance_time", "mean_time", "p95_time", "overload_time"):
            cols[k + "_min"] = v / 60.0
        elif k == "overload_duration":
            cols["overload_duration_edge_min"] = v / 60.0
        elif k == "exposure":
            cols["danger_exposure_ped_min"] = v / 60.0
        elif k == "area_above_max":
            cols["danger_area_max_m2"] = v
        elif k == "congestion_exposure":
            cols["congestion_exposure_ped_min"] = v / 60.0
        elif k == "congestion_area_max":
            cols["congestion_area_max_m2"] = v
        else:
            cols[k] = v
    for j, name in enumerate(zone_names or []):
        cols[f"zone_full_{name}_min"] = m["zone_full_time"][:, j] / 60.0
    return pd.DataFrame(cols)
