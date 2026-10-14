"""Станции транспорта (метро, остановка шаттлов): цепочка звеньев за входом.

Вход станции — сток CTM вида ``entrance`` (sim/network.py); его ``capacity`` — пропускная
способность дверей, накопитель стока обязан быть нулевым: место для людей за дверями —
звенья станции. За дверями — цепочка звеньев, например турникеты → вестибюль/эскалатор →
платформа. У звена k вместимость накопителя B_k (чел) и пропуск μ_k (чел/с) в следующее
звено; у последнего звена (платформа) вместо μ — отток в транспорт.

Шаг dt (все потоки — по состоянию начала шага, как в CTM):
- поток k → k+1:  f_k = min(x_k, μ_k·dt, B_{k+1} − x_{k+1});
- отток с платформы:
  ``continuous`` — min(x, κ · вместимость / интервал · dt), κ — калиброванный множитель
  эффективной пропускной способности (``throughput_factor``, по умолчанию 1; калибровка
  модели планирования по «реальности» с дискретными поездами — experiments/calibrate_model.py);
  ``discrete``   — поезд, прибывший в [t, t + dt), забирает min(x, вместимость);
- приём дверей на следующий шаг CTM: min(уровень·μ_двери·dt, B_0 − x_0) — когда первое
  звено заполнено, вход не принимает людей и очередь встаёт на ребре «площадка → вход»,
  а дальше распространяется на улицу (механизм стока CTM с нулевым накопителем).
  Свободное место первого звена делится между входами станции пропорционально уровень·μ.

Дополнительные поезда: интервал задан уровнями (уровень 0 — штатный, далее — меньшие
интервалы). Запрос переводит станцию на следующий уровень через задержку ввода; число смен
уровня ограничено (ресурс). Запрос сверх ресурса игнорируется и учитывается в ``rejected``.

Все массивы состояния — с пакетной осью первой (как в CTM). Одна станция — одно
назначение; людей внутри станции по назначениям не различаем.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from sim.network import Network
from sim.units import Section

VEHICLE_MODELS = ("continuous", "discrete")


@dataclass(frozen=True)
class LinkSpec:
    name: str
    storage: float          # вместимость накопителя, чел
    capacity: float | None  # пропуск в следующее звено, чел/с; None — у последнего звена


@dataclass(frozen=True)
class VehicleSpec:
    capacity: float                  # свободных мест в прибывающем поезде / автобусе, чел
    headway_levels: tuple[float, ...]  # интервалы по уровням, с (уровень 0 — штатный)
    min_headway: float               # минимальный интервал линии, с
    max_level_changes: int           # сколько раз можно запросить следующий уровень
    activation_delay: float          # задержка ввода дополнительных поездов, с

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "VehicleSpec":
        sec = Section(d, where)
        cap = sec.quantity("capacity", "count")
        levels = sec.get("headway_levels_s")
        h_min = sec.quantity("min_headway", "time")
        changes = sec.get("max_level_changes")
        delay = sec.quantity("activation_delay", "time")
        sec.finish()
        levels = tuple(float(h) for h in levels)
        if not cap > 0:
            raise ValueError(f"{where}: capacity должна быть > 0")
        if not levels or any(b >= a for a, b in zip(levels, levels[1:])):
            raise ValueError(f"{where}: headway_levels_s — непустой строго убывающий список")
        if levels[-1] < h_min:
            raise ValueError(f"{where}: интервал {levels[-1]} с меньше минимального {h_min} с")
        if not isinstance(changes, int) or not 0 <= changes <= len(levels) - 1:
            raise ValueError(f"{where}: max_level_changes — целое в [0, {len(levels) - 1}]")
        if delay < 0:
            raise ValueError(f"{where}: activation_delay ≥ 0")
        return cls(cap, levels, h_min, changes, delay)


@dataclass(frozen=True)
class StationSpec:
    id: str
    entrances: tuple[str, ...]  # id стоков-входов (на этом этапе — ровно один)
    alternative: str            # назначение, куда идут люди от закрытого входа
    direction: str              # направление (линия): закрыть все станции направления нельзя
    links: tuple[LinkSpec, ...]
    vehicle: VehicleSpec

    @classmethod
    def from_dict(cls, d: dict) -> "StationSpec":
        sec = Section(d, f"станция {d.get('id')}")
        sid = str(sec.get("id"))
        entrances = tuple(str(e) for e in sec.get("entrances"))
        alternative = str(sec.get("alternative"))
        direction = str(sec.get("direction"))
        raw_links = sec.get("links")
        vehicle = VehicleSpec.from_dict(sec.get("vehicle"), f"станция {sid}, vehicle")
        sec.finish()
        links = []
        for k, ld in enumerate(raw_links):
            ls = Section(ld, f"станция {sid}, звено {k}")
            name = str(ls.get("name"))
            storage = ls.quantity("storage", "count")
            last = k == len(raw_links) - 1
            cap = None if last else ls.quantity("capacity", "rate")
            ls.finish()
            if not storage > 0 or (cap is not None and not cap > 0):
                raise ValueError(f"станция {sid}, звено {name}: storage и capacity > 0")
            links.append(LinkSpec(name, storage, cap))
        if not links:
            raise ValueError(f"станция {sid}: нужно хотя бы одно звено (платформа)")
        if len(entrances) != 1:
            # Решение студента (DECISIONS.md, этап 2): входы хранятся списком, но при закрытии
            # одного из нескольких входов той же станции нужен разворот внутри слоя
            # назначения — в CTM его нет.
            raise ValueError(f"станция {sid}: на этом этапе поддерживается ровно один вход")
        return cls(sid, entrances, alternative, direction, tuple(links), vehicle)


@dataclass
class StationState:
    x: np.ndarray            # (B, L)  людей в звеньях
    admitted: np.ndarray     # (B, S)  накоплено вошло через двери
    departed: np.ndarray     # (B, S)  накоплено уехало
    activation: np.ndarray   # (B, S, K) время ввода принятых смен уровня (inf — нет)
    requests: np.ndarray     # (B, S)  принято запросов
    rejected: np.ndarray     # (B, S)  отклонено запросов (сверх ресурса)
    next_departure: np.ndarray  # (B, S) время следующего поезда (только discrete)

    def copy(self) -> "StationState":
        return StationState(*(getattr(self, f).copy() for f in self.__dataclass_fields__))

    @property
    def batch(self) -> int:
        return self.x.shape[0]


class Stations:
    """Все станции сценария; звенья всех станций — в одном плоском массиве."""

    def __init__(self, specs: list[StationSpec], net: Network, dt: float,
                 vehicle_model: str = "continuous", throughput_factor: dict | None = None):
        if vehicle_model not in VEHICLE_MODELS:
            raise ValueError(f"vehicle_model: одно из {VEHICLE_MODELS}")
        self.specs = list(specs)
        self.dt = dt
        self.vehicle_model = vehicle_model
        self.ids = [s.id for s in specs]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("повторяющиеся id станций")
        sinks = net.nodes_of_type("sink")
        sink_pos = {net.nodes[v].id: k for k, v in enumerate(sinks)}

        storage, cap, first, last = [], [], [], []
        ent_sink, ent_station, ent_cap, dest = [], [], [], []
        seen_dest, seen_ent = set(), set()
        for s, sp in enumerate(specs):
            first.append(len(storage))
            for lk in sp.links:
                storage.append(lk.storage)
                cap.append(np.inf if lk.capacity is None else lk.capacity)
            last.append(len(storage) - 1)
            d = None
            for eid in sp.entrances:
                if eid not in sink_pos:
                    raise ValueError(f"станция {sp.id}: {eid} — не сток сети")
                node = net.nodes[net.node_index[eid]]
                if node.kind != "entrance":
                    raise ValueError(f"станция {sp.id}: сток {eid} должен быть kind: entrance")
                if node.buffer != 0:
                    raise ValueError(f"станция {sp.id}: у входа {eid} накопитель стока должен "
                                     f"быть 0 — место за дверями задаётся звеньями станции")
                if not math.isfinite(node.capacity):
                    raise ValueError(f"станция {sp.id}: у входа {eid} нужна конечная capacity")
                if eid in seen_ent:
                    raise ValueError(f"вход {eid} принадлежит нескольким станциям")
                seen_ent.add(eid)
                de = net.sink_destination(net.node_index[eid])
                if d is not None and de != d:
                    raise ValueError(f"станция {sp.id}: у входов разные назначения")
                d = de
                ent_sink.append(sink_pos[eid])
                ent_station.append(s)
                ent_cap.append(node.capacity)
            if d in seen_dest:
                raise ValueError(f"назначение {d} у нескольких станций")
            seen_dest.add(d)
            if sp.alternative not in net.dest_index or sp.alternative == d:
                raise ValueError(f"станция {sp.id}: альтернатива {sp.alternative!r} — "
                                 f"другое назначение сети")
            if sp.vehicle.min_headway < dt:
                raise ValueError(f"станция {sp.id}: минимальный интервал меньше шага dt")
            dest.append(net.dest_index[d])

        self.storage = np.array(storage)
        self.link_cap = np.array(cap)
        self.first = np.array(first)
        self.last = np.array(last)
        self.n_links = len(storage)
        self.is_last = np.zeros(self.n_links, dtype=bool)
        self.is_last[self.last] = True
        self.up = np.flatnonzero(~self.is_last)     # звенья с последующим звеном
        self.ent_sink = np.array(ent_sink, dtype=int)      # позиция стока в порядке CTM
        self.ent_station = np.array(ent_station, dtype=int)
        self.ent_cap = np.array(ent_cap)
        self.ent_ids = [e for sp in specs for e in sp.entrances]
        self.dest = np.array(dest, dtype=int)
        self.alternative = [net.dest_index[sp.alternative] for sp in specs]
        # Станция назначения (или −1, если назначение — не станция, например уход из района).
        self.station_of_dest = np.full(net.n_dest, -1, dtype=int)
        self.station_of_dest[self.dest] = np.arange(len(specs))
        self.directions = sorted({sp.direction for sp in specs})
        self.direction_of = np.array([self.directions.index(sp.direction) for sp in specs], dtype=int)
        n_lev = max(len(sp.vehicle.headway_levels) for sp in specs) if specs else 1
        # Уровни, дополненные последним значением (дальше уровня max_level_changes не уйти).
        self.headways = np.array([list(sp.vehicle.headway_levels)
                                  + [sp.vehicle.headway_levels[-1]] * (n_lev - len(sp.vehicle.headway_levels))
                                  for sp in specs]).reshape(len(specs), n_lev)
        self.veh_cap = np.array([sp.vehicle.capacity for sp in specs])
        self.max_changes = np.array([sp.vehicle.max_level_changes for sp in specs], dtype=int)
        self.delay = np.array([sp.vehicle.activation_delay for sp in specs])
        self.K = max(1, int(self.max_changes.max())) if specs else 1
        tf = dict(throughput_factor or {})
        if set(tf) - set(self.ids):
            raise ValueError(f"throughput_factor: неизвестные станции {sorted(set(tf) - set(self.ids))}")
        self.throughput_factor = np.array([float(tf.get(i, 1.0)) for i in self.ids])
        if np.any(self.throughput_factor <= 0):
            raise ValueError("throughput_factor: множители > 0")
        if vehicle_model != "continuous" and np.any(self.throughput_factor != 1.0):
            raise ValueError("throughput_factor задаётся только для vehicle_model: continuous "
                             "(калибровка модели планирования)")

    @property
    def n_stations(self) -> int:
        return len(self.specs)

    @property
    def n_entrances(self) -> int:
        return self.ent_sink.size

    def initial_state(self, batch: int, first_departure: np.ndarray | None = None) -> StationState:
        """``first_departure`` (B, S) — время первого поезда, с (только discrete);
        по умолчанию — через штатный интервал."""
        B, S = batch, self.n_stations
        if first_departure is None:
            first_departure = np.broadcast_to(self.headways[:, 0], (B, S))
        nd = np.asarray(first_departure, dtype=float)
        if nd.shape != (B, S):
            raise ValueError(f"first_departure: ожидается форма {(B, S)}")
        return StationState(x=np.zeros((B, self.n_links)), admitted=np.zeros((B, S)),
                            departed=np.zeros((B, S)), activation=np.full((B, S, self.K), np.inf),
                            requests=np.zeros((B, S), dtype=int), rejected=np.zeros((B, S), dtype=int),
                            next_departure=nd.copy())

    def level(self, st: StationState, t: float) -> np.ndarray:
        """(B, S) действующий уровень интервала в момент t."""
        return (st.activation <= t).sum(axis=-1)

    def headway(self, st: StationState, t: float) -> np.ndarray:
        """(B, S) действующий интервал, с."""
        return self._headway_at(st, np.full(st.requests.shape, float(t)))

    def request_extra(self, st: StationState, t: float, mask: np.ndarray) -> StationState:
        """Запрос дополнительных поездов в момент t; ``mask`` (B, S) bool. Принятый запрос
        переводит станцию на следующий уровень в момент t + задержка ввода."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != st.requests.shape:
            raise ValueError(f"extra_trains: ожидается форма {st.requests.shape}")
        if not mask.any():
            return st
        new = st.copy()
        ok = mask & (st.requests < self.max_changes)
        b, s = np.nonzero(ok)
        new.activation[b, s, st.requests[b, s]] = t + self.delay[s]
        new.requests += ok
        new.rejected += mask & ~ok
        return new

    def advance(self, st: StationState, t: float) -> StationState:
        """Потоки внутри станций на шаге [t, t + dt)."""
        dt = self.dt
        x = st.x
        f = np.zeros_like(x)
        f[:, self.up] = np.minimum(np.minimum(x[:, self.up], self.link_cap[self.up] * dt),
                                   self.storage[self.up + 1] - x[:, self.up + 1])
        plat = x[:, self.last]
        new = st.copy()
        if self.vehicle_model == "continuous":
            out = np.minimum(plat, self.throughput_factor * self.veh_cap / self.headway(st, t) * dt)
        else:
            arrive = st.next_departure < t + dt
            out = np.where(arrive, np.minimum(plat, self.veh_cap), 0.0)
            if arrive.any():  # следующий поезд — через интервал, действующий в момент прибытия
                h_at = self._headway_at(st, st.next_departure)
                new.next_departure = np.where(arrive, st.next_departure + h_at, st.next_departure)
        f[:, self.last] = out
        inflow = np.zeros_like(x)
        inflow[:, self.up + 1] = f[:, self.up]
        new.x = x - f + inflow
        new.departed = st.departed + out
        return new

    def _headway_at(self, st: StationState, times: np.ndarray) -> np.ndarray:
        """(B, S) интервал, действующий в моменты ``times`` (B, S)."""
        lev = (st.activation <= times[..., None]).sum(axis=-1)
        return np.take_along_axis(np.broadcast_to(self.headways, lev.shape + self.headways.shape[1:]),
                                  lev[..., None], axis=-1)[..., 0]

    def entrance_capacity(self, st: StationState, level: np.ndarray) -> np.ndarray:
        """(B, Ne) пропускная способность входов на шаг CTM, чел/с: уровень входа × μ дверей,
        но не больше свободного места в первом звене станции (делится между входами
        станции пропорционально уровень·μ)."""
        want = level * self.ent_cap                                  # (B, Ne)
        room = np.maximum(self.storage[self.first] - st.x[:, self.first], 0.0)  # (B, S)
        tot = np.zeros_like(room)
        np.add.at(tot, (slice(None), self.ent_station), want)
        share = np.divide(want, tot[:, self.ent_station], out=np.zeros_like(want),
                          where=tot[:, self.ent_station] > 0)
        return np.minimum(want, share * room[:, self.ent_station] / self.dt)

    def admit(self, st: StationState, entered: np.ndarray) -> StationState:
        """Люди ``entered`` (B, Ne), прошедшие двери за шаг CTM, входят в первое звено."""
        by_st = np.zeros_like(st.admitted)
        np.add.at(by_st, (slice(None), self.ent_station), entered)
        x = st.x.copy()
        x[:, self.first] += by_st
        return replace(st, x=x, admitted=st.admitted + by_st)

    def is_open(self, level: np.ndarray) -> np.ndarray:
        """(B, S) у станции открыт хотя бы один вход (уровень > 0)."""
        out = np.zeros((level.shape[0], self.n_stations), dtype=bool)
        np.logical_or.at(out, (slice(None), self.ent_station), level > 0)
        return out

    def in_stations(self, st: StationState) -> np.ndarray:
        """(B,) людей внутри станций."""
        return st.x.sum(axis=1)
