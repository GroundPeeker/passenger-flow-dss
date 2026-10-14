"""Базовые стратегии управления (постановка задачи, «Методы», п. 1–3а).

Интерфейс стратегии (пакетный: B эпизодов сразу):
    policy.reset(sim, B)
    actions = policy.act(k, obs)   # k — номер интервала управления, obs — наблюдение
                                   # в конце интервала k − 1 (None при k = 0)
Стратегия видит только шумные показания счётчиков (``Observation.values``), номер
интервала и известное оператору устройство района (сеть, пропускные способности,
штатное расписание) — не истинное состояние и не параметры эпизода (c, N, профиль).
Рекомендация по наблюдению конца интервала k − 1 действует весь интервал k: запаздывание
на один интервал присуще всем стратегиям; задержка оператора τ — этап 7а.

Рычаги (``Lever``):
- ``route``  — указатель маршрута (``Actions.recommend``), вариант по имени;
- ``switch`` — указатель смены станции (``Actions.switch``), вариант по имени;
- ``entrance`` — уровень входа станции (``Actions.entrance_level``).

Стратегии:
1. ``NoControl`` — все входы открыты, рекомендаций нет.
2. ``FixedSchedule`` — заранее спланированная схема: вариант показывается в доле f
   интервалов, равномерно по времени (алгоритм Брезенхэма), независимо от наблюдений.
3. ``ThresholdRule`` — пороговое правило с гистерезисом: включить действие, когда
   измеренная плотность участка выше ``on``, выключить — когда ниже ``off``.
3а. ``LeastLoaded`` — правило наименее загруженного пути (Mayr & Köster, PLOS ONE 2022):
   каждые ``every`` интервалов на указателе выбирается вариант с наименьшей загрузкой:
   - ``LeastLoadedRoute`` (указатель маршрута) — по наибольшей измеренной плотности
     наблюдаемых участков маршрута (люди / площадь);
   - ``LeastLoadedStation`` (указатель смены станции) — по ожидаемому времени ожидания
     станции W = (люди на площади перед входом + очередь на подходах) / пропускная
     способность станции (решение студента, DECISIONS.md, этап 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.metrics import MetricsConfig, Recorder, Trajectory
from sim.simulator import Actions, EpisodeParams, Simulator


@dataclass(frozen=True)
class Lever:
    kind: str   # route | switch | entrance
    target: str  # узел указателя или вход станции

    def __post_init__(self):
        if self.kind not in ("route", "switch", "entrance"):
            raise ValueError(f"рычаг: route, switch или entrance, а не {self.kind!r}")

    def set(self, sim: Simulator, a: Actions, value, mask=None) -> None:
        """Записывает значение рычага (имя варианта / None или уровень входа) в действия
        пакета; ``mask`` (B,) — только для части эпизодов."""
        rows = slice(None) if mask is None else mask
        if self.kind == "route":
            a.recommend[rows, sim.signs.nodes.index(self.target)] = \
                sim.signs.option_index(self.target, value)
        elif self.kind == "switch":
            a.switch[rows, sim.switch_signs.nodes.index(self.target)] = \
                sim.switch_signs.option_index(self.target, value)
        else:
            a.entrance_level[rows, sim.stations.ent_ids.index(self.target)] = float(value)


class Policy:
    name = "policy"

    def reset(self, sim: Simulator, B: int) -> None:
        self.sim, self.B = sim, B

    def act(self, k: int, obs) -> Actions:
        raise NotImplementedError

    def _default(self) -> Actions:
        return Actions.default(self.sim, self.B)

    def _sensor(self, edge: str) -> int:
        names = [e for e, kind in self.sim.observer.sensors if kind == "count"]
        if edge not in names:
            raise ValueError(f"{self.name}: на участке {edge} нет счётчика count")
        return [e for e, _ in self.sim.observer.sensors].index(edge)

    def _edge_area(self, edge: str) -> float:
        k = self.sim.net.edge_index[edge]
        c = self.sim.ctm.cells
        return float(self.sim.ctm.area[c.first[k]:c.last[k] + 1].sum())


class NoControl(Policy):
    name = "no_control"

    def act(self, k, obs):
        return self._default()


def schedule_active(k: int, fraction: float) -> bool:
    """Интервал k входит в равномерную долю ``fraction`` интервалов (Брезенхэм):
    за первые n интервалов активны ⌊n·f⌋."""
    eps = 1e-9
    return math.floor((k + 1) * fraction + eps) > math.floor(k * fraction + eps)


class FixedSchedule(Policy):
    """Вариант ``option`` рычага ``lever`` — в доле ``fraction`` интервалов с начала
    эпизода, в остальные — без рекомендации (для входа — ``off_value``)."""
    name = "fixed"

    def __init__(self, lever: Lever, option, fraction: float, off_value=None):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction ∈ [0, 1]")
        self.lever, self.option, self.fraction = lever, option, float(fraction)
        self.off_value = 1.0 if (off_value is None and lever.kind == "entrance") else off_value

    def act(self, k, obs):
        a = self._default()
        if schedule_active(k, self.fraction):
            self.lever.set(self.sim, a, self.option)
        elif self.off_value is not None:
            self.lever.set(self.sim, a, self.off_value)
        return a


@dataclass(frozen=True)
class ThresholdSpec:
    edge: str        # наблюдаемый участок (счётчик count)
    on: float        # чел/м²: включить при измеренной плотности > on
    off: float       # чел/м²: выключить при < off (off ≤ on — гистерезис)
    lever: Lever
    value: object    # вариант рекомендации или уровень входа

    def __post_init__(self):
        if not self.off <= self.on:
            raise ValueError("пороговое правило: off ≤ on")


class ThresholdRule(Policy):
    """Правила применяются по порядку; позднее правило на том же рычаге перекрывает
    раннее. Плотность участка — показание счётчика / площадь участка."""
    name = "threshold"

    def __init__(self, rules: list[ThresholdSpec]):
        self.rules = list(rules)

    def reset(self, sim, B):
        super().reset(sim, B)
        self.idx = [self._sensor(r.edge) for r in self.rules]
        self.area = np.array([self._edge_area(r.edge) for r in self.rules])
        self.active = np.zeros((B, len(self.rules)), dtype=bool)

    def act(self, k, obs):
        if obs is not None:
            rho = obs.values[:, self.idx] / self.area
            on = np.array([r.on for r in self.rules])
            off = np.array([r.off for r in self.rules])
            self.active = np.where(self.active, rho >= off, rho > on)
        a = self._default()
        for j, r in enumerate(self.rules):
            if self.active[:, j].any():
                r.lever.set(self.sim, a, r.value, mask=self.active[:, j])
        return a


class LeastLoadedRoute(Policy):
    """``routes`` {вариант: [наблюдаемые участки маршрута]}; загрузка варианта — наибольшая
    измеренная плотность его участков; при равенстве — первый вариант. До первого
    наблюдения рекомендации нет."""
    name = "least_loaded"

    def __init__(self, sign: str, routes: dict[str, list[str]], every: int = 1):
        self.lever = Lever("route", sign)
        self.routes = {k: list(v) for k, v in routes.items()}
        self.every = int(every)

    def reset(self, sim, B):
        super().reset(sim, B)
        self.opts = list(self.routes)
        self.idx = [[self._sensor(e) for e in es] for es in self.routes.values()]
        self.area = [np.array([self._edge_area(e) for e in es]) for es in self.routes.values()]
        self.choice = np.full(B, -1, dtype=int)

    def loads(self, obs) -> np.ndarray:
        """(B, n_вариантов) загрузка, чел/м²."""
        return np.stack([(obs.values[:, i] / ar).max(axis=1)
                         for i, ar in zip(self.idx, self.area)], axis=1)

    def act(self, k, obs):
        if obs is not None and k % self.every == 0:
            self.choice = np.argmin(self.loads(obs), axis=1)
        a = self._default()
        for j, opt in enumerate(self.opts):
            m = self.choice == j
            if m.any():
                self.lever.set(self.sim, a, opt, mask=m)
        return a


@dataclass(frozen=True)
class StationQueue:
    """Наблюдаемая очередь станции: площадь перед входом (накопитель) и подходы к ней."""
    plaza: str
    approach: tuple[str, ...] = ()


def nominal_throughput(sim: Simulator, station: str) -> float:
    """Штатная пропускная способность станции, чел/с: наименьшая из дверей входов,
    звеньев станции и вывоза (вместимость поезда / штатный интервал). Известна оператору
    (устройство станции и расписание), от параметров эпизода не зависит."""
    st = sim.stations
    s = st.ids.index(station)
    doors = st.ent_cap[st.ent_station == s].sum()
    links = st.link_cap[st.first[s]:st.last[s] + 1]
    links = links[np.isfinite(links)]
    vehicle = st.veh_cap[s] / st.headways[s, 0]
    return float(min([doors, vehicle] + links.tolist()))


class LeastLoadedStation(Policy):
    """Указатель смены станции: вариант ``option`` (уйти со станции ``from_station`` на
    ``to_station``) рекомендуется, когда ожидаемое ожидание на ``to_station`` меньше.

    Ожидаемое время ожидания W_s = Q_s / μ_s: Q_s — измеренное число людей на площади
    перед входом; если площадь заполнена (≥ ``full_fraction`` вместимости A·ρ_queue),
    очередь вышла на улицы, и к Q_s добавляются показания подходов (как в определении
    очереди у входа, sim/metrics.py); μ_s — штатная пропускная способность станции."""
    name = "least_loaded"

    def __init__(self, sign: str, option: str, from_station: str, to_station: str,
                 queues: dict[str, StationQueue], every: int = 1, full_fraction: float = 0.95):
        self.lever = Lever("switch", sign)
        self.option = option
        self.stations = (from_station, to_station)
        self.queues = queues
        self.every = int(every)
        self.full_fraction = float(full_fraction)

    def reset(self, sim, B):
        super().reset(sim, B)
        self.mu = np.array([nominal_throughput(sim, s) for s in self.stations])
        self.plaza = [self._sensor(self.queues[s].plaza) for s in self.stations]
        self.approach = [[self._sensor(e) for e in self.queues[s].approach]
                         for s in self.stations]
        cap = []
        for s in self.stations:
            k = sim.net.edge_index[self.queues[s].plaza]
            c = sim.ctm.cells.first[k]
            if c not in sim.ctm.zone_cells:
                raise ValueError(f"площадь станции {s} ({self.queues[s].plaza}) — не накопитель")
            cap.append(sim.ctm.cell_cap[c])
        self.full = self.full_fraction * np.array(cap)
        self.on = np.zeros(B, dtype=bool)

    def waits(self, obs) -> np.ndarray:
        """(B, 2) ожидаемое время ожидания, с: [from_station, to_station]."""
        out = []
        for j in range(2):
            q = obs.values[:, self.plaza[j]].copy()
            if self.approach[j]:
                spill = q >= self.full[j]
                q += spill * obs.values[:, self.approach[j]].sum(axis=1)
            out.append(q / self.mu[j])
        return np.stack(out, axis=1)

    def act(self, k, obs):
        if obs is not None and k % self.every == 0:
            w = self.waits(obs)
            self.on = w[:, 1] < w[:, 0]
        a = self._default()
        if self.on.any():
            self.lever.set(self.sim, a, self.option, mask=self.on)
        return a


# --- прогон ------------------------------------------------------------------------------

def run_policy(sim: Simulator, params: EpisodeParams, rng: np.random.Generator,
               n_steps: int, policy: Policy, cfg: MetricsConfig | None = None,
               cells: bool = False, log_actions: bool = False):
    """Прогон пакета эпизодов со стратегией и записью траектории (sim.metrics.Recorder).
    Возвращает Trajectory, а при ``log_actions`` — ещё список действий по интервалам."""
    B = params.batch
    rec = Recorder(sim, cfg, cells=cells)
    state = sim.reset(params)
    rec.record(state)
    policy.reset(sim, B)
    obs, log = None, []
    for k in range(n_steps):
        a = policy.act(k, obs)
        if log_actions:
            log.append(a)
        state, obs = sim.step(state, a, rng)
        rec.record(state)
    traj: Trajectory = rec.result()
    return (traj, log) if log_actions else traj
