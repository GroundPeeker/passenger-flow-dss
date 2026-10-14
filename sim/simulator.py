"""Симулятор сценария: CTM + станции + спрос + поведение + наблюдения; интерфейс управления.

Функциональное ядро (для фильтра частиц и MPC — копирование состояний, пакетные прогоны):
    sim = Simulator.from_yaml("configs/scenario_toy.yaml", fidelity="truth")
    params = sim.sample_params(rng, B)        # параметры эпизода по элементам пакета
    state = sim.reset(params)
    state, obs = sim.step(state, actions, rng)  # один интервал управления Δt
Обёртка для экспериментов и модели оператора: ``Episode(sim, params, rng).step(actions)``.

Точность (``fidelity``, обязательный аргумент; блок ``fidelity`` конфига сценария):
- ``truth`` — «реальность» экспериментов: случайный выпуск со стадиона и дискретные поезда;
- ``model`` — модель фильтра частиц и MPC: непрерывный отток с платформы и шум процесса.
Модель контроллера намеренно отличается от «реальности» — защита от inverse crime
(DECISIONS.md, этап 2а).

Действия (``Actions``, все массивы — с пакетной осью первой) — полный вектор управлений
на интервал, неявного «оставить как было» нет:
- ``entrance_level`` (B, Ne) — уровень входа станции из ``entrance_levels`` конфига
  (0 — закрыт, 0.5 — ограничен, 1 — открыт);
- ``recommend``      (B, n_signs) — вариант рекомендации на указателе, −1 — нет;
- ``extra_trains``   (B, n_stations) bool — запрос дополнительных поездов (событие);
- ``switch``         (B, n_switch) — вариант на указателе смены станции, −1 — нет
  (необязательно: None — нет рекомендаций смены станции).
Закрыть (уровень 0) все станции одного направления (``direction`` станции) нельзя — ошибка;
любой ненулевой уровень (в том числе 0.5) доступен любой станции, даже единственной
в своём направлении: станция с уровнем > 0 считается открытой.

Мониторинг плотности (``SimState.interval``): на каждом шаге CTM внутри интервала
управления считаются наибольшая плотность ячейки по участкам, экспозиция (человеко-секунды
в ячейках с ρ > порога) и площадь ячеек с ρ > порога — для порога опасной плотности ρ_крит
и порога затора, по типу ячейки (улица / зона ожидания), а также время заполнения зон
ожидания ≥ 95 %. Пороги — ``Simulator(metrics=MetricsConfig)``, по умолчанию
configs/metrics.yaml. Сохраняются агрегаты за интервал (метрики — sim/metrics.py).

Порядок внутри шага CTM: потоки внутри станций → приём дверей на этот шаг → выпуск
секторов → перенос людей от закрытых дверей (модель осведомлённости) → шаг CTM →
вошедшие через двери попадают в первое звено станции.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from sim.behavior import (CLOSURE_MODELS, BehaviorParams, Compliance, ComplianceParams,
                          SignSpec, Signs, SwitchSigns, SwitchSignSpec, effective_compliance)
from sim.ctm import CTM, CTMParams, CTMState
from sim.demand import Demand, DemandParams, ReleaseNoise
from sim.network import Network
from sim.observation import Observation, Observer
from sim.station import StationSpec, Stations, StationState
from sim.units import Section

ROOT = Path(__file__).resolve().parents[1]
FIDELITIES = ("truth", "model")


@dataclass
class Actions:
    entrance_level: np.ndarray  # (B, Ne)
    recommend: np.ndarray       # (B, n_signs) int, −1 — нет рекомендации
    extra_trains: np.ndarray    # (B, n_stations) bool
    switch: np.ndarray | None = None  # (B, n_switch) int, −1 — нет; None — все −1

    @classmethod
    def default(cls, sim: "Simulator", B: int) -> "Actions":
        """Без управления: все входы открыты, рекомендаций нет, запросов нет."""
        return cls(np.ones((B, sim.stations.n_entrances)),
                   np.full((B, sim.signs.n_signs), -1, dtype=int),
                   np.zeros((B, sim.stations.n_stations), dtype=bool),
                   np.full((B, sim.switch_signs.n_signs), -1, dtype=int))

    @classmethod
    def from_dict(cls, sim: "Simulator", B: int, *, entrances: dict | None = None,
                  recommend: dict | None = None, extra_trains=(),
                  switch: dict | None = None) -> "Actions":
        """Одинаковые для всего пакета действия по именам: ``entrances`` {вход: уровень},
        ``recommend`` {узел указателя: имя варианта или None}, ``extra_trains`` [станции],
        ``switch`` {узел указателя смены станции: имя варианта или None}."""
        a = cls.default(sim, B)
        for node, opt in (switch or {}).items():
            a.switch[:, sim.switch_signs.nodes.index(node)] = sim.switch_signs.option_index(node, opt)
        for eid, lev in (entrances or {}).items():
            a.entrance_level[:, sim.stations.ent_ids.index(eid)] = lev
        for node, opt in (recommend or {}).items():
            a.recommend[:, sim.signs.nodes.index(node)] = sim.signs.option_index(node, opt)
        for sid in extra_trains:
            a.extra_trains[:, sim.stations.ids.index(sid)] = True
        return a


@dataclass
class EpisodeParams:
    """Неизвестные контроллеру параметры эпизода, по элементам пакета."""
    demand: DemandParams
    compliance: ComplianceParams
    first_departure: np.ndarray  # (B, n_stations) время первого поезда, с (discrete)

    @property
    def batch(self) -> int:
        return self.demand.batch


@dataclass
class IntervalStats:
    """Агрегаты по шагам CTM за последний интервал управления, по пакету."""
    edge_max: np.ndarray     # (B, E) наибольшая плотность ячейки участка, чел/м²
    exposure: np.ndarray     # (B, E) человеко-секунды в ячейках с ρ > ρ_крит
    area_max: np.ndarray     # (B,)  наибольшая площадь ячеек с ρ > ρ_крит, м²
    area_time: np.ndarray    # (B,)  ∫ площадь ячеек с ρ > ρ_крит dt, м²·с
    over_time: np.ndarray    # (B, E) время, когда на участке есть ячейка с ρ > ρ_крит, с
    any_time: np.ndarray     # (B,)  время, когда перегружен хоть один участок, с
    exposure_cong: np.ndarray  # (B, E) человеко-секунды в ячейках выше порога затора
    area_cong_max: np.ndarray  # (B,)  наибольшая площадь ячеек выше порога затора, м²
    zone_full_time: np.ndarray  # (B, Z) время заполнения зоны ≥ zone_full_fraction, с

    @classmethod
    def zeros(cls, B: int, E: int, Z: int) -> "IntervalStats":
        return cls(np.zeros((B, E)), np.zeros((B, E)), np.zeros(B), np.zeros(B),
                   np.zeros((B, E)), np.zeros(B), np.zeros((B, E)), np.zeros(B),
                   np.zeros((B, Z)))


@dataclass
class SimState:
    t: float                  # с от окончания матча
    ctm: CTMState
    stations: StationState
    params: EpisodeParams
    walk: np.ndarray          # (B, n_compliance) дрейф c на логит-шкале
    edge_exit: np.ndarray     # (B, E) прошло через конец рёбер за последний интервал
    stadium: np.ndarray       # (B, P, Ns) людей ещё на стадионе
    release_factor: np.ndarray  # (B, Ns) множитель интенсивности выпуска G (шум выпуска)
    interval: IntervalStats | None = None  # мониторинг плотности за последний интервал

    @property
    def batch(self) -> int:
        return self.ctm.batch


class Simulator:
    def __init__(self, cfg: dict, base: Path = ROOT, ctm_params: CTMParams | None = None, *,
                 fidelity: str, metrics=None, calibrated: bool = True,
                 throughput_factor: dict | None = None):
        """``calibrated`` — применять калибровку из блока ``fidelity.<точность>.calibration``
        (файл с ``vehicle_throughput_factor``); ``throughput_factor`` {станция: κ} — задать
        множители явно (перекрывает файл; для процедуры калибровки)."""
        if fidelity not in FIDELITIES:
            raise ValueError(f"fidelity: одно из {FIDELITIES}")
        self.fidelity = fidelity
        # Ключи верхнего уровня с «_» — заготовки YAML-якорей (&имя), в модель не входят.
        sec = Section({k: v for k, v in cfg.items() if not str(k).startswith("_")}, "сценарий")
        fid_all = Section(sec.get("fidelity"), "fidelity")
        fid = Section(fid_all.get(fidelity), f"fidelity.{fidelity}")
        for other in FIDELITIES:
            fid_all.get(other)
        fid_all.finish()
        vehicle_model = fid.get("vehicle_model")
        self.release_noise = ReleaseNoise.from_dict(fid.get("release_noise"))
        calib_path = fid.get("calibration", None)
        fid.finish()
        self.calibration = None
        if throughput_factor is None and calibrated and calib_path:
            with open(base / calib_path, encoding="utf-8") as f:
                cal = Section(yaml.safe_load(f), f"калибровка {calib_path}")
            throughput_factor = {str(k): float(v)
                                 for k, v in cal.get("vehicle_throughput_factor").items()}
            for k in list(cal.d):
                cal.get(k)            # прочие ключи — описание процедуры и результатов
            cal.finish()
            self.calibration = str(calib_path)
        self.throughput_factor = dict(throughput_factor or {})
        self.net = Network.from_yaml(base / sec.get("network"))
        ctm_path = base / sec.get("ctm")
        self.ctm_params = ctm_params or CTMParams.from_yaml(ctm_path)
        self.ctm = CTM(self.net, self.ctm_params)
        self.dt = self.ctm.dt
        behavior = BehaviorParams.from_yaml(base / sec.get("behavior"))
        self.detour_rate = behavior.detour_rate
        self.control_step = sec.quantity("control_step", "time")
        n_sub = self.control_step / self.dt
        if abs(n_sub - round(n_sub)) > 1e-9 or round(n_sub) < 1:
            raise ValueError("control_step должен быть кратен шагу CTM dt")
        self.n_sub = int(round(n_sub))
        self.entrance_levels = np.array([float(x) for x in sec.get("entrance_levels")])
        if not (np.all((self.entrance_levels >= 0) & (self.entrance_levels <= 1))
                and 0.0 in self.entrance_levels and 1.0 in self.entrance_levels):
            raise ValueError("entrance_levels: значения в [0, 1], включая 0 и 1")
        self.stations = Stations([StationSpec.from_dict(d) for d in sec.get("stations")],
                                 self.net, self.dt, vehicle_model, self.throughput_factor)
        entr = {self.net.nodes[v].id for v in self.net.nodes_of_type("sink")
                if self.net.nodes[v].kind == "entrance"}
        if entr != set(self.stations.ent_ids):
            raise ValueError(f"каждый вход сети должен принадлежать станции: "
                             f"без станции {sorted(entr - set(self.stations.ent_ids))}")
        self.signs = Signs([SignSpec.from_dict(d) for d in (sec.get("signs") or [])], self.net)
        self.switch_signs = SwitchSigns([SwitchSignSpec.from_dict(d)
                                         for d in (sec.get("station_signs", None) or [])],
                                        self.ctm)
        self.demand = Demand(sec.get("demand"), self.net)
        self.compliance = Compliance(sec.get("compliance"))
        model = sec.get("closure_awareness")
        if model not in CLOSURE_MODELS:
            raise ValueError(f"closure_awareness: одна из {sorted(CLOSURE_MODELS)}")
        self.closure = CLOSURE_MODELS[model](self.ctm, self.stations)
        self.observer = Observer(sec.get("observation"), self.net, self.ctm.cells)
        sec.finish()
        if self.release_noise.has_factor:
            r = self.release_noise.correlation_time / self.dt
            if abs(r - round(r)) > 1e-9:
                raise ValueError("release_noise.correlation_time должно быть кратно dt")
            self.factor_every = int(round(r))
        if metrics is None:
            from sim.metrics import MetricsConfig
            metrics = MetricsConfig.from_yaml()
        self.set_metrics(metrics)
        self.station_sinks = np.zeros(len(self.ctm.sink_nodes), dtype=bool)
        self.station_sinks[self.stations.ent_sink] = True

    def set_metrics(self, cfg) -> None:
        """Пороги мониторинга (sim.metrics.MetricsConfig) → порог каждой ячейки по типу
        элемента: ``rho_crit_cells`` — опасная плотность ρ_крит (ограничение MPC),
        ``congestion_cells`` — затор."""
        self.metrics_cfg = cfg
        self.rho_crit = cfg.rho_crit

        def by_type(street, zone):
            x = np.full(self.ctm.n_cells, float(street))
            x[self.ctm.zone_cells] = float(zone)
            return x
        self.rho_crit_cells = by_type(*cfg.rho_crit)
        self.congestion_cells = by_type(*cfg.congestion)
        self.zone_full_level = cfg.zone_full_fraction * self.ctm.zone_cap

    @classmethod
    def from_yaml(cls, path: str | Path, *, fidelity: str,
                  ctm_params: CTMParams | None = None,
                  metrics=None, calibrated: bool = True) -> "Simulator":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f), ROOT, ctm_params, fidelity=fidelity,
                       metrics=metrics, calibrated=calibrated)

    # --- эпизод -----------------------------------------------------------------------

    @property
    def n_compliance(self) -> int:
        """Число указателей со своей c: указатели маршрута, затем смены станции."""
        return self.signs.n_signs + self.switch_signs.n_signs

    def sample_params(self, rng: np.random.Generator, B: int) -> EpisodeParams:
        """Параметры эпизода: спрос, степень следования, фаза поездов ~ U[0, штатный интервал)."""
        demand = self.demand.sample(rng, B)
        comp = self.compliance.sample(rng, B, self.n_compliance)
        phase = rng.uniform(0.0, 1.0, (B, self.stations.n_stations)) * self.stations.headways[:, 0]
        return EpisodeParams(demand, comp, phase)

    def fixed_params(self, B: int, c=1.0, total: float | None = None,
                     profile: dict | None = None) -> EpisodeParams:
        """Параметры без неопределённости (тесты, рисунки проверки): средние доли,
        заданная c на всех указателях, первый поезд — через штатный интервал."""
        return EpisodeParams(self.demand.fixed(B, total, profile),
                             ComplianceParams.constant(c, B, self.n_compliance),
                             np.broadcast_to(self.stations.headways[:, 0],
                                             (B, self.stations.n_stations)).copy())

    def reset(self, params: EpisodeParams) -> SimState:
        B = params.batch
        return SimState(t=0.0, ctm=self.ctm.initial_state(B),
                        stations=self.stations.initial_state(B, params.first_departure),
                        params=params, walk=np.zeros((B, self.n_compliance)),
                        edge_exit=np.zeros((B, len(self.net.edges))),
                        stadium=self.demand.initial_pool(params.demand, self.release_noise),
                        release_factor=np.ones((B, self.demand.n_sectors)),
                        interval=IntervalStats.zeros(B, len(self.net.edges),
                                                     self.ctm.zone_cells.size))

    def _check_actions(self, a: Actions, B: int) -> None:
        shapes = {"entrance_level": (B, self.stations.n_entrances),
                  "recommend": (B, self.signs.n_signs),
                  "extra_trains": (B, self.stations.n_stations)}
        if a.switch is not None:
            shapes["switch"] = (B, self.switch_signs.n_signs)
        for name, shape in shapes.items():
            if np.shape(getattr(a, name)) != shape:
                raise ValueError(f"actions.{name}: ожидается форма {shape} (пакет первой осью)")
        if not np.all(np.isin(a.entrance_level, self.entrance_levels)):
            raise ValueError(f"actions.entrance_level: допустимы {self.entrance_levels.tolist()}")
        is_open = self.stations.is_open(np.asarray(a.entrance_level))
        for d, name in enumerate(self.stations.directions):
            of_dir = self.stations.direction_of == d
            if np.any(~is_open[:, of_dir].any(axis=1)):
                raise ValueError(f"actions.entrance_level: закрыты все станции направления "
                                 f"{name!r} — запрещено")

    def step(self, state: SimState, actions: Actions | None, rng: np.random.Generator
             ) -> tuple[SimState, Observation]:
        """Один интервал управления Δt = ``control_step``."""
        B = state.batch
        a = actions if actions is not None else Actions.default(self, B)
        self._check_actions(a, B)
        p = state.params
        st = self.stations.request_extra(state.stations, state.t, a.extra_trains)
        walk = self.compliance.drift(state.walk, rng, self.control_step)
        c = effective_compliance(p.compliance, walk)
        ns = self.signs.n_signs
        split = self.signs.split(np.asarray(a.recommend, dtype=int), c[:, :ns])
        switch = None
        if self.switch_signs.rules.size:
            choice = (np.full((B, self.switch_signs.n_signs), -1, dtype=int) if a.switch is None
                      else np.asarray(a.switch, dtype=int))
            frac = self.switch_signs.fractions(choice, c[:, ns:])
            if np.any(frac > 0):
                switch = (self.switch_signs.rules, frac)
        level = np.asarray(a.entrance_level, dtype=float)
        sink_cap = self.ctm.batch(self.ctm.sink_capacity, B)
        detour = np.full(B, self.detour_rate) if self.detour_rate > 0 else None
        ent = self.stations.ent_sink
        last = self.ctm.cells.last
        cs = state.ctm
        stadium, factor = state.stadium, state.release_factor
        noise = self.release_noise
        edge_exit = np.zeros((B, len(self.net.edges)))
        mon = IntervalStats.zeros(B, len(self.net.edges), self.ctm.zone_cells.size)
        first = self.ctm.cells.first
        area = self.ctm.area
        for k in range(self.n_sub):
            t = state.t + k * self.dt
            if noise.has_factor and int(round(t / self.dt)) % self.factor_every == 0:
                factor = noise.draw_factor(rng, factor.shape)
            st = self.stations.advance(st, t)
            sink_cap[:, ent] = self.stations.entrance_capacity(st, level)
            out = self.demand.release(p.demand, stadium, t, t + self.dt, noise, rng, factor)
            stadium = stadium - out
            inflow = out / self.dt
            cs = self.closure.apply(cs, level, split)
            served0 = cs.served[:, :, ent].sum(axis=1)
            cs = self.ctm.step(cs, inflow=inflow, split=split, sink_capacity=sink_cap,
                               detour_rate=detour, dest_switch=switch)
            st = self.stations.admit(st, cs.served[:, :, ent].sum(axis=1) - served0)
            edge_exit += cs.cell_outflow[:, :, last].sum(axis=1)
            n_c = cs.n.sum(axis=1)
            rho = n_c / area
            np.maximum(mon.edge_max, np.maximum.reduceat(rho, first, axis=1), out=mon.edge_max)
            above = rho > self.rho_crit_cells
            mon.exposure += np.add.reduceat(n_c * above, first, axis=1) * self.dt
            over = np.logical_or.reduceat(above, first, axis=1)
            mon.over_time += over * self.dt
            mon.any_time += over.any(axis=1) * self.dt
            a_above = (area * above).sum(axis=1)
            np.maximum(mon.area_max, a_above, out=mon.area_max)
            mon.area_time += a_above * self.dt
            cong = rho > self.congestion_cells
            mon.exposure_cong += np.add.reduceat(n_c * cong, first, axis=1) * self.dt
            np.maximum(mon.area_cong_max, (area * cong).sum(axis=1), out=mon.area_cong_max)
            if self.ctm.zone_cells.size:
                mon.zone_full_time += (n_c[:, self.ctm.zone_cells]
                                       >= self.zone_full_level) * self.dt
        new = SimState(t=state.t + self.control_step, ctm=cs, stations=st, params=p,
                       walk=walk, edge_exit=edge_exit, stadium=stadium, release_factor=factor,
                       interval=mon)
        true = self.observer.true_values(cs.n.sum(axis=1), edge_exit)
        return new, Observation(new.t, self.observer.noisy(true, rng), true)

    # --- баланс -----------------------------------------------------------------------

    def balance(self, state: SimState) -> dict[str, np.ndarray]:
        """Учёт людей, (B,) по статьям. Выпущено = в сети + в станциях + уехало + ушло
        через стоки ``exit``; всего зрителей (начальный пул) = на стадионе + выпущено."""
        cs = state.ctm
        released = cs.generated.sum(axis=(1, 2))
        served = cs.served.sum(axis=1)                      # (B, Nk)
        out = {
            "released": released,
            "on_network": cs.in_system(),                   # включая очереди источников
            "in_stations": self.stations.in_stations(state.stations),
            "departed": state.stations.departed.sum(axis=1),
            "exited": served[:, ~self.station_sinks].sum(axis=1),
            "in_stadium": state.stadium.sum(axis=(1, 2)),
        }
        out["residual"] = released - (out["on_network"] + out["in_stations"]
                                      + out["departed"] + out["exited"])
        return out


class Episode:
    """Прогон одного пакета с хранением состояния: ``step(actions)`` → наблюдение."""

    def __init__(self, sim: Simulator, params: EpisodeParams, rng: np.random.Generator):
        self.sim = sim
        self.rng = rng
        self.state = sim.reset(params)

    def step(self, actions: Actions | None = None) -> Observation:
        self.state, obs = self.sim.step(self.state, actions, self.rng)
        return obs
