"""Наблюдения: шумные счётчики на подмножестве рёбер, раз в шаг управления.

Типы датчиков:
- ``count`` — число людей на ребре в конце интервала управления (камера);
- ``flow``  — число людей, прошедших через конец ребра за интервал (линейный счётчик).

Шум (одна модель на сценарий):
- ``none``      — точные значения;
- ``poisson``   — y ~ Poisson(x);
- ``lognormal`` — y = x·exp(σε − σ²/2), σ² = ln(1 + cv²): несмещённый мультипликативный
  шум с коэффициентом вариации cv.
Вместе с шумными значениями возвращаются истинные (для тестов и анализа фильтра).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.units import Section

SENSOR_KINDS = ("count", "flow")
NOISE_MODELS = ("none", "poisson", "lognormal")


@dataclass(frozen=True)
class Observation:
    time: float           # с, конец интервала управления
    values: np.ndarray    # (B, n_sensors) показания с шумом
    true: np.ndarray      # (B, n_sensors) истинные значения


class Observer:
    def __init__(self, cfg: dict, net, cells):
        sec = Section(cfg, "observation")
        sensors = []
        for sd in sec.get("sensors"):
            ss = Section(sd, f"датчик {sd.get('edge')}")
            edge, kind = str(ss.get("edge")), ss.get("kind")
            ss.finish()
            if edge not in net.edge_index:
                raise ValueError(f"датчик: неизвестное ребро {edge}")
            if kind not in SENSOR_KINDS:
                raise ValueError(f"датчик {edge}: kind из {SENSOR_KINDS}")
            sensors.append((edge, kind))
        if len(set(sensors)) != len(sensors):
            raise ValueError("повторяющиеся датчики")
        noise = Section(sec.get("noise"), "observation.noise")
        self.noise = noise.get("model")
        if self.noise not in NOISE_MODELS:
            raise ValueError(f"observation.noise.model: одно из {NOISE_MODELS}")
        self.cv = float(noise.get("cv")) if self.noise == "lognormal" else 0.0
        noise.finish()
        sec.finish()
        if self.cv < 0:
            raise ValueError("observation.noise.cv ≥ 0")
        self.sensors = sensors
        self.edges = np.array([net.edge_index[e] for e, _ in sensors], dtype=int)
        self.is_flow = np.array([k == "flow" for _, k in sensors], dtype=bool)
        # Матрица «ячейка → ребро датчика» для счётчиков вида count.
        self.cell_to_sensor = np.zeros((cells.n_cells, len(sensors)))
        for j, k in enumerate(self.edges):
            if not self.is_flow[j]:
                self.cell_to_sensor[cells.first[k]:cells.last[k] + 1, j] = 1.0

    @property
    def n_sensors(self) -> int:
        return len(self.sensors)

    @property
    def names(self) -> list[str]:
        return [f"{k}:{e}" for e, k in self.sensors]

    def true_values(self, n_cells_total: np.ndarray, edge_exit: np.ndarray) -> np.ndarray:
        """(B, n_sensors). ``n_cells_total`` (B, C) — людей в ячейках в конце интервала;
        ``edge_exit`` (B, E) — прошло через конец рёбер за интервал."""
        x = n_cells_total @ self.cell_to_sensor
        x[:, self.is_flow] = edge_exit[:, self.edges[self.is_flow]]
        return np.maximum(x, 0.0)

    def noisy(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.noise == "none":
            return x.copy()
        if self.noise == "poisson":
            return rng.poisson(x).astype(float)
        s2 = np.log1p(self.cv ** 2)
        return x * np.exp(rng.normal(-s2 / 2, np.sqrt(s2), x.shape))
