"""Спрос: выход зрителей со стадиона по секторам.

Сектор — узел-источник сети. На эпизод (и на каждый элемент пакета — частицу фильтра,
сценарий MPC) семплируются:
- общее число зрителей N — логнормально с заданными средним и коэффициентом вариации;
- доли секторов — Дирихле(κ_s · средние доли);
- доли направлений сектора — Дирихле(κ_d · средние доли), средние — ``destinations``
  источника в сети; назначения с нулевой средней долей остаются нулевыми;
- параметры профиля выхода (время выхода после финального свистка).

Выпуск идёт из «пула» — людей, ещё остающихся на стадионе, по (назначение, сектор).
За шаг [t0, t1) каждый оставшийся выходит с вероятностью (интенсивностью отказа профиля)
h = (F(t1) − F(t0)) / (1 − F(t0)), где F — накопленная функция профиля выхода. Профиль —
сменный (реестр ``PROFILES``) и зависит от времени: так позже добавится удержание сектора
(изменение профиля), не меняя интерфейса источника CTM (поток ``inflow`` на каждом шаге).

Случайность выпуска внутри эпизода — модель ``ReleaseNoise`` (ключ ``release_noise``
в блоке точности сценария, см. sim/simulator.py):
- ``none``           — детерминированный выпуск: pool·h (ровно N·(F(t1) − F(t0)));
- ``binomial``       — люди выходят независимо: Binomial(pool, h) при целом пуле. Это
  точный аналог пуассоновского выпуска при конечном N (в пределе малых h — Пуассон)
  и сохраняет N: никто не выходит дважды;
- ``overdispersed``  — выход группами: Binomial(pool, min(1, h·G)), G ~ Гамма(среднее 1,
  CV = cv), G обновляется раз в ``correlation_time`` (дважды стохастический выпуск,
  избыточная дисперсия: Var ≈ m + m²·cv² на интервале корреляции);
- ``multiplicative`` — непрерывный выпуск pool·min(1, h·G) с тем же множителем G —
  шум процесса для модели фильтра/MPC (без целочисленности).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammainc

from sim.network import Network
from sim.units import Section


# --- профили выхода ------------------------------------------------------------------

class GammaProfile:
    """Время выхода зрителя после окончания матча ~ Гамма(форма k, среднее m);
    неопределённость: m ~ U[m_lo, m_hi], k ~ U[k_lo, k_hi] (на эпизод, общие для секторов)."""

    name = "gamma"

    def __init__(self, sec: Section):
        self.mean_lo = sec.quantity("mean_low", "time")
        self.mean_hi = sec.quantity("mean_high", "time")
        self.shape_lo = float(sec.get("shape_low"))
        self.shape_hi = float(sec.get("shape_high"))
        if not 0 < self.mean_lo <= self.mean_hi or not 0 < self.shape_lo <= self.shape_hi:
            raise ValueError("exit_profile gamma: 0 < low ≤ high")

    def sample(self, rng: np.random.Generator, B: int) -> dict[str, np.ndarray]:
        return {"mean": rng.uniform(self.mean_lo, self.mean_hi, B),
                "shape": rng.uniform(self.shape_lo, self.shape_hi, B)}

    def nominal(self, B: int) -> dict[str, np.ndarray]:
        """Середины диапазонов."""
        return {"mean": np.full(B, (self.mean_lo + self.mean_hi) / 2),
                "shape": np.full(B, (self.shape_lo + self.shape_hi) / 2)}

    @staticmethod
    def cdf(p: dict[str, np.ndarray], t: float, n_sectors: int) -> np.ndarray:
        """(B, Ns) доля выпущенных к моменту t."""
        k, m = p["shape"], p["mean"]
        F = gammainc(k, max(t, 0.0) * k / m)
        return np.broadcast_to(F[:, None], (F.size, n_sectors))


class TableProfile:
    """Заданная таблицей интенсивность выхода (доля зрителей в минуту) на отрезках времени,
    одинаковая для секторов, без неопределённости. F — кусочно-линейная, нормирована на 1."""

    name = "table"

    def __init__(self, sec: Section):
        self.times = np.array([float(x) for x in sec.get("breakpoints_min")]) * 60.0
        rates = np.array([float(x) for x in sec.get("relative_rate")])
        if self.times.size != rates.size + 1 or np.any(np.diff(self.times) <= 0) or np.any(rates < 0):
            raise ValueError("exit_profile table: breakpoints_min — возрастающие, "
                             "на одну точку больше, чем relative_rate ≥ 0")
        mass = np.concatenate([[0.0], np.cumsum(rates * np.diff(self.times))])
        if mass[-1] <= 0:
            raise ValueError("exit_profile table: нулевая интенсивность")
        self.F = mass / mass[-1]

    def sample(self, rng: np.random.Generator, B: int) -> dict[str, np.ndarray]:
        return {"batch": np.zeros(B)}

    def nominal(self, B: int) -> dict[str, np.ndarray]:
        return {"batch": np.zeros(B)}

    def cdf(self, p: dict[str, np.ndarray], t: float, n_sectors: int) -> np.ndarray:
        F = float(np.interp(t, self.times, self.F))
        return np.full((p["batch"].size, n_sectors), F)


PROFILES = {c.name: c for c in (GammaProfile, TableProfile)}


# --- случайность выпуска -------------------------------------------------------------

RELEASE_NOISE = ("none", "binomial", "overdispersed", "multiplicative")


@dataclass(frozen=True)
class ReleaseNoise:
    model: str = "none"
    cv: float = 0.0                 # CV множителя G (overdispersed, multiplicative)
    correlation_time: float = 0.0   # период обновления G, с

    @classmethod
    def from_dict(cls, d: dict) -> "ReleaseNoise":
        sec = Section(d, "release_noise")
        model = sec.get("model")
        if model not in RELEASE_NOISE:
            raise ValueError(f"release_noise.model: одно из {RELEASE_NOISE}")
        cv, tau = 0.0, 0.0
        if model in ("overdispersed", "multiplicative"):
            cv = float(sec.get("cv"))
            tau = sec.quantity("correlation_time", "time")
            if not (cv > 0 and tau > 0):
                raise ValueError("release_noise: cv > 0 и correlation_time > 0")
        sec.finish()
        return cls(model, cv, tau)

    @property
    def integer(self) -> bool:
        """Пул и выпуск — целые числа людей."""
        return self.model in ("binomial", "overdispersed")

    @property
    def has_factor(self) -> bool:
        return self.model in ("overdispersed", "multiplicative")

    def draw_factor(self, rng: np.random.Generator, shape) -> np.ndarray:
        """Множитель G ~ Гамма(форма 1/cv², масштаб cv²): среднее 1, CV = cv."""
        k = 1.0 / self.cv ** 2
        return rng.gamma(k, 1.0 / k, shape)


# --- спрос ---------------------------------------------------------------------------

@dataclass
class DemandParams:
    """Параметры спроса на эпизод, по элементам пакета."""
    total: np.ndarray        # (B,)       N, чел
    sector_share: np.ndarray  # (B, Ns)    доли секторов, Σ = 1
    dest_share: np.ndarray   # (B, P, Ns) доли направлений сектора, Σ_p = 1
    profile: dict[str, np.ndarray]  # параметры профиля выхода, массивы (B,)

    @property
    def batch(self) -> int:
        return self.total.size


class Demand:
    def __init__(self, cfg: dict, net: Network):
        sec = Section(cfg, "demand")
        att = Section(sec.get("attendance"), "demand.attendance")
        self.mean_total = att.quantity("mean", "count")
        self.cv_total = float(att.get("cv"))
        att.finish()
        sources = [net.nodes[v].id for v in net.nodes_of_type("source")]
        shares = {str(k): float(v) for k, v in sec.get("sector_shares").items()}
        if set(shares) != set(sources):
            raise ValueError(f"demand.sector_shares: нужны доли ровно по источникам {sources}")
        w = np.array([shares[s] for s in sources])
        if np.any(w < 0) or abs(w.sum() - 1) > 1e-9:
            raise ValueError("demand.sector_shares: доли ≥ 0 с суммой 1")
        self.sector_mean = w
        self.sector_conc = sec.get("sector_concentration")
        self.dest_conc = sec.get("destination_concentration")
        for name, k in (("sector_concentration", self.sector_conc),
                        ("destination_concentration", self.dest_conc)):
            if k is not None and not float(k) > 0:
                raise ValueError(f"demand.{name}: > 0 или null (без неопределённости)")
        prof = Section(sec.get("exit_profile"), "demand.exit_profile")
        model = prof.get("model")
        if model not in PROFILES:
            raise ValueError(f"exit_profile.model: одно из {sorted(PROFILES)}")
        self.profile = PROFILES[model](prof)
        prof.finish()
        sec.finish()
        if self.mean_total <= 0 or self.cv_total < 0:
            raise ValueError("demand.attendance: mean > 0, cv ≥ 0")
        self.sources = sources
        self.dest_mean = net.source_shares       # (P, Ns) из сети
        self.n_dest = net.n_dest

    @property
    def n_sectors(self) -> int:
        return len(self.sources)

    def sample(self, rng: np.random.Generator, B: int) -> DemandParams:
        """Параметры спроса для пакета из B эпизодов. Порядок обращений к rng фиксирован."""
        if self.cv_total > 0:
            s2 = np.log1p(self.cv_total ** 2)
            total = rng.lognormal(np.log(self.mean_total) - s2 / 2, np.sqrt(s2), B)
        else:
            total = np.full(B, self.mean_total)
        sector = _dirichlet(rng, self.sector_mean, self.sector_conc, B)          # (B, Ns)
        dest = np.stack([_dirichlet(rng, self.dest_mean[:, s], self.dest_conc, B)
                         for s in range(self.n_sectors)], axis=-1)              # (B, P, Ns)
        return DemandParams(total, sector, dest, self.profile.sample(rng, B))

    def fixed(self, B: int, total: float | None = None, profile: dict | None = None) -> DemandParams:
        """Параметры без неопределённости (средние доли, середины диапазонов профиля,
        если не заданы явно) — для тестов и рисунков проверки."""
        t = np.full(B, self.mean_total if total is None else float(total))
        prof = ({k: np.full(B, float(v)) for k, v in profile.items()} if profile is not None
                else self.profile.nominal(B))
        return DemandParams(t, np.broadcast_to(self.sector_mean, (B, self.n_sectors)).copy(),
                            np.broadcast_to(self.dest_mean, (B,) + self.dest_mean.shape).copy(), prof)

    def released(self, p: DemandParams, t: float) -> np.ndarray:
        """(B, Ns) ожидаемое накопленное число выпущенных из секторов к моменту t."""
        return p.total[:, None] * p.sector_share * self.profile.cdf(p.profile, t, self.n_sectors)

    def initial_pool(self, p: DemandParams, noise: ReleaseNoise) -> np.ndarray:
        """(B, P, Ns) людей на стадионе в начале эпизода; при целочисленном выпуске
        округляется до целых (суммарное N отличается от заданного не больше чем на P·Ns/2)."""
        pool = p.total[:, None, None] * p.sector_share[:, None, :] * p.dest_share
        return np.round(pool) if noise.integer else pool

    def hazard(self, p: DemandParams, t0: float, t1: float) -> np.ndarray:
        """(B, Ns) доля оставшихся на стадионе, выходящих за [t0, t1)."""
        F0 = self.profile.cdf(p.profile, t0, self.n_sectors)
        F1 = self.profile.cdf(p.profile, t1, self.n_sectors)
        rest = 1.0 - F0
        h = np.ones_like(F0)  # профиль исчерпан — выходят все оставшиеся
        np.divide(F1 - F0, rest, out=h, where=rest > 1e-12)
        return np.clip(h, 0.0, 1.0)

    def release(self, p: DemandParams, pool: np.ndarray, t0: float, t1: float,
                noise: ReleaseNoise, rng: np.random.Generator,
                factor: np.ndarray | None = None) -> np.ndarray:
        """(B, P, Ns) вышло со стадиона за [t0, t1) (не больше, чем осталось в ``pool``).
        ``factor`` (B, Ns) — множитель G (overdispersed, multiplicative)."""
        h = self.hazard(p, t0, t1)
        if noise.has_factor:
            h = np.minimum(1.0, h * factor)
        h = np.broadcast_to(h[:, None, :], pool.shape)
        if noise.integer:
            return rng.binomial(pool.astype(np.int64), h).astype(float)
        return pool * h


def _dirichlet(rng: np.random.Generator, mean: np.ndarray, conc, B: int) -> np.ndarray:
    """(B, K) выборка Дирихле(conc·mean) по ненулевым компонентам mean; conc = None —
    без неопределённости (возвращается mean)."""
    out = np.zeros((B, mean.size))
    supp = mean > 0
    if conc is None or supp.sum() == 1:
        out[:] = mean
        return out
    out[:, supp] = rng.dirichlet(float(conc) * mean[supp], B)
    return out
