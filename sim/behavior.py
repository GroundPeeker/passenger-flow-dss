"""Поведение пешеходов.

1. Спонтанный обход затора (используется ядром CTM): люди, ждущие у забитого выхода узла
   (φ_o < 1), с интенсивностью θ [1/с] переходят на другие выходы того же входа, ведущие
   к той же станции (тот же слой назначения) и не забитые на этом шаге. По умолчанию θ = 0:
   люди к забитой ветви ждут, остальные проходят (узел без FIFO, см. sim/ctm.py).

2. Привычные маршруты и рекомендации. Привычные доли — матрица сети ``net.split``
   (кратчайший путь или переопределение в YAML). Указатель — развилка со списком вариантов
   рекомендации; вариант — {назначение: исходящее ребро}. Для назначений варианта доля c
   (степень следования) идёт по рекомендованному ребру, остальные 1 − c — по привычным долям:
   β = c·рекомендация + (1 − c)·привычка. Матрица передаётся в CTM и действует в момент
   фиксации выбора (DECISIONS.md, этап 1в(г)).

3. Степень следования c. На эпизод: базовое c ~ U[lo, hi] или Beta(a, b); различие
   по указателям — смещение на логит-шкале N(0, σ_узел²); дрейф — случайное блуждание
   на логит-шкале со стандартным отклонением σ_дрейф·√(Δt, ч) за шаг управления.
   c = 0 и c = 1 точны (логит ±∞ без смещений).

3а. Указатель «смена станции» (этап 4): на развилке вариант рекомендации — {станция:
   другая станция}. Доля c людей станции из варианта, проходящих развилку, в момент
   фиксации выбора движения меняет назначение (``CTM.step(dest_switch=…)``), остальные
   идут к своей станции. Степень следования — своя на каждом указателе: индексы c
   указателей смены станции идут после указателей маршрута.

4. Кто и когда узнаёт о закрытии входа — сменная модель (реестр ``CLOSURE_MODELS``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from sim.network import Network
from sim.units import Section


@dataclass(frozen=True)
class BehaviorParams:
    detour_rate: float = 0.0  # θ, 1/с: интенсивность спонтанного обхода затора

    @classmethod
    def from_dict(cls, d: dict) -> "BehaviorParams":
        sec = Section(d, "behavior")
        rate = sec.quantity("detour_rate", "inv_time")
        sec.finish()
        if rate < 0:
            raise ValueError("behavior.detour_rate: должна быть ≥ 0")
        return cls(detour_rate=rate)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BehaviorParams":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


def detour_transfer(waiting: np.ndarray, blocked: np.ndarray, reach: np.ndarray,
                    weight: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Перенос выбора движения ждущими у забитых выходов.

    Массивы по входам узлов I и движениям входа K (дополнение до K — фиктивные движения
    с reach = False, blocked = False, waiting = 0):
    waiting (B, P, I, K) — люди, выбравшие движение и не ушедшие на этом шаге;
    blocked (B, I, K)    — выход движения забит (φ_o < 1);
    reach   (P, I, K)    — по выходу движения можно дойти до назначения p;
    weight  (B, P, I, K) — предпочтения при выборе замены (доли маршрутов β);
    delta   (B,)         — доля ждущих, меняющих выбор за шаг, 1 − exp(−θ·dt).
    Возвращает изменение числа выбравших каждое движение (B, P, I, K); сумма по K — 0
    (люди не покидают ячейку и не меняют назначение).
    Замена — незабитые движения того же входа, ведущие к p, пропорционально weight;
    если у них всех weight = 0 — поровну. Нет замены — никто не переходит.
    """
    elig = (~blocked)[:, None] & reach[None]
    w = np.where(elig, weight, 0.0)
    ws = w.sum(axis=-1, keepdims=True)
    eq = elig.astype(float)
    eqs = eq.sum(axis=-1, keepdims=True)
    w = np.divide(w, ws, out=np.divide(eq, eqs, out=np.zeros_like(eq), where=eqs > 0),
                  where=ws > 0)
    moved = delta[:, None, None, None] * waiting * blocked[:, None] * (eqs > 0)
    return moved.sum(axis=-1, keepdims=True) * w - moved


# --- указатели и рекомендации ---------------------------------------------------------

@dataclass(frozen=True)
class SignSpec:
    node: str
    options: tuple[tuple[str, dict[str, str]], ...]  # (имя варианта, {назначение: ребро})

    @classmethod
    def from_dict(cls, d: dict) -> "SignSpec":
        sec = Section(d, f"указатель {d.get('node')}")
        node = str(sec.get("node"))
        opts = []
        for od in sec.get("options"):
            os_ = Section(od, f"указатель {node}, вариант {od.get('name')}")
            opts.append((str(os_.get("name")),
                         {str(k): str(v) for k, v in os_.get("route").items()}))
            os_.finish()
        sec.finish()
        if not opts or len({n for n, _ in opts}) != len(opts):
            raise ValueError(f"указатель {node}: нужны варианты с разными именами")
        return cls(node, tuple(opts))


class Signs:
    """Указатели сети. Эффективная матрица долей (B, P, E):
    β = H + Σ_s c_s · D[s, o_s], где H — привычные доли, D[s, o] — разность
    «рекомендация − привычка» в строках (узел указателя, назначения варианта),
    o_s = −1 — рекомендации нет (D = 0)."""

    def __init__(self, specs: list[SignSpec], net: Network):
        self.specs = list(specs)
        self.net = net
        self.nodes = [s.node for s in specs]
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("несколько указателей на одном узле")
        H = net.split
        P, E = H.shape
        n_opt = max((len(s.options) for s in specs), default=0)
        # D[s, 0] — нет рекомендации; D[s, o + 1] — вариант o.
        self.D = np.zeros((len(specs), n_opt + 1, P, E))
        self.n_options = np.array([len(s.options) for s in specs], dtype=int)
        union = H.copy()
        for s, sp in enumerate(specs):
            if sp.node not in net.node_index or net.nodes[net.node_index[sp.node]].type != "junction":
                raise ValueError(f"указатель {sp.node}: нужна развилка сети")
            outs = net.out_edges(sp.node)
            for o, (name, route) in enumerate(sp.options):
                for dest, eid in route.items():
                    if dest not in net.dest_index:
                        raise ValueError(f"указатель {sp.node}, {name}: неизвестное назначение {dest}")
                    p = net.dest_index[dest]
                    k = net.edge_index.get(eid)
                    if k is None or k not in outs:
                        raise ValueError(f"указатель {sp.node}, {name}: {eid} — не исходящее ребро")
                    if not net.edge_reaches(k, p):
                        raise ValueError(f"указатель {sp.node}, {name}: по {eid} не дойти до {dest}")
                    rec = np.zeros(E)
                    rec[k] = 1.0
                    self.D[s, o + 1, p, outs] = rec[outs] - H[p, outs]
                    union[p, k] += 1.0
        # Любая комбинация вариантов и любая c дают маршруты внутри объединения
        # «привычка ∪ все варианты» — оно обязано быть ациклическим.
        for p, dest in enumerate(net.destinations):
            cyc = net.routing_cycle(union, p)
            if cyc:
                raise ValueError(f"указатели: цикл маршрутов {dest} при объединении привычных "
                                 f"долей и всех вариантов: {' → '.join(cyc)}")

    @property
    def n_signs(self) -> int:
        return len(self.specs)

    def option_index(self, sign: str, option: str | None) -> int:
        """Индекс варианта для массива действий (−1 — нет рекомендации)."""
        sp = self.specs[self.nodes.index(sign)]
        if option is None:
            return -1
        names = [n for n, _ in sp.options]
        if option not in names:
            raise ValueError(f"указатель {sign}: нет варианта {option!r}, есть {names}")
        return names.index(option)

    def split(self, recommend: np.ndarray, c: np.ndarray) -> np.ndarray:
        """(B, P, E) эффективные доли. ``recommend`` (B, n_signs) — индекс варианта или −1,
        ``c`` (B, n_signs) — степень следования на указателе."""
        B = recommend.shape[0]
        H = np.broadcast_to(self.net.split, (B,) + self.net.split.shape)
        if self.n_signs == 0:
            return H.copy()
        if np.any(recommend < -1) or np.any(recommend >= self.n_options):
            raise ValueError("recommend: индекс варианта вне диапазона")
        D = self.D[np.arange(self.n_signs), recommend + 1]       # (B, n_signs, P, E)
        out = H + np.einsum("bs,bspe->bpe", c, D)
        np.clip(out, 0.0, 1.0, out=out)  # убирает округление вида −1e-17
        return out


# --- указатели смены станции ----------------------------------------------------------

@dataclass(frozen=True)
class SwitchSignSpec:
    node: str
    options: tuple[tuple[str, dict[str, str]], ...]  # (имя варианта, {станция: другая станция})

    @classmethod
    def from_dict(cls, d: dict) -> "SwitchSignSpec":
        sec = Section(d, f"указатель смены станции {d.get('node')}")
        node = str(sec.get("node"))
        opts = []
        for od in sec.get("options"):
            os_ = Section(od, f"указатель смены станции {node}, вариант {od.get('name')}")
            opts.append((str(os_.get("name")),
                         {str(k): str(v) for k, v in os_.get("switch").items()}))
            os_.finish()
        sec.finish()
        if not opts or len({n for n, _ in opts}) != len(opts):
            raise ValueError(f"указатель смены станции {node}: нужны варианты с разными именами")
        return cls(node, tuple(opts))


class SwitchSigns:
    """Указатели смены станции. Правила CTM (``DestSwitchRules``) всех вариантов всех
    указателей — одним массивом; доли переноса (B, R) = c указателя × [вариант выбран]."""

    def __init__(self, specs: list[SwitchSignSpec], ctm):
        net = ctm.net
        self.specs = list(specs)
        self.nodes = [s.node for s in specs]
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("несколько указателей смены станции на одном узле")
        self.n_options = np.array([len(s.options) for s in specs], dtype=int)
        parts, sign_of, opt_of = [], [], []
        for s, sp in enumerate(specs):
            if sp.node not in net.node_index or net.nodes[net.node_index[sp.node]].type != "junction":
                raise ValueError(f"указатель смены станции {sp.node}: нужна развилка сети")
            for o, (name, sw) in enumerate(sp.options):
                for src, dst in sw.items():
                    for d in (src, dst):
                        if d not in net.dest_index:
                            raise ValueError(f"указатель смены станции {sp.node}, {name}: "
                                             f"неизвестное назначение {d}")
                    r = ctm.dest_switch_rules(sp.node, src, dst)
                    parts.append(r)
                    sign_of += [s] * r.size
                    opt_of += [o] * r.size
        from sim.ctm import DestSwitchRules
        self.rules = DestSwitchRules.concat(parts)
        self.sign_of = np.array(sign_of, dtype=int)
        self.opt_of = np.array(opt_of, dtype=int)

    @property
    def n_signs(self) -> int:
        return len(self.specs)

    def option_index(self, sign: str, option: str | None) -> int:
        sp = self.specs[self.nodes.index(sign)]
        if option is None:
            return -1
        names = [n for n, _ in sp.options]
        if option not in names:
            raise ValueError(f"указатель смены станции {sign}: нет варианта {option!r}, есть {names}")
        return names.index(option)

    def fractions(self, choice: np.ndarray, c: np.ndarray) -> np.ndarray:
        """(B, R) доли переноса. ``choice`` (B, n_signs) — вариант или −1,
        ``c`` (B, n_signs) — степень следования на указателях смены станции."""
        if np.any(choice < -1) or np.any(choice >= self.n_options):
            raise ValueError("switch: индекс варианта вне диапазона")
        active = choice[:, self.sign_of] == self.opt_of[None, :]
        return np.where(active, c[:, self.sign_of], 0.0)


# --- степень следования ---------------------------------------------------------------

@dataclass
class ComplianceParams:
    """Степень следования на эпизод, по элементам пакета."""
    base: np.ndarray    # (B,)          базовая c
    offset: np.ndarray  # (B, n_signs)  смещение указателя на логит-шкале

    @property
    def batch(self) -> int:
        return self.base.size

    @classmethod
    def constant(cls, c, B: int, n_signs: int) -> "ComplianceParams":
        """Одинаковая c на всех указателях (скаляр или (B,))."""
        base = np.broadcast_to(np.asarray(c, dtype=float), (B,)).copy()
        if np.any((base < 0) | (base > 1)):
            raise ValueError("c ∈ [0, 1]")
        return cls(base, np.zeros((B, n_signs)))


class Compliance:
    def __init__(self, cfg: dict):
        sec = Section(cfg, "compliance")
        self.distribution = sec.get("distribution")
        if self.distribution == "uniform":
            self.a, self.b = float(sec.get("low")), float(sec.get("high"))
            if not 0 <= self.a <= self.b <= 1:
                raise ValueError("compliance uniform: 0 ≤ low ≤ high ≤ 1")
        elif self.distribution == "beta":
            self.a, self.b = float(sec.get("a")), float(sec.get("b"))
            if not (self.a > 0 and self.b > 0):
                raise ValueError("compliance beta: a, b > 0")
        else:
            raise ValueError("compliance.distribution: uniform или beta")
        self.node_sd = float(sec.get("node_sd_logit"))
        self.drift_sd = float(sec.get("drift_sd_logit_per_sqrt_h"))
        sec.finish()
        if self.node_sd < 0 or self.drift_sd < 0:
            raise ValueError("compliance: стандартные отклонения ≥ 0")

    def sample(self, rng: np.random.Generator, B: int, n_signs: int) -> ComplianceParams:
        base = (rng.uniform(self.a, self.b, B) if self.distribution == "uniform"
                else rng.beta(self.a, self.b, B))
        offset = (rng.normal(0.0, self.node_sd, (B, n_signs)) if self.node_sd > 0
                  else np.zeros((B, n_signs)))
        return ComplianceParams(base, offset)

    def drift(self, walk: np.ndarray, rng: np.random.Generator, dt_control: float) -> np.ndarray:
        """Шаг случайного блуждания на логит-шкале за интервал управления."""
        if self.drift_sd == 0:
            return walk
        return walk + rng.normal(0.0, self.drift_sd * np.sqrt(dt_control / 3600.0), walk.shape)


def effective_compliance(p: ComplianceParams, walk: np.ndarray) -> np.ndarray:
    """(B, n_signs) c на указателях: sigmoid(logit(base) + смещение + блуждание).
    c = 0 и c = 1 сохраняются точно."""
    with np.errstate(divide="ignore"):
        logit = np.log(p.base) - np.log1p(-p.base)
    z = logit[:, None] + p.offset + walk
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


# --- осведомлённость о закрытии входа ---------------------------------------------------

class AtDoor:
    """О закрытии входа узнают у двери: на каждом шаге CTM люди слоя станции на ребре
    «площадка → закрытый вход» переносятся на альтернативную станцию
    (``CTM.transfer_destination`` с разворотом в площадку). Остальные идут по привычке,
    пока не дойдут до двери (или их не перенаправит указатель).

    Если альтернатива — станция и она тоже закрыта, перенос не выполняется: люди ждут
    у двери (решение студента, DECISIONS.md, этап 2а)."""

    name = "at_door"

    def __init__(self, ctm, stations):
        net = ctm.net
        self.ctm = ctm
        self.stations = stations
        self.items = []   # (вход, назначение, альтернатива, рёбра, ячейки, слой, станция альтернативы)
        for e, eid in enumerate(stations.ent_ids):
            s = stations.ent_station[e]
            p, q = stations.dest[s], stations.alternative[s]
            edges = net.in_edges(eid)
            for k in edges:
                plaza = net.edges[k].source
                if not any(j != k and net.edge_reaches(j, q) for j in net.out_edges(plaza)):
                    raise ValueError(f"вход {eid}: с площадки {plaza} не дойти до альтернативы "
                                     f"{net.destinations[q]}")
            cells = np.concatenate([np.arange(ctm.cells.first[k], ctm.cells.last[k] + 1)
                                    for k in edges])
            self.items.append((e, net.destinations[p], net.destinations[q],
                               [net.edges[k].id for k in edges], cells, p,
                               int(stations.station_of_dest[q])))

    def apply(self, state, level: np.ndarray, split: np.ndarray):
        """``level`` (B, Ne) — уровни входов; ``split`` (B, P, E) — текущие доли."""
        closed = level == 0.0
        if not closed.any():
            return state
        is_open = self.stations.is_open(level)
        for e, dest, alt, edges, cells, p, s_alt in self.items:
            mask = closed[:, e] & (is_open[:, s_alt] if s_alt >= 0 else True)
            if not mask.any() or not np.any(state.n[mask][:, p, cells] > 0):
                continue
            state = self.ctm.transfer_destination(state, dest, alt, mask.astype(float), edges,
                                                  split=split)
        return state


CLOSURE_MODELS = {c.name: c for c in (AtDoor,)}
