"""Проверка детерминизма: два прогона одного сценария с одним seed дают побитово
одинаковые результаты (постановка задачи, «Проверка этапов»).

1. Ядро CTM: эталонная сеть и эпизод из configs/bench.yaml; сеть каждый раз строится
   заново из seed. Сравниваются все массивы конечного состояния и история чисел людей
   в ячейках. Режимы: узлы без FIFO и с FIFO, пакет из нескольких прогонов с разными
   потоками и обходом затора.
2. Полный сценарий (configs/scenario_toy.yaml): семплирование спроса и степени следования,
   дрейф c и различие по указателям, шум счётчиков, случайный выпуск со стадиона,
   дискретные поезда (точность truth), рекомендации,
   закрытие входа, дополнительные поезда — все источники случайности этапа 2.
3. Сценарий «стадион» (configs/scenario_stadium.yaml, этап 3): эпизод без управления
   с записью траектории и метрик (sim/metrics.py).

Запуск:  uv run python -m experiments.check_determinism
Код возврата 0 — побитово совпадает, 1 — нет.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import yaml

from sim.bench import CTM_CONFIG, load_bench, reference_network
from sim.ctm import CTM, CTMParams, CTMState
from sim.simulator import Actions, Episode, Simulator, ROOT

SCENARIO = ROOT / "configs" / "scenario_toy.yaml"
STADIUM = ROOT / "configs" / "scenario_stadium.yaml"

STATE_FIELDS = [f.name for f in dataclasses.fields(CTMState) if f.name != "step"]


def run_scenario(cfg: dict, fifo: bool, n_steps: int | None = None) -> dict[str, np.ndarray]:
    """Один прогон сценария; возвращает массивы для сравнения."""
    params = dataclasses.replace(CTMParams.from_yaml(CTM_CONFIG), fifo=fifo)
    m = CTM(reference_network(cfg), params)
    n_steps = n_steps or int(round(cfg["duration"] / params.dt))
    B = 3
    inflow = np.array([0.5, 1.0, 2.0])[:, None, None] * m.source_inflow
    detour = np.array([0.0, 0.0, 0.01])
    s, hist = m.run(m.initial_state(batch=B), n_steps, record=True,
                    inflow=inflow, detour_rate=detour)
    out = {f: getattr(s, f) for f in STATE_FIELDS}
    out["history"] = hist
    return out


def run_simulator(seed: int, n_control: int = 60) -> dict[str, np.ndarray]:
    """Эпизод полного сценария со всеми источниками случайности; пакет из 3 эпизодов."""

    with open(SCENARIO, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["compliance"].update(node_sd_logit=0.5, drift_sd_logit_per_sqrt_h=0.5)
    sim = Simulator(cfg, fidelity="truth")   # случайный выпуск + дискретные поезда
    rng = np.random.default_rng(seed)
    B = 3
    ep = Episode(sim, sim.sample_params(rng, B), rng)
    obs = []
    for k in range(n_control):
        a = Actions.from_dict(sim, B, recommend={"J1": "south" if k % 20 < 10 else "north"},
                              entrances={"M1": 0.0} if 15 <= k < 25 else {},
                              extra_trains=["B"] if k == 5 else [])
        obs.append(ep.step(a).values)
    s = ep.state
    out = {f"ctm.{f}": getattr(s.ctm, f) for f in STATE_FIELDS}
    out.update({"stations.x": s.stations.x, "stations.departed": s.stations.departed,
                "walk": s.walk, "observations": np.stack(obs),
                "demand.total": s.params.demand.total, "stadium": s.stadium})
    return out


def run_stadium(seed: int, n_control: int = 60) -> dict[str, np.ndarray]:
    """Эпизод сценария «стадион» (truth) с записью траектории и метриками."""
    from sim.metrics import episode_metrics, run_episode
    sim = Simulator.from_yaml(STADIUM, fidelity="truth")
    rng = np.random.default_rng(seed)
    tr = run_episode(sim, sim.sample_params(rng, 2), rng, n_control, cells=True)
    out = {f"traj.{k}": v for k, v in vars(tr).items() if isinstance(v, np.ndarray)}
    out.update({f"metrics.{k}": np.asarray(v) for k, v in episode_metrics(tr).items()})
    return out


def compare(a: dict, b: dict) -> list[str]:
    """Имена массивов, которые не совпадают побитово."""
    return [k for k in a if a[k].shape != b[k].shape or a[k].tobytes() != b[k].tobytes()]


def check(cfg: dict | None = None, n_steps: int | None = None, verbose: bool = True) -> bool:
    cfg = cfg or load_bench()
    ok = True
    for fifo in (False, True):
        r1 = run_scenario(cfg, fifo, n_steps)
        r2 = run_scenario(cfg, fifo, n_steps)
        bad = compare(r1, r2)
        ok &= not bad
        if verbose:
            label = "FIFO" if fifo else "без FIFO"
            status = "совпадает побитово" if not bad else f"РАСХОЖДЕНИЕ: {', '.join(bad)}"
            print(f"seed {cfg['seed']}, узлы {label}: {len(r1)} массивов, "
                  f"история {r1['history'].shape} — {status}")
    # Контроль чувствительности: другой seed даёт другую сеть и другой результат.
    other = dict(cfg, seed=cfg["seed"] + 1)
    differs = bool(compare(run_scenario(cfg, False, n_steps), run_scenario(other, False, n_steps)))
    if verbose:
        print(f"seed {cfg['seed']} и {other['seed']}: результаты "
              f"{'различаются (ожидаемо)' if differs else 'СОВПАДАЮТ — seed не влияет'}")
    # Полный сценарий этапа 2.
    seed = cfg["seed"]
    n_control = 60 if n_steps is None else max(5, n_steps // 12)
    r1, r2 = run_simulator(seed, n_control), run_simulator(seed, n_control)
    bad = compare(r1, r2)
    ok &= not bad
    sim_differs = bool(compare(r1, run_simulator(seed + 1, n_control)))
    if verbose:
        status = "совпадает побитово" if not bad else f"РАСХОЖДЕНИЕ: {', '.join(bad)}"
        print(f"сценарий {SCENARIO.relative_to(ROOT)}, seed {seed}: {len(r1)} массивов, "
              f"{n_control} интервалов управления — {status}")
        print(f"сценарий, seed {seed} и {seed + 1}: результаты "
              f"{'различаются (ожидаемо)' if sim_differs else 'СОВПАДАЮТ — seed не влияет'}")
    # Сценарий «стадион» с метриками (этап 3).
    r1, r2 = run_stadium(seed, n_control), run_stadium(seed, n_control)
    bad = compare(r1, r2)
    ok &= not bad
    st_differs = bool(compare(r1, run_stadium(seed + 1, n_control)))
    if verbose:
        status = "совпадает побитово" if not bad else f"РАСХОЖДЕНИЕ: {', '.join(bad)}"
        print(f"сценарий {STADIUM.relative_to(ROOT)}, seed {seed}: {len(r1)} массивов "
              f"(траектория и метрики), {n_control} интервалов управления — {status}")
        print(f"«стадион», seed {seed} и {seed + 1}: результаты "
              f"{'различаются (ожидаемо)' if st_differs else 'СОВПАДАЮТ — seed не влияет'}")
    return ok and differs and sim_differs and st_differs


def main() -> int:
    ok = check()
    print("ИТОГ:", "детерминизм подтверждён" if ok else "детерминизм НАРУШЕН")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
