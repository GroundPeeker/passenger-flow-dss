"""Бенчмарк симулятора: время эпизода на эталонной сети (configs/bench.yaml).

Запуск:  uv run python -m sim.bench [--repeats N] [--batches 1 100]
Цель (постановка задачи): эпизод 1–2 ч модельного времени на 50–80 рёбер — < 1 с.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import yaml

from sim.ctm import CTM, CTMParams
from sim.network import Network, grid_network
from sim.units import Section

ROOT = Path(__file__).resolve().parents[1]
BENCH_CONFIG = ROOT / "configs" / "bench.yaml"
CTM_CONFIG = ROOT / "configs" / "ctm.yaml"


def load_bench(path: Path = BENCH_CONFIG) -> dict:
    """Читает конфиг эталонной сети и эпизода (величины в СИ)."""
    with open(path, encoding="utf-8") as f:
        top = Section(yaml.safe_load(f), "bench")
    net = Section(top.get("network"), "bench.network")
    if net.get("generator") != "grid":
        raise ValueError("bench.network.generator: поддерживается только 'grid'")
    cfg = {
        "rows": int(net.get("rows")), "cols": int(net.get("cols")), "seed": int(net.get("seed")),
        "inflow": net.quantity("inflow", "rate"),
        "sink_capacity": net.quantity("sink_capacity", "rate"),
        "duration": top.quantity("duration", "time"),
        "repeats": int(top.get("repeats")),
        "batches": [int(b) for b in top.get("batches")],
    }
    net.finish()
    top.finish()
    return cfg


def reference_network(cfg: dict) -> Network:
    """Эталонная сеть: решётка, детерминированная по seed из конфига."""
    return grid_network(cfg["rows"], cfg["cols"], np.random.default_rng(cfg["seed"]),
                        inflow=cfg["inflow"], sink_capacity=cfg["sink_capacity"])


def main(argv=None) -> None:
    cfg = load_bench()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=cfg["repeats"])
    ap.add_argument("--batches", type=int, nargs="+", default=cfg["batches"])
    args = ap.parse_args(argv)

    params = CTMParams.from_yaml(CTM_CONFIG)
    net = reference_network(cfg)
    m = CTM(net, params)
    n_steps = int(round(cfg["duration"] / params.dt))
    print(f"{platform.machine()}, numpy {np.__version__}")
    print(f"сеть: {len(net.edges)} рёбер, {len(net.nodes)} узлов, {m.n_cells} ячеек, "
          f"{m.n_dest} назначений; узлы: {'FIFO' if params.fifo else 'без FIFO'}; "
          f"debug: {params.debug}; dt = {params.dt} с, {n_steps} шагов "
          f"({cfg['duration'] / 3600:g} ч)")
    for B in args.batches:
        times = []
        for _ in range(args.repeats):
            s = m.initial_state(batch=B)
            t0 = time.perf_counter()
            s, _ = m.run(s, n_steps)
            times.append(time.perf_counter() - t0)
        t = float(np.median(times))
        print(f"B = {B:4d}: {t:.3f} с на пакет, {t / B * 1e3:.2f} мс на эпизод, "
              f"{t / n_steps * 1e6:.0f} мкс на шаг пакета "
              f"(в системе к концу: {s.in_system().mean():.0f} чел)")


if __name__ == "__main__":
    main()
