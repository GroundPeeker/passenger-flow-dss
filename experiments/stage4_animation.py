"""Этап 4: один эпизод при c = 0,3 и c = 1,0 для стратегий LeastLoaded и фиксированной
схемы, рассчитанной на c = 1 — плотности ячеек для анимации.

Запуск:  uv run python -m experiments.stage4_animation
Конфиг — configs/experiment_stage4_animation.yaml. Результат —
experiments/results/stage4/animation.npz (плотности ячеек, траектории, рекомендации)
и animation_meta.json; рисунок — stage4/animation_compliance (analysis/fig_stage4.py).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import yaml

from control.baselines import run_policy
from experiments import stage4_compliance as exp4
from experiments.stage3_scenario import _git_rev
from sim.metrics import episode_metrics
from sim.simulator import ROOT
from sim.units import Section

CONFIG = ROOT / "configs" / "experiment_stage4_animation.yaml"
OUT = exp4.OUT_DIR / "animation.npz"


def load_config(path: Path = CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        sec = Section(yaml.safe_load(f), "experiment_stage4_animation")
    cfg = {"experiment": sec.get("experiment"), "seed": int(sec.get("seed")),
           "compliance": [float(c) for c in sec.get("compliance")],
           "strategies": list(sec.get("strategies")), "seats_pct": int(sec.get("seats_pct"))}
    sec.finish()
    return cfg


def run(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    ecfg = exp4.load_config(ROOT / cfg["experiment"])
    plan = exp4.load_plan(ecfg)
    sim = exp4.make_sim(ecfg, ecfg["evaluation"]["fidelity"], cfg["seats_pct"])
    strat = exp4.strategies(ecfg, plan, cfg["seats_pct"])
    n = int(round(ecfg["duration"] / sim.control_step))
    arrays, runs = {}, []
    for name in cfg["strategies"]:
        for c in cfg["compliance"]:
            p = exp4.episode_params(sim, [cfg["seed"]], c)
            traj, log = run_policy(sim, p, exp4.episode_rng([cfg["seed"]]), n, strat[name][1](),
                                   cfg=sim.metrics_cfg, cells=True, log_actions=True)
            m = episode_metrics(traj, sim.metrics_cfg)
            key = f"{name}|{c:g}"
            arrays[f"{key}|density"] = traj.cell_density[:, 0]
            arrays[f"{key}|in_area"] = traj.in_area[:, 0]
            arrays[f"{key}|exposure_cum"] = np.cumsum(traj.exposure[:, 0].sum(axis=1)) / 60
            arrays[f"{key}|switch"] = np.array([a.switch[0, 0] for a in log])
            runs.append({"strategy": name, "c": c, "key": key,
                         "total_ped": float(m["total_ped"][0]),
                         "mean_time_min": float(m["mean_time"][0] / 60),
                         "danger_exposure_ped_min": float(m["exposure"][0] / 60),
                         "clearance_time_min": float(m["clearance_time"][0] / 60),
                         "max_density": float(m["max_density"][0])})
    arrays["t"] = traj.t
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    meta = {"config": cfg, "runs": runs, "plan": plan["chosen"], "git": _git_rev(),
            "created": dt.datetime.now().isoformat(timespec="seconds")}
    OUT.with_name("animation_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    for r in runs:
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
    return OUT


if __name__ == "__main__":
    run()
