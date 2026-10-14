"""Этап 3: прогон сценария «стадион» без управления по нескольким seed.

Запуск:  uv run python -m experiments.stage3_scenario [--config configs/experiment_stage3.yaml]
Результаты (experiments/results/stage3/):
- trajectories.npz — траектории всех эпизодов (sim.metrics.Trajectory) + плотности ячеек
  эпизода анимации (``anim_cell_density``);
- metrics.csv — скалярные метрики по эпизодам;
- meta.json — конфиг эксперимента, параметры эпизодов, версия кода, дата.
Рисунки stage3/* строятся из этих файлов (analysis/fig_stage3.py).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from sim.metrics import MetricsConfig, Trajectory, episode_metrics, run_episode, scalar_table
from sim.simulator import ROOT, Simulator
from sim.units import Section

CONFIG = ROOT / "configs" / "experiment_stage3.yaml"
OUT_DIR = ROOT / "experiments" / "results" / "stage3"


def load_config(path: Path = CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    sec = Section(raw, "experiment_stage3")
    cfg = {"scenario": sec.get("scenario"), "metrics": sec.get("metrics"),
           "fidelity": sec.get("fidelity"), "n_seeds": int(sec.get("n_seeds")),
           "duration": sec.quantity("duration", "time"),
           "animation_seed": int(sec.get("animation_seed")),
           "key_edges": list(sec.get("key_edges"))}
    sec.finish()
    if not 0 <= cfg["animation_seed"] < cfg["n_seeds"]:
        raise ValueError("animation_seed должен быть среди seed 0 … n_seeds − 1")
    return cfg


def config_hash(cfg: dict) -> str:
    """Хэш конфига эксперимента и всех входящих в него файлов (сценарий, сеть, CTM,
    поведение, метрики) — чтобы рисунки не строились по устаревшим результатам."""
    with open(ROOT / cfg["scenario"], encoding="utf-8") as f:
        scen = yaml.safe_load(f)
    files = [cfg["scenario"], cfg["metrics"], scen["network"], scen["ctm"], scen["behavior"]]
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    for p in files:
        h.update((ROOT / p).read_bytes())
    return h.hexdigest()[:16]


def _git_rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(cfg: dict, out_dir: Path = OUT_DIR, verbose: bool = True) -> Path:
    mcfg = MetricsConfig.from_yaml(ROOT / cfg["metrics"])
    sim = Simulator.from_yaml(ROOT / cfg["scenario"], fidelity=cfg["fidelity"], metrics=mcfg)
    zone_names = [sim.net.edges[int(sim.ctm.cells.edge[c])].id for c in sim.ctm.zone_cells]
    n_steps = int(round(cfg["duration"] / sim.control_step))
    trajs, params, anim = [], [], None
    t0 = time.perf_counter()
    for seed in range(cfg["n_seeds"]):
        rng = np.random.default_rng(seed)
        p = sim.sample_params(rng, 1)
        tr = run_episode(sim, p, rng, n_steps, cfg=mcfg, cells=seed == cfg["animation_seed"])
        if tr.cell_density is not None:
            anim = tr.cell_density[:, 0]
            tr.cell_density = None
        trajs.append(tr)
        params.append({"seed": seed, "total_ped": float(p.demand.total[0]),
                       "sector_share": p.demand.sector_share[0].tolist(),
                       "exit_profile": {k: float(v[0]) for k, v in p.demand.profile.items()}})
    elapsed = time.perf_counter() - t0
    traj = Trajectory.concat(trajs)
    m = episode_metrics(traj, mcfg)
    table = scalar_table(m, zone_names)
    table.insert(0, "seed", np.arange(cfg["n_seeds"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    traj.save(out_dir / "trajectories.npz", anim_cell_density=anim)
    table.to_csv(out_dir / "metrics.csv", index=False)
    net = sim.net
    meta = {"experiment": "stage3_scenario", "config": cfg, "config_hash": config_hash(cfg),
            "git": _git_rev(), "created": dt.datetime.now().isoformat(timespec="seconds"),
            "policy": "без управления (все входы открыты, рекомендаций нет)",
            "control_step_s": sim.control_step, "rho_crit_street_ped_per_m2": mcfg.rho_crit_street,
            "rho_crit_zone_ped_per_m2": mcfg.rho_crit_zone,
            "congestion_street_ped_per_m2": mcfg.congestion_street,
            "congestion_zone_ped_per_m2": mcfg.congestion_zone,
            "zone_full_fraction": mcfg.zone_full_fraction,
            "zones": zone_names,
            "queue_density_ped_per_m2": (sim.ctm.fd.rho_crit if mcfg.queue_density is None
                                         else mcfg.queue_density),
            "edges": [e.id for e in net.edges], "stations": sim.stations.ids,
            "station_links": [f"{sp.id}:{lk.name}" for sp in sim.stations.specs for lk in sp.links],
            "link_storage_ped": sim.stations.storage.tolist(),
            "n_cells": sim.ctm.n_cells, "episodes": params, "elapsed_s": round(elapsed, 2)}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    if verbose:
        print(f"{cfg['n_seeds']} эпизодов × {n_steps} интервалов, {elapsed:.1f} с → {out_dir}")
        cols = ["seed", "total_ped", "clearance_time_min", "mean_time_min", "p95_time_min",
                "danger_exposure_ped_min", "danger_area_max_m2", "congestion_exposure_ped_min",
                "zone_full_e_sq_min", "max_density"]
        print(table[cols].round(1).to_string(index=False))
        print("доля эпизодов с перегрузкой:", float(np.mean(m["overload_any"])))
    return out_dir


def load(cfg: dict | None = None, out_dir: Path = OUT_DIR, rerun_if_stale: bool = True):
    """Результаты эксперимента: (Trajectory, плотности ячеек эпизода анимации, meta).
    Если результатов нет или они получены с другим конфигом — пересчитываются
    (при ``rerun_if_stale``), иначе ошибка."""
    cfg = cfg or load_config()
    meta_path = out_dir / "meta.json"
    stale = (not meta_path.exists() or
             json.loads(meta_path.read_text(encoding="utf-8")).get("config_hash") != config_hash(cfg))
    if stale:
        if not rerun_if_stale:
            raise FileNotFoundError(f"нет актуальных результатов в {out_dir}: запустите "
                                    f"uv run python -m experiments.stage3_scenario")
        print(f"результаты этапа 3 отсутствуют или устарели — пересчёт ({out_dir})")
        run(cfg, out_dir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    traj = Trajectory.load(out_dir / "trajectories.npz")
    with np.load(out_dir / "trajectories.npz") as z:
        anim = z["anim_cell_density"]
    return traj, anim, meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    run(load_config(args.config), args.out)


if __name__ == "__main__":
    main()
