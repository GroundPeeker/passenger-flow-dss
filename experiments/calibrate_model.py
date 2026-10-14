"""Калибровка эффективной пропускной способности станций в модели планирования.

Запуск:  uv run python -m experiments.calibrate_model
Конфиг — configs/experiment_calibration.yaml (процедура описана там). Результаты:
- configs/calibration_stadium.yaml — множители κ (подключается в блоке
  ``fidelity.model.calibration`` сценария), процедура и результат проверки;
- experiments/results/calibration/loss.csv — отклонение очереди от κ по станциям и режимам;
- experiments/results/calibration/validation.csv — метрики «реальности», модели без
  калибровки и с калибровкой на проверочных seed по эпизодам;
- experiments/results/calibration/meta.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from control.baselines import (FixedSchedule, Lever, LeastLoadedStation, NoControl,
                               StationQueue, run_policy)
from experiments.stage3_scenario import _git_rev
from experiments.stage4_compliance import episode_params, episode_rng
from sim.metrics import MetricsConfig, episode_metrics
from sim.simulator import ROOT, Simulator
from sim.units import Section

CONFIG = ROOT / "configs" / "experiment_calibration.yaml"
OUT_DIR = ROOT / "experiments" / "results" / "calibration"
LL_QUEUES = {"A": StationQueue("e_sq", ("e_av", "e_pj")), "B": StationQueue("e_mb", ("e_pb",))}


def load_config(path: Path = CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        sec = Section(yaml.safe_load(f), "experiment_calibration")
    seeds = lambda d: list(range(int(d["first"]), int(d["first"]) + int(d["count"])))
    cfg = {"scenario": sec.get("scenario"), "metrics": sec.get("metrics"),
           "duration": sec.quantity("duration", "time"),
           "calibration_seeds": seeds(sec.get("calibration_seeds")),
           "validation_seeds": seeds(sec.get("validation_seeds")),
           "factors": [float(x) for x in sec.get("factors")],
           "regimes": list(sec.get("regimes")), "validation_extra": list(sec.get("validation_extra")),
           "output": sec.get("output")}
    sec.finish()
    if set(cfg["calibration_seeds"]) & set(cfg["validation_seeds"]):
        raise ValueError("seed калибровки и проверки пересекаются")
    if set(cfg["calibration_seeds"] + cfg["validation_seeds"]) & set(range(50)) | \
            set(cfg["calibration_seeds"] + cfg["validation_seeds"]) & set(range(1000, 1010)):
        raise ValueError("seed калибровки/проверки пересекаются с оценочными или планировочными")
    return cfg


def make_sim(cfg, fidelity, factor: dict | None = None, calibrated: bool = False) -> Simulator:
    with open(ROOT / cfg["scenario"], encoding="utf-8") as f:
        scen = yaml.safe_load(f)
    return Simulator(scen, ROOT, fidelity=fidelity, calibrated=calibrated,
                     throughput_factor=factor,
                     metrics=MetricsConfig.from_yaml(ROOT / cfg["metrics"]))


def regime_policy(r: dict):
    """(стратегия, c) режима."""
    if r["name"] == "no_control":
        return NoControl(), 1.0
    if r["name"].startswith("least_loaded"):
        return LeastLoadedStation("JN", "to_B", "A", "B", LL_QUEUES), float(r["compliance"])
    return (FixedSchedule(Lever("switch", r["sign"]), r["option"], float(r["fraction"])),
            float(r["compliance"]))


def simulate(sim, cfg, seeds, r):
    pol, c = regime_policy(r)
    n = int(round(cfg["duration"] / sim.control_step))
    return run_policy(sim, episode_params(sim, seeds, c), episode_rng(seeds), n, pol,
                      cfg=sim.metrics_cfg)


def summary(traj, cfg) -> dict[str, np.ndarray]:
    m = episode_metrics(traj, cfg)
    return {"danger_exposure_ped_min": m["exposure"] / 60, "mean_time_min": m["mean_time"] / 60,
            "clearance_time_min": m["clearance_time"] / 60,
            "congestion_exposure_ped_min": m["congestion_exposure"] / 60,
            "max_queue_A": m["max_queue_outside"][:, 0], "max_queue_B": m["max_queue_outside"][:, 1],
            "max_queue_C": m["max_queue_outside"][:, 2]}


def run(cfg: dict, out_dir: Path = OUT_DIR) -> dict:
    t0 = time.perf_counter()
    seeds = cfg["calibration_seeds"]
    truth = make_sim(cfg, "truth")
    ids = truth.stations.ids
    q_true = {r["name"]: simulate(truth, cfg, seeds, r).queue_outside for r in cfg["regimes"]}
    rows = []
    for k in cfg["factors"]:
        sim = make_sim(cfg, "model", {s: k for s in ids})
        for r in cfg["regimes"]:
            q = simulate(sim, cfg, seeds, r).queue_outside        # (T, B, S)
            mse = ((q - q_true[r["name"]]) ** 2).mean(axis=(0, 1))
            bias = (q - q_true[r["name"]]).mean(axis=(0, 1))
            for s, sid in enumerate(ids):
                rows.append({"factor": k, "regime": r["name"], "station": sid,
                             "rmse_ped": float(np.sqrt(mse[s])), "bias_ped": float(bias[s]),
                             "mean_queue_truth": float(q_true[r["name"]][..., s].mean())})
    loss = pd.DataFrame(rows)
    # критерий: средний квадрат отклонения по режимам → минимум по κ для каждой станции
    tot = (loss.assign(mse=loss.rmse_ped ** 2).groupby(["station", "factor"]).mse.mean()
           .reset_index())
    chosen = {s: float(g.loc[g.mse.idxmin(), "factor"]) for s, g in tot.groupby("station")}

    # проверка на отдельных seed: «реальность», модель без калибровки, модель с калибровкой
    vseeds = cfg["validation_seeds"]
    sims = {"truth": truth, "model_uncalibrated": make_sim(cfg, "model"),
            "model_calibrated": make_sim(cfg, "model", chosen)}
    vrows = []
    for r in cfg["regimes"] + cfg["validation_extra"]:
        for name, sim in sims.items():
            m = summary(simulate(sim, cfg, vseeds, r), sim.metrics_cfg)
            for j, s in enumerate(vseeds):
                vrows.append({"regime": r["name"], "model": name, "seed": s,
                              **{k: float(v[j]) for k, v in m.items()}})
    val = pd.DataFrame(vrows)
    agg = val.groupby(["regime", "model"]).mean(numeric_only=True).drop(columns="seed")
    ratio = {}
    for reg, g in agg.groupby(level="regime"):
        t = g.loc[(reg, "truth")]
        ratio[reg] = {mdl: {k: float(g.loc[(reg, mdl), k] / t[k]) if t[k] else None
                            for k in ("danger_exposure_ped_min", "mean_time_min",
                                      "clearance_time_min", "congestion_exposure_ped_min")}
                      for mdl in ("model_uncalibrated", "model_calibrated")}
    out_dir.mkdir(parents=True, exist_ok=True)
    loss.to_csv(out_dir / "loss.csv", index=False)
    val.to_csv(out_dir / "validation.csv", index=False)
    meta = {"config": cfg, "chosen": chosen, "ratio_to_truth": ratio, "git": _git_rev(),
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(time.perf_counter() - t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    write_calibration(cfg, chosen, ratio, meta)
    print("κ:", chosen)
    print(agg.round(1).to_string())
    print(json.dumps(ratio, ensure_ascii=False, indent=1))
    return meta


def write_calibration(cfg, chosen, ratio, meta) -> None:
    lines = [
        "# Калибровка модели планирования (fidelity: model) сценария «стадион».",
        "# Создан experiments/calibrate_model.py (configs/experiment_calibration.yaml) —",
        "# вручную не править. Процедура и результат — docs/DECISIONS.md, этап 4 (доработка).",
        "#",
        "# κ — множитель эффективной пропускной способности вывоза в модели с непрерывным",
        "# оттоком (вместимость / интервал · κ): подобран по очередям перед входами станций",
        f"# «реальности» (дискретные поезда) на seed {cfg['calibration_seeds'][0]}–"
        f"{cfg['calibration_seeds'][-1]}, режимы: "
        + ", ".join(r["name"] for r in cfg["regimes"]) + ".",
        "vehicle_throughput_factor:",
        *[f"  {s}: {k:.2f}" for s, k in chosen.items()],
        "# Проверка на seed "
        f"{cfg['validation_seeds'][0]}–{cfg['validation_seeds'][-1]}: отношение средних метрик",
        "# модели к «реальности» (без калибровки → с калибровкой).",
        "validation_ratio_to_truth:",
    ]
    for reg, d in ratio.items():
        lines.append(f"  {reg}:")
        for k in d["model_uncalibrated"]:
            u, c = d["model_uncalibrated"][k], d["model_calibrated"][k]
            fmt = lambda x: "null" if x is None else f"{x:.3f}"
            lines.append(f"    {k}: {{uncalibrated: {fmt(u)}, calibrated: {fmt(c)}}}")
    lines += [f"created: '{meta['created']}'", f"git: '{meta['git']}'"]
    (ROOT / cfg["output"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
