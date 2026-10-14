"""Этап 4: базовые стратегии при разной степени следования c (мотивирующий эксперимент).

Два шага, по порядку (схемы фиксируются в DECISIONS.md до оценочных прогонов):
  uv run python -m experiments.stage4_compliance plan   # подбор фиксированных схем
  uv run python -m experiments.stage4_compliance run    # оценочные прогоны
Конфиг — configs/experiment_stage4.yaml. Результаты (experiments/results/stage4/):
- plan.csv  — f → средние по seed планирования экспозиция и среднее время, по схемам;
- plan_episodes.csv — то же по seed, с экспозицией и максимумом плотности по участкам
  (``exposure_<ребро>``, чел·мин; ``max_density_<ребро>``, чел/м²) — для рисунка
  stage4/fixed_tradeoff и разбора механизма;
- plan.json — выбранные f, хэш конфига;
- metrics.csv — метрики эпизодов: вариант мест × порог ρ_крит × стратегия × c × seed;
- summary.csv — среднее и 95 % ДИ (t-распределение по seed) по стратегиям и c;
- paired.csv — парные разности со стратегией «без управления» на тех же эпизодах
  (основной статистический анализ): среднее, 95 % ДИ, число эпизодов лучше / хуже;
- metrics_attendance.csv, paired_attendance.csv — то же при фиксированной явке N;
- n10/ — результаты первой серии (10 seed), сохранены для сравнения;
- meta.json — конфиг, схемы, версия кода, дата, время счёта.
Рисунки stage4/* строятся из этих файлов (analysis/fig_stage4.py).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from control.baselines import (FixedSchedule, Lever, LeastLoadedRoute, LeastLoadedStation,
                               NoControl, StationQueue, run_policy)
from experiments.stage3_scenario import _git_rev
from sim.behavior import ComplianceParams
from sim.demand import DemandParams
from sim.metrics import MetricsConfig, episode_metrics
from sim.simulator import ROOT, EpisodeParams, Simulator
from sim.units import Section

CONFIG = ROOT / "configs" / "experiment_stage4.yaml"
OUT_DIR = ROOT / "experiments" / "results" / "stage4"

# Метрики для таблиц и рисунков: колонка → (ключ episode_metrics, множитель).
METRICS = {
    "mean_time_min": ("mean_time", 1 / 60),
    "danger_exposure_ped_min": ("exposure", 1 / 60),
    "clearance_time_min": ("clearance_time", 1 / 60),
    "max_density": ("max_density", 1.0),
    "p95_time_min": ("p95_time", 1 / 60),
    "overload_any": ("overload_any", 1.0),
    "congestion_exposure_ped_min": ("congestion_exposure", 1 / 60),
    "total_ped": ("total_ped", 1.0),
}


def _seeds(x) -> list[int]:
    """Список seed или {first, count}."""
    if isinstance(x, dict):
        return list(range(int(x["first"]), int(x["first"]) + int(x["count"])))
    return [int(s) for s in x]


def load_config(path: Path = CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    sec = Section(raw, "experiment_stage4")
    ev, pl = Section(sec.get("evaluation"), "evaluation"), Section(sec.get("planning"), "planning")
    lv, sv = Section(sec.get("levers"), "levers"), Section(sec.get("seat_variants"), "seat_variants")
    cfg = {
        "scenario": sec.get("scenario"), "metrics": sec.get("metrics"),
        "duration": sec.quantity("duration", "time"),
        "evaluation": {"fidelity": ev.get("fidelity"), "seeds": _seeds(ev.get("seeds")),
                       "compliance": [float(c) for c in ev.get("compliance")],
                       "attendance": [float(n) for n in ev.get("attendance_ped")]},
        "planning": {"fidelity": pl.get("fidelity"), "calibrated": bool(pl.get("calibrated")),
                     "seeds": [int(s) for s in pl.get("seeds")],
                     "fractions": [float(f) for f in pl.get("fractions")],
                     "tolerance": float(pl.get("tolerance")),
                     "design_compliance_switch": [float(c) for c in pl.get("design_compliance_switch")],
                     "design_compliance_route": [float(c) for c in pl.get("design_compliance_route")]},
        "levers": {"switch": lv.get("switch"), "route": lv.get("route")},
        "least_loaded_every": int(sec.get("least_loaded_every")),
        "seat_variants": {"base": int(sv.get("base")), "stations": list(sv.get("stations")),
                          "capacity_ped": {int(k): float(v)
                                           for k, v in sv.get("capacity_ped").items()}},
        "exposure_thresholds_street": [float(x) for x in sec.get("exposure_thresholds_street")],
    }
    for s in (sec, ev, pl, lv, sv):
        s.finish()
    if set(cfg["evaluation"]["seeds"]) & set(cfg["planning"]["seeds"]):
        raise ValueError("seed планирования и оценки пересекаются")
    if cfg["seat_variants"]["base"] not in cfg["seat_variants"]["capacity_ped"]:
        raise ValueError("seat_variants.base — один из вариантов capacity_ped")
    return cfg


def config_hash(cfg: dict) -> str:
    with open(ROOT / cfg["scenario"], encoding="utf-8") as f:
        scen = yaml.safe_load(f)
    files = [cfg["scenario"], cfg["metrics"], scen["network"], scen["ctm"], scen["behavior"]]
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode())
    for p in files:
        h.update((ROOT / p).read_bytes())
    return h.hexdigest()[:16]


# --- сценарий и параметры эпизодов ---------------------------------------------------------

def make_sim(cfg: dict, fidelity: str, seats: int, rho_crit_street: float | None = None,
             calibrated: bool | None = None) -> Simulator:
    """Симулятор сценария с вариантом свободных мест ``seats`` (%) и, при необходимости,
    другим порогом ρ_крит улиц (только мониторинг — на динамику не влияет)."""
    with open(ROOT / cfg["scenario"], encoding="utf-8") as f:
        scen = json.loads(json.dumps(yaml.safe_load(f)))   # копия без общих якорей YAML
    cap = cfg["seat_variants"]["capacity_ped"][seats]
    for st in scen["stations"]:
        if st["id"] in cfg["seat_variants"]["stations"]:
            st["vehicle"]["capacity_ped"] = cap
    mcfg = MetricsConfig.from_yaml(ROOT / cfg["metrics"])
    if rho_crit_street is not None:
        mcfg = dataclasses.replace(mcfg, rho_crit_street=rho_crit_street)
    if calibrated is None:     # модель планирования этапа 4 — без калибровки (конфиг)
        calibrated = cfg["planning"]["calibrated"]
    return Simulator(scen, ROOT, fidelity=fidelity, metrics=mcfg, calibrated=calibrated)


def concat_params(parts: list[EpisodeParams]) -> EpisodeParams:
    cat = lambda xs: np.concatenate(xs, axis=0)
    d = [p.demand for p in parts]
    demand = DemandParams(cat([x.total for x in d]), cat([x.sector_share for x in d]),
                          cat([x.dest_share for x in d]),
                          {k: cat([x.profile[k] for x in d]) for k in d[0].profile})
    comp = ComplianceParams(cat([p.compliance.base for p in parts]),
                            cat([p.compliance.offset for p in parts]))
    return EpisodeParams(demand, comp, cat([p.first_departure for p in parts]))


def episode_params(sim: Simulator, seeds: list[int], c: float,
                   attendance: float | None = None) -> EpisodeParams:
    """Параметры эпизодов: seed s → Generator(s) → ``sample_params``; c — постоянная
    на всех указателях (одинаковые N, профиль, доли, фаза поездов для всех c);
    ``attendance`` — фиксированная явка N вместо случайной (остальное — из seed)."""
    p = concat_params([sim.sample_params(np.random.default_rng(s), 1) for s in seeds])
    p.compliance = ComplianceParams.constant(c, p.batch, sim.n_compliance)
    if attendance is not None:
        p.demand.total = np.full(p.batch, float(attendance))
    return p


def episode_rng(seeds: list[int]) -> np.random.Generator:
    """Случайность внутри эпизодов (выпуск, поезда, шум счётчиков) — одна и та же для
    всех c и всех стратегий (общие случайные числа; расходится по ходу эпизода, когда
    действия меняют число вызовов генератора, например пуассоновский шум счётчиков)."""
    return np.random.default_rng(seeds)


# --- стратегии ------------------------------------------------------------------------------

def switch_lever(cfg):
    sw = cfg["levers"]["switch"]
    return Lever("switch", sw["sign"]), sw["option"]


def route_lever(cfg):
    rt = cfg["levers"]["route"]
    return Lever("route", rt["sign"]), rt["option"]


def strategies(cfg: dict, plan: dict, seats: int) -> dict:
    """Стратегии оценочных прогонов: имя → (рычаг, конструктор)."""
    sw, rt = cfg["levers"]["switch"], cfg["levers"]["route"]
    every = cfg["least_loaded_every"]
    out = {"no_control": ("—", NoControl)}
    lev, opt = switch_lever(cfg)
    for c in cfg["planning"]["design_compliance_switch"]:
        f = plan["chosen"][scheme_key("switch", seats, c)]["fraction"]
        out[f"fixed_c{c:g}"] = ("switch", lambda f=f: FixedSchedule(lev, opt, f))
    queues = {s: StationQueue(q["plaza"], tuple(q["approach"])) for s, q in sw["queues"].items()}
    out["least_loaded"] = ("switch", lambda: LeastLoadedStation(
        sw["sign"], sw["option"], sw["from_station"], sw["to_station"], queues, every,
        sw["full_fraction"]))
    if seats == cfg["seat_variants"]["base"]:
        rlev, ropt = route_lever(cfg)
        for c in cfg["planning"]["design_compliance_route"]:
            f = plan["chosen"][scheme_key("route", seats, c)]["fraction"]
            out[f"route_fixed_c{c:g}"] = ("route", lambda f=f: FixedSchedule(rlev, ropt, f))
        out["route_least_loaded"] = ("route", lambda: LeastLoadedRoute(
            rt["sign"], rt["routes"], every))
    return out


def scheme_key(lever: str, seats: int, c: float) -> str:
    return f"{lever}|{seats}|{c:g}"


def run_batch(sim, cfg, seeds, c, policy, by_edge: bool = False,
              attendance: float | None = None) -> dict[str, np.ndarray]:
    n_steps = int(round(cfg["duration"] / sim.control_step))
    traj = run_policy(sim, episode_params(sim, seeds, c, attendance), episode_rng(seeds), n_steps,
                      policy, cfg=sim.metrics_cfg)
    m = episode_metrics(traj, sim.metrics_cfg)
    out = {col: np.asarray(m[key], dtype=float) * mul for col, (key, mul) in METRICS.items()}
    if by_edge:
        for j, e in enumerate(sim.net.edges):
            out[f"exposure_{e.id}"] = m["exposure_edge"][:, j] / 60.0
            out[f"max_density_{e.id}"] = m["max_density_edge"][:, j]
    return out


# --- шаг 1: подбор схем ---------------------------------------------------------------------

def choose(table: pd.DataFrame, tol: float) -> pd.Series:
    """Лексикографический выбор: минимум экспозиции, среди f с экспозицией
    ≤ минимум·(1 + tol) — минимум среднего времени; при равенстве — меньшая f."""
    e_min = table["danger_exposure_ped_min"].min()
    ok = table[table["danger_exposure_ped_min"] <= e_min * (1 + tol) + 1e-9]
    return ok.sort_values(["mean_time_min", "fraction"]).iloc[0]


def plan(cfg: dict, out_dir: Path = OUT_DIR, verbose: bool = True) -> dict:
    pl = cfg["planning"]
    rows, episodes, chosen = [], [], {}
    t0 = time.perf_counter()
    jobs = [("switch", s, c) for s in cfg["seat_variants"]["capacity_ped"]
            for c in pl["design_compliance_switch"]]
    jobs += [("route", cfg["seat_variants"]["base"], c) for c in pl["design_compliance_route"]]
    for lever, seats, c in jobs:
        sim = make_sim(cfg, pl["fidelity"], seats)
        lev, opt = switch_lever(cfg) if lever == "switch" else route_lever(cfg)
        part = []
        for f in pl["fractions"]:
            m = run_batch(sim, cfg, pl["seeds"], c, FixedSchedule(lev, opt, f), by_edge=True)
            for j, s in enumerate(pl["seeds"]):
                episodes.append({"lever": lever, "seats_pct": seats, "design_c": c,
                                 "fraction": f, "seed": s,
                                 **{k: float(v[j]) for k, v in m.items()}})
            m = {k: m[k] for k in METRICS}
            part.append({"lever": lever, "seats_pct": seats, "design_c": c, "fraction": f,
                         **{k: float(np.mean(v)) for k, v in m.items()}})
        tab = pd.DataFrame(part)
        best = choose(tab, pl["tolerance"])
        tab["chosen"] = tab["fraction"] == best["fraction"]
        rows.append(tab)
        chosen[scheme_key(lever, seats, c)] = {
            "lever": lever, "seats_pct": seats, "design_c": c, "fraction": float(best["fraction"]),
            "danger_exposure_ped_min": float(best["danger_exposure_ped_min"]),
            "mean_time_min": float(best["mean_time_min"])}
        if verbose:
            print(f"{lever}, мест {seats} %, c = {c:g}: f = {best['fraction']:g}")
            print(tab[["fraction", "danger_exposure_ped_min", "mean_time_min",
                       "clearance_time_min"]].round(1).to_string(index=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(out_dir / "plan.csv", index=False)
    pd.DataFrame(episodes).to_csv(out_dir / "plan_episodes.csv", index=False)
    res = {"config_hash": config_hash(cfg), "git": _git_rev(),
           "created": dt.datetime.now().isoformat(timespec="seconds"),
           "elapsed_s": round(time.perf_counter() - t0, 1), "chosen": chosen}
    (out_dir / "plan.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    return res


def load_plan(cfg: dict, out_dir: Path = OUT_DIR) -> dict:
    path = out_dir / "plan.json"
    if not path.exists():
        raise FileNotFoundError("нет plan.json: сначала uv run python -m "
                                "experiments.stage4_compliance plan")
    res = json.loads(path.read_text(encoding="utf-8"))
    if res["config_hash"] != config_hash(cfg):
        raise ValueError("plan.json получен с другим конфигом — повторите шаг plan")
    return res


# --- шаг 2: оценочные прогоны ---------------------------------------------------------------

def ci95(x: np.ndarray) -> tuple[float, float, float]:
    """Среднее и 95 % доверительный интервал среднего (t-распределение); при inf — inf."""
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    if not np.all(np.isfinite(x)) or x.size < 2:
        return m, m, m
    h = float(stats.t.ppf(0.975, x.size - 1) * np.std(x, ddof=1) / np.sqrt(x.size))
    return m, m - h, m + h


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["seats_pct", "rho_crit_street", "strategy", "lever", "c"]
    for k, g in df.groupby(keys, sort=False):
        row = dict(zip(keys, k))
        for col in METRICS:
            m, lo, hi = ci95(g[col].to_numpy())
            row[col] = m
            row[col + "_ci_lo"], row[col + "_ci_hi"] = lo, hi
            row[col + "_min"], row[col + "_max"] = float(g[col].min()), float(g[col].max())
        rows.append(row)
    return pd.DataFrame(rows)


PAIRED_METRICS = ("mean_time_min", "danger_exposure_ped_min", "clearance_time_min",
                  "max_density", "p95_time_min")


def paired(df: pd.DataFrame, keys: list[str], duration_min: float) -> pd.DataFrame:
    """Парные разности метрик стратегии и «без управления» на тех же эпизодах (keys + c +
    seed): среднее, 95 % ДИ (t), эпизодов лучше / хуже (разность < 0 / > 0). Не рассосавшиеся
    эпизоды (inf) — со временем рассасывания = длительности эпизода (оценка снизу)."""
    ref = df[df.strategy == "no_control"].set_index(keys + ["c", "seed"])
    rows = []
    for k, g in df[df.strategy != "no_control"].groupby(keys + ["strategy", "lever", "c"],
                                                        sort=False):
        g = g.set_index(keys + ["c", "seed"])
        r = ref.loc[g.index]
        row = dict(zip(keys + ["strategy", "lever", "c"], k))
        for col in PAIRED_METRICS:
            a = np.minimum(g[col].to_numpy(float), duration_min)
            b = np.minimum(r[col].to_numpy(float), duration_min)
            d = a - b if col == "clearance_time_min" else g[col].to_numpy(float) - r[col].to_numpy(float)
            m, lo, hi = ci95(d)
            row.update({col: m, col + "_ci_lo": lo, col + "_ci_hi": hi,
                        col + "_n_better": int((d < 0).sum()), col + "_n_worse": int((d > 0).sum())})
        row["n"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def run(cfg: dict, out_dir: Path = OUT_DIR, verbose: bool = True) -> Path:
    pl = load_plan(cfg, out_dir)
    ev = cfg["evaluation"]
    base = cfg["seat_variants"]["base"]
    base_thr = MetricsConfig.from_yaml(ROOT / cfg["metrics"]).rho_crit_street
    jobs = [(s, base_thr) for s in cfg["seat_variants"]["capacity_ped"]]
    jobs += [(base, thr) for thr in cfg["exposure_thresholds_street"] if thr != base_thr]
    rows = []
    t0 = time.perf_counter()
    for seats, thr in jobs:
        sim = make_sim(cfg, ev["fidelity"], seats, rho_crit_street=thr)
        for name, (lever, make) in strategies(cfg, pl, seats).items():
            for c in ev["compliance"]:
                m = run_batch(sim, cfg, ev["seeds"], c, make())
                for j, s in enumerate(ev["seeds"]):
                    rows.append({"seats_pct": seats, "rho_crit_street": thr, "strategy": name,
                                 "lever": lever, "c": c, "seed": s,
                                 **{k: float(v[j]) for k, v in m.items()}})
        if verbose:
            print(f"мест {seats} %, ρ_крит улиц {thr:g}: готово, {time.perf_counter() - t0:.0f} с")
    # явка как отдельная ось: базовый вариант мест и порог, стратегии смены станции
    att_rows = []
    sim = make_sim(cfg, ev["fidelity"], base, rho_crit_street=base_thr)
    for n_att in ev["attendance"]:
        for name, (lever, make) in strategies(cfg, pl, base).items():
            if lever == "route":
                continue
            for c in ev["compliance"]:
                m = run_batch(sim, cfg, ev["seeds"], c, make(), attendance=n_att)
                for j, s in enumerate(ev["seeds"]):
                    att_rows.append({"attendance_ped": n_att, "strategy": name, "lever": lever,
                                     "c": c, "seed": s, **{k: float(v[j]) for k, v in m.items()}})
        if verbose:
            print(f"явка {n_att:.0f}: готово, {time.perf_counter() - t0:.0f} с")
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    summ = summarize(df)
    att = pd.DataFrame(att_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics.csv", index=False)
    summ.to_csv(out_dir / "summary.csv", index=False)
    paired(df, ["seats_pct", "rho_crit_street"], cfg["duration"] / 60).to_csv(out_dir / "paired.csv", index=False)
    att.to_csv(out_dir / "metrics_attendance.csv", index=False)
    paired(att, ["attendance_ped"], cfg["duration"] / 60).to_csv(out_dir / "paired_attendance.csv", index=False)
    meta = {"experiment": "stage4_compliance", "config": cfg, "config_hash": config_hash(cfg),
            "plan": pl["chosen"], "base_rho_crit_street": base_thr, "git": _git_rev(),
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(elapsed, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2,
                                                  default=str), encoding="utf-8")
    if verbose:
        sel = summ[(summ.seats_pct == base) & (summ.rho_crit_street == base_thr)]
        cols = ["strategy", "c", "mean_time_min", "danger_exposure_ped_min",
                "clearance_time_min", "max_density"]
        print(sel[cols].round(2).to_string(index=False))
        print(f"{len(df)} эпизодов, {elapsed:.0f} с → {out_dir}")
    return out_dir


def load(cfg: dict | None = None, out_dir: Path = OUT_DIR):
    """Результаты: (metrics, summary, plan.csv, meta). Без пересчёта: оценочные прогоны
    запускаются только явно (схемы фиксируются в DECISIONS.md до них)."""
    cfg = cfg or load_config()
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"нет результатов в {out_dir}: uv run python -m "
                                f"experiments.stage4_compliance plan, затем run")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["config_hash"] != config_hash(cfg):
        raise ValueError(f"результаты {out_dir} получены с другим конфигом — повторите plan и run")
    return (pd.read_csv(out_dir / "metrics.csv"), pd.read_csv(out_dir / "summary.csv"),
            pd.read_csv(out_dir / "plan.csv"), meta)


def load_extra(out_dir: Path = OUT_DIR) -> dict[str, pd.DataFrame]:
    """Парные разности, серия с фиксированной явкой и первая серия (10 seed)."""
    return {"paired": pd.read_csv(out_dir / "paired.csv"),
            "attendance": pd.read_csv(out_dir / "metrics_attendance.csv"),
            "paired_attendance": pd.read_csv(out_dir / "paired_attendance.csv"),
            "n10": pd.read_csv(out_dir / "n10" / "metrics.csv")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["plan", "run"])
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    (plan if args.step == "plan" else run)(cfg, args.out)


if __name__ == "__main__":
    main()
