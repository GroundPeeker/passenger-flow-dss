"""Этап 4: анализ — почему схема, рассчитанная на c = 0,5, лучше схемы на c = 1 даже при
c = 1. Выбранные схемы (plan.json) НЕ меняются: это анализ, а не повторный подбор.

Запуск:  uv run python -m experiments.stage4_scheme_analysis
Процедура подбора — та же, что в stage4_compliance.plan (сетка f, лексикографический
критерий «экспозиция, затем время при допуске 5 %»), базовый вариант мест, рычаг смены
станции, расчётная c ∈ {1; 0,5}. Варианты «планировщика»:
- model_10   — модель без калибровки, seed 1000–1009 (как при подборе схем этапа 4);
- model_50   — модель без калибровки, seed 1000–1049 (больше seed планирования);
- calib_50   — откалиброванная модель, seed 1000–1049;
- truth_50   — «реальность», seed 1000–1049 (не оценочные seed) — что выбрал бы
               планировщик без расхождения модели.
Результаты — experiments/results/stage4/scheme_analysis/: episodes.csv (по seed),
table.csv (средние), chosen.json (выбранные f), eval_c1.csv (оценочные прогоны при c = 1).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from control.baselines import FixedSchedule
from experiments import stage4_compliance as exp4

OUT = exp4.OUT_DIR / "scheme_analysis"
VARIANTS = {"model_10": ("model", False, 10), "model_50": ("model", False, 50),
            "calib_50": ("model", True, 50), "truth_50": ("truth", None, 50)}


def run() -> dict:
    cfg = exp4.load_config()
    base = cfg["seat_variants"]["base"]
    lev, opt = exp4.switch_lever(cfg)
    first = cfg["planning"]["seeds"][0]
    rows = []
    t0 = time.perf_counter()
    for name, (fid, calibrated, n) in VARIANTS.items():
        sim = exp4.make_sim(cfg, fid, base, calibrated=bool(calibrated))
        seeds = list(range(first, first + n))
        for c in cfg["planning"]["design_compliance_switch"]:
            for f in cfg["planning"]["fractions"]:
                m = exp4.run_batch(sim, cfg, seeds, c, FixedSchedule(lev, opt, f), by_edge=True)
                for j, s in enumerate(seeds):
                    rows.append({"variant": name, "design_c": c, "fraction": f, "seed": s,
                                 **{k: float(v[j]) for k, v in m.items()}})
        print(f"{name}: {time.perf_counter() - t0:.0f} с")
    ep = pd.DataFrame(rows)
    tab = (ep.groupby(["variant", "design_c", "fraction"])
           [["danger_exposure_ped_min", "mean_time_min", "clearance_time_min"]]
           .agg(["mean", "std"]).reset_index())
    tab.columns = ["_".join(c).rstrip("_") for c in tab.columns]
    chosen = {}
    for (v, c), g in tab.groupby(["variant", "design_c"]):
        g2 = g.rename(columns={"danger_exposure_ped_min_mean": "danger_exposure_ped_min",
                               "mean_time_min_mean": "mean_time_min"})
        best = exp4.choose(g2, cfg["planning"]["tolerance"])
        e_min = g2.danger_exposure_ped_min.min()
        ok = g2[g2.danger_exposure_ped_min <= e_min * (1 + cfg["planning"]["tolerance"]) + 1e-9]
        chosen[f"{v}|{c:g}"] = {
            "fraction": float(best.fraction),
            "argmin_exposure": float(g2.loc[g2.danger_exposure_ped_min.idxmin(), "fraction"]),
            "argmin_mean_time": float(g2.loc[g2.mean_time_min.idxmin(), "fraction"]),
            "within_tolerance": sorted(float(x) for x in ok.fraction)}
    # оценочные прогоны при c = 1 (уже посчитаны в серии 50 seed): f = 0,1 и 0,15
    df, _, _, meta = exp4.load(cfg)
    plan = meta["plan"]
    sel = df[(df.seats_pct == base) & (df.rho_crit_street == meta["base_rho_crit_street"])
             & (df.c == 1.0) & df.strategy.isin(["no_control", "fixed_c1", "fixed_c0.5"])]
    f_of = {"no_control": 0.0, "fixed_c1": plan[exp4.scheme_key("switch", base, 1.0)]["fraction"],
            "fixed_c0.5": plan[exp4.scheme_key("switch", base, 0.5)]["fraction"]}
    ev = []
    for name, g in sel.groupby("strategy"):
        row = {"strategy": name, "fraction": f_of[name], "n": len(g)}
        for col in ("danger_exposure_ped_min", "mean_time_min", "clearance_time_min"):
            m, lo, hi = exp4.ci95(g[col].to_numpy(float))
            row.update({col: m, col + "_ci_lo": lo, col + "_ci_hi": hi})
        ev.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    ep.to_csv(OUT / "episodes.csv", index=False)
    tab.to_csv(OUT / "table.csv", index=False)
    pd.DataFrame(ev).sort_values("fraction").to_csv(OUT / "eval_c1.csv", index=False)
    (OUT / "chosen.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(json.dumps(chosen, ensure_ascii=False, indent=1))
    print(pd.DataFrame(ev).round(2).to_string(index=False))
    return chosen


if __name__ == "__main__":
    run()
