"""Рисунки этапа 4 — базовые стратегии при разной степени следования c.

Данные — из результатов experiments/stage4_compliance.py (experiments/results/stage4/);
рисунки результатов не пересчитывают эксперимент: схемы фиксируются в DECISIONS.md до
оценочных прогонов, поэтому прогоны запускаются только явно (plan, затем run).

- fixed_tradeoff — подбор фиксированной схемы: среднее время и экспозиция от доли f
  (прогоны планирования, c = 1 и c = 0,5);
- fixed_tradeoff_edges — экспозиция по участкам от f (где появляется опасная плотность);
- metrics_vs_compliance — главный рисунок: среднее время в районе, экспозиция опасной
  плотности, время рассасывания от c;
- metrics_vs_compliance_seats — то же при 15/20/25/30 % свободных мест;
- max_density_vs_compliance — максимум плотности от c (вспомогательный);
- exposure_thresholds — экспозиция от c при ρ_крит улиц 3,5 и 4,0;
- route_lever_effect — второй рычаг (указатель маршрута JN);
- animation_compliance — анимация одного эпизода (seed 13, N ≈ 39 тыс.) при c = 0,3
  и c = 1,0 для схемы c = 1 и LeastLoaded (.mp4 и .gif);
- paired_difference — разность метрик стратегии и «без управления» на тех же seed
  (парное сравнение: эпизоды с одинаковыми N, профилем, долями и фазой поездов).

Полосы — 95 % доверительный интервал среднего по seed (t-распределение) на рисунках
от c; для неотрицательных метрик нижняя граница полосы обрезается нулём (экспозиция
сильно скошена: среднее определяют 1–2 эпизода с большой явкой, t-интервал уходит ниже 0);
эпизоды, не рассосавшиеся за эпизод, рисуются со временем рассасывания = длительности; разброс по seed планирования (минимум–максимум) — на рисунках подбора схемы.
Функции возвращают (рисунок, данные pandas.DataFrame, метаданные dict).
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style
from experiments import stage4_compliance as exp4

OUT = exp4.OUT_DIR

# Подписи и вид стратегий: цвет — постоянный цвет стратегии (analysis/style.py), две
# фиксированные схемы — один цвет, различаются типом линии (расчётная c).
STRATEGIES = {
    "no_control": ("без управления", style.STRATEGY_COLORS["no_control"], "-", "o"),
    "fixed_c1": ("фикс. схема, расчётная c = 1", style.STRATEGY_COLORS["fixed"], "-", "s"),
    "fixed_c0.5": ("фикс. схема, расчётная c = 0,5", style.STRATEGY_COLORS["fixed"], "--", "D"),
    "least_loaded": ("наименее загруженный путь", style.STRATEGY_COLORS["least_loaded"], "-", "^"),
    "route_fixed_c1": ("фикс. схема JN, расчётная c = 1", style.STRATEGY_COLORS["fixed"], "-", "s"),
    "route_least_loaded": ("наименее загруженный путь, JN",
                           style.STRATEGY_COLORS["least_loaded"], "-", "^"),
}
MAIN = ["no_control", "fixed_c1", "fixed_c0.5", "least_loaded"]
# Расчётная c фиксированных схем: на кривых выделяется точка «фактическая c = расчётной».
DESIGN_C = {"fixed_c1": 1.0, "fixed_c0.5": 0.5, "route_fixed_c1": 1.0}
DESIGN_NOTE = ("Крупный незакрашенный маркер — фактическая c равна расчётной c схемы"
               "{interp}.")
DESIGN_INTERP = (" (c = 0,5 нет в сетке оценки: точка — линейная интерполяция между "
                 "c = 0,4 и 0,6)")


def _design_point(ax, name, x, y):
    """Крупный незакрашенный маркер на кривой фиксированной схемы при c = расчётной."""
    if name not in DESIGN_C:
        return
    label, color, ls, mk = STRATEGIES[name]
    dc = DESIGN_C[name]
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not x.min() <= dc <= x.max():
        return
    ax.plot([dc], [np.interp(dc, x, y)], marker=mk, ms=11, mfc="none", mec=color, mew=1.6,
            ls="", zorder=6, clip_on=False)


def _design_note(fig, leg, interp: bool = True):
    """Одна строка под легендой: что означает крупный маркер."""
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(0.5, bb.y0 - 0.01, DESIGN_NOTE.format(interp=DESIGN_INTERP if interp else ""),
             ha="center", va="top", fontsize=7.5, color=style.TEXT_SECONDARY)
ROUTE = ["no_control", "route_fixed_c1", "route_least_loaded"]
PANELS = [("mean_time_min", "Среднее время в районе, мин"),
          ("danger_exposure_ped_min", "Экспозиция опасной\nплотности, тыс. чел·мин"),
          ("clearance_time_min", "Время рассасывания, мин")]
SCALE = {"danger_exposure_ped_min": 1e-3}
EDGE_LABELS = {"e_av": "проспект (к A)", "e_pb": "бульвар к B", "e_ns": "обход стадиона (на север)",
               "e_sn": "обход стадиона (на юг)", "e_sc": "подход к автобусам",
               "e_pk": "аллея парка", "e_pj": "аллея JP → PA"}


def _results():
    cfg = exp4.load_config()
    df, summ, plan, meta = exp4.load(cfg)
    return cfg, df, summ, meta


def _plan():
    cfg = exp4.load_config()
    pj = exp4.load_plan(cfg)
    ep = pd.read_csv(OUT / "plan_episodes.csv")
    return cfg, ep, pj


def _meta(cfg, meta, **kw):
    return {"source": "experiments/results/stage4", "experiment_config": cfg,
            "experiment_git": meta.get("git"), "experiment_created": meta.get("created"),
            "config_hash": meta.get("config_hash"), **kw}


def _finite(x, cap):
    """inf (не рассосалось за эпизод) → длительность эпизода, для рисования."""
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), x, cap)


def _ci_rows(df, keys, col, cap):
    """Среднее и 95 % ДИ по seed для каждой группы keys."""
    rows = []
    for k, g in df.groupby(keys, sort=True):
        m, lo, hi = exp4.ci95(_finite(g[col].to_numpy(), cap))
        rows.append((*np.atleast_1d(k), m, lo, hi, int(np.sum(~np.isfinite(g[col])))))
    return pd.DataFrame(rows, columns=[*keys, "mean", "ci_lo", "ci_hi", "n_inf"])


def _vs_c(ax, df, strategies, col, cap):
    """Кривые «метрика от c» со стратегиями; возвращает данные рисунка."""
    out = []
    k = SCALE.get(col, 1.0)
    for name in strategies:
        g = df[df.strategy == name]
        if g.empty:
            continue
        label, color, ls, mk = STRATEGIES[name]
        r = _ci_rows(g, ["c"], col, cap)
        ax.fill_between(r.c, np.maximum(r.ci_lo, 0.0) * k, r.ci_hi * k, color=color,
                        alpha=0.15, lw=0)
        ax.plot(r.c, r["mean"] * k, color=color, ls=ls, marker=mk, ms=4, label=label)
        _design_point(ax, name, r.c, r["mean"] * k)
        r.insert(0, "strategy", name)
        r.insert(1, "metric", col)
        out.append(r)
    ax.set_xticks(sorted(df.c.unique()))
    return pd.concat(out)


def _cap(cfg):
    return cfg["duration"] / 60.0


# --- подбор фиксированной схемы ----------------------------------------------------------

def fixed_tradeoff():
    """Среднее время и экспозиция опасной плотности от доли f интервалов с рекомендацией
    «на B» (прогоны планирования, модель, seed 1000–1009, базовый вариант мест);
    линия — среднее, полоса — минимум–максимум по seed; отмечены выбранные f."""
    cfg, ep, pj = _plan()
    seats = cfg["seat_variants"]["base"]
    d = ep[(ep.lever == "switch") & (ep.seats_pct == seats)]
    fig, axs = plt.subplots(1, 2, figsize=style.a4(6.5))
    rows = []
    styles = {1.0: ("-", "s"), 0.5: ("--", "D")}
    color = style.STRATEGY_COLORS["fixed"]
    for c in cfg["planning"]["design_compliance_switch"]:
        g = d[d.design_c == c]
        f_star = pj["chosen"][exp4.scheme_key("switch", seats, c)]["fraction"]
        ls, mk = styles.get(c, ("-", "o"))
        for ax, (col, ylabel) in zip(axs, PANELS[:2]):
            k = SCALE.get(col, 1.0)
            a = g.groupby("fraction")[col].agg(["mean", "min", "max"]).reset_index()
            ax.fill_between(a.fraction, a["min"] * k, a["max"] * k, color=color,
                            alpha=0.12 if c == 1.0 else 0.08, lw=0)
            ax.plot(a.fraction, a["mean"] * k, color=color, ls=ls,
                    label=f"c = {c:g}".replace(".", ","))
            y = float(a.loc[np.isclose(a.fraction, f_star), "mean"].iloc[0]) * k
            ax.plot([f_star], [y], marker=mk, color=color, ms=6, mfc="white", mew=1.4)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Доля интервалов с рекомендацией «на B» f")
            for _, r in a.iterrows():
                rows.append({"design_c": c, "metric": col, "fraction": r.fraction,
                             "mean": r["mean"], "min": r["min"], "max": r["max"],
                             "chosen": bool(np.isclose(r.fraction, f_star))})
    axs[1].set_yscale("log")
    axs[0].legend(title="Расчётная c", loc="upper left")
    for ax, t in zip(axs, ("Среднее время в районе", "Экспозиция опасной плотности")):
        ax.set_title(t, loc="left", color=style.TEXT)
    fig.text(0.0, -0.04, "Маркер — выбранная f (минимум экспозиции; при равенстве в пределах "
             "5 % — минимум времени). Полоса — минимум–максимум по 10 seed планирования.",
             fontsize=7, color=style.TEXT_SECONDARY)
    fig.tight_layout()
    return fig, pd.DataFrame(rows), _meta(cfg, pj, seeds=cfg["planning"]["seeds"],
                                          seats_pct=seats, fidelity=cfg["planning"]["fidelity"],
                                          chosen={k: v for k, v in pj["chosen"].items()
                                                  if v["seats_pct"] == seats})


def fixed_tradeoff_edges():
    """Экспозиция опасной плотности по участкам от f (средняя по seed планирования):
    где появляется опасная плотность при малой и большой доле переключения."""
    cfg, ep, pj = _plan()
    seats = cfg["seat_variants"]["base"]
    d = ep[(ep.lever == "switch") & (ep.seats_pct == seats)]
    cols = [c for c in d.columns if c.startswith("exposure_")]
    designs = cfg["planning"]["design_compliance_switch"]
    fig, axs = plt.subplots(1, len(designs), figsize=style.a4(6.5), sharey=True)
    rows = []
    for ax, c in zip(np.atleast_1d(axs), designs):
        g = d[d.design_c == c].groupby("fraction")[cols].mean() / 1000
        top = [e for e in g.columns if g[e].max() > 0.05]
        edges = [e.removeprefix("exposure_") for e in top]
        order = sorted(range(len(top)), key=lambda j: -g[top[j]].max())
        colors = {}
        for j, idx in enumerate(order):
            colors[top[idx]] = style.SERIES[j % len(style.SERIES)]
        ax.stackplot(g.index, *[g[top[j]] for j in order],
                     labels=[EDGE_LABELS.get(edges[j], edges[j]) for j in order],
                     colors=[colors[top[j]] for j in order], alpha=0.85)
        ax.set_title(f"Расчётная c = {c:g}".replace(".", ","), loc="left", color=style.TEXT)
        ax.set_xlabel("Доля интервалов f")
        for f, r in g.iterrows():
            for e in top:
                rows.append({"design_c": c, "fraction": f, "edge": e.removeprefix("exposure_"),
                             "exposure_kped_min": r[e]})
    np.atleast_1d(axs)[0].set_ylabel("Экспозиция опасной\nплотности, тыс. чел·мин")
    np.atleast_1d(axs)[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig, pd.DataFrame(rows), _meta(cfg, pj, seeds=cfg["planning"]["seeds"],
                                          seats_pct=seats, fidelity=cfg["planning"]["fidelity"])


# --- оценочные прогоны -------------------------------------------------------------------

def _base(df, cfg, meta):
    """Базовый вариант мест и базовый порог ρ_крит улиц (configs/metrics.yaml)."""
    return df[(df.seats_pct == cfg["seat_variants"]["base"])
              & (df.rho_crit_street == meta["base_rho_crit_street"])]


def metrics_vs_compliance():
    """Главный рисунок этапа 4: среднее время в районе, экспозиция опасной плотности,
    время рассасывания от истинной c; стратегии рычага смены станции; базовый вариант
    мест; среднее по 10 seed и 95 % ДИ."""
    cfg, df, summ, meta = _results()
    d = _base(df, cfg, meta)
    fig, axs = plt.subplots(1, 3, figsize=style.a4(6.5))
    data = []
    for ax, (col, ylabel) in zip(axs, PANELS):
        data.append(_vs_c(ax, d, MAIN, col, _cap(cfg)))
        ax.set_ylabel(ylabel)
    h, l = axs[0].get_legend_handles_labels()
    fig.supxlabel("Фактическая степень следования c", fontsize=9, y=0.06)
    leg = fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _design_note(fig, leg)
    return fig, pd.concat(data), _meta(cfg, meta, seeds=cfg["evaluation"]["seeds"],
                                       seats_pct=cfg["seat_variants"]["base"],
                                       band="95 % ДИ среднего по seed (t)", plan=meta["plan"])


def metrics_vs_compliance_seats():
    """То же, что главный рисунок, при 15/20/25/30 % свободных мест (обязательство 2б):
    строки — метрики, столбцы — варианты мест."""
    cfg, df, summ, meta = _results()
    thr = meta["base_rho_crit_street"]
    seats = sorted(cfg["seat_variants"]["capacity_ped"])
    fig, axs = plt.subplots(3, len(seats), figsize=style.a4(17), sharex=True, sharey="row")
    data = []
    for j, s in enumerate(seats):
        d = df[(df.seats_pct == s) & (df.rho_crit_street == thr)]
        for i, (col, ylabel) in enumerate(PANELS):
            r = _vs_c(axs[i, j], d, MAIN, col, _cap(cfg))
            r.insert(0, "seats_pct", s)
            data.append(r)
            if j == 0:
                axs[i, j].set_ylabel(ylabel)
        axs[0, j].set_title(f"{s} %", loc="left", color=style.TEXT)
    fig.supxlabel("Фактическая степень следования c (столбцы — доля свободных мест в поездах A и B)",
                  fontsize=9, y=0.035)
    h, l = axs[0, 0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _design_note(fig, leg)
    return fig, pd.concat(data), _meta(cfg, meta, seeds=cfg["evaluation"]["seeds"],
                                       band="95 % ДИ среднего по seed (t)", plan=meta["plan"])


def max_density_vs_compliance():
    """Максимум плотности ячейки за эпизод от c (вспомогательная метрика: на улицах
    стоячая очередь в CTM сжимается до ρ_jam, DECISIONS.md, этап 3б)."""
    cfg, df, summ, meta = _results()
    d = _base(df, cfg, meta)
    fig, ax = plt.subplots(figsize=style.a4(7))
    data = _vs_c(ax, d, MAIN, "max_density", _cap(cfg))
    ax.axhline(4.0, color=style.TEXT_SECONDARY, lw=0.8, ls=":")
    ax.text(0.21, 4.05, "ρ_крит улиц 4,0", fontsize=7, color=style.TEXT_SECONDARY)
    ax.set_ylabel("Максимальная плотность, чел/м²")
    ax.set_xlabel("Фактическая степень следования c")
    h, l = ax.get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _design_note(fig, leg)
    return fig, data, _meta(cfg, meta, seeds=cfg["evaluation"]["seeds"],
                            band="95 % ДИ среднего по seed (t)")


def exposure_thresholds():
    """Экспозиция опасной плотности от c при ρ_крит улиц 3,5 и 4,0 чел/м² (порог —
    только мониторинг, траектории те же); базовый вариант мест."""
    cfg, df, summ, meta = _results()
    d = df[df.seats_pct == cfg["seat_variants"]["base"]]
    thrs = sorted(d.rho_crit_street.unique())
    fig, axs = plt.subplots(1, len(thrs), figsize=style.a4(6.5))
    data = []
    for ax, thr in zip(np.atleast_1d(axs), thrs):
        r = _vs_c(ax, d[d.rho_crit_street == thr], MAIN, "danger_exposure_ped_min", _cap(cfg))
        r.insert(0, "rho_crit_street", thr)
        data.append(r)
        ax.set_title(f"ρ_крит улиц = {thr:g} чел/м²".replace(".", ","), loc="left",
                     color=style.TEXT)
        ax.set_ylabel(PANELS[1][1])
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.supxlabel("Фактическая степень следования c", fontsize=9, y=0.06)
    leg = fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _design_note(fig, leg)
    return fig, pd.concat(data), _meta(cfg, meta, seeds=cfg["evaluation"]["seeds"],
                                       band="95 % ДИ среднего по seed (t)")


def route_lever_effect():
    """Второй рычаг — указатель маршрута JN (проспект / парк, оба к станции A): те же
    метрики от c для фиксированной схемы и правила наименее загруженного пути."""
    cfg, df, summ, meta = _results()
    d = _base(df, cfg, meta)
    fig, axs = plt.subplots(1, 3, figsize=style.a4(6.5))
    data = []
    for ax, (col, ylabel) in zip(axs, PANELS):
        data.append(_vs_c(ax, d, ROUTE, col, _cap(cfg)))
        ax.set_ylabel(ylabel)
    h, l = axs[0].get_legend_handles_labels()
    fig.supxlabel("Фактическая степень следования c", fontsize=9, y=0.06)
    leg = fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _design_note(fig, leg, interp=False)
    return fig, pd.concat(data), _meta(cfg, meta, seeds=cfg["evaluation"]["seeds"],
                                       band="95 % ДИ среднего по seed (t)", plan=meta["plan"])


def _paired_panel(ax, pr, col, strategies=None):
    """Кривые парной разности (стратегия − без управления) от c из таблицы paired."""
    k = SCALE.get(col, 1.0)
    for name in strategies or MAIN[1:]:
        r = pr[pr.strategy == name].sort_values("c")
        if r.empty:
            continue
        label, color, ls, mk = STRATEGIES[name]
        ax.fill_between(r.c, r[col + "_ci_lo"] * k, r[col + "_ci_hi"] * k, color=color,
                        alpha=0.15, lw=0)
        ax.plot(r.c, r[col] * k, color=color, ls=ls, marker=mk, ms=4, label=label)
        _design_point(ax, name, r.c, r[col] * k)
    ax.axhline(0, color=style.STRATEGY_COLORS["no_control"], lw=1.0)
    ax.set_xticks(sorted(pr.c.unique()))


def _paired_rows(pr, extra: dict):
    cols = ["strategy", "c"] + [c for c in pr.columns if any(
        c.startswith(m) for m, _ in PANELS)]
    out = pr[cols].copy()
    for k, v in extra.items():
        out.insert(0, k, v)
    return out


def paired_difference():
    """Основной статистический анализ: разность метрики стратегии и «без управления» на тех
    же эпизодах (одинаковые параметры), среднее и 95 % ДИ разности (t); ниже нуля — лучше.
    Строки — первая серия (10 seed) и основная (50 seed); базовый вариант мест."""
    cfg, df, summ, meta = _results()
    ex = exp4.load_extra()
    base, thr = cfg["seat_variants"]["base"], meta["base_rho_crit_street"]
    sel = lambda pr: pr[(pr.seats_pct == base) & (pr.rho_crit_street == thr)]
    p50 = sel(ex["paired"])
    p10 = sel(exp4.paired(ex["n10"], ["seats_pct", "rho_crit_street"], _cap(cfg)))
    n50 = len(cfg["evaluation"]["seeds"])
    fig, axs = plt.subplots(2, 3, figsize=style.a4(12), sharex=True, sharey="col")
    data = []
    for i, (pr, n) in enumerate(((p10, 10), (p50, n50))):
        units = ["Δ, мин", "Δ, тыс. чел·мин", "Δ, мин"]
        titles = ["Среднее время в районе", "Экспозиция ρ > ρ_крит", "Время рассасывания"]
        for j, (col, ylabel) in enumerate(PANELS):
            _paired_panel(axs[i, j], pr, col)
            axs[i, j].set_ylabel(f"{n} seed: {units[j]}" if j == 0 else units[j])
            if i == 0:
                axs[i, j].set_title(titles[j], loc="left", color=style.TEXT, fontsize=9)
        data.append(_paired_rows(pr, {"n_seeds": n}))
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.supxlabel("Фактическая степень следования c", fontsize=9, y=0.06)
    leg = fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _design_note(fig, leg)
    return fig, pd.concat(data), _meta(cfg, meta, seeds=cfg["evaluation"]["seeds"],
                                       seeds_first_series=list(range(10)), seats_pct=base,
                                       band="95 % ДИ средней парной разности по seed (t)",
                                       reference="no_control")


def paired_by_attendance():
    """Парные разности со «без управления» при фиксированной явке N = 25/30/35/40 тыс.
    (остальные параметры эпизода — из seed; схемы спланированы на случайную явку со средним
    30 тыс.): строки — метрики, столбцы — уровни явки."""
    cfg, df, summ, meta = _results()
    pr = exp4.load_extra()["paired_attendance"]
    levels = sorted(pr.attendance_ped.unique())
    fig, axs = plt.subplots(3, len(levels), figsize=style.a4(17), sharex=True, sharey="row")
    data = []
    for j, n in enumerate(levels):
        g = pr[pr.attendance_ped == n]
        for i, (col, ylabel) in enumerate(PANELS):
            _paired_panel(axs[i, j], g, col)
            if j == 0:
                axs[i, j].set_ylabel("Δ " + ylabel[0].lower() + ylabel[1:])
        axs[0, j].set_title(f"N = {n / 1000:.0f} тыс.", loc="left", color=style.TEXT)
        data.append(_paired_rows(g, {"attendance_ped": n}))
    fig.supxlabel("Фактическая степень следования c (столбцы — явка N)", fontsize=9, y=0.035)
    h, l = axs[0, 0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _design_note(fig, leg)
    return fig, pd.concat(data), _meta(cfg, meta, seeds=cfg["evaluation"]["seeds"],
                                       attendance_ped=levels,
                                       band="95 % ДИ средней парной разности по seed (t)",
                                       reference="no_control")


def model_calibration():
    """Калибровка модели планирования: отклонение очереди перед входом станции модели от
    «реальности» в зависимости от множителя κ (seed калибровки, среднее по режимам) и
    отношение средних метрик модели к «реальности» на проверочных seed до и после."""
    from experiments import calibrate_model as cal
    ccfg = cal.load_config()
    loss = pd.read_csv(cal.OUT_DIR / "loss.csv")
    meta = json.loads((cal.OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    fig, axs = plt.subplots(1, 2, figsize=style.a4(7), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axs[0]
    agg = (loss.assign(mse=loss.rmse_ped ** 2).groupby(["station", "factor"]).mse.mean()
           .pow(0.5).rename("rmse").reset_index())
    for i, (st, g) in enumerate(agg.groupby("station")):
        color = style.SERIES[i]
        ax.plot(g.factor, g.rmse, color=color, label=f"станция {st}")
        k = meta["chosen"][st]
        ax.plot([k], [g.loc[np.isclose(g.factor, k), "rmse"].iloc[0]], "o", color=color,
                mfc="white", mew=1.4)
    ax.set_xlabel("Множитель пропускной способности κ")
    ax.set_ylabel("СКО очереди перед входом, чел")
    ax.set_title("Калибровка", loc="left", color=style.TEXT)
    ax.legend()
    ax = axs[1]
    metrics = [("danger_exposure_ped_min", "экспозиция"), ("mean_time_min", "ср. время"),
               ("clearance_time_min", "рассасывание")]
    regimes = list(meta["ratio_to_truth"])
    names = {"no_control": "без упр.", "switch_f0.3": "смена f = 0,3",
             "least_loaded_c1": "наим. загр., c = 1"}
    rows, w = [], 0.13
    x = np.arange(len(regimes))
    for j, (m, mname) in enumerate(metrics):
        for t, (mdl, alpha) in enumerate((("model_uncalibrated", 0.35), ("model_calibrated", 1.0))):
            v = [meta["ratio_to_truth"][r][mdl][m] for r in regimes]
            off = (j * 2 + t - 2.5) * w
            ax.bar(x + off, v, w, color=style.SERIES[j], alpha=alpha,
                   label=f"{mname}, {'без калибровки' if t == 0 else 'с калибровкой'}")
            rows += [{"regime": r, "metric": m, "model": mdl, "ratio": vv} for r, vv in zip(regimes, v)]
    ax.axhline(1.0, color=style.TEXT_SECONDARY, lw=0.8)
    ax.set_xticks(x, [names.get(r, r) for r in regimes])
    ax.set_ylim(0.6, 1.15)
    ax.set_ylabel("Модель / «реальность»")
    ax.set_title("Проверка на отдельных seed", loc="left", color=style.TEXT)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    data = pd.concat([agg.assign(kind="rmse"), pd.DataFrame(rows).assign(kind="validation")])
    return fig, data, {"source": "experiments/results/calibration", "experiment_config": ccfg,
                       "experiment_git": meta["git"], "experiment_created": meta["created"],
                       "chosen": meta["chosen"]}


def _dec(x: float, nd: int = 1) -> str:
    """Число с десятичной запятой (подписи для текста по-русски)."""
    return f"{x:.{nd}f}".replace(".", ",")


def _animation_build():
    """Фигура и функция кадра анимации одного эпизода (experiments/stage4_animation.py):
    панели — схема, рассчитанная на c = 1, и LeastLoaded при c = 0,3 и c = 1,0. В подписи
    панели — текущие значения (люди в районе, накопленная экспозиция опасной плотности,
    действующая рекомендация на JN) и итог эпизода. Кадр — интервал управления (1 мин)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from analysis import netplot
    from experiments import stage4_animation as an
    acfg = an.load_config()
    meta = json.loads(an.OUT.with_name("animation_meta.json").read_text(encoding="utf-8"))
    z = np.load(an.OUT)
    ecfg = exp4.load_config(exp4.ROOT / acfg["experiment"])
    sim = exp4.make_sim(ecfg, ecfg["evaluation"]["fidelity"], acfg["seats_pct"])
    geo = netplot.NetGeometry(sim.net, sim.ctm.cells)
    t = z["t"] / 60
    runs = meta["runs"]
    keys = [r["key"] for r in runs]
    in_area = np.stack([z[f"{k}|in_area"] for k in keys])
    alive = np.flatnonzero((in_area > 0.5).any(axis=0))
    n_frames = int(min(len(t), alive[-1] + 6))
    combos = [(name, c) for name in acfg["strategies"] for c in acfg["compliance"]]
    fig, axs = plt.subplots(1, len(combos), figsize=(12.8, 7.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.30, wspace=0.04)
    title = fig.suptitle("", x=0.02, ha="left", fontsize=13, color=style.TEXT)
    panels = []
    for ax, (name, c) in zip(axs, combos):
        k = f"{name}|{c:g}"
        r = next(r for r in runs if r["key"] == k)
        lc = netplot.draw_network(ax, geo, z[f"{k}|density"][0], stadium_label=False)
        netplot.draw_nodes(ax, geo, sim, labels=False, signs=False, scale=0.7)
        label = STRATEGIES[name][0].replace("фикс. схема, ", "фикс. схема,\n")
        label = label.replace("наименее загруженный путь", "наименее\nзагруженный путь")
        ax.set_title(f"{label}\nфактическая c = " + f"{c:g}".replace(".", ","),
                     loc="left", color=style.TEXT, fontsize=10)
        txt = ax.text(0.0, -0.02, "", transform=ax.transAxes, fontsize=9, color=style.TEXT,
                      va="top")
        ax.text(0.0, -0.2, f"итог эпизода:\nэкспозиция {_dec(r['danger_exposure_ped_min'] / 1000)} "
                f"тыс. чел·мин\nрассасывание {r['clearance_time_min']:.0f} мин",
                transform=ax.transAxes, fontsize=8, color=style.TEXT_SECONDARY, va="top")
        panels.append((lc, txt, k))
    sm = ScalarMappable(Normalize(0, sim.ctm.fd.rho_jam), style.SEQUENTIAL)
    cax = fig.add_axes([0.35, 0.075, 0.3, 0.018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Плотность, чел/м²")
    N = runs[0]["total_ped"]

    def update(f):
        title.set_text(f"t = {t[f]:.0f} мин после окончания матча · seed {acfg['seed']}, "
                       f"N = {N:,.0f}".replace(",", " "))
        out = [title]
        for lc, txt, k in panels:
            lc.set_array(z[f"{k}|density"][f])
            sw = z[f"{k}|switch"]
            rec = "на B" if f > 0 and sw[min(f - 1, len(sw) - 1)] >= 0 else "—"
            txt.set_text(f"в районе {_dec(z[f'{k}|in_area'][f] / 1000)} тыс. чел\n"
                         f"экспозиция {_dec(z[f'{k}|exposure_cum'][f] / 1000)} тыс. чел·мин\n"
                         f"рекомендация на JN: {rec}")
            out += [lc, txt]
        return out

    rows = []
    for r in runs:
        d = z[f"{r['key']}|density"][:n_frames]
        rows.append(pd.DataFrame({"strategy": r["strategy"], "c": r["c"],
                                  "t_min": np.repeat(t[:n_frames], d.shape[1]),
                                  "cell": np.tile(np.arange(d.shape[1]), n_frames),
                                  "density_ped_m2": d.reshape(-1)}))
    out_meta = {"source": "experiments/results/stage4/animation.npz",
                "experiment_config": acfg, "runs": runs, "experiment_git": meta["git"],
                "frames": n_frames, "fps": 8, "frame_step_min": float(t[1] - t[0])}
    return fig, update, n_frames, t, pd.concat(rows), out_meta


def animation_compliance():
    """Анимация одного эпизода при c = 0,3 и 1,0 (схема c = 1 и LeastLoaded), .mp4 и .gif."""
    from matplotlib.animation import FuncAnimation
    fig, update, n_frames, t, data, meta = _animation_build()
    anim = FuncAnimation(fig, update, frames=n_frames, interval=125, blit=False)
    return anim, data, meta


def animation_frame(minute: int):
    """Статичный кадр той же анимации на заданной минуте (для слайдов)."""
    fig, update, n_frames, t, data, meta = _animation_build()
    k = int(np.argmin(np.abs(t - minute)))
    update(k)
    return fig, data[np.isclose(data.t_min, t[k])], {**meta, "frame_minute": float(t[k])}
