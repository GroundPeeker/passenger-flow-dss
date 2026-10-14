"""Рисунки этапа 3 — описание сценария «стадион» без управления (план работ: timeseries,
network_map, animation; плюс density_snapshots — снимки плотности для текста диплома,
где анимацию не вставить).

Данные — из результатов experiments/stage3_scenario.py (experiments/results/stage3/);
если их нет или конфиг изменился, эксперимент пересчитывается и сохраняется (постановка задачи:
рисунки строятся из сохранённых результатов). network_map — только схема, без симуляции.

Функции возвращают (рисунок или анимация, данные pandas.DataFrame, метаданные dict).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from analysis import netplot, style
from experiments import stage3_scenario as exp3
from sim.metrics import MetricsConfig
from sim.simulator import ROOT, Simulator

BAND = (10, 90)          # полоса разброса по эпизодам — 10–90 % квантили
EDGE_LABELS = {"e_n0": "северная эспланада [Н]", "e_s0": "южная площадь [Н]",
               "e_av": "проспект (маршрут 1 к A)", "e_pk": "аллея парка (маршрут 2 к A)",
               "e_pj": "аллея JP → PA", "e_sq": "привокзальная площадь A [Н]",
               "e_mb": "площадка B [Н]", "e_mc": "площадка автобусов [Н]",
               "e_pb": "бульвар к B", "e_sc": "подход к автобусам", "e_ns": "обход стадиона (на север)",
               "e_sn": "обход стадиона (на юг)"}
STATION_LABELS = {"A": "метро A", "B": "метро B", "C": "автобусы C"}


def _load():
    cfg = exp3.load_config()
    traj, anim, meta = exp3.load(cfg)
    sim = Simulator.from_yaml(ROOT / cfg["scenario"], fidelity=cfg["fidelity"],
                              metrics=MetricsConfig.from_yaml(ROOT / cfg["metrics"]))
    return cfg, traj, anim, meta, sim


def _band(ax, t, X, color, label, lw=1.5, ls="-"):
    """Среднее по эпизодам и полоса 10–90 %. X — (T, B)."""
    lo, hi = np.percentile(X, BAND, axis=1)
    ax.fill_between(t, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(t, X.mean(axis=1), color=color, lw=lw, ls=ls, label=label)
    return X.mean(axis=1), lo, hi


def _base_meta(cfg, meta):
    return {"source": "experiments/results/stage3", "experiment_config": cfg,
            "experiment_git": meta["git"], "experiment_created": meta["created"],
            "config_hash": meta["config_hash"], "seeds": list(range(cfg["n_seeds"])),
            "policy": meta["policy"], "band": f"среднее по эпизодам и {BAND[0]}–{BAND[1]} % квантили"}


def timeseries():
    """Люди в районе, плотности ключевых участков, очереди у входов станций — среднее
    по 20 эпизодам и полоса 10–90 %."""
    cfg, traj, _, meta, sim = _load()
    t = traj.t / 60
    ids = meta["edges"]
    st = sim.stations
    rs, rz = meta["rho_crit_street_ped_per_m2"], meta["rho_crit_zone_ped_per_m2"]
    cs_, cz = meta["congestion_street_ped_per_m2"], meta["congestion_zone_ped_per_m2"]
    fig, axs = plt.subplots(4, 1, figsize=style.a4(22), sharex=True)
    data = {"t_min": t}

    ax = axs[0]
    series = [("на стадионе", traj.in_stadium, style.NEUTRAL),
              ("на улицах района", traj.on_network, style.SERIES[0]),
              ("внутри станций", traj.in_stations, style.SERIES[1])]
    for name, X, c in series:
        m, lo, hi = _band(ax, t, X / 1000, c, name)
        data[f"{name}_mean_kped"], data[f"{name}_p10"], data[f"{name}_p90"] = m, lo, hi
    ax.set_ylabel("Людей, тыс. чел")
    ax.set_title("Люди в районе стадиона", loc="left", color=style.TEXT)
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, None)

    ax = axs[1]
    for j, eid in enumerate(cfg["key_edges"]):
        k = ids.index(eid)
        m, lo, hi = _band(ax, t, traj.edge_max[:, :, k], style.SERIES[j], EDGE_LABELS.get(eid, eid))
        data[f"rho_max_{eid}_mean"], data[f"rho_max_{eid}_p10"], data[f"rho_max_{eid}_p90"] = m, lo, hi
    for thr, name, ls in ((cs_, "затор, улицы", ":"), (rs, "ρ_крит, улицы", "--"),
                          (rz, "ρ_крит, зоны [Н]", "--")):
        ax.axhline(thr, color=style.TEXT_SECONDARY, lw=0.9, ls=ls)
        ax.annotate(f"{name} {thr:.2f}", (0.5, thr), xytext=(0, 3),
                    textcoords="offset points", ha="left", color=style.TEXT_SECONDARY, fontsize=7)
    ax.axhline(sim.ctm.fd.rho_jam, color=style.NEUTRAL, lw=0.8, ls=":")
    ax.annotate(f"плотность затора ρ_jam = {sim.ctm.fd.rho_jam:g}", (t[-1], sim.ctm.fd.rho_jam),
                xytext=(0, -10), textcoords="offset points", ha="right",
                color=style.TEXT_SECONDARY, fontsize=8)
    ax.set_ylim(0, 6)
    ax.set_ylabel("Плотность, чел/м²")
    ax.set_title("Наибольшая плотность ячейки на ключевых участках", loc="left", color=style.TEXT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    ax = axs[2]
    for s, sid in enumerate(st.ids):
        m, lo, hi = _band(ax, t, traj.queue_outside[:, :, s] / 1000, style.SERIES[s],
                          STATION_LABELS.get(sid, sid))
        data[f"queue_out_{sid}_mean_kped"], data[f"queue_out_{sid}_p10"], data[f"queue_out_{sid}_p90"] = m, lo, hi
    ax.set_ylabel("Очередь, тыс. чел")
    ax.set_title("Очередь перед входом (на улице, область затора у входа)", loc="left",
                 color=style.TEXT)
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, None)

    ax = axs[3]
    for s, sid in enumerate(st.ids):
        sl = slice(st.first[s], st.last[s] + 1)
        X = traj.station_links[:, :, sl].sum(axis=2)
        m, lo, hi = _band(ax, t, X / 1000, style.SERIES[s], STATION_LABELS.get(sid, sid))
        cap = st.storage[sl].sum() / 1000
        ax.axhline(cap, color=style.SERIES[s], lw=0.7, ls=":")
        data[f"inside_{sid}_mean_kped"], data[f"inside_{sid}_p10"], data[f"inside_{sid}_p90"] = m, lo, hi
    ax.set_ylabel("Внутри станции, тыс. чел")
    ax.set_title("Заполнение станций (пунктир — вместимость всех звеньев)", loc="left",
                 color=style.TEXT)
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, None)

    ax.set_xlabel("Время после окончания матча, мин")
    ax.set_xlim(0, t[-1])
    fig.tight_layout()

    clr = np.asarray(traj.in_area)
    meta_out = {**_base_meta(cfg, meta), "rho_crit_street_ped_per_m2": rs,
                "rho_crit_zone_ped_per_m2": rz, "congestion_street_ped_per_m2": cs_,
                "congestion_zone_ped_per_m2": cz,
                "queue_density_ped_per_m2": meta["queue_density_ped_per_m2"],
                "key_edges": cfg["key_edges"],
                "description": "Без управления. Плотность — наибольшая по ячейкам участка и по "
                               "шагам CTM за минуту; накопители [Н] — плотность = люди / площадь; "
                               "очередь — люди станции в связной области ячеек с ρ > ρc перед "
                               "входом (sim/metrics.py)",
                "peak_in_area_ped_mean": float(clr.max(axis=0).mean())}
    return fig, pd.DataFrame(data), meta_out


def safety():
    """Метрики безопасности во времени: экспозиция опасной плотности (главная) и затора
    (вспомогательная), площадь выше порогов, заполнение зон ожидания."""
    cfg, traj, _, meta, sim = _load()
    t = traj.t / 60
    rs, rz = meta["rho_crit_street_ped_per_m2"], meta["rho_crit_zone_ped_per_m2"]
    cs_, cz = meta["congestion_street_ped_per_m2"], meta["congestion_zone_ped_per_m2"]
    fig, axs = plt.subplots(3, 1, figsize=style.a4(18), sharex=True)
    data = {"t_min": t}

    # экспозиция за минуту / 60 с = среднее за минуту число людей выше порога
    ax = axs[0]
    for X, c, name, key in ((traj.exposure, style.SERIES[0],
                             f"опасная плотность (ρ_крит: улицы {rs:g}, зоны {rz:g})", "danger"),
                            (traj.exposure_congestion, style.SERIES[1],
                             f"затор (улицы {cs_:.2f}, зоны {cz:.2f})", "congestion")):
        m, lo, hi = _band(ax, t, X.sum(axis=2) / 60.0 / 1000, c, name)
        data[f"{key}_exposure_mean_kped"], data[f"{key}_exposure_p10"], data[f"{key}_exposure_p90"] = m, lo, hi
    ax.set_ylabel("Людей выше порога, тыс.")
    ax.set_title("Экспозиция: людей выше порога (среднее за минуту; интеграл — чел·мин)",
                 loc="left", color=style.TEXT)
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)

    ax = axs[1]
    for X, c, name, key in ((traj.area_above, style.SERIES[0], "опасная плотность", "danger"),
                            (traj.area_congestion, style.SERIES[1], "затор", "congestion")):
        m, lo, hi = _band(ax, t, X / 1000, c, name)
        data[f"{key}_area_mean_km2"], data[f"{key}_area_p10"], data[f"{key}_area_p90"] = m, lo, hi
    ax.set_ylabel("Площадь, тыс. м²")
    ax.set_title("Площадь ячеек выше порога (наибольшая за минуту)", loc="left", color=style.TEXT)
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)

    ax = axs[2]
    zc = sim.ctm.zone_cells
    zids = [meta["edges"][int(sim.ctm.cells.edge[c])] for c in zc]
    for j, (c, eid) in enumerate(zip(zc, zids)):
        k = meta["edges"].index(eid)
        fill = traj.edge_mean[:, :, k] * sim.ctm.area[c] / sim.ctm.zone_cap[j] * 100
        m, lo, hi = _band(ax, t, fill, style.SERIES[j], EDGE_LABELS.get(eid, eid))
        data[f"zone_fill_{eid}_mean_pct"], data[f"zone_fill_{eid}_p10"], data[f"zone_fill_{eid}_p90"] = m, lo, hi
    ax.axhline(100 * sim.metrics_cfg.zone_full_fraction, color=style.TEXT_SECONDARY, lw=0.9, ls="--")
    ax.annotate(f"«заполнена» ≥ {100 * sim.metrics_cfg.zone_full_fraction:.0f} %",
                (t[-1], 100 * sim.metrics_cfg.zone_full_fraction), xytext=(0, 3),
                textcoords="offset points", ha="right", color=style.TEXT_SECONDARY, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Заполнение, %")
    ax.set_title("Заполнение зон ожидания (снимок раз в минуту)", loc="left", color=style.TEXT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    ax.set_xlabel("Время после окончания матча, мин")
    ax.set_xlim(0, t[-1])
    fig.tight_layout()
    meta_out = {**_base_meta(cfg, meta), "rho_crit_street_ped_per_m2": rs,
                "rho_crit_zone_ped_per_m2": rz, "congestion_street_ped_per_m2": cs_,
                "congestion_zone_ped_per_m2": cz,
                "zone_full_fraction": sim.metrics_cfg.zone_full_fraction,
                "description": "Без управления. Экспозиция и площади — по шагам CTM (5 с), "
                               "агрегаты за минуту; заполнение зон — снимок раз в минуту"}
    return fig, pd.DataFrame(data), meta_out


def network_map():
    """Схема сети «стадион»: участки (толщина — ширина), станции, указатель, камеры."""
    sim = Simulator.from_yaml(ROOT / "configs" / "scenario_stadium.yaml", fidelity="model")
    geo = netplot.NetGeometry(sim.net, sim.ctm.cells)
    fig, ax = plt.subplots(figsize=style.a4(17))
    netplot.draw_network(ax, geo)
    sensors = set(sim.observer.edges.tolist())
    rows = []
    for k, e in enumerate(sim.net.edges):
        mid, ang = geo.edge_midpoint(k)
        is_pair = any(k in pr for pr in sim.net.bidirectional_pairs())
        short = e.length < 50     # короткое ребро: на схеме — под значком станции
        if k in sensors:
            if short:             # камера выносится в сторону (на 90 м схемы) с выноской
                cam = (mid[0] - 20.0, mid[1] - 110.0)
                ax.plot([mid[0], cam[0]], [mid[1], cam[1]], color=style.SERIES[2], lw=1.0,
                        zorder=8)
                ax.plot(*cam, "o", ms=7, color=style.SERIES[2], markeredgecolor="white",
                        markeredgewidth=1.0, zorder=8)
            else:
                ax.plot(*mid, "o", ms=7, color=style.SERIES[2], markeredgecolor="white",
                        markeredgewidth=1.0, zorder=8)
        if short and e.zone:      # накопитель на коротком ребре — подпись рядом
            ax.annotate(f"{e.id} [Н]\n{e.length:.0f}×{e.width:g} м",
                        (mid[0] - 20.0, mid[1] - 110.0) if k in sensors else mid,
                        xytext=(-8, -4) if k in sensors else (10, -14),
                        ha="right" if k in sensors else "left", va="top" if k in sensors else "baseline",
                        textcoords="offset points", fontsize=7, color=style.TEXT_SECONDARY,
                        zorder=9)
        if not short and not (is_pair and e.id in ("e_sn", "e_ba")):
            label = (e.id if not is_pair else f"{e.id} ⇅") + (" [Н]" if e.zone else "")
            # обход стадиона подписывается снаружи (слева), улица на восток — над линией
            off, ha = {"e_ns": ((-10, -2), "right"), "e_e": ((0, 8), "center")}.get(
                e.id, ((7, -2), "left"))
            ax.annotate(f"{label}\n{e.length:.0f}×{e.width:g} м", mid, xytext=off, ha=ha,
                        textcoords="offset points", fontsize=7, color=style.TEXT_SECONDARY,
                        zorder=9)
        rows.append({"edge": e.id, "from": e.source, "to": e.target, "length_m": e.length,
                     "width_m": e.width, "area_m2": e.area, "zone": e.zone, "cells": int(sim.ctm.cells.n_per_edge[k]),
                     "sensor": k in sensors})
    netplot.draw_nodes(ax, geo, sim)
    for n in ("JN", "JS", "JP", "PA", "PB", "PC"):
        ax.annotate(n, geo.pos[n], xytext=(-16, -10) if n.startswith("P") else (-14, 4), textcoords="offset points", fontsize=7,
                    color=style.TEXT, zorder=9)
    # легенда обозначений
    h = [plt.Line2D([], [], color="#c9c7c1", lw=5, label="участок (толщина — ширина)"),
         plt.Line2D([], [], marker="s", ls="", color=style.TEXT, label="вход станции"),
         plt.Line2D([], [], marker="D", ls="", color="white", markeredgecolor=style.TEXT,
                    label="выход из района (пешком)"),
         plt.Line2D([], [], marker="^", ls="", ms=9, color=style.SERIES[3],
                    markeredgecolor=style.TEXT,
                    label="указатель JN: смена станции A→B\nи выбор маршрута (проспект / парк)"),
         plt.Line2D([], [], marker="o", ls="", ms=7, color=style.SERIES[2], label="камера (счётчик)")]
    ax.legend(handles=h, loc="lower left", fontsize=7)
    n_zone = sum(e.zone for e in sim.net.edges)
    ax.set_title(f"Синтетическая сеть «стадион»: {len(sim.net.edges)} участков "
                 f"(из них {n_zone} накопителей [Н]), 2 станции метро и остановка автобусов", loc="left", color=style.TEXT)
    fig.tight_layout()
    meta = {"network": "networks/stadium.yaml", "scenario": "configs/scenario_stadium.yaml",
            "seed": None, "n_cells": sim.ctm.n_cells,
            "description": "Схема (координаты layout); длины рёбер — из конфига сети"}
    return fig, pd.DataFrame(rows), meta


def _frame_title(t_min):
    return f"t = {t_min:.0f} мин после окончания матча"


def density_snapshots():
    """Плотность по сети в четыре момента эпизода анимации (для текста диплома)."""
    cfg, traj, anim, meta, sim = _load()
    geo = netplot.NetGeometry(sim.net, sim.ctm.cells)
    t = traj.t / 60
    moments = [10, 20, 40, 70]
    fig, axs = plt.subplots(2, 2, figsize=style.a4(22))
    axs = axs.ravel()
    rows = []
    for ax, tm in zip(axs, moments):
        k = int(np.argmin(np.abs(t - tm)))
        netplot.draw_network(ax, geo, anim[k], stadium_label=False)
        netplot.draw_nodes(ax, geo, sim, labels=False, signs=False, scale=0.7)
        ax.set_title(f"{t[k]:.0f} мин", color=style.TEXT)
        rows += [{"t_min": t[k], "cell": c, "edge": meta["edges"][int(sim.ctm.cells.edge[c])],
                  "density_ped_m2": float(anim[k][c])} for c in range(anim.shape[1])]
    sm = ScalarMappable(Normalize(0, sim.ctm.fd.rho_jam), style.SEQUENTIAL)
    cb = fig.colorbar(sm, ax=list(axs), fraction=0.03, pad=0.02, shrink=0.6)
    cb.set_label("Плотность, чел/м²")
    meta_out = {**_base_meta(cfg, meta), "animation_seed": cfg["animation_seed"],
                "moments_min": moments,
                "episode": meta["episodes"][cfg["animation_seed"]]}
    return fig, pd.DataFrame(rows), meta_out


def animation():
    """Анимация плотности по сети (эпизод animation_seed) + люди в районе и очереди.
    Кадр — интервал управления (1 мин). Сохраняется как .gif (и .mp4 при наличии ffmpeg)."""
    cfg, traj, anim, meta, sim = _load()
    b = cfg["animation_seed"]
    geo = netplot.NetGeometry(sim.net, sim.ctm.cells)
    t = traj.t / 60
    st = sim.stations
    fig = plt.figure(figsize=(12.8, 7.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], hspace=0.55, wspace=0.12,
                          left=0.02, right=0.97, top=0.92, bottom=0.08)
    axm = fig.add_subplot(gs[:, 0])
    lc = netplot.draw_network(axm, geo, anim[0], stadium_label=False)
    netplot.draw_nodes(axm, geo, sim, signs=False, scale=0.8)
    sm = ScalarMappable(Normalize(0, sim.ctm.fd.rho_jam), style.SEQUENTIAL)
    cb = fig.colorbar(sm, ax=axm, fraction=0.04, pad=0.02, shrink=0.7, location="bottom")
    cb.set_label("Плотность, чел/м²")
    title = fig.suptitle(_frame_title(0), x=0.02, ha="left", fontsize=13, color=style.TEXT)

    ax1 = fig.add_subplot(gs[0, 1])
    lines1 = []
    for name, X, c in (("на стадионе", traj.in_stadium, style.NEUTRAL),
                       ("на улицах", traj.on_network, style.SERIES[0]),
                       ("в станциях", traj.in_stations, style.SERIES[1])):
        ax1.plot(t, X[:, b] / 1000, color=c, lw=0.8, alpha=0.25)
        lines1.append((ax1.plot([], [], color=c, lw=1.8, label=name)[0], X[:, b] / 1000))
    ax1.set_ylabel("тыс. чел")
    ax1.set_title("Люди в районе", loc="left", color=style.TEXT)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    lines2 = []
    for s, sid in enumerate(st.ids):
        X = traj.queue_outside[:, b, s] / 1000
        ax2.plot(t, X, color=style.SERIES[s], lw=0.8, alpha=0.25)
        lines2.append((ax2.plot([], [], color=style.SERIES[s], lw=1.8,
                                label=STATION_LABELS.get(sid, sid))[0], X))
    ax2.set_ylabel("тыс. чел")
    ax2.set_title("Очередь перед входом", loc="left", color=style.TEXT)
    ax2.legend(loc="upper right", fontsize=8)

    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    lines3 = []
    ids = meta["edges"]
    for j, eid in enumerate(cfg["key_edges"]):
        X = traj.edge_max[:, b, ids.index(eid)]
        ax3.plot(t, X, color=style.SERIES[j], lw=0.8, alpha=0.25)
        lines3.append((ax3.plot([], [], color=style.SERIES[j], lw=1.6, label=eid)[0], X))
    ax3.axhline(meta["congestion_street_ped_per_m2"], color=style.TEXT_SECONDARY, lw=0.8, ls=":")
    for thr in (meta["rho_crit_street_ped_per_m2"], meta["rho_crit_zone_ped_per_m2"]):
        ax3.axhline(thr, color=style.TEXT_SECONDARY, lw=0.8, ls="--")
    ax3.set_ylim(0, 6)
    ax3.set_ylabel("чел/м²")
    ax3.set_title("Наибольшая плотность на ключевых участках", loc="left", color=style.TEXT)
    ax3.legend(loc="upper right", fontsize=7, ncol=3)
    ax3.set_xlabel("Время, мин")
    cursors = [ax.axvline(0, color=style.TEXT, lw=0.8) for ax in (ax1, ax2, ax3)]

    # кадры — до рассасывания плюс 5 мин (дальше сеть пуста)
    empty = np.flatnonzero(traj.in_area[:, b] > 0.5)
    n_frames = int(min(len(t), (empty[-1] + 6) if empty.size else len(t)))
    ax3.set_xlim(0, t[n_frames - 1])

    def update(k):
        lc.set_array(anim[k])
        title.set_text(_frame_title(t[k]))
        for lines in (lines1, lines2, lines3):
            for ln, X in lines:
                ln.set_data(t[:k + 1], X[:k + 1])
        for c in cursors:
            c.set_xdata([t[k], t[k]])
        return [lc, title, *cursors]

    anim_obj = FuncAnimation(fig, update, frames=n_frames, interval=125, blit=False)
    data = pd.DataFrame({"t_min": np.repeat(t[:n_frames], anim.shape[1]),
                         "cell": np.tile(np.arange(anim.shape[1]), n_frames),
                         "edge": np.tile([ids[int(e)] for e in sim.ctm.cells.edge], n_frames),
                         "density_ped_m2": anim[:n_frames].reshape(-1)})
    meta_out = {**_base_meta(cfg, meta), "animation_seed": b, "frames": int(n_frames),
                "fps": 8, "frame_step_min": float(t[1] - t[0]),
                "episode": meta["episodes"][b]}
    return anim_obj, data, meta_out
