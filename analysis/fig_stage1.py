"""Рисунки этапа 1 — проверка модели CTM (план работ: fundamental_diagram, conservation,
bottleneck_queue; дополнительно spacetime_spillback). По постановке задачи рисунки проверки модели
могут пересчитывать симуляцию внутри скрипта (результаты экспериментов здесь не нужны).

Каждая функция возвращает (рисунок, данные pandas.DataFrame, метаданные dict).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import style
from sim.bench import CTM_CONFIG
from sim.ctm import CTM, CTMParams
from sim.network import Network


def _params() -> CTMParams:
    return dataclasses.replace(CTMParams.from_yaml(CTM_CONFIG), debug=False)


def _line(length, width, inflow, capacity=None):
    return Network.from_dict({
        "nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": inflow},
                  {"id": "K", "type": "sink", "kind": "exit",
                   "capacity_ped_per_s": capacity, "buffer_ped": 0}],
        "edges": [{"id": "e", "from": "S", "to": "K", "length_m": length, "width_m": width}],
    })


def _ctm_meta(p: CTMParams) -> dict:
    return {"ctm_config": str(CTM_CONFIG.relative_to(style.ROOT)), "dt_s": p.dt,
            "fd": dataclasses.asdict(p.fd), "fifo": p.fifo, "seed": None}


def fundamental_diagram():
    """Фундаментальная диаграмма Weidmann и стационарные состояния CTM на длинном ребре:
    свободная ветвь (поток источника q·W) и заторная (сток пропускает q·W)."""
    p = _params()
    fd = p.fd
    W = 4.0
    rows = []
    for q in [0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.05, 1.15]:
        m = CTM(_line(1000.0, W, q * W), p)
        rows.append(("свободная", q, *_steady(m, 2000)))
    for q in [0.2, 0.35, 0.5, 0.7, 0.9, 1.05]:
        m = CTM(_line(300.0, W, fd.q_max * W, capacity=q * W), p)
        rows.append(("заторная", q, *_steady(m, 4000)))
    sim = pd.DataFrame(rows, columns=["ветвь", "q_задано", "rho_сим", "q_сим", "v_сим"])

    rho = np.linspace(1e-3, fd.rho_jam, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.a4(6.5))
    ax1.plot(rho, fd.flow(rho), color=style.SERIES[0], label="Weidmann (1993)")
    ax1.plot(sim["rho_сим"], sim["q_сим"], "o", color=style.SERIES[1], markersize=5,
             markeredgecolor="white", markeredgewidth=0.8, label="CTM, стационар")
    ax1.axvline(fd.rho_crit, color=style.NEUTRAL, lw=0.8, ls="--")
    ax1.annotate(f"ρc ≈ {fd.rho_crit:.2f}", (fd.rho_crit, 0.08), xytext=(4, 0),
                 textcoords="offset points", color=style.TEXT_SECONDARY, fontsize=8)
    ax1.set_xlabel("Плотность, чел/м²")
    ax1.set_ylabel("Удельный поток, чел/(м·с)")
    ax1.set_xlim(0, fd.rho_jam)
    ax1.set_ylim(0, None)
    ax2.plot(rho, fd.speed(rho), color=style.SERIES[0], label="Weidmann (1993)")
    ax2.plot(sim["rho_сим"], sim["v_сим"], "o", color=style.SERIES[1], markersize=5,
             markeredgecolor="white", markeredgewidth=0.8, label="CTM, стационар")
    ax2.set_xlabel("Плотность, чел/м²")
    ax2.set_ylabel("Скорость, м/с")
    ax2.set_xlim(0, fd.rho_jam)
    ax2.set_ylim(0, None)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    err = float(np.max(np.abs(sim["q_сим"] - fd.flow(sim["rho_сим"]))))
    meta = {**_ctm_meta(p), "width_m": W, "max_abs_flow_error": err,
            "description": "Стационарные точки CTM на ребре 1000 м (свободная ветвь) и "
                           "300 м с узким стоком (заторная ветвь) против диаграммы Weidmann"}
    return fig, sim, meta


def _steady(m: CTM, n_steps: int):
    s, _ = m.run(m.initial_state(), n_steps)
    mid = slice(m.n_cells // 4, 3 * m.n_cells // 4)
    rho = float(m.density(s)[0, mid].mean())
    q = float((s.cell_outflow[0, :, mid].sum(0) / (m.cells.width[mid] * m.dt)).mean())
    return rho, q, q / rho


def bottleneck_queue():
    """Очередь у узкого стока (план работ, этап 1): пока приток λ больше пропускной
    способности μ, число людей в системе растёт линейно с темпом λ − μ; после
    прекращения притока очередь рассасывается с темпом μ. При λ < μ очереди нет."""
    p = _params()
    mu, t_stop_min, horizon_min = 1.0, 20.0, 70.0
    cases = [(3.0, style.SERIES[0]), (1.5, style.SERIES[1]), (0.8, style.SERIES[2])]
    n_steps = int(horizon_min * 60 / p.dt)
    n_on = int(t_stop_min * 60 / p.dt)
    t_min = np.arange(1, n_steps + 1) * p.dt / 60
    rows = []
    fig, ax = plt.subplots(figsize=style.a4(7))
    for lam, color in cases:
        m = CTM(_line(200.0, 4.0, lam, capacity=mu), p)
        s = m.initial_state()
        off = np.zeros((1, m.n_dest, len(m.source_nodes)))
        n_sys = np.empty(n_steps)
        for k in range(n_steps):
            s = m.step(s, inflow=None if k < n_on else off)
            n_sys[k] = s.in_system()[0]
        # ориентир точечной очереди: рост (λ − μ)·t до остановки, затем убывание с темпом μ
        peak = max(lam - mu, 0.0) * t_stop_min * 60
        ref = np.where(t_min <= t_stop_min, max(lam - mu, 0.0) * t_min * 60,
                       np.maximum(peak - mu * (t_min - t_stop_min) * 60, 0.0))
        ax.plot(t_min, n_sys, color=color, label=f"λ = {lam:g} чел/с")
        if lam > mu:
            ax.plot(t_min, ref, color=color, lw=0.9, ls=":")
        rows += [{"lambda_ped_s": lam, "mu_ped_s": mu, "t_min": t, "in_system": v,
                  "point_queue_ref": r} for t, v, r in zip(t_min, n_sys, ref)]
    ax.plot([], [], color=style.NEUTRAL, lw=0.9, ls=":",
            label="точечная очередь: (λ − μ)·t, затем −μ·t")
    ax.axvline(t_stop_min, color=style.NEUTRAL, lw=0.8, ls="--")
    ax.annotate("приток прекращён", (t_stop_min, 0.45), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", color=style.TEXT_SECONDARY,
                fontsize=8)
    ax.set_xlabel("Время, мин")
    ax.set_ylabel("Людей в системе, чел")
    ax.set_xlim(0, horizon_min)
    ax.set_ylim(0, None)
    ax.set_title(f"Узкий сток μ = {mu:g} чел/с, ребро 200 × 4 м", loc="left", color=style.TEXT)
    ax.legend(loc="upper right")
    fig.tight_layout()
    meta = {**_ctm_meta(p), "mu_ped_s": mu, "lambdas_ped_s": [c[0] for c in cases],
            "inflow_stop_min": t_stop_min, "horizon_min": horizon_min,
            "edge": {"length_m": 200, "width_m": 4}, "sink_buffer_ped": 0}
    return fig, pd.DataFrame(rows), meta


def conservation():
    """Сохранение людей во времени (план работ, этап 1): невязка
    «вышло + перенесено − (на сети + в очередях + в накопителях) − ушло» на тестовой сети
    и на эталонной решётке; должна быть нулём до ошибки округления."""
    from sim.bench import load_bench, reference_network
    p = _params()
    horizon_min = 60.0
    n_steps = int(horizon_min * 60 / p.dt)
    t_min = np.arange(1, n_steps + 1) * p.dt / 60
    # Обе невязки — ноль, вторая линия пунктиром, чтобы не скрывать первую.
    nets = [("тестовая сеть",
             Network.from_yaml(style.ROOT / "networks" / "toy_fork.yaml"), style.SERIES[0], "-"),
            ("решётка, 50 рёбер",
             reference_network(load_bench()), style.SERIES[1], "--")]
    rows = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.a4(7.5))
    for label, net, color, ls in nets:
        m = CTM(net, p)
        s = m.initial_state()
        gen, sys_, srv, res = (np.empty(n_steps) for _ in range(4))
        for k in range(n_steps):
            s = m.step(s)
            gen[k] = s.generated.sum() + s.transferred.sum()
            sys_[k] = s.in_system()[0]
            srv[k] = s.served.sum()
            res[k] = gen[k] - sys_[k] - srv[k]
        rows += [{"network": label, "t_min": t, "generated": g, "in_system": i, "served": v,
                  "residual": r} for t, g, i, v, r in zip(t_min, gen, sys_, srv, res)]
        ax2.plot(t_min, res / gen.max(), color=color, ls=ls, label=label)
        if net is nets[0][1]:
            ax1.plot(t_min, gen, color=style.SERIES[0], label="вышло")
            ax1.plot(t_min, sys_, color=style.SERIES[2], label="в районе")
            ax1.plot(t_min, srv, color=style.SERIES[3], label="покинули район")
    ax1.set_title("Тестовая сеть", loc="left", color=style.TEXT)
    ax1.set_xlabel("Время, мин")
    ax1.set_ylabel("Людей, чел")
    ax1.set_xlim(0, horizon_min)
    ax1.set_ylim(0, None)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax2.set_title("Расхождение баланса", loc="left", color=style.TEXT)
    ax2.set_xlabel("Время, мин")
    ax2.set_ylabel("Расхождение / вышло за эпизод")
    ax2.set_xlim(0, horizon_min)
    ax2.set_ylim(-1e-12, 1e-12)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    data = pd.DataFrame(rows)
    meta = {**_ctm_meta(p), "horizon_min": horizon_min,
            "networks": {"тестовая сеть": "networks/toy_fork.yaml",
                         "решётка, 50 рёбер": "configs/bench.yaml"},
            "max_abs_residual_ped": float(data["residual"].abs().max()),
            "max_rel_residual": float((data["residual"].abs()
                                       / data.groupby("network")["generated"].transform("max")).max())}
    return fig, data, meta


def spacetime_spillback():
    """Диаграмма «время — положение» плотности на ребре перед узким стоком:
    затор возникает у стока и распространяется назад до источника."""
    p = _params()
    L, W, lam, mu = 300.0, 4.0, 3.0, 2.0
    m = CTM(_line(L, W, lam, capacity=mu), p)
    n_steps = int(60 * 60 / p.dt)
    _, hist = m.run(m.initial_state(), n_steps, record=True)
    rho = hist[:, 0, :] / m.area                       # (T, C)
    x = (np.arange(m.n_cells) + 0.5) * m.cells.length[0]
    t_min = np.arange(1, n_steps + 1) * p.dt / 60

    fig, ax = plt.subplots(figsize=style.a4(7))
    im = ax.imshow(rho.T, origin="lower", aspect="auto", cmap=style.SEQUENTIAL,
                   vmin=0, vmax=p.fd.rho_jam,
                   extent=[0, t_min[-1], 0, L], interpolation="nearest")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Плотность, чел/м²")
    cb.ax.axhline(p.fd.rho_crit, color="white", lw=1)
    ax.set_xlabel("Время, мин")
    ax.set_ylabel("Положение от источника, м")
    ax.set_title(f"Приток λ = {lam:g} чел/с, сток μ = {mu:g} чел/с, ребро {L:g} × {W:g} м",
                 loc="left", color=style.TEXT)
    fig.tight_layout()
    tt, xx = np.meshgrid(t_min, x, indexing="ij")
    data = pd.DataFrame({"t_min": tt.ravel(), "x_m": xx.ravel(), "rho": rho.ravel()})
    meta = {**_ctm_meta(p), "lambda_ped_s": lam, "mu_ped_s": mu,
            "edge": {"length_m": L, "width_m": W}, "n_cells": m.n_cells}
    return fig, data, meta
