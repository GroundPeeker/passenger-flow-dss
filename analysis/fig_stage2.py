"""Рисунки этапа 2 — проверка модулей поведения и станции (план работ: compliance_split,
station_chain). Рисунки проверки модели считают симуляцию внутри скрипта (постановка задачи).

Каждая функция возвращает (рисунок, данные pandas.DataFrame, метаданные dict).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from analysis import style
from sim.ctm import CTMParams
from sim.simulator import Actions, Simulator

SCENARIO = style.ROOT / "configs" / "scenario_toy.yaml"


def _sim(changes=None) -> Simulator:
    """Модель (fidelity="model") без шума процесса: рисунки проверки модели
    детерминированы и сравниваются с точными ожидаемыми значениями."""
    with open(SCENARIO, encoding="utf-8") as f:
        cfg = json.loads(json.dumps(yaml.safe_load(f)))   # без общих YAML-якорей
    cfg["fidelity"]["model"]["release_noise"] = {"model": "none"}
    if changes:
        changes(cfg)
    params = dataclasses.replace(CTMParams.from_yaml(style.ROOT / cfg["ctm"]), debug=False)
    return Simulator(cfg, style.ROOT, ctm_params=params, fidelity="model")


def _run(sim, params, n_steps, actions=None):
    rng = np.random.default_rng(0)  # шум наблюдений не используется в рисунках
    s = sim.reset(params)
    exit_sum = np.zeros_like(s.edge_exit)
    hist = []
    for k in range(n_steps):
        s, _ = sim.step(s, actions(k) if actions else None, rng)
        exit_sum += s.edge_exit
        hist.append(s)
    return s, exit_sum, hist


def compliance_split():
    """Доли по ветвям развилки J1 при разной степени следования c (план работ, этап 2).
    Рекомендация «north»: всем, кто идёт в метро, — по e2. Привычка: M1 — по e2,
    M2 — по e3. Доля людей M2, ушедших по e2, должна быть равна c."""
    sim = _sim()
    N, n_steps = 6000.0, 150
    cs = [0.0, 0.25, 0.5, 0.75, 1.0]
    e, p = sim.net.edge_index, sim.net.dest_index
    s1 = sim.net.source_shares[:, 0] * sim.demand.sector_mean[0] * N   # (P,) люди S1
    act = Actions.from_dict(sim, 1, recommend={"J1": "north"})
    rows = []
    for c in cs:
        _, exit_sum, _ = _run(sim, sim.fixed_params(1, c=c, total=N), n_steps, lambda k: act)
        e2, e3 = exit_sum[0, e["e2"]], exit_sum[0, e["e3"]]
        # M1 при рекомендации north идут по e2 и по привычке, и по рекомендации:
        # всё сверх них на e2 — люди M2, последовавшие рекомендации.
        m2_e2 = e2 - s1[p["M1"]]
        rows.append({"c": c, "e2_ped": e2, "e3_ped": e3, "M2_total_ped": s1[p["M2"]],
                     "M2_share_e2": m2_e2 / s1[p["M2"]], "expected_share": c})
    data = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.a4(7))
    x = np.arange(len(cs))
    w = 0.38
    ax1.bar(x - w / 2 - 0.01, data["e2_ped"], w, color=style.SERIES[0], label="e2 (north)")
    ax1.bar(x + w / 2 + 0.01, data["e3_ped"], w, color=style.SERIES[1], label="e3 (south)")
    ax1.set_xticks(x, [f"{c:g}" for c in cs])
    ax1.set_xlabel("Степень следования c")
    ax1.set_ylabel("Прошло по ветви, чел")
    ax1.set_title("Потоки по ветвям J1", loc="left", color=style.TEXT)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax1.grid(axis="x", visible=False)

    cc = np.linspace(0, 1, 50)
    ax2.plot(cc, cc, color=style.NEUTRAL, lw=1.0, ls="--", label="ожидаемо: доля = c")
    ax2.plot(data["c"], data["M2_share_e2"], "o", color=style.SERIES[0], markersize=7,
             markeredgecolor="white", markeredgewidth=1.0, label="симуляция")
    ax2.annotate("привычка", (0, 0), xytext=(6, 4), textcoords="offset points",
                 color=style.TEXT_SECONDARY, fontsize=8)
    ax2.annotate("строго по\nрекомендации", (1, 1), xytext=(-62, -22), textcoords="offset points",
                 color=style.TEXT_SECONDARY, fontsize=8)
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_xlabel("Степень следования c")
    ax2.set_ylabel("Доля людей M2 по e2")
    ax2.set_title("Следование рекомендации", loc="left", color=style.TEXT)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    meta = {"scenario": str(SCENARIO.relative_to(style.ROOT)), "seed": None,
            "fidelity": "model", "release_noise": "none", "total_ped": N, "horizon_min": n_steps, "recommendation": {"J1": "north"},
            "c_values": cs,
            "max_abs_share_error": float((data["M2_share_e2"] - data["c"]).abs().max()),
            "description": "Параметры спроса — средние (без неопределённости); "
                           "доля M2 по e2 восстановлена как (поток e2 − люди M1) / люди M2"}
    return fig, data, meta


def station_chain():
    """ПРОВЕРОЧНЫЙ СЦЕНАРИЙ (не типичный): заполнение звеньев станции A при малом вывозе
    (план работ, этап 2): очередь растёт
    сначала на платформе, потом на эскалаторе, потом у турникетов, потом выходит на улицу.

    Сценарий подобран так, чтобы узким местом был только вывоз: спрос растянут (среднее
    время выхода 20 мин, форма 2, N = 7000), пик подхода к M1 ниже пропускной способности
    турникетов 200 чел/мин; вывоз искусственно мал — 450 свободных мест раз в 30 мин
    (15 чел/мин). Тогда очередь у входа возникает только после заполнения станции.
    Поведение в типичном сценарии (смена узких мест) — тест
    tests/stage2/test_station.py::test_typical_scenario_bottleneck_shifts."""
    headway, capacity = 1800.0, 450.0
    profile = {"mean": 20 * 60.0, "shape": 2.0}

    def slow(d):
        for st in d["stations"]:
            st["vehicle"].update(headway_levels_s=[headway], max_level_changes=0,
                                 capacity_ped=capacity)
    sim = _sim(slow)
    N, n_steps = 7000.0, 90
    _, _, hist = _run(sim, sim.fixed_params(1, c=0.0, total=N, profile=profile), n_steps)
    st = sim.stations
    a = st.ids.index("A")
    links = list(range(st.first[a], st.last[a] + 1))
    names = [lk.name for lk in st.specs[a].links]
    labels = {"turnstiles": "турникеты", "escalator": "вестибюль/эскалатор", "platform": "платформа"}
    t = np.arange(1, n_steps + 1) * sim.control_step / 60
    e = sim.net.edge_index
    def rho(s, k):  # средняя плотность на ребре, чел/м²
        cells = sim.ctm.cells.edge == k
        return s.ctm.n[0, :, cells].sum() / sim.ctm.area[cells].sum()
    rho_e2a = np.array([rho(s, e["e2a"]) for s in hist])
    rho_e2 = np.array([rho(s, e["e2"]) for s in hist])
    adm = np.diff([0.0] + [s.stations.admitted[0, a] for s in hist]) / (sim.control_step / 60)
    fill = np.array([[s.stations.x[0, L] / st.storage[L] for L in links] for s in hist])

    door = sim.stations.ent_cap[sim.stations.ent_ids.index("M1")] * 60
    train = st.veh_cap[a] / headway * 60
    turn = st.link_cap[st.first[a]] * 60
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=style.a4(15), sharex=True)
    colors = {"platform": style.SERIES[0], "escalator": style.SERIES[1],
              "turnstiles": style.SERIES[2]}
    for j, nm in reversed(list(enumerate(names))):
        ax1.plot(t, 100 * fill[:, j], color=colors.get(nm, style.SERIES[3]), label=labels.get(nm, nm))
    ax1.set_ylabel("Заполнение, %")
    ax1.set_ylim(0, 135)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_title("Звенья станции A — проверочный сценарий (вывоз 15 чел/мин)", loc="left",
                  color=style.TEXT)
    ax1.legend(loc="upper left", ncol=3)
    ax2.plot(t, adm, color=style.SERIES[0], label="приём дверей")
    ax2.axhline(door, color=style.NEUTRAL, lw=0.9, ls="--")
    ax2.axhline(train, color=style.NEUTRAL, lw=0.9, ls=":")
    ax2.annotate(f"пропускная способность дверей {door:.0f}", (t[-1], door), xytext=(0, 3),
                 textcoords="offset points", ha="right", color=style.TEXT_SECONDARY, fontsize=8)
    ax2.annotate(f"вывоз поездами {train:.0f}", (t[-1], train), xytext=(0, 3),
                 textcoords="offset points", ha="right", color=style.TEXT_SECONDARY, fontsize=8)
    ax2.set_ylabel("Приём, чел/мин")
    ax2.axhline(turn, color=style.NEUTRAL, lw=0.9, ls="-.")
    ax2.annotate(f"турникеты {turn:.0f}", (t[-1], turn), xytext=(0, 3),
                 textcoords="offset points", ha="right", color=style.TEXT_SECONDARY, fontsize=8)
    ax2.set_ylim(0, door * 1.15)
    ax2.set_title("Вход M1", loc="left", color=style.TEXT)
    ax3.plot(t, rho_e2a, color=style.SERIES[0], label="e2a: площадка → вход (12 м)")
    ax3.plot(t, rho_e2, color=style.SERIES[1], label="e2: улица к площадке (180 м)")
    ax3.axhline(sim.ctm.fd.rho_crit, color=style.NEUTRAL, lw=0.9, ls="--")
    ax3.annotate(f"ρc ≈ {sim.ctm.fd.rho_crit:.2f}", (t[-1], sim.ctm.fd.rho_crit), xytext=(0, 3),
                 textcoords="offset points", ha="right", color=style.TEXT_SECONDARY, fontsize=8)
    ax3.set_ylabel("Плотность, чел/м²")
    ax3.set_xlabel("Время после окончания матча, мин")
    ax3.set_title("Подход к входу", loc="left", color=style.TEXT)
    ax3.legend(loc="upper left")
    ax3.set_xlim(0, t[-1])
    ax3.set_ylim(0, None)
    fig.tight_layout()
    data = pd.DataFrame({"t_min": t, "admitted_ped_per_min": adm, "rho_e2a_ped_m2": rho_e2a, "rho_e2_ped_m2": rho_e2,
                         **{f"fill_{nm}": fill[:, j] for j, nm in enumerate(names)}})
    meta = {"verification_scenario": True, "fidelity": "model", "release_noise": "none",
            "scenario": str(SCENARIO.relative_to(style.ROOT)), "seed": None, "total_ped": N,
            "exit_profile": {"mean_s": profile["mean"], "shape": profile["shape"]},
            "headway_s": headway, "vehicle_capacity_ped": float(st.veh_cap[a]),
            "door_capacity_ped_per_min": door, "storage_ped": st.storage[links].tolist(),
            "description": "ПРОВЕРОЧНЫЙ СЦЕНАРИЙ, не типичный: доли спроса — средние; c = 0; "
                           "вывоз 15 чел/мин искусственно много меньше пропускной способности "
                           "турникетов 200 чел/мин"}
    return fig, data, meta
