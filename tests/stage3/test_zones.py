"""Накопители (широкие зоны ожидания, этап 3а), мониторинг плотности по шагам CTM,
использование маршрута через парк."""

import dataclasses

import numpy as np
import pytest

from sim.ctm import CTM, CTMParams
from sim.metrics import MetricsConfig, foreign_arrivals, run_episode
from sim.network import Network
from sim.simulator import Actions


def zone_line(inflow, cap, zone_len=60.0, zone_w=40.0):
    """S → a (улица 100 × 5) → Z (накопитель) → K (сток с пропускной способностью cap)."""
    return Network.from_dict({
        "nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": inflow},
                  {"id": "J", "type": "junction"},
                  {"id": "K", "type": "sink", "kind": "exit", "capacity_ped_per_s": cap}],
        "edges": [{"id": "a", "from": "S", "to": "J", "length_m": 100, "width_m": 5},
                  {"id": "z", "from": "J", "to": "K", "length_m": zone_len, "width_m": zone_w,
                   "zone": True}],
    })


@pytest.fixture
def zparams(params):
    return dataclasses.replace(params, queue_density=3.0)


def test_zone_is_one_cell(zparams):
    m = CTM(zone_line(1.0, 1.0), zparams)
    k = m.net.edge_index["z"]
    assert m.cells.n_per_edge[k] == 1
    assert m.zone_cap[0] == pytest.approx(3.0 * 60 * 40)


def test_zone_requires_queue_density(params):
    with pytest.raises(ValueError, match="queue_density"):
        CTM(zone_line(1.0, 1.0), dataclasses.replace(params, queue_density=None))


def test_ctm_yaml_has_queue_density(root):
    p = CTMParams.from_yaml(root / "configs" / "ctm.yaml")
    assert 2.0 <= p.queue_density <= 4.0   # диапазон чувствительности (решение студента)


def test_zone_fills_to_capacity_then_spills_back(zparams):
    """Приток 3 чел/с, сток 0.5 чел/с: накопитель заполняется ровно до A·ρ_queue,
    дальше затор уходит на улицу; плотность накопителя = люди / площадь ≤ ρ_queue;
    люди сохраняются (проверки режима debug на каждом шаге)."""
    m = CTM(zone_line(3.0, 0.5), zparams)
    s = m.initial_state()
    z = m.zone_cells[0]
    a = np.arange(m.cells.first[0], m.cells.last[0] + 1)
    hist_z, hist_a = [], []
    for _ in range(2400):                      # 200 мин
        s = m.step(s)
        hist_z.append(s.n[0, :, z].sum())
        hist_a.append(s.n[0, :, a].sum())
    hist_z, hist_a = np.array(hist_z), np.array(hist_a)
    cap = m.zone_cap[0]
    assert hist_z.max() <= cap * (1 + 1e-12)
    # приём — по состоянию начала шага: в равновесии свободно ровно место оттока за шаг
    full = np.flatnonzero(hist_z >= cap - 0.5 * m.dt * (1 + 1e-9))
    assert full.size > 0
    # до заполнения улица свободна (ρ < ρc), после — затор на улице
    first_full = full[0]
    rho_a_before = hist_a[:first_full - 20].max() / m.area[a].sum()
    assert rho_a_before < m.fd.rho_crit
    assert hist_a[-1] / m.area[a].sum() > m.fd.rho_crit
    np.testing.assert_allclose(m.density(s)[0, z], hist_z[-1] / m.area[z])


def test_zone_free_flow_outflow(zparams):
    """Пустой сток без ограничения: выход из накопителя за шаг = n·v0·dt/L (свободное
    движение через площадь)."""
    m = CTM(zone_line(0.0, None), dataclasses.replace(zparams, debug=False))
    s = m.initial_state()
    z = m.zone_cells[0]
    s.n[0, 0, z] = 300.0
    k = m.edge_in_slot[m.net.edge_index["z"]]
    s.committed[0, 0, np.flatnonzero(m.mov_in == k)] = 300.0
    s2 = m.step(s)
    expected = 300.0 * m.fd.v0 * m.dt / 60.0
    assert s2.served[0].sum() == pytest.approx(expected)


def test_zone_without_zones_matches_street_cells(params):
    """Сеть без накопителей — ядро ведёт себя как на этапе 1 (вместимость ячейки ρ_jam·A)."""
    from tests.helpers import line_network
    m = CTM(line_network(200, 4, 1.0, capacity=0.5), params)
    assert m.zone_cells.size == 0
    np.testing.assert_array_equal(m.cell_cap, m.fd.rho_jam * m.area)


def test_closed_entrance_with_zone_redirects(stadium):
    """Вход A закрыт 15 мин: люди A с площади (накопителя) уходят к станции B, никто
    не теряется, «чужих» прибытий нет; после открытия вход снова принимает."""
    sim = stadium()
    rng = np.random.default_rng(0)
    s = sim.reset(sim.fixed_params(1))
    total = s.stadium.sum()
    closed = Actions.from_dict(sim, 1, entrances={"MA": 0.0})
    admitted = []
    for k in range(120):
        s, _ = sim.step(s, closed if 15 <= k < 30 else None, rng)
        b = sim.balance(s)
        assert abs(b["residual"][0]) < 1e-6 * total
        admitted.append(s.stations.admitted[0, 0])
    a = np.array(admitted)
    assert a[29] - a[16] == pytest.approx(0, abs=1e-6)      # закрыт — никто не вошёл
    assert a[60] > a[30]
    assert s.ctm.transferred[0, sim.net.dest_index["MB"]] > 0
    assert s.ctm.uturned[0, :, sim.net.edge_index["e_sq"]].sum() > 0
    assert foreign_arrivals(sim.ctm, s.ctm).sum() == pytest.approx(0, abs=1e-9)


def test_monitor_matches_snapshots_when_interval_is_one_ctm_step(stadium):
    """Шаг управления = шагу CTM: агрегаты интервала совпадают со снимком — экспозиция
    = люди в ячейках выше порога × dt, площадь выше порога — сумма площадей таких ячеек."""
    def one_step(cfg):
        del cfg["control_step_min"]
        cfg["control_step_s"] = 5
    sim = stadium(debug=False, changes=one_step)
    mc = MetricsConfig(0.3, 0.8, 0.5, 1.0, 0.95, 0.01, None)
    sim.set_metrics(mc)
    rng = np.random.default_rng(0)
    tr = run_episode(sim, sim.fixed_params(1), rng, 300, cfg=mc, cells=True)
    rho = tr.cell_density[1:, 0].astype(float)                  # (T, C)
    area = sim.ctm.area
    thr = np.full(sim.ctm.n_cells, 0.5)
    thr[sim.ctm.zone_cells] = 1.0
    above = rho > thr
    np.testing.assert_allclose(tr.area_above[1:, 0], (above * area).sum(axis=1), rtol=1e-5)
    exp_cells = rho * area * above * sim.dt
    exp_edge = np.add.reduceat(exp_cells, sim.ctm.cells.first, axis=1)
    np.testing.assert_allclose(tr.exposure[1:, 0], exp_edge, rtol=1e-5, atol=1e-6)
    assert tr.exposure.sum() > 0
    # затор — свой порог (0.3 / 0.8), экспозиция затора ≥ экспозиции опасной плотности
    cthr = np.full(sim.ctm.n_cells, 0.3)
    cthr[sim.ctm.zone_cells] = 0.8
    cong = rho > cthr
    np.testing.assert_allclose(tr.area_congestion[1:, 0], (cong * area).sum(axis=1), rtol=1e-5)
    assert np.all(tr.exposure_congestion >= tr.exposure - 1e-6)
    # заполнение зон ≥ 95 %: интервал = один шаг CTM → dt, если зона заполнена в снимке
    zc = sim.ctm.zone_cells
    full = rho[:, zc] * area[zc] >= 0.95 * sim.ctm.zone_cap * (1 - 1e-6)
    np.testing.assert_allclose(tr.zone_full_time[1:, 0], full * sim.dt)


def test_park_unused_by_habit_not_by_routing_error(stadium):
    """Парк почти не используется из-за привычных долей (кратчайший путь к A — проспект),
    а не из-за ошибки маршрутов. Через развилку JN идут все люди метро — и с севера, и
    с юга (по обходу стадиона). Без рекомендации по e_pk идут только люди B, по e_pj —
    никто; при рекомендации «через парк» и c = 1 по e_pk идут все люди метро, по e_pj —
    все люди A; при c = 0.5 — половина людей A."""
    sim = stadium()
    e, p = sim.net.edge_index, sim.net.dest_index
    N = 30000.0
    metro = sim.net.source_shares @ sim.demand.sector_mean * N    # (P,) все зрители по назначениям

    def flows(c, rec):
        rng = np.random.default_rng(0)
        s = sim.reset(sim.fixed_params(1, c=c, total=N))
        act = Actions.from_dict(sim, 1, recommend={"JN": rec})
        tot = np.zeros(len(sim.net.edges))
        for _ in range(150):
            s, _ = sim.step(s, act, rng)
            tot += s.edge_exit[0]
        return tot

    habit = flows(0.0, None)
    assert habit[e["e_pk"]] == pytest.approx(metro[p["MB"]], rel=1e-6)
    assert habit[e["e_pj"]] == pytest.approx(0.0, abs=1e-6)
    park = flows(1.0, "park")
    assert park[e["e_pk"]] == pytest.approx(metro[p["MA"]] + metro[p["MB"]], rel=1e-6)
    assert park[e["e_pj"]] == pytest.approx(metro[p["MA"]], rel=1e-6)
    half = flows(0.5, "park")
    assert half[e["e_pj"]] == pytest.approx(0.5 * metro[p["MA"]], rel=1e-6)


def test_thresholds_by_element_type(stadium):
    """Пороги ячейки — по типу элемента (накопитель / улица), для затора и опасной плотности;
    полная зона ожидания (ρ_queue) не считается ни затором, ни опасной."""
    sim = stadium(debug=False)
    z = sim.ctm.zone_cells
    mask = np.ones(sim.ctm.n_cells, dtype=bool)
    mask[z] = False
    for cells, (street, zone) in ((sim.rho_crit_cells, sim.metrics_cfg.rho_crit),
                                  (sim.congestion_cells, sim.metrics_cfg.congestion)):
        assert np.all(cells[z] == zone) and np.all(cells[mask] == street)
        assert sim.ctm.params.queue_density < zone
    assert np.all(sim.rho_crit_cells > sim.congestion_cells)
