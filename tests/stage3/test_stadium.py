"""Сеть «стадион», запись траектории, очередь у входа, правдоподобие сценария."""

import numpy as np
import pytest

from sim.metrics import MetricsConfig, Recorder, episode_metrics, foreign_arrivals, run_episode
from sim.network import Network

CFG = MetricsConfig.from_yaml()


def test_network_structure(root):
    net = Network.from_yaml(root / "networks" / "stadium.yaml")
    assert 10 <= len(net.edges) <= 20                       # план работ, этап 3
    zones = {e.id for e in net.edges if e.zone}
    assert zones == {"e_n0", "e_s0", "e_sq", "e_mb", "e_mc"}  # широкие зоны ожидания
    assert all(e.width <= 10 for e in net.edges if not e.zone)   # улицы — в CTM
    assert {"MA", "MB", "MC", "OUT"} <= set(net.destinations)
    assert len(net.nodes_of_type("source")) == 2
    assert net.layout and set(net.layout["nodes"]) == {n.id for n in net.nodes}


def test_two_alternative_routes_to_main_station(root):
    """От развилки JN к метро A — два маршрута без общих рёбер до площади (проспект, парк);
    привычный — короче."""
    net = Network.from_yaml(root / "networks" / "stadium.yaml")
    e = net.edge_index
    p = net.dest_index["MA"]
    avenue = net.edges[e["e_av"]].length
    park = net.edges[e["e_pk"]].length + net.edges[e["e_pj"]].length
    assert net.edge_reaches(e["e_av"], p) and net.edge_reaches(e["e_pk"], p)
    assert avenue < park < 1.5 * avenue
    assert net.split[p, e["e_av"]] == 1.0 and net.split[p, e["e_pk"]] == 0.0


def test_sign_options_switch_routes(stadium):
    sim = stadium()
    e, p = sim.net.edge_index, sim.net.dest_index
    rec = np.array([[sim.signs.option_index("JN", "park")]])
    S = sim.signs.split(rec, np.ones((1, 1)))
    assert S[0, p["MA"], e["e_pk"]] == 1.0 and S[0, p["MB"], e["e_pk"]] == 1.0
    S = sim.signs.split(np.array([[sim.signs.option_index("JN", "avenue")]]), np.ones((1, 1)))
    assert S[0, p["MB"], e["e_av"]] == 1.0


def test_episode_conservation_and_no_wrong_arrivals(stadium):
    """Без управления: учёт людей сходится на каждом шаге, никто не пришёл не на свою
    станцию и не разворачивался, все ушли из района за 2,5 ч."""
    sim = stadium()
    rng = np.random.default_rng(0)
    p = sim.fixed_params(1)
    s = sim.reset(p)
    total = s.stadium.sum()
    for _ in range(150):
        s, _ = sim.step(s, None, rng)
        b = sim.balance(s)
        assert abs(b["residual"][0]) < 1e-6 * total
        assert abs(b["released"][0] + b["in_stadium"][0] - total) < 1e-6 * total
    assert foreign_arrivals(sim.ctm, s.ctm).sum() == pytest.approx(0, abs=1e-9)
    assert s.ctm.uturned.sum() == 0
    assert b["departed"][0] + b["exited"][0] == pytest.approx(total, rel=1e-9)


def test_trajectory_accounting(stadium):
    """В районе + ушло = всего зрителей в каждом снимке; плотности в [0, ρ_jam]."""
    sim = stadium(debug=False)
    rng = np.random.default_rng(1)
    tr = run_episode(sim, sim.sample_params(rng, 2), rng, 60, cfg=CFG, cells=True)
    assert tr.t.shape == (61,) and tr.edge_max.shape == (61, 2, len(sim.net.edges))
    np.testing.assert_allclose(tr.in_area + tr.left, np.broadcast_to(tr.total, tr.left.shape),
                               rtol=1e-9)
    assert tr.cell_density.min() >= 0
    assert tr.cell_density.max() <= sim.ctm.fd.rho_jam + 1e-6
    # максимум за интервал (по шагам CTM) не меньше плотности в снимке
    assert np.all(tr.edge_mean <= tr.edge_max + 1e-6)
    snap_max = np.zeros_like(tr.edge_max)
    np.maximum.at(snap_max, (slice(None), slice(None), sim.ctm.cells.edge), tr.cell_density)
    assert np.all(snap_max <= tr.edge_max + 1e-5)
    zc = sim.ctm.zone_cells
    assert tr.cell_density[..., zc].max() <= sim.ctm.params.queue_density + 1e-9


def test_nominal_scenario_is_plausible(stadium):
    """Средние параметры, «реальность» без управления: рассасывание — десятки минут,
    время в районе — минуты, а не секунды или часы (план работ, этап 3)."""
    sim = stadium(fidelity="truth", exact=False, debug=False)
    rng = np.random.default_rng(0)
    tr = run_episode(sim, sim.fixed_params(1), rng, 150, cfg=CFG)
    m = episode_metrics(tr, CFG)
    assert 30 * 60 < m["clearance_time"][0] < 150 * 60
    assert 5 * 60 < m["mean_time"][0] < 60 * 60
    assert m["mean_time"][0] < m["p95_time"][0] < 120 * 60
    assert m["censored_ped"][0] == pytest.approx(0, abs=1e-6)
    assert 0 < m["max_density"][0] <= sim.ctm.fd.rho_jam + 1e-6


def test_run_episode_deterministic(stadium):
    sim = stadium(fidelity="truth", exact=False, debug=False)

    def once(seed):
        rng = np.random.default_rng(seed)
        return run_episode(sim, sim.sample_params(rng, 1), rng, 40, cfg=CFG)
    a, b, c = once(3), once(3), once(4)
    assert a.edge_max.tobytes() == b.edge_max.tobytes()
    assert a.queue_outside.tobytes() == b.queue_outside.tobytes()
    assert a.edge_max.tobytes() != c.edge_max.tobytes()


def _state_with(sim, fill):
    """Начальное состояние с людьми в ячейках: fill — {(ребро, назначение): плотность}
    на всём ребре или {(ребро, назначение, ячейки ребра): плотность}."""
    s = sim.reset(sim.fixed_params(1))
    cells = sim.ctm.cells
    for key, rho in fill.items():
        eid, dest = key[:2]
        k, p = sim.net.edge_index[eid], sim.net.dest_index[dest]
        c = np.arange(cells.first[k], cells.last[k] + 1)
        if len(key) == 3:
            c = c[key[2]]
        s.ctm.n[0, p, c] = rho * sim.ctm.area[c]
    return s


def test_queue_outside_zone_and_spillback(stadium):
    """Очередь A: люди A на площади (накопителе) — всегда; выше по течению — только при
    заполненной площади, и только связная заторная часть проспекта (хвост); отдельный
    затор выше разрыва и люди других станций не считаются."""
    sim = stadium(debug=False)
    rec = Recorder(sim, CFG)
    cells, area = sim.ctm.cells, sim.ctm.area
    k_av = sim.net.edge_index["e_av"]
    n_av = int(cells.n_per_edge[k_av])
    rq = sim.ctm.params.queue_density
    s = _state_with(sim, {("e_sq", "MA"): rq,                            # площадь заполнена
                          ("e_av", "MA", slice(n_av - 3, None)): 2.5,    # хвост очереди
                          ("e_av", "MA", slice(0, n_av - 5)): 3.0,       # отдельный затор выше
                          ("e_pj", "MB"): 3.0})                          # чужая станция
    rec.record(s)
    q = rec.result().queue_outside[0, 0]
    A = sim.stations.ids.index("A")

    def ppl(eid, dens, sl=slice(None)):
        k = sim.net.edge_index[eid]
        c = np.arange(cells.first[k], cells.last[k] + 1)[sl]
        return dens * area[c].sum()
    expected = ppl("e_sq", rq) + ppl("e_av", 2.5, slice(n_av - 3, None))
    assert q[A] == pytest.approx(expected)
    assert q[sim.stations.ids.index("B")] == 0


def test_queue_outside_stops_at_partly_filled_zone(stadium):
    """Площадь заполнена наполовину — в очередь входит только она, затор на проспекте
    не связан с дверью."""
    sim = stadium(debug=False)
    rec = Recorder(sim, CFG)
    rq = sim.ctm.params.queue_density
    rec.record(_state_with(sim, {("e_sq", "MA"): rq / 2, ("e_av", "MA"): 3.0}))
    k = sim.net.edge_index["e_sq"]
    c = sim.ctm.cells.first[k]
    assert rec.result().queue_outside[0, 0, 0] == pytest.approx(rq / 2 * sim.ctm.area[c])


def test_netplot_geometry(stadium):
    from analysis import netplot
    sim = stadium(debug=False)
    geo = netplot.NetGeometry(sim.net, sim.ctm.cells)
    assert len(geo.cell_segments) == sim.ctm.n_cells
    # встречные рёбра нарисованы раздельно
    k1, k2 = sim.net.edge_index["e_ab"], sim.net.edge_index["e_ba"]
    assert not np.allclose(geo.polylines[k1][0], geo.polylines[k2][-1])


def test_stage3_figures_registered():
    from analysis import make_figures
    stage3 = {n for n, (st, _) in make_figures.REGISTRY.items() if st == "3"}
    assert {"timeseries", "network_map", "animation", "safety"} <= stage3   # план работ, этап 3
