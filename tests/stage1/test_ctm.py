import dataclasses

import numpy as np
import pytest

from sim.ctm import CTM, WeidmannFD
from sim.metrics import fallback_mass, foreign_arrivals, uturns
from sim.network import InfiniteBufferWarning, Network, grid_network
from tests.helpers import (bidir_network, diamond_network, fork_network, line_network,
                           movement, same_station_fork)


def _with_fifo(params, fifo):
    return dataclasses.replace(params, fifo=fifo)


def _no_debug(params):
    """Для тестов с вручную заданным состоянием (люди без выпуска из источника) или
    с намеренно неверной матрицей долей: отладочные проверки здесь заведомо сработали бы."""
    return dataclasses.replace(params, debug=False)


def _congested_grid(params, seed=0, **kw):
    net = grid_network(5, 5, np.random.default_rng(seed), inflow=4.0, sink_capacity=1.5, **kw)
    return CTM(net, params)


# --- фундаментальная диаграмма и условие Куранта ---------------------------------

def test_weidmann_reference_values():
    fd = WeidmannFD()
    assert fd.speed(0.0) == pytest.approx(1.34)
    assert fd.speed(5.4) == pytest.approx(0.0, abs=1e-12)
    assert fd.rho_crit == pytest.approx(1.75, abs=0.01)  # Weidmann: ρc ≈ 1.75 чел/м²
    assert fd.q_max == pytest.approx(1.225, abs=0.005)   # ≈ 1.22 чел/(м·с)
    assert WeidmannFD(q_cap=1.0).q_max == 1.0


def test_flow_derivative_matches_finite_difference():
    fd = WeidmannFD()
    rho = np.linspace(0.2, 5.3, 50)
    h = 1e-6
    num = (fd.flow(rho + h) - fd.flow(rho - h)) / (2 * h)
    np.testing.assert_allclose(fd.flow_derivative(rho), num, rtol=1e-6, atol=1e-8)


def test_backward_wave_speed():
    fd = WeidmannFD()
    # наибольшая скорость обратной волны — у плотности затора: v0·γ/ρ_jam
    assert fd.backward_wave_speed == pytest.approx(1.34 * 1.913 / 5.4, rel=1e-6)
    assert fd.max_wave_speed == fd.v0
    steep = WeidmannFD(gamma=10.0)  # обратная волна быстрее v0
    assert steep.backward_wave_speed == pytest.approx(1.34 * 10 / 5.4, rel=1e-6)
    assert steep.max_wave_speed == steep.backward_wave_speed


def test_courant_uses_backward_wave(params):
    p = dataclasses.replace(params, fd=WeidmannFD(gamma=10.0))
    m = CTM(_congested_grid(params).net, p)
    assert np.all(m.cells.length >= p.fd.backward_wave_speed * p.dt - 1e-9)
    assert np.any(m.cells.length < 1.0 * p.fd.backward_wave_speed * p.dt * 1.5)
    # с крутой заторной ветвью модель остаётся в допустимых пределах
    s = m.initial_state()
    for _ in range(1500):
        s = m.step(s, sink_capacity=np.zeros((1, len(m.sink_nodes))))
        rho = m.density(s)
        assert rho.min() >= -1e-12 and rho.max() <= p.fd.rho_jam * (1 + 1e-12)
    assert rho.max() > 0.99 * p.fd.rho_jam


# --- сохранение и допустимые значения ---------------------------------------------

def _check_committed(m, s):
    """Сумма выбравших движения входа = людям в последней ячейке / очереди источника."""
    by_in = np.add.reduceat(s.committed, m.mov_in_starts, axis=-1)
    np.testing.assert_allclose(by_in[..., m.in_cell_slot], s.n[..., m.in_cell], atol=1e-8)
    np.testing.assert_allclose(by_in[..., m.source_slot], s.source_queue, atol=1e-8)
    assert s.committed.min() >= -1e-9


@pytest.mark.parametrize("detour", [0.0, 0.05])
@pytest.mark.parametrize("fifo", [False, True])
@pytest.mark.parametrize("buffer", [0.0, 50.0])
def test_conservation_by_destination(params, fifo, buffer, detour):
    m = CTM(_congested_grid(params, sink_buffer=buffer).net, _with_fifo(params, fifo))
    s = m.initial_state(batch=3)
    inflow = np.array([1.0, 3.0, 6.0])[:, None, None] * m.net.source_shares
    for _ in range(1500):
        s = m.step(s, inflow=inflow, detour_rate=np.full(3, detour))
        _check_committed(m, s)
        lhs = s.generated.sum(axis=-1) + s.transferred       # (B, P)
        rhs = s.in_system_by_dest() + s.served.sum(axis=-1)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-11, atol=1e-8)
    assert np.all(s.served.sum(axis=(1, 2)) > 0)
    # через сток уходят только люди с его назначением
    for k, v in enumerate(m.sink_nodes):
        p = m.net.dest_index[m.net.sink_destination(v)]
        assert np.all(np.delete(s.served[:, :, k], p, axis=1) == 0)


@pytest.mark.parametrize("fifo", [False, True])
def test_density_bounds_under_heavy_congestion(params, fifo):
    m = CTM(_congested_grid(params).net, _with_fifo(params, fifo))
    s = m.initial_state(batch=2)
    # второй прогон: все стоки закрыты — сеть заполняется до затора
    caps = np.stack([m.sink_capacity, np.zeros_like(m.sink_capacity)])
    rho_max = 0.0
    for _ in range(2000):
        s = m.step(s, inflow=m.batch(2 * m.source_inflow, 2), sink_capacity=caps)
        rho = m.density(s)
        assert rho.min() >= -1e-12
        assert rho.max() <= params.fd.rho_jam * (1 + 1e-12)
        assert s.n.min() >= -1e-9
        assert s.source_queue.min() >= -1e-9 and s.sink_buffer.min() >= -1e-12
        rho_max = max(rho_max, rho.max())
    assert rho_max > 0.99 * params.fd.rho_jam
    assert np.all(s.served[1] == 0)


# --- фундаментальная диаграмма на длинном ребре -------------------------------------

def _steady_interior(m, s, n_steps):
    s, _ = m.run(s, n_steps)
    W, dt = m.cells.width, m.dt
    mid = slice(m.n_cells // 4, 3 * m.n_cells // 4)
    rho = m.density(s)[0, mid]
    q = s.cell_outflow[0, :, mid].sum(0) / (W[mid] * dt)  # чел/(м·с)
    return rho, q


@pytest.mark.parametrize("q_target", [0.2, 0.5, 0.9, 1.15])
def test_free_flow_matches_fd(params, q_target):
    W = 4.0
    m = CTM(line_network(1000.0, W, inflow=q_target * W), params)
    rho, q = _steady_interior(m, m.initial_state(), 2000)
    fd = params.fd
    rho_ref = fd.density_for_flow(q_target, "free")
    np.testing.assert_allclose(q, q_target, rtol=1e-8)
    np.testing.assert_allclose(rho, rho_ref, rtol=1e-6)
    np.testing.assert_allclose(q / rho, fd.speed(rho_ref), rtol=1e-6)
    assert np.all(rho <= fd.rho_crit)


@pytest.mark.parametrize("q_target", [0.3, 0.6, 1.0])
def test_congested_flow_matches_fd(params, q_target):
    W = 4.0
    fd = params.fd
    m = CTM(line_network(300.0, W, inflow=fd.q_max * W, capacity=q_target * W), params)
    rho, q = _steady_interior(m, m.initial_state(), 4000)
    rho_ref = fd.density_for_flow(q_target, "congested")
    np.testing.assert_allclose(q, q_target, rtol=1e-6)
    np.testing.assert_allclose(rho, rho_ref, rtol=1e-6)
    np.testing.assert_allclose(q / rho, fd.speed(rho_ref), rtol=1e-6)
    assert np.all(rho >= fd.rho_crit)


def test_free_flow_travel_time(params):
    """Фронт малой плотности проходит ребро примерно за L / v0."""
    L, W = 670.0, 4.0
    m = CTM(line_network(L, W, inflow=0.05 * W), params)
    s = m.initial_state()
    for k in range(1, 400):
        s = m.step(s)
        if s.served.sum() > 1e-3 * s.generated.sum():
            break
    assert abs(k * m.dt - L / params.fd.v0) <= 2 * m.dt


# --- узлы -------------------------------------------------------------------------

def _edge_outflow(m, s, eid):
    """(B, P) людей, вышедших из ребра за последний шаг."""
    return s.cell_outflow[:, :, m.cells.last[m.net.edge_index[eid]]]


@pytest.mark.parametrize("fifo", [False, True])
def test_diverge_by_destination(params, fifo):
    m = CTM(fork_network(inflow=3.0), _with_fifo(params, fifo))
    s, _ = m.run(m.initial_state(), 400)
    fa, fb = _edge_outflow(m, s, "a")[0], _edge_outflow(m, s, "b")[0]
    # по ребру a идут только люди в A, по b — только в B, в долях источника
    assert fa[1] == 0 and fb[0] == 0
    assert fa.sum() / (fa.sum() + fb.sum()) == pytest.approx(0.3, rel=1e-9)
    assert (fa.sum() + fb.sum()) / m.dt == pytest.approx(3.0, rel=1e-9)


def _blocked_fork_state(m):
    """Последняя ячейка e0: 20 чел в A (выбрали ребро a) и 20 в B (выбрали b);
    первая ячейка ребра a забита (ρ_jam)."""
    s = m.initial_state()
    c, ei = m.cells, m.net.edge_index
    last0, first_a = c.last[ei["e0"]], c.first[ei["a"]]
    s.n[0, :, last0] = [20.0, 20.0]
    s.committed[0, 0, movement(m, "e0", "a")] = 20.0
    s.committed[0, 1, movement(m, "e0", "b")] = 20.0
    s.n[0, 0, first_a] = m.fd.rho_jam * m.area[first_a]
    return s, last0


def test_node_without_fifo_lets_other_destination_pass(params):
    m = CTM(fork_network(inflow=0.0), _no_debug(_with_fifo(params, False)))
    s, last0 = _blocked_fork_state(m)
    S = min(float(m.fd.flow(40.0 / m.area[last0])) * m.width_dt[last0],
            m.capacity_cell[last0], 40.0)
    s1 = m.step(s)
    out = s1.cell_outflow[0, :, last0]
    assert out[0] == 0.0                         # в A — некуда
    assert out[1] == pytest.approx(S * 0.5)      # в B — вся доля предложения
    assert s1.n[0, 1, last0] == pytest.approx(20.0 - S * 0.5)


def test_node_with_fifo_blocks_whole_input(params):
    m = CTM(fork_network(inflow=0.0), _no_debug(_with_fifo(params, True)))
    s, last0 = _blocked_fork_state(m)
    s1 = m.step(s)
    np.testing.assert_array_equal(s1.cell_outflow[0, :, last0], [0.0, 0.0])


def test_closed_destination_fifo_vs_non_fifo(params):
    """Сток A закрыт. Без FIFO люди в B проходят, пока очередь в A не дойдёт до развилки
    и не заполнит подход; с FIFO развилка останавливается раньше."""
    served_b = {}
    for fifo in (False, True):
        m = CTM(fork_network(inflow=2.0, shares=(0.2, 0.8), cap_a=0.0, len_a=10.0, width_a=1.0),
                _with_fifo(params, fifo))
        s, _ = m.run(m.initial_state(), 300)
        served_b[fifo] = s.served[0, 1, 1]  # назначение B, сток B
        np.testing.assert_allclose(s.generated.sum(), s.in_system()[0] + s.served.sum())
    assert served_b[False] > 1.5 * served_b[True]


@pytest.mark.parametrize("fifo", [False, True])
def test_merge_capacity_and_shares(params, fifo):
    W1, W2, W_out = 4.0, 2.0, 3.0
    net = Network.from_dict({
        "nodes": [{"id": "S1", "type": "source", "inflow_ped_per_s": 5.0},
                  {"id": "S2", "type": "source", "inflow_ped_per_s": 5.0},
                  {"id": "J", "type": "junction"},
                  {"id": "K", "type": "sink", "kind": "exit"}],
        "edges": [{"id": "i1", "from": "S1", "to": "J", "length_m": 100, "width_m": W1},
                  {"id": "i2", "from": "S2", "to": "J", "length_m": 100, "width_m": W2},
                  {"id": "o", "from": "J", "to": "K", "length_m": 100, "width_m": W_out}],
    })
    m = CTM(net, _with_fifo(params, fifo))
    s, _ = m.run(m.initial_state(), 2000)
    q = params.fd.q_max
    f1, f2 = _edge_outflow(m, s, "i1")[0, 0], _edge_outflow(m, s, "i2")[0, 0]
    # суммарный поток = пропускная способность выходного ребра
    assert (f1 + f2) / m.dt == pytest.approx(q * W_out, rel=1e-6)
    # оба входа в заторе (предложение = q_max·W) → доли пропорциональны ширине
    assert f1 / f2 == pytest.approx(W1 / W2, rel=1e-6)
    assert np.all(s.source_queue.sum(axis=1) > 0)


def test_route_split_override_batched(params):
    """Доли маршрутов внутри одного назначения, разные для каждого прогона пакета."""
    net = diamond_network(inflow=2.0)
    m = CTM(net, params)
    fr = [0.1, 0.5, 0.9]
    split = np.stack([net.split_with({"J": {"K": {"a": f, "b": 1 - f}}}) for f in fr])
    s, _ = m.run(m.initial_state(batch=3), 300, split=split)
    fa, fb = _edge_outflow(m, s, "a")[:, 0], _edge_outflow(m, s, "b")[:, 0]
    np.testing.assert_allclose(fa / (fa + fb), fr, rtol=1e-9)


# --- сток с ограниченной пропускной способностью -------------------------------------

def _sink_queue_run(params, lam, mu, buffer):
    m = CTM(line_network(200.0, 4.0, inflow=lam, capacity=mu, buffer=buffer), params)
    s, _ = m.run(m.initial_state(), 300)
    a, b = s.in_system()[0], s.served.sum()
    n = 300
    s, _ = m.run(s, n)
    T = n * m.dt
    assert (s.served.sum() - b) / T == pytest.approx(mu, rel=1e-9)
    assert (s.in_system()[0] - a) / T == pytest.approx(lam - mu, rel=1e-9)
    return m, s


def test_sink_queue_spills_back_when_overloaded(params):
    m, s = _sink_queue_run(params, lam=3.0, mu=1.0, buffer=0.0)
    rho = m.density(s)[0]
    # очередь — затор на ребре перед входом, распространяется до источника
    assert rho[-1] > params.fd.rho_crit
    assert s.source_queue.sum() > 0 and s.sink_buffer.sum() == 0


def test_sink_queue_infinite_buffer(params):
    with pytest.warns(InfiniteBufferWarning):
        m, s = _sink_queue_run(params, lam=3.0, mu=1.0, buffer=np.inf)
    assert s.sink_buffer.sum() > 0
    assert m.density(s).max() < params.fd.rho_crit  # точечная очередь не перекрывает ребро


def test_sink_finite_buffer_fills_then_spills(params):
    m, s = _sink_queue_run(params, lam=3.0, mu=1.0, buffer=200.0)
    assert s.sink_buffer.sum() == pytest.approx(200.0)
    assert m.density(s)[0, -1] > params.fd.rho_crit


def test_sink_no_queue_when_underloaded(params):
    m = CTM(line_network(200.0, 4.0, inflow=0.8, capacity=1.0), params)
    s, _ = m.run(m.initial_state(), 500)
    a = s.in_system()[0]
    s, _ = m.run(s, 500)
    assert s.in_system()[0] == pytest.approx(a, rel=1e-9)
    assert m.density(s).max() < params.fd.rho_crit


# --- детерминизм и пакетный режим ----------------------------------------------------

def test_determinism(params):
    def episode(seed):
        m = _congested_grid(params, seed=seed)
        s, hist = m.run(m.initial_state(), 500, record=True)
        return hist, s.n

    h1, n1 = episode(3)
    h2, n2 = episode(3)
    np.testing.assert_array_equal(h1, h2)
    np.testing.assert_array_equal(n1, n2)
    h3, _ = episode(4)
    assert h1.shape != h3.shape or not np.array_equal(h1, h3)


@pytest.mark.parametrize("fifo", [False, True])
def test_batch_equals_individual_runs(params, fifo):
    m = CTM(_congested_grid(params).net, _with_fifo(params, fifo))
    scale = [0.5, 1.0, 2.0]
    caps = np.array([1.0, 2.0, 0.5])[:, None] * m.sink_capacity
    inflow = np.array(scale)[:, None, None] * m.source_inflow
    sb, _ = m.run(m.initial_state(batch=3), 600, inflow=inflow, sink_capacity=caps)
    for b in range(3):
        s1, _ = m.run(m.initial_state(), 600, inflow=inflow[b:b + 1],
                      sink_capacity=caps[b:b + 1])
        np.testing.assert_allclose(sb.n[b], s1.n[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(sb.served[b], s1.served[0], rtol=1e-12, atol=1e-12)


def test_state_batch_first(params):
    m = _congested_grid(params)
    s = m.initial_state(batch=7)
    for arr in (s.n, s.committed, s.source_queue, s.sink_buffer, s.generated, s.served,
                s.transferred, s.cell_outflow):
        assert arr.shape[:2] == (7, m.n_dest)
    assert m.step(s).n.shape == (7, m.n_dest, m.n_cells)


@pytest.mark.parametrize("kw", [
    {"inflow": "unbatched_inflow"}, {"split": "unbatched_split"},
    {"sink_capacity": "unbatched_caps"}, {"detour_rate": 0.0},
])
def test_overrides_require_batch_axis(params, kw):
    m = _congested_grid(params)
    vals = {"unbatched_inflow": m.source_inflow, "unbatched_split": m.net.split,
            "unbatched_caps": m.sink_capacity}
    (key, v), = kw.items()
    with pytest.raises(ValueError, match="пакет первой осью"):
        m.step(m.initial_state(batch=2), **{key: vals.get(v, v)})


# --- ожидание у забитой ветви и обход затора (behavior) --------------------------------

def test_waiting_for_blocked_branch_same_station(params):
    """Узел без FIFO: выбравшие забитую ветвь ждут её, а не перераспределяются."""
    m = CTM(same_station_fork(inflow=0.0), _no_debug(params))
    s = m.initial_state()
    c, ei = m.cells, m.net.edge_index
    last0, first_a = c.last[ei["e0"]], c.first[ei["a"]]
    s.n[0, 0, last0] = 40.0
    s.committed[0, 0, movement(m, "e0", "a")] = 20.0
    s.committed[0, 0, movement(m, "e0", "b")] = 20.0
    s.n[0, 0, first_a] = m.fd.rho_jam * m.area[first_a]
    S = min(float(m.fd.flow(40.0 / m.area[last0])) * m.width_dt[last0], m.capacity_cell[last0])
    s1 = m.step(s)
    assert s1.committed[0, 0, movement(m, "e0", "a")] == pytest.approx(20.0)
    assert s1.committed[0, 0, movement(m, "e0", "b")] == pytest.approx(20.0 - S / 2)
    # обход затора: доля δ = 1 − exp(−θ·dt) ждущих переходит на ветвь b
    theta = 0.02
    s2 = m.step(s, detour_rate=np.array([theta]))
    d = 1 - np.exp(-theta * m.dt)
    assert s2.committed[0, 0, movement(m, "e0", "a")] == pytest.approx(20.0 * (1 - d))
    assert s2.committed[0, 0, movement(m, "e0", "b")] == pytest.approx(20.0 - S / 2 + 20.0 * d)
    np.testing.assert_allclose(s2.n, s1.n)  # обход меняет только выбор, не положение


def test_detour_default_off_and_sensitivity(params):
    """Вход A1 станции закрыт, короткое ребро a быстро забивается. Считаем, сколько людей
    выбрало каждую ветвь: прошли по ребру (в ребре + в стоке) + ждут у развилки.
    θ = 0: ровно поровну (доли 0.5/0.5 сохраняются, никто не перераспределяется);
    θ > 0: часть ждущих у a переходит на b."""
    chose = {}
    for theta in (0.0, 0.01):
        m = CTM(same_station_fork(inflow=2.0, len_a=10.0, width_a=1.0), params)
        s, _ = m.run(m.initial_state(), 200, detour_rate=np.array([theta]))
        c, ei = m.cells, m.net.edge_index
        chose[theta] = {}
        for k, e in enumerate(("a", "b")):
            on_edge = s.n[0, 0, c.first[ei[e]]:c.last[ei[e]] + 1].sum()
            reached = s.served[0, 0, k] + s.sink_buffer[0, 0, k]
            chose[theta][e] = on_edge + reached + s.committed[0, 0, movement(m, "e0", e)]
        np.testing.assert_allclose(s.generated.sum(), s.in_system()[0] + s.served.sum())
    assert chose[0.0]["a"] == pytest.approx(chose[0.0]["b"], rel=1e-9)
    assert chose[0.01]["b"] > 1.2 * chose[0.01]["a"]


def test_detour_only_to_same_destination(params):
    """Люди в A у забитой ветви a не уходят в B: другой станции обход не касается."""
    m = CTM(fork_network(inflow=0.0), _no_debug(params))
    s, _ = _blocked_fork_state(m)
    s1 = m.step(s, detour_rate=np.array([1.0]))
    assert s1.committed[0, 0, movement(m, "e0", "a")] == pytest.approx(20.0)
    assert s1.committed[0, 0, movement(m, "e0", "b")] == 0.0


# --- «чужие» прибытия и явный перенос между слоями назначений ---------------------------

@pytest.mark.parametrize("fifo", [False, True])
def test_no_foreign_arrivals(params, fifo, root):
    for net in (_congested_grid(params).net, Network.from_yaml(root / "networks" / "toy_fork.yaml"),
                bidir_network(3.0)):
        m = CTM(net, _with_fifo(params, fifo))
        s, _ = m.run(m.initial_state(batch=2), 1000, detour_rate=np.array([0.0, 0.05]))
        assert s.served.sum() > 0
        np.testing.assert_array_equal(foreign_arrivals(m, s), 0.0)


def test_foreign_arrivals_detect_bad_routing(params):
    """Матрица долей в обход проверки (A → ребро b) — метрика это ловит."""
    m = CTM(fork_network(inflow=2.0), _no_debug(params))
    bad = m.net.split.copy()
    ei = m.net.edge_index
    bad[0, ei["a"]], bad[0, ei["b"]] = 0.0, 1.0
    with pytest.raises(ValueError):
        m.net.validate_split(bad)
    s, _ = m.run(m.initial_state(), 300, split=m.batch(bad, 1))
    fa = foreign_arrivals(m, s)[0]
    assert fa[1] > 0 and fa[0] == 0


def test_transfer_destination(params):
    m = CTM(fork_network(inflow=2.0), params)
    s, _ = m.run(m.initial_state(batch=2), 100)
    before = s.n.sum(axis=(1, 2))
    s = m.transfer_destination(s, "A", "B", np.array([0.0, 1.0]), edges=["e0"])
    np.testing.assert_allclose(s.n.sum(axis=(1, 2)), before)
    np.testing.assert_allclose(s.transferred.sum(axis=1), 0.0, atol=1e-9)
    assert s.transferred[1, 1] > 0 and s.transferred[0, 1] == 0
    _check_committed(m, s)
    s, _ = m.run(s, 300)
    np.testing.assert_allclose(s.generated.sum(-1) + s.transferred,
                               s.in_system_by_dest() + s.served.sum(-1), atol=1e-8)
    np.testing.assert_array_equal(foreign_arrivals(m, s), 0.0)


def test_transfer_uturn_from_entrance(params, root):
    """Вход M1 закрыт; люди на ребре площадка → вход (e2a) переносятся в M2: вперёд пути нет,
    они разворачиваются на площадку PM1 и уходят по её ветви e7."""
    net = Network.from_yaml(root / "networks" / "toy_fork.yaml")
    m = CTM(net, params)
    caps = m.batch(m.sink_capacity, 1)
    caps[0, m.sink_nodes.index(net.node_index["M1"])] = 0.0
    s, _ = m.run(m.initial_state(), 150, sink_capacity=caps)
    ei, c = net.edge_index, m.cells
    on_entrance = s.n[0, 0, c.first[ei["e2a"]]:c.last[ei["e2a"]] + 1].sum()
    assert on_entrance > 0
    s = m.transfer_destination(s, "M1", "M2", 1.0, edges=["e2a"])
    uturned = s.uturned[0, 0, ei["e2a"]]
    assert 0 < uturned <= on_entrance + 1e-9
    assert s.transferred[0, 1] == pytest.approx(uturned)
    left = s.n[0, 0, c.first[ei["e2a"]]:c.last[ei["e2a"]] + 1].sum()
    assert left == pytest.approx(on_entrance - uturned)   # не поместившиеся остались
    assert s.n[0, 1, c.first[ei["e7"]]] >= uturned - 1e-9    # ушли на связку к PM2
    s, _ = m.run(s, 300, sink_capacity=caps)
    np.testing.assert_array_equal(foreign_arrivals(m, s), 0.0)
    assert uturns(s)[0] == pytest.approx(uturned)          # вне переноса разворотов нет


def test_transfer_forward_has_no_uturn(params, root):
    net = Network.from_yaml(root / "networks" / "toy_fork.yaml")
    m = CTM(net, params)
    s, _ = m.run(m.initial_state(), 100)
    s = m.transfer_destination(s, "M1", "M2", 1.0, edges=["e1", "e2"])  # дальше через PM1 → e7
    assert s.uturned.sum() == 0 and s.transferred[0, 1] > 0
    s, _ = m.run(s, 400)
    np.testing.assert_array_equal(foreign_arrivals(m, s), 0.0)


def _entrance_net(len_entrance):
    return Network.from_dict({
        "nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": 1.0,
                   "destinations": {"A": 0.5, "B": 0.5}},
                  {"id": "P", "type": "junction"},
                  {"id": "A", "type": "sink", "kind": "entrance", "capacity_ped_per_s": 1.0},
                  {"id": "B", "type": "sink", "kind": "exit"}],
        "edges": [{"id": "s", "from": "S", "to": "P", "length_m": 50, "width_m": 4},
                  {"id": "pa", "from": "P", "to": "A", "length_m": len_entrance, "width_m": 4},
                  {"id": "pb", "from": "P", "to": "B", "length_m": 50, "width_m": 4}]})


def test_uturn_threshold_checked_on_load(params):
    limit = params.max_uturn_cells * params.fd.max_wave_speed * params.dt   # 13.4 м
    m = CTM(_entrance_net(limit - 0.1), params)
    assert m.uturn_allowed[m.net.edge_index["pa"]]
    with pytest.raises(ValueError, match="порога мгновенного разворота"):
        CTM(_entrance_net(limit + 1.0), params)
    # порог — параметр конфига
    CTM(_entrance_net(limit + 1.0), dataclasses.replace(params, max_uturn_cells=3))


def test_uturn_refused_on_long_edge(params):
    """С ребра J→X (100 м) до B только разворотом, а ребро длиннее порога — ошибка."""
    m = CTM(bidir_network(2.0), params)
    s, _ = m.run(m.initial_state(), 50)
    with pytest.raises(ValueError, match="порога мгновенного разворота"):
        m.transfer_destination(s, "A", "B", 1.0, edges=["jx"])


def test_transfer_refused_without_any_route(params):
    """С ребра S1 → A нет пути в B ни вперёд, ни разворотом (у S1 нет других рёбер)."""
    net = Network.from_dict({
        "nodes": [{"id": "S1", "type": "source", "inflow_ped_per_s": 1.0},
                  {"id": "S2", "type": "source", "inflow_ped_per_s": 1.0},
                  {"id": "A", "type": "sink", "kind": "exit"},
                  {"id": "B", "type": "sink", "kind": "exit"}],
        "edges": [{"id": "x", "from": "S1", "to": "A", "length_m": 50, "width_m": 3},
                  {"id": "y", "from": "S2", "to": "B", "length_m": 50, "width_m": 3}]})
    m = CTM(net, params)
    s, _ = m.run(m.initial_state(), 20)
    with pytest.raises(ValueError, match="даже разворотом"):
        m.transfer_destination(s, "A", "B", 1.0, edges=["x"])


# --- запасное правило выбора (fallback) ---------------------------------------------

@pytest.mark.parametrize("fifo", [False, True])
def test_no_fallback_in_normal_scenarios(params, fifo, root):
    for net in (_congested_grid(params).net, Network.from_yaml(root / "networks" / "toy_fork.yaml"),
                bidir_network(3.0), diamond_network(2.0)):
        m = CTM(net, _with_fifo(params, fifo))
        s, _ = m.run(m.initial_state(batch=2), 800, detour_rate=np.array([0.0, 0.05]))
        np.testing.assert_array_equal(fallback_mass(m, s), 0.0)
        np.testing.assert_array_equal(s.transferred, 0.0)
        np.testing.assert_array_equal(uturns(s), 0.0)


def test_fallback_uses_habitual_shares(params):
    """Матрица долей без долей слоя K на развилке J (в обход проверки — так бывает,
    если рекомендация сменилась, пока люди шли по ребру): выбор — по привычным долям
    сети (0.5/0.5 при равных путях), люди учитываются в fallback_mass."""
    net = diamond_network(2.0)
    m = CTM(net, _no_debug(params))
    bad = m.batch(net.split, 1)
    bad[0, 0, [net.edge_index["a"], net.edge_index["b"]]] = 0.0   # нет долей на J
    s, _ = m.run(m.initial_state(), 100, split=bad)
    fa = s.cell_outflow[0, 0, m.cells.last[net.edge_index["a"]]]
    fb = s.cell_outflow[0, 0, m.cells.last[net.edge_index["b"]]]
    assert fa / (fa + fb) == pytest.approx(0.5, rel=1e-9)   # привычные доли 0.5/0.5
    assert fallback_mass(m, s)[0] > 0
    np.testing.assert_array_equal(foreign_arrivals(m, s), 0.0)


def test_fallback_redirects_to_nearest_reachable_station(params):
    """Люди в B оказались на ребре J→X (переносом в обход проверки): без разворота
    до B не дойти — при входе в последнюю ячейку назначение явно переносится
    на ближайшую достижимую станцию A."""
    net = bidir_network(0.0)
    m = CTM(net, _no_debug(params))
    s = m.initial_state()
    k = net.edge_index["jx"]
    s.n[0, net.dest_index["B"], m.cells.first[k]] = 10.0
    s, _ = m.run(s, 200)
    assert s.transferred[0, net.dest_index["A"]] == pytest.approx(10.0)
    assert fallback_mass(m, s)[0] == pytest.approx(10.0)
    assert s.served[0, net.dest_index["A"]].sum() == pytest.approx(10.0)
    np.testing.assert_array_equal(foreign_arrivals(m, s), 0.0)


# --- рекомендации применяются в момент фиксации выбора ---------------------------------

def test_split_change_applies_at_commit_time(params):
    """Смена долей не меняет выбор уже зафиксировавших (люди в последней ячейке e0),
    а новые фиксируют выбор по новым долям."""
    net = diamond_network(2.0)
    m = CTM(net, params)
    s, _ = m.run(m.initial_state(), 60)                     # доли по умолчанию 0.5/0.5
    ma, mb = movement(m, "e0", "a"), movement(m, "e0", "b")
    assert s.committed[0, 0, ma] == pytest.approx(s.committed[0, 0, mb])
    to_a = m.batch(net.split_with({"J": {"K": {"a": 1.0, "b": 0.0}}}), 1)
    before_b = s.committed[0, 0, mb]
    s1 = m.step(s, split=to_a)
    last0 = m.cells.last[net.edge_index["e0"]]
    assert before_b - s1.committed[0, 0, mb] > 0             # зафиксировавшие b уходят в b
    assert s1.cell_outflow[0, 0, last0] > 0
    s2, _ = m.run(s1, 40, split=to_a)
    assert s2.committed[0, 0, mb] == pytest.approx(0.0, abs=1e-9)   # новые — только в a


# --- развороты# --- развороты ---------------------------------------------------------------------

def test_no_uturn_movements(params):
    m = CTM(bidir_network(), params)
    E = len(m.net.edges)
    in_edge = {m.edge_in_slot[e]: e for e in range(E)}
    for k in range(m.n_mov):
        ei, eo = in_edge.get(m.mov_in[k]), m.mov_src[k]
        if ei is not None and eo < E:
            assert not m.net.is_uturn(ei, eo)


# --- отладочный режим ---------------------------------------------------------------

def test_debug_mode_on_in_tests(params):
    assert params.debug


def test_debug_validates_split_every_step(params):
    m = CTM(fork_network(inflow=2.0), params)
    bad = m.batch(m.net.split, 2)
    ei = m.net.edge_index
    bad[1, 0, ei["a"]], bad[1, 0, ei["b"]] = 0.0, 1.0      # A → по ребру b: неверно
    s = m.step(m.initial_state(batch=2), split=m.batch(m.net.split, 2))
    with pytest.raises(ValueError, match="нельзя дойти"):
        m.step(s, split=bad)
    m_fast = CTM(fork_network(inflow=2.0), _no_debug(params))
    m_fast.step(s, split=bad)                               # без debug не проверяется


def test_debug_checks_invariants(params):
    m = CTM(fork_network(inflow=2.0), params)
    s = m.initial_state()
    s.n[0, 0, 3] = 5.0                                      # люди «из ниоткуда»
    with pytest.raises(AssertionError, match="баланс|committed"):
        m.step(s)
