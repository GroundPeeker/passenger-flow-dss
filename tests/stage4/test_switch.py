"""Указатель «смена станции»: доля c людей станции, проходящих развилку, меняет
назначение в момент фиксации выбора; уже выбравшие не меняются; сохранение людей."""
import numpy as np
import pytest

from sim.ctm import CTM
from sim.simulator import Actions
from tests.helpers import fork_network

N_STEPS = 150  # 2,5 ч


def run(sim, params, actions, n=N_STEPS, seed=0):
    rng = np.random.default_rng(seed)
    s = sim.reset(params)
    for k in range(n):
        s, _ = sim.step(s, actions(k), rng)
    return s


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_switch_moves_fraction_c_of_station_flow(stadium, c):
    """Рекомендация to_B весь эпизод: на B переходит ровно доля c людей станции A
    (через JN к A идут все — и северные, и южные по обходу); баланс сохраняется."""
    sim = stadium()
    p = sim.fixed_params(1, c=c)
    a = Actions.from_dict(sim, 1, switch={"JN": "to_B"})
    s = run(sim, p, lambda k: a)
    d = sim.net.dest_index
    gen_a = s.ctm.generated[0, d["MA"]].sum()
    assert s.ctm.transferred[0, d["MB"]] == pytest.approx(c * gen_a, rel=1e-9, abs=1e-6)
    assert s.ctm.transferred[0, d["MA"]] == pytest.approx(-c * gen_a, rel=1e-9, abs=1e-6)
    ia, ib = sim.stations.ids.index("A"), sim.stations.ids.index("B")
    assert s.stations.departed[0, ia] == pytest.approx((1 - c) * gen_a, rel=1e-6, abs=1e-6)
    assert s.stations.departed[0, ib] == pytest.approx(
        s.ctm.generated[0, d["MB"]].sum() + c * gen_a, rel=1e-6)
    assert abs(sim.balance(s)["residual"][0]) < 1e-8 * gen_a


def test_no_switch_without_recommendation(stadium):
    """Без рекомендации (и при switch = None) переносов нет при любой c."""
    sim = stadium()
    p = sim.fixed_params(1, c=1.0)
    a = Actions.default(sim, 1)
    a.switch = None
    s = run(sim, p, lambda k: a, n=60)
    assert np.all(s.ctm.transferred == 0.0)


def test_switch_applies_only_at_choice_moment(params):
    """CTM: за шаг меняют назначение ровно люди слоя p, вошедшие в последнюю ячейку
    входящего ребра на этом шаге; уже стоящие в ней (выбор зафиксирован) — нет."""
    net = fork_network(inflow=3.0, shares=(0.6, 0.4))
    ctm = CTM(net, params)
    rules = ctm.dest_switch_rules("J", "A", "B")
    s = ctm.initial_state(1)
    for _ in range(60):
        s = ctm.step(s)
    a, b = net.dest_index["A"], net.dest_index["B"]
    last = ctm.cells.last[net.edge_index["e0"]]
    assert s.n[0, a, last] > 0
    new = ctm.step(s, dest_switch=(rules, np.ones((1, rules.size))))
    entered = new.cell_outflow[0, a, last - 1]           # вошли в последнюю ячейку
    assert entered > 0
    assert new.transferred[0, b] == pytest.approx(entered, rel=1e-12)
    # уже стоявшие в последней ячейке слоя A остались в A (кроме ушедших через узел)
    stayed = s.n[0, a, last] - new.cell_outflow[0, a, last]
    assert new.n[0, a, last] == pytest.approx(stayed, rel=1e-12)


def test_switch_rules_validation(stadium):
    def bad(dest):
        def ch(cfg):
            cfg["station_signs"][0]["options"][0]["switch"] = {"MA": dest}
        return ch
    with pytest.raises(ValueError, match="неизвестное назначение"):
        stadium(changes=bad("MX"))
    with pytest.raises(ValueError, match="одно и то же"):
        stadium(changes=bad("MA"))

    def not_junction(cfg):
        cfg["station_signs"][0]["node"] = "SN"
    with pytest.raises(ValueError, match="развилка"):
        stadium(changes=not_junction)


def test_compliance_per_sign_includes_switch(stadium):
    """c — своя у каждого указателя: маршрута, затем смены станции."""
    sim = stadium()
    assert sim.n_compliance == sim.signs.n_signs + sim.switch_signs.n_signs == 2
    p = sim.sample_params(np.random.default_rng(1), 3)
    assert p.compliance.offset.shape == (3, 2)


def test_switch_deterministic(stadium):
    sim = stadium(fidelity="truth", exact=False, debug=False)
    p = sim.sample_params(np.random.default_rng(3), 2)
    act = lambda k: Actions.from_dict(sim, 2, switch={"JN": "to_B" if k % 3 else None})
    s1, s2 = run(sim, p, act, n=40, seed=7), run(sim, p, act, n=40, seed=7)
    assert np.array_equal(s1.ctm.n, s2.ctm.n)
    assert np.array_equal(s1.ctm.transferred, s2.ctm.transferred)
