"""Поведение: привычные доли, рекомендации, степень следования c, дрейф, различие по узлам."""
import numpy as np
import pytest
import yaml

from sim.behavior import ComplianceParams, SignSpec, Signs, effective_compliance
from sim.network import Network
from sim.simulator import Actions
from tests.stage2.conftest import run

# Сеть toy_fork: через развилку J1 проходят только люди сектора S1. Привычка на J1:
# M1 → e2, M2 → e3, OUT → e3. Указатель J1: north = {M1: e2, M2: e2}, south = {M1: e3, M2: e3}.
N = 6000.0
STEPS = 150  # интервалов по 1 мин: к концу все выпущенные прошли развилку


def s1_by_dest(sim):
    p = sim.net.dest_index
    sh = sim.net.source_shares[:, 0] * sim.demand.sector_mean[0] * N
    return {d: sh[p[d]] for d in ("M1", "M2", "OUT")}


def branch_flows(sim, c, option):
    params = sim.fixed_params(1, c=c, total=N)
    act = Actions.from_dict(sim, 1, recommend={"J1": option})
    s, exit_sum, _ = run(sim, params, STEPS, actions=lambda k: act)
    assert s.ctm.in_system()[0] < 1e-6 * N        # всё прошло
    e = sim.net.edge_index
    return exit_sum[0, e["e2"]], exit_sum[0, e["e3"]]


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_split_follows_compliance_north(make_sim, c):
    """На e2 идут все M1 и доля c людей M2; при c = 1 — строго по рекомендации,
    при c = 0 — строго по привычке."""
    sim = make_sim()
    s1 = s1_by_dest(sim)
    e2, e3 = branch_flows(sim, c, "north")
    assert e2 == pytest.approx(s1["M1"] + c * s1["M2"], rel=1e-6)
    assert e3 == pytest.approx((1 - c) * s1["M2"] + s1["OUT"], rel=1e-6)


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_split_follows_compliance_south(make_sim, c):
    sim = make_sim()
    s1 = s1_by_dest(sim)
    e2, e3 = branch_flows(sim, c, "south")
    assert e2 == pytest.approx((1 - c) * s1["M1"], rel=1e-6, abs=1e-6)
    assert e3 == pytest.approx(c * s1["M1"] + s1["M2"] + s1["OUT"], rel=1e-6)


def test_no_recommendation_is_habit(make_sim):
    """Без рекомендации c не влияет: доли — привычные."""
    sim = make_sim()
    s1 = s1_by_dest(sim)
    for c in (0.0, 1.0):
        s, exit_sum, _ = run(sim, sim.fixed_params(1, c=c, total=N), STEPS)
        assert exit_sum[0, sim.net.edge_index["e2"]] == pytest.approx(s1["M1"], rel=1e-6)


def test_effective_split_rows(make_sim):
    """Эффективная матрица долей = c·рекомендация + (1 − c)·привычка в строках указателя,
    остальные строки — привычные; уходящих из района рекомендация не касается."""
    sim = make_sim()
    net = sim.net
    p, e = net.dest_index, net.edge_index
    c = np.array([[0.0], [0.3], [1.0]])
    rec = np.full((3, 1), sim.signs.option_index("J1", "south"))
    S = sim.signs.split(rec, c)
    for b, cb in enumerate(c[:, 0]):
        assert S[b, p["M1"], e["e3"]] == pytest.approx(cb)
        assert S[b, p["M1"], e["e2"]] == pytest.approx(1 - cb)
        assert S[b, p["M2"], e["e3"]] == pytest.approx(1.0)
        np.testing.assert_array_equal(S[b, p["OUT"]], net.split[p["OUT"]])
        net.validate_split(S[b])
    others = [k for k in range(len(net.edges)) if net.edges[k].source != "J1"]
    np.testing.assert_array_equal(S[:, :, others], np.broadcast_to(net.split[:, others], (3, net.n_dest, len(others))))


def test_compliance_endpoints_exact_and_offsets():
    p = ComplianceParams.constant([0.0, 0.5, 1.0], 3, 2)
    c = effective_compliance(p, np.zeros((3, 2)))
    np.testing.assert_array_equal(c, [[0, 0], [0.5, 0.5], [1, 1]])
    p.offset[:] = [[1.0, -1.0]] * 3
    c = effective_compliance(p, np.zeros((3, 2)))
    assert c[1, 0] > 0.5 > c[1, 1]                        # указатели различаются
    np.testing.assert_array_equal(c[[0, 2]], [[0, 0], [1, 1]])  # края сохраняются


def test_compliance_sampling_node_sd_and_drift(make_sim):
    def cfg(d):
        d["compliance"].update(node_sd_logit=1.0, drift_sd_logit_per_sqrt_h=1.0)
    sim = make_sim(cfg)
    rng = np.random.default_rng(3)
    params = sim.sample_params(rng, 200)
    c0 = params.compliance.base
    assert 0 <= c0.min() and c0.max() <= 1 and c0.std() > 0.2   # U[0, 1]
    assert params.compliance.offset.std() == pytest.approx(1.0, rel=0.2)
    walk = np.zeros((200, 1))
    for _ in range(60):
        walk = sim.compliance.drift(walk, rng, 60.0)
    assert walk.std() == pytest.approx(1.0, rel=0.2)       # σ·√(1 ч) после часа


def test_sign_union_cycle_rejected():
    """Вариант указателя, который вместе с привычными долями даёт цикл маршрутов, — ошибка
    (тогда смесь c·рекомендация + (1 − c)·привычка может водить людей по кругу)."""
    net = Network.from_dict({
        "nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": 1.0,
                   "destinations": {"A": 0.5, "B": 0.5}},
                  {"id": "J", "type": "junction"}, {"id": "X", "type": "junction"},
                  {"id": "Y", "type": "junction"},
                  {"id": "A", "type": "sink", "kind": "exit"},
                  {"id": "B", "type": "sink", "kind": "exit"}],
        "edges": [{"id": "s0", "from": "S", "to": "J", "length_m": 100, "width_m": 5},
                  {"id": "a", "from": "J", "to": "X", "length_m": 100, "width_m": 3},
                  {"id": "b", "from": "X", "to": "Y", "length_m": 100, "width_m": 3},
                  {"id": "c", "from": "Y", "to": "J", "length_m": 100, "width_m": 3},
                  {"id": "xa", "from": "X", "to": "A", "length_m": 100, "width_m": 3},
                  {"id": "yb", "from": "Y", "to": "B", "length_m": 100, "width_m": 3}],
    })
    # привычка: Y → c → J → a → X → A; рекомендация на X: A по b (к Y) — цикл X → Y → J → X
    spec = SignSpec("X", (("loop", {"A": "b"}),))
    with pytest.raises(ValueError, match="цикл"):
        Signs([spec], net)
    Signs([SignSpec("X", (("ok", {"A": "xa"}),))], net)   # без цикла — допустимо


def test_sign_option_must_reach(make_sim):
    def cfg(d):
        # e2a ведёт прямо во вход M1 — по нему не дойти до M2
        d["signs"].append({"node": "PM1", "options": [{"name": "z", "route": {"M2": "e2a"}}]})
    with pytest.raises(ValueError, match="не дойти"):
        make_sim(cfg)


HABIT = {"M1": {"e2": 0.8, "e3": 0.2}, "M2": {"e2": 0.3, "e3": 0.7}}


@pytest.fixture
def habit_sim(make_sim, root, tmp_path):
    """Тестовый сценарий, где привычные доли на J1 ненулевые для обеих ветвей:
    M1 — 0.8 по e2 и 0.2 по e3, M2 — 0.3 по e2 и 0.7 по e3 (переопределение в YAML сети)."""
    with open(root / "networks" / "toy_fork.yaml", encoding="utf-8") as f:
        net = yaml.safe_load(f)
    for node in net["nodes"]:
        if node["id"] == "J1":
            node["split"] = HABIT
    path = tmp_path / "toy_fork_habit.yaml"
    path.write_text(yaml.safe_dump(net, allow_unicode=True), encoding="utf-8")

    def cfg(d):
        d["network"] = str(path)
    return make_sim(cfg)


@pytest.mark.parametrize("option", ["north", "south"])
@pytest.mark.parametrize("c", [0.0, 0.3, 0.5, 1.0])
def test_split_with_nonzero_habit(habit_sim, c, option):
    """β = c·рекомендация + (1 − c)·привычка при ненулевой привычной доле обеих ветвей:
    при c = 0 — ровно привычные доли, при c = 1 — строго по рекомендации, между ними —
    линейно (доля следующих добавляется к привычной, а не заменяет её)."""
    sim = habit_sim
    np.testing.assert_allclose(sim.net.split[sim.net.dest_index["M2"], sim.net.edge_index["e2"]], 0.3)
    s1 = s1_by_dest(sim)
    rec_e2 = 1.0 if option == "north" else 0.0          # рекомендованная доля e2 для M1 и M2
    share_e2 = {d: c * rec_e2 + (1 - c) * HABIT[d]["e2"] for d in ("M1", "M2")}
    e2, e3 = branch_flows(sim, c, option)
    exp_e2 = sum(s1[d] * share_e2[d] for d in ("M1", "M2"))
    exp_e3 = sum(s1[d] * (1 - share_e2[d]) for d in ("M1", "M2")) + s1["OUT"]
    assert e2 == pytest.approx(exp_e2, rel=1e-6, abs=1e-6)
    assert e3 == pytest.approx(exp_e3, rel=1e-6)


def test_effective_split_nonzero_habit(habit_sim):
    """Строки матрицы долей на J1 при c = 0.3: смесь привычки и рекомендации."""
    sim = habit_sim
    p, e = sim.net.dest_index, sim.net.edge_index
    S = sim.signs.split(np.full((1, 1), sim.signs.option_index("J1", "north")), np.full((1, 1), 0.3))
    assert S[0, p["M2"], e["e2"]] == pytest.approx(0.3 + 0.7 * 0.3)
    assert S[0, p["M2"], e["e3"]] == pytest.approx(0.7 * 0.7)
    assert S[0, p["M1"], e["e2"]] == pytest.approx(0.3 + 0.7 * 0.8)
    sim.net.validate_split(S[0])
