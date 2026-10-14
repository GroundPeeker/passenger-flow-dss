import warnings

import numpy as np
import pytest

from sim.network import InfiniteBufferWarning, Network, grid_network
from tests.helpers import (bidir_network, diamond_network, fork_network, line_network,
                           same_station_fork)


def test_toy_network_discretization(root, params):
    net = Network.from_yaml(root / "networks" / "toy_fork.yaml")
    min_len = params.fd.max_wave_speed * params.dt
    cells = net.discretize(min_len)
    assert np.all(cells.length >= min_len - 1e-9)
    for k, e in enumerate(net.edges):
        sel = cells.edge == k
        assert np.isclose(cells.length[sel].sum(), e.length)
        assert np.all(cells.width[sel] == e.width)
        assert sel.sum() == int(e.length // min_len)  # максимум ячеек при условии Куранта
        assert cells.first[k] == np.flatnonzero(sel)[0]
        assert cells.last[k] == np.flatnonzero(sel)[-1]


def test_toy_network_routing(root):
    net = Network.from_yaml(root / "networks" / "toy_fork.yaml")
    assert net.destinations == ["M1", "M2", "OUT"]
    e = net.edge_index
    # с площадки PM1 к M2 и OUT — по связке e7 (для перенесённых), с PM2 к M1 — по e8;
    # штатно люди в M2 и OUT на площадку PM1 не идут: из J1 их путь — e3
    assert net.split[:, e["e7"]].tolist() == [0, 1, 1]
    assert net.split[:, e["e8"]].tolist() == [1, 0, 0]
    assert net.split[1:, e["e2"]].tolist() == [0, 0]
    # J1: в M1 — по e2, в M2 и OUT — по e3
    assert net.split[:, e["e2"]].tolist() == [1, 0, 0]
    assert net.split[:, e["e3"]].tolist() == [0, 1, 1]
    np.testing.assert_allclose(net.source_shares, [[0.6, 0.0], [0.3, 0.7], [0.1, 0.3]])


def test_shortest_path_default_and_ties():
    net = diamond_network(1.0, len_a=100, len_b=150)
    e = net.edge_index
    assert net.split[0, e["a"]] == 1.0 and net.split[0, e["b"]] == 0.0
    net = diamond_network(1.0, len_a=100, len_b=100)
    assert net.split[0, e["a"]] == 0.5 and net.split[0, e["b"]] == 0.5


def test_split_with_validation():
    net = diamond_network(1.0)
    S = net.split_with({"J": {"K": {"a": 0.2, "b": 0.8}}})
    assert S[0, net.edge_index["a"]] == pytest.approx(0.2)
    for bad in [{"J": {"K": {"a": 0.5}}},                 # сумма ≠ 1
                {"J": {"K": {"a": -0.5, "b": 1.5}}},      # отрицательная доля
                {"J": {"K": {"e3": 1.0}}},                # не исходящее ребро
                {"J": {"X": {"a": 1.0}}}]:                # неизвестное назначение
        with pytest.raises(ValueError):
            net.split_with(bad)


def test_split_to_unreachable_destination_rejected():
    net = fork_network(1.0)
    with pytest.raises(ValueError, match="нельзя дойти"):
        net.split_with({"J": {"A": {"a": 0.5, "b": 0.5}}})


def test_source_shares_validation():
    d = {
        "nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": 1.0},
                  {"id": "J", "type": "junction"},
                  {"id": "A", "type": "sink", "kind": "exit"}, {"id": "B", "type": "sink", "kind": "exit"}],
        "edges": [{"id": "e0", "from": "S", "to": "J", "length_m": 50, "width_m": 3},
                  {"id": "a", "from": "J", "to": "A", "length_m": 50, "width_m": 3},
                  {"id": "b", "from": "J", "to": "B", "length_m": 50, "width_m": 3}],
    }
    with pytest.raises(ValueError, match="destinations"):
        Network.from_dict(d)  # два назначения — доли обязательны
    d["nodes"][0]["destinations"] = {"A": 0.5, "B": 0.6}
    with pytest.raises(ValueError, match="сумме"):
        Network.from_dict(d)


def test_shared_destination_for_station_entrances():
    net = Network.from_dict({
        "nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": 1.0},
                  {"id": "J", "type": "junction"},
                  {"id": "A1", "type": "sink", "kind": "exit", "destination": "A"},
                  {"id": "A2", "type": "sink", "kind": "exit", "destination": "A"}],
        "edges": [{"id": "e0", "from": "S", "to": "J", "length_m": 50, "width_m": 3},
                  {"id": "a1", "from": "J", "to": "A1", "length_m": 50, "width_m": 3},
                  {"id": "a2", "from": "J", "to": "A2", "length_m": 80, "width_m": 3}],
    })
    assert net.destinations == ["A"]
    assert net.split[0].tolist() == [1.0, 1.0, 0.0]  # ближайший вход


def test_short_edge_rejected():
    with pytest.raises(ValueError, match="Куранта"):
        line_network(5.0, 3.0, 1.0).discretize(6.7)


def test_topology_checks():
    with pytest.raises(ValueError, match="сток"):
        Network.from_dict({
            "nodes": [{"id": "S", "type": "source"}, {"id": "K", "type": "sink", "kind": "exit"},
                      {"id": "J", "type": "junction"}],
            "edges": [{"id": "e", "from": "S", "to": "K", "length_m": 50, "width_m": 3},
                      {"id": "x", "from": "K", "to": "J", "length_m": 50, "width_m": 3}]})


def test_infinite_buffer_warns():
    with pytest.warns(InfiniteBufferWarning, match="только для тестов"):
        line_network(100.0, 3.0, 1.0, capacity=1.0, buffer=np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line_network(100.0, 3.0, 1.0, capacity=1.0, buffer=100.0)


def test_grid_network_size_and_seed():
    a = grid_network(5, 5, np.random.default_rng(1))
    b = grid_network(5, 5, np.random.default_rng(1))
    c = grid_network(5, 5, np.random.default_rng(2))
    assert len(a.edges) == 50 and a.n_dest == 5
    assert a.edges == b.edges and np.array_equal(a.split, b.split)
    assert a.edges != c.edges
    assert np.any((a.split > 0) & (a.split < 1))  # есть дробные маршруты


# --- граф маршрутов: без циклов и разворотов ---------------------------------------

def _bidir_grid(rows, cols, rng=None):
    """Двунаправленная решётка (все улицы в обе стороны), источник в углу (0, 0),
    по стоку-станции в двух других углах. Без rng — равные длины (много равных путей)."""
    def jid(i, j):
        return f"J{i}_{j}"
    edges = []

    def add(u, v, L=100.0):
        edges.append({"id": f"e{len(edges)}", "from": u, "to": v, "length_m": L, "width_m": 4})

    def street(u, v):  # два направления одной улицы — одинаковые длина и ширина
        L = 100.0 if rng is None else float(rng.uniform(50, 150))
        add(u, v, L); add(v, u, L)
    for i in range(rows):
        for j in range(cols):
            if j + 1 < cols:
                street(jid(i, j), jid(i, j + 1))
            if i + 1 < rows:
                street(jid(i, j), jid(i + 1, j))
    add("S", jid(0, 0))
    add(jid(rows - 1, cols - 1), "A")
    add(jid(0, cols - 1), "B")
    nodes = ([{"id": "S", "type": "source", "inflow_ped_per_s": 2.0,
               "destinations": {"A": 0.5, "B": 0.5}}]
             + [{"id": jid(i, j), "type": "junction"} for i in range(rows) for j in range(cols)]
             + [{"id": "A", "type": "sink", "kind": "exit"}, {"id": "B", "type": "sink", "kind": "exit"}])
    return Network.from_dict({"nodes": nodes, "edges": edges})


def _all_networks(root):
    return [Network.from_yaml(root / "networks" / "toy_fork.yaml"),
            grid_network(5, 5, np.random.default_rng(0)),
            diamond_network(1.0), fork_network(1.0), same_station_fork(1.0),
            bidir_network(), _bidir_grid(4, 5), _bidir_grid(4, 5, np.random.default_rng(1))]


def test_routing_graph_acyclic_for_every_destination(root):
    for net in _all_networks(root):
        for p in range(net.n_dest):
            assert net.routing_cycle(net.split, p) is None
        for k_in in range(len(net.edges)):       # следствие: маршрут не ведёт в разворот
            for k_out in range(len(net.edges)):
                if net.is_uturn(k_in, k_out):
                    assert not np.any((net.split[:, k_in] > 0) & (net.split[:, k_out] > 0))


def test_bidir_grid_ties_split_evenly():
    net = _bidir_grid(3, 3)
    assert np.any(net.split == 0.5)  # равные пути — поровну, и всё равно без циклов


def _edge_between(net, u, v):
    (k,) = [e.id for e in net.edges if e.source == u and e.target == v]
    return k


def test_cyclic_split_rejected():
    net = _bidir_grid(3, 3)
    ring = ["J0_0", "J0_1", "J1_1", "J1_0", "J0_0"]
    over = {}
    for u, v in zip(ring, ring[1:]):
        eid = _edge_between(net, u, v)
        over[u] = {"A": {e: float(e == eid) for e in
                         (net.edges[k].id for k in net.out_edges(u))}}
    with pytest.raises(ValueError, match="цикл"):
        net.split_with(over)
    # доля на ребро, с которого до назначения не дойти без разворота
    with pytest.raises(ValueError, match="нельзя дойти"):
        bidir_network(xj_split={"A": {"xj": 0.5, "xa": 0.5}})


def test_edge_reachability_respects_uturn_ban():
    net = bidir_network()
    ei, p = net.edge_index, net.dest_index
    assert not net.edge_reaches(ei["jx"], p["B"])   # J→X, дальше в B только через X→J
    assert net.edge_reaches(ei["s0"], p["B"])
    assert net.edge_reaches(ei["xj"], p["B"])
    assert not net.edge_reaches(ei["xj"], p["A"])   # X→J, обратно в X — разворот


def test_entrance_requires_plaza(root):
    net = Network.from_yaml(root / "networks" / "toy_fork.yaml")
    assert {n.id for n in net.nodes if n.kind == "entrance"} == {"M1", "M2"}
    d = {"nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": 1.0,
                    "destinations": {"A": 0.5, "B": 0.5}},
                   {"id": "J", "type": "junction"},
                   {"id": "A", "type": "sink", "kind": "entrance"},
                   {"id": "B", "type": "sink", "kind": "exit"}],
         "edges": [{"id": "s", "from": "S", "to": "J", "length_m": 50, "width_m": 3},
                   {"id": "a", "from": "J", "to": "A", "length_m": 50, "width_m": 3},
                   {"id": "b", "from": "J", "to": "B", "length_m": 50, "width_m": 3}]}
    Network.from_dict(d)  # J — площадка: у неё есть ветвь b к другому назначению
    d["nodes"][0]["destinations"] = {"A": 1.0}
    d["nodes"].insert(1, {"id": "P", "type": "junction"})
    d["edges"][1] = {"id": "a", "from": "P", "to": "A", "length_m": 50, "width_m": 3}
    d["edges"].append({"id": "jp", "from": "J", "to": "P", "length_m": 50, "width_m": 3})
    with pytest.raises(ValueError, match="площадка"):
        Network.from_dict(d)  # перед входом A узел P без ветви к другим назначениям
    with pytest.raises(ValueError, match="kind"):
        Network.from_dict({"nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": 1.0},
                                     {"id": "K", "type": "sink"}],
                           "edges": [{"id": "e", "from": "S", "to": "K",
                                      "length_m": 50, "width_m": 3}]})


def test_dead_end_junction_rejected():
    with pytest.raises(ValueError, match="тупик"):
        Network.from_dict({
            "nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": 1.0},
                      {"id": "J", "type": "junction"}, {"id": "D", "type": "junction"},
                      {"id": "K", "type": "sink", "kind": "exit"}],
            "edges": [{"id": "s", "from": "S", "to": "J", "length_m": 50, "width_m": 3},
                      {"id": "jd", "from": "J", "to": "D", "length_m": 50, "width_m": 3},
                      {"id": "dj", "from": "D", "to": "J", "length_m": 50, "width_m": 3},
                      {"id": "jk", "from": "J", "to": "K", "length_m": 50, "width_m": 3}]})


def test_behavior_config(root):
    from sim.behavior import BehaviorParams
    bp = BehaviorParams.from_yaml(root / "configs" / "behavior.yaml")
    assert bp.detour_rate == 0.0
    assert BehaviorParams.from_dict({"detour_rate_per_min": 6}).detour_rate == pytest.approx(0.1)
    with pytest.raises(ValueError, match="не указаны единицы"):
        BehaviorParams.from_dict({"detour_rate": 0.1})


def test_toy_network_streets(root):
    """Временное правило до этапа 10: основные улицы выхода односторонние, встречные
    рёбра двунаправленного участка — по половине ширины улицы (6 м → 3 + 3)."""
    net = Network.from_yaml(root / "networks" / "toy_fork.yaml")
    pairs = {frozenset((net.edges[a].id, net.edges[b].id)) for a, b in net.bidirectional_pairs()}
    assert pairs == {frozenset(("e7", "e8"))}
    e = net.edge_index
    assert net.edges[e["e7"]].width + net.edges[e["e8"]].width == 6.0


def test_bidirectional_pair_must_match():
    d = {"nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": 1.0},
                   {"id": "U", "type": "junction"}, {"id": "V", "type": "junction"},
                   {"id": "K", "type": "sink", "kind": "exit"}],
         "edges": [{"id": "s", "from": "S", "to": "U", "length_m": 50, "width_m": 3},
                   {"id": "uv", "from": "U", "to": "V", "length_m": 50, "width_m": 3},
                   {"id": "vu", "from": "V", "to": "U", "length_m": 50, "width_m": 6},
                   {"id": "uk", "from": "U", "to": "K", "length_m": 50, "width_m": 3},
                   {"id": "vk", "from": "V", "to": "K", "length_m": 50, "width_m": 3}]}
    with pytest.raises(ValueError, match="встречные"):
        Network.from_dict(d)
    d["edges"][2]["width_m"] = 3
    Network.from_dict(d)
