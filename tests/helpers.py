"""Небольшие сети для тестов CTM (величины в СИ, ключи с единицами)."""
from sim.network import Network


def line_network(length, width, inflow, capacity=None, buffer=0.0):
    """Источник → одно ребро → сток."""
    return Network.from_dict({
        "nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": inflow},
                  {"id": "K", "type": "sink", "kind": "exit", "capacity_ped_per_s": capacity,
                   "buffer_ped": buffer}],
        "edges": [{"id": "e", "from": "S", "to": "K", "length_m": length, "width_m": width}],
    })


def fork_network(inflow, shares=(0.3, 0.7), cap_a=None, cap_b=None,
                 len_a=100.0, width_a=3.0):
    """S → e0 → J → a → A,  J → b → B; назначения A и B."""
    return Network.from_dict({
        "nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": inflow,
                   "destinations": {"A": shares[0], "B": shares[1]}},
                  {"id": "J", "type": "junction"},
                  {"id": "A", "type": "sink", "kind": "exit", "capacity_ped_per_s": cap_a},
                  {"id": "B", "type": "sink", "kind": "exit", "capacity_ped_per_s": cap_b}],
        "edges": [{"id": "e0", "from": "S", "to": "J", "length_m": 100, "width_m": 5},
                  {"id": "a", "from": "J", "to": "A", "length_m": len_a, "width_m": width_a},
                  {"id": "b", "from": "J", "to": "B", "length_m": 100, "width_m": 3}],
    })


def diamond_network(inflow, len_a=100.0, len_b=100.0):
    """S → e0 → J → (a | b) → J2 → e3 → K; одно назначение K, два пути."""
    return Network.from_dict({
        "nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": inflow},
                  {"id": "J", "type": "junction"}, {"id": "J2", "type": "junction"},
                  {"id": "K", "type": "sink", "kind": "exit"}],
        "edges": [{"id": "e0", "from": "S", "to": "J", "length_m": 100, "width_m": 5},
                  {"id": "a", "from": "J", "to": "J2", "length_m": len_a, "width_m": 3},
                  {"id": "b", "from": "J", "to": "J2", "length_m": len_b, "width_m": 3},
                  {"id": "e3", "from": "J2", "to": "K", "length_m": 100, "width_m": 5}],
    })


def same_station_fork(inflow, cap_1=0.0, cap_2=None, shares=(0.5, 0.5), len_a=100.0, width_a=3.0):
    """S → e0 → J → a → A1,  J → b → A2; оба стока — входы одной станции «A»;
    на развилке доли shares по a и b."""
    return Network.from_dict({
        "nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": inflow},
                  {"id": "J", "type": "junction",
                   "split": {"A": {"a": shares[0], "b": shares[1]}}},
                  {"id": "A1", "type": "sink", "kind": "exit", "destination": "A", "capacity_ped_per_s": cap_1},
                  {"id": "A2", "type": "sink", "kind": "exit", "destination": "A", "capacity_ped_per_s": cap_2}],
        "edges": [{"id": "e0", "from": "S", "to": "J", "length_m": 100, "width_m": 5},
                  {"id": "a", "from": "J", "to": "A1", "length_m": len_a, "width_m": width_a},
                  {"id": "b", "from": "J", "to": "A2", "length_m": 100, "width_m": 3}],
    })


def bidir_network(inflow=1.0, xj_split=None):
    """Сеть с встречными рёбрами J⇄X:  S → s0 → J,  J → jx → X → xa → A,
    X → xj → J,  J → jb → B. Назначения A, B."""
    x = {"id": "X", "type": "junction"}
    if xj_split:
        x["split"] = xj_split
    return Network.from_dict({
        "nodes": [{"id": "S", "type": "source", "inflow_ped_per_s": inflow,
                   "destinations": {"A": 0.5, "B": 0.5}},
                  {"id": "J", "type": "junction"}, x,
                  {"id": "A", "type": "sink", "kind": "exit"}, {"id": "B", "type": "sink", "kind": "exit"}],
        "edges": [{"id": "s0", "from": "S", "to": "J", "length_m": 100, "width_m": 5},
                  {"id": "jx", "from": "J", "to": "X", "length_m": 100, "width_m": 3},
                  {"id": "xj", "from": "X", "to": "J", "length_m": 100, "width_m": 3},
                  {"id": "xa", "from": "X", "to": "A", "length_m": 100, "width_m": 3},
                  {"id": "jb", "from": "J", "to": "B", "length_m": 100, "width_m": 3}],
    })


def movement(m, in_edge, out_edge):
    """Индекс движения CTM «ребро in_edge → ребро out_edge»."""
    ei = m.net.edge_index
    slot = m.edge_in_slot[ei[in_edge]]
    (k,) = [k for k in range(m.n_mov) if m.mov_in[k] == slot and m.mov_src[k] == ei[out_edge]]
    return k
