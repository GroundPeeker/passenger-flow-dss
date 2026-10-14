import math

import pytest

from sim.ctm import CTMParams
from sim.network import Network
from sim.units import Section


@pytest.mark.parametrize("key,kind,val,si", [
    ("x_ped_per_min", "rate", 90, 1.5),
    ("x_ped_per_h", "rate", 7200, 2.0),
    ("x_ped_per_s", "rate", 2, 2.0),
    ("x_km", "length", 0.2, 200.0),
    ("x_min", "time", 2, 120.0),
    ("x_km_per_h", "speed", 4.824, 1.34),
    ("x_ped_per_m_per_min", "specific_flow", 60, 1.0),
    ("x_ped_per_m2", "density", 5.4, 5.4),
])
def test_conversion_to_si(key, kind, val, si):
    sec = Section({key: val}, "t")
    assert sec.quantity("x", kind) == pytest.approx(si)
    sec.finish()


def test_inf_and_null():
    assert Section({"x_ped": ".inf"}, "t").quantity("x", "count") == math.inf
    assert Section({"x_ped": float("inf")}, "t").quantity("x", "count") == math.inf
    assert Section({"x_ped": None}, "t").quantity("x", "count") is None
    assert Section({}, "t").quantity("x", "count", 0.0) == 0.0


def test_missing_units_rejected():
    with pytest.raises(ValueError, match="не указаны единицы"):
        Section({"capacity": 2}, "t").quantity("capacity", "rate")


def test_duplicate_units_rejected():
    with pytest.raises(ValueError, match="несколько раз"):
        Section({"x_ped_per_s": 1, "x_ped_per_min": 60}, "t").quantity("x", "rate")


def test_wrong_dimension_rejected():
    sec = Section({"length_s": 5}, "t")
    with pytest.raises(ValueError, match="нет обязательной"):
        sec.quantity("length", "length")
    with pytest.raises(ValueError, match="неизвестные ключи"):
        sec.finish()


def _net(sink):
    return {
        "nodes": [{"id": "S", "type": "source", "inflow_ped_per_h": 3600},
                  {"id": "K", "type": "sink", "kind": "exit", **sink}],
        "edges": [{"id": "e", "from": "S", "to": "K", "length_km": 0.1, "width_m": 4}],
    }


def test_network_loader_converts_units():
    net = Network.from_dict(_net({"capacity_ped_per_min": 120, "buffer_ped": 30}))
    assert net.nodes[0].inflow == pytest.approx(1.0)
    assert net.nodes[1].capacity == pytest.approx(2.0)
    assert net.nodes[1].buffer == 30.0
    assert net.edges[0].length == pytest.approx(100.0)
    with pytest.raises(ValueError, match="не указаны единицы"):
        Network.from_dict(_net({"capacity": 2.0}))
    with pytest.raises(ValueError, match="неизвестные ключи"):
        Network.from_dict(_net({"capacity_ped_per_m": 2.0}))


def test_ctm_params_units_equivalent(params):
    alt = CTMParams.from_dict({
        "dt_s": 5.0,
        "fundamental_diagram": {"model": "weidmann", "v0_km_per_h": 1.34 * 3.6,
                                "rho_jam_ped_per_m2": 5.4, "gamma_per_m2": 1.913,
                                "q_max_ped_per_m_per_s": None},
        "node_model": {"fifo": False}, "debug": False,
        "transfer": {"max_uturn_cells": 2},
    })
    assert alt.dt == params.dt and alt.fifo == params.fifo
    assert alt.fd.v0 == pytest.approx(params.fd.v0, rel=1e-15)
    alt2 = CTMParams.from_dict({
        "dt_s": 6, "fundamental_diagram": {"model": "weidmann", "v0_m_per_s": 1.34,
                                           "rho_jam_ped_per_m2": 5.4, "gamma_per_m2": 1.913,
                                           "q_max_ped_per_m_per_min": 60},
        "node_model": {"fifo": True}, "debug": False,
        "transfer": {"max_uturn_cells": 2}})
    assert alt2.fd.q_max == pytest.approx(1.0) and alt2.fifo


def test_ctm_params_require_all_keys():
    with pytest.raises(ValueError, match="node_model"):
        CTMParams.from_dict({"dt_s": 5, "fundamental_diagram": {
            "model": "weidmann", "v0_m_per_s": 1.34, "rho_jam_ped_per_m2": 5.4,
            "gamma_per_m2": 1.913, "q_max_ped_per_m_per_s": None}})
    with pytest.raises(ValueError, match="dt"):
        CTMParams.from_dict({"dt": 5})
