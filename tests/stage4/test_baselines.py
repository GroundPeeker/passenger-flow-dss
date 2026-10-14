"""Базовые стратегии (control/baselines.py) и вспомогательные функции эксперимента этапа 4."""
import math

import numpy as np
import pandas as pd
import pytest

from control.baselines import (FixedSchedule, Lever, LeastLoadedRoute, LeastLoadedStation,
                               NoControl, StationQueue, ThresholdRule, ThresholdSpec,
                               nominal_throughput, run_policy, schedule_active)
from sim.observation import Observation
from sim.simulator import Actions


def obs_for(sim, B, **edges):
    """Наблюдение с заданными показаниями счётчиков (остальные — 0)."""
    names = [e for e, _ in sim.observer.sensors]
    v = np.zeros((B, len(names)))
    for e, x in edges.items():
        v[:, names.index(e)] = x
    return Observation(60.0, v, v.copy())


def area(sim, edge):
    k = sim.net.edge_index[edge]
    c = sim.ctm.cells
    return sim.ctm.area[c.first[k]:c.last[k] + 1].sum()


@pytest.mark.parametrize("f", [0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
def test_schedule_fraction(f):
    """За первые n интервалов активны ⌊n·f⌋ — равномерно по времени."""
    act = [schedule_active(k, f) for k in range(150)]
    for n in (10, 60, 150):
        assert sum(act[:n]) == math.floor(n * f + 1e-9)


def test_no_control_is_default(stadium):
    sim = stadium()
    pol = NoControl()
    pol.reset(sim, 2)
    a, d = pol.act(0, None), Actions.default(sim, 2)
    for x, y in zip((a.entrance_level, a.recommend, a.extra_trains, a.switch),
                    (d.entrance_level, d.recommend, d.extra_trains, d.switch)):
        assert np.array_equal(x, y)


def test_fixed_schedule_sets_lever(stadium):
    sim = stadium()
    pol = FixedSchedule(Lever("switch", "JN"), "to_B", 0.5)
    pol.reset(sim, 1)
    acts = [pol.act(k, None).switch[0, 0] for k in range(10)]
    assert acts == [-1, 0] * 5
    pol = FixedSchedule(Lever("entrance", "MA"), 0.5, 1.0)
    pol.reset(sim, 1)
    assert pol.act(0, None).entrance_level[0, sim.stations.ent_ids.index("MA")] == 0.5


def test_least_loaded_route_picks_lower_density(stadium):
    sim = stadium()
    pol = LeastLoadedRoute("JN", {"avenue": ["e_av"], "park": ["e_pk", "e_pj"]})
    pol.reset(sim, 2)
    assert np.all(pol.act(0, None).recommend == -1)          # до первого наблюдения
    a_av, a_pk = area(sim, "e_av"), area(sim, "e_pk")
    o = obs_for(sim, 2, e_av=np.array([2.0, 0.5]) * a_av, e_pk=np.array([1.0, 1.0]) * a_pk)
    rec = pol.act(1, o).recommend[:, 0]
    assert rec.tolist() == [sim.signs.option_index("JN", "park"),
                            sim.signs.option_index("JN", "avenue")]


def test_least_loaded_station_wait_time(stadium):
    """W = (площадь + подходы, если площадь заполнена) / пропускная способность."""
    sim = stadium()
    q = {"A": StationQueue("e_sq", ("e_av", "e_pj")), "B": StationQueue("e_mb", ("e_pb",))}
    pol = LeastLoadedStation("JN", "to_B", "A", "B", q)
    pol.reset(sim, 3)
    mu_a, mu_b = nominal_throughput(sim, "A"), nominal_throughput(sim, "B")
    assert mu_a * 60 == pytest.approx(173.0, abs=0.1) and mu_b * 60 == pytest.approx(129.75)
    full_a = pol.full[0]
    assert full_a == pytest.approx(0.95 * 3.0 * area(sim, "e_sq"))
    o = obs_for(sim, 3, e_sq=np.array([1000.0, full_a, 100.0]),
                e_av=np.array([5000.0, 5000.0, 0.0]), e_mb=np.array([900.0, 900.0, 1000.0]))
    w = pol.waits(o)
    assert w[0, 0] == pytest.approx(1000 / mu_a)                 # площадь не заполнена
    assert w[1, 0] == pytest.approx((full_a + 5000) / mu_a)      # очередь вышла на проспект
    choice = pol.act(1, o).switch[:, 0]
    assert choice.tolist() == [-1, 0, -1]


def test_threshold_rule_hysteresis(stadium):
    sim = stadium()
    a_av = area(sim, "e_av")
    pol = ThresholdRule([ThresholdSpec("e_av", on=1.5, off=1.0, lever=Lever("route", "JN"),
                                       value="park")])
    pol.reset(sim, 1)
    seq = [0.5, 1.6, 1.2, 0.9, 1.2]
    got = [pol.act(k + 1, obs_for(sim, 1, e_av=r * a_av)).recommend[0, 0] for k, r in enumerate(seq)]
    park = sim.signs.option_index("JN", "park")
    assert got == [-1, park, park, -1, -1]


def test_run_policy_conservation_and_determinism(stadium):
    sim = stadium(fidelity="truth", exact=False, debug=False)
    p = sim.sample_params(np.random.default_rng(0), 2)
    q = {"A": StationQueue("e_sq", ("e_av", "e_pj")), "B": StationQueue("e_mb", ("e_pb",))}
    mk = lambda: LeastLoadedStation("JN", "to_B", "A", "B", q)
    t1, log = run_policy(sim, p, np.random.default_rng(5), 60, mk(), log_actions=True)
    t2 = run_policy(sim, p, np.random.default_rng(5), 60, mk())
    assert np.array_equal(t1.in_area, t2.in_area)
    assert len(log) == 60 and any((a.switch == 0).any() for a in log)
    lhs = t1.released[-1] + t1.in_stadium[-1]
    assert np.allclose(lhs, t1.total)
    assert np.allclose(t1.released[-1], t1.on_network[-1] + t1.in_stations[-1] + t1.left[-1])


def test_choose_lexicographic():
    from experiments.stage4_compliance import choose
    t = pd.DataFrame({"fraction": [0.0, 0.1, 0.2, 0.3],
                      "danger_exposure_ped_min": [100.0, 10.0, 10.4, 30.0],
                      "mean_time_min": [20.0, 30.0, 25.0, 10.0]})
    assert choose(t, 0.05)["fraction"] == 0.2      # 10.4 ≤ 10·1.05, время меньше
    assert choose(t, 0.0)["fraction"] == 0.1


def test_episode_params_common_across_c():
    from experiments.stage4_compliance import episode_params, load_config, make_sim
    cfg = load_config()
    sim = make_sim(cfg, "truth", 20)
    p1, p2 = episode_params(sim, [0, 1, 2], 0.2), episode_params(sim, [0, 1, 2], 1.0)
    assert np.array_equal(p1.demand.total, p2.demand.total)
    assert np.array_equal(p1.first_departure, p2.first_departure)
    assert np.all(p1.compliance.base == 0.2) and np.all(p2.compliance.base == 1.0)
    one = sim.sample_params(np.random.default_rng(1), 1)
    assert p1.demand.total[1] == one.demand.total[0]
    assert sim.stations.veh_cap[sim.stations.ids.index("A")] == 519
    assert make_sim(cfg, "truth", 30).stations.veh_cap[0] == 779
