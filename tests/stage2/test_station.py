"""Станции: цепочка звеньев, отток в транспорт, дополнительные поезда, распространение затора."""
import numpy as np
import pytest

from sim.simulator import Actions
from tests.stage2.conftest import run


def slow_trains(headway_s=900):
    """Редкие поезда на обеих станциях: вывоз (520 мест / 15 мин ≈ 35 чел/мин) много меньше
    пропускной способности турникетов (200 чел/мин)."""
    def cfg(d):
        for st in d["stations"]:
            st["vehicle"].update(headway_levels_s=[headway_s], max_level_changes=0)
    return cfg


def per_min(sim, rate_per_s):
    return rate_per_s * sim.control_step


def test_spillback_order(make_sim):
    """Проверочный сценарий: спрос растянут (среднее время выхода 20 мин, форма 2,
    N = 10 000 → пик подхода к A ≈ 150 чел/мин ниже пропускной способности турникетов
    200 чел/мин), поезда редкие. Тогда узкое место — только вывоз, и звенья заполняются
    от платформы назад: платформа → эскалатор → турникеты; после заполнения цепочки
    приём дверей падает до темпа вывоза — затор выходит на улицу.
    (В пик типичного сценария первыми заполняются турникеты — см.
    test_typical_scenario_bottleneck_shifts.)"""
    sim = make_sim(slow_trains())
    params = sim.fixed_params(1, c=0.0, total=10000, profile={"mean": 1200.0, "shape": 2.0})
    _, _, hist = run(sim, params, 60)
    st = sim.stations
    a = st.ids.index("A")
    links = list(range(st.first[a], st.last[a] + 1))
    full_at = {}
    for k, (s, _) in enumerate(hist):
        for L in links:
            # «заполнено»: при заполненном звене за шаг dt освобождается и сразу занимается
            # место, равное оттоку, поэтому после шага x = B − отток
            if L not in full_at and s.stations.x[0, L] >= 0.95 * st.storage[L]:
                full_at[L] = k
    order = [full_at[L] for L in reversed(links)]
    assert order == sorted(order) and len(set(order)) == len(order), full_at
    assert np.all(hist[-1][0].stations.x[0] <= st.storage + 1e-9)

    admitted = np.diff([0.0] + [s.stations.admitted[0, a] for s, _ in hist])   # чел/мин
    turnstiles = per_min(sim, st.link_cap[st.first[a]])
    train = per_min(sim, st.veh_cap[a] / st.headways[a, 0])
    k_plat, k_first = full_at[st.last[a]], full_at[st.first[a]]
    e2a = sim.ctm.cells.edge == sim.net.edge_index["e2a"]
    rho_door = np.array([s.ctm.n[0, :, e2a].sum() for s, _ in hist]) / sim.ctm.area[e2a].sum()
    rho_c = sim.ctm.fd.rho_crit
    # пока цепочка не заполнена, станция не ограничивает: приём ниже турникетов,
    # у двери свободное движение (ρ < ρc)
    assert np.all(admitted[:k_plat] < turnstiles) and rho_door[:k_plat].max() < rho_c
    # цепочка заполнена — узкое место вывоз, у двери затор (ρ > ρc)
    assert admitted[k_first + 2:k_first + 8] == pytest.approx(train, rel=1e-9)
    assert rho_door[k_first + 8] > rho_c


def test_continuous_outflow_rate(make_sim):
    """Отток с платформы при очереди на ней = вместимость / интервал."""
    sim = make_sim(slow_trains(900))
    _, _, hist = run(sim, sim.fixed_params(1, c=0.0, total=20000), 60)
    a = sim.stations.ids.index("A")
    dep = np.array([s.stations.departed[0, a] for s, _ in hist])
    rate = np.diff(dep)[40:]                       # платформа уже заполнена
    np.testing.assert_allclose(rate, per_min(sim, sim.stations.veh_cap[a] / 900), rtol=1e-12)


def test_discrete_trains(make_sim):
    """Дискретные поезда: отъезд только в интервалы, где по расписанию есть поезд,
    не больше вместимости; при заполненной платформе поезд уходит полным."""
    def cfg(d):
        slow_trains(900)(d)
        d["fidelity"]["model"]["vehicle_model"] = "discrete"
    sim = make_sim(cfg)
    params = sim.fixed_params(1, c=0.0, total=20000)
    params.first_departure[:] = 30.0
    _, _, hist = run(sim, params, 60)
    a = sim.stations.ids.index("A")
    cap = sim.stations.veh_cap[a]
    dep = np.diff([0.0] + [s.stations.departed[0, a] for s, _ in hist])
    trains = 30.0 + 900.0 * np.arange(10)
    has_train = np.array([np.any((trains >= 60 * k) & (trains < 60 * (k + 1))) for k in range(60)])
    assert np.all(dep[~has_train] == 0)
    assert np.all(dep <= cap + 1e-9)
    assert np.sum(dep == cap) >= 2                 # при заполненной платформе поезд уходит полным


def test_extra_trains_delay_and_budget(make_sim):
    sim = make_sim()
    st = sim.stations
    s = st.initial_state(1)
    s = st.request_extra(s, 0.0, np.array([[True, False]]))
    s = st.request_extra(s, 60.0, np.array([[True, False]]))
    s = st.request_extra(s, 120.0, np.array([[True, False]]))     # сверх ресурса (2)
    h = lambda t: st.headway(s, t)[0]
    np.testing.assert_array_equal(h(599.0), [240, 240])
    np.testing.assert_array_equal(h(600.0), [180, 240])          # ровно через задержку 10 мин
    np.testing.assert_array_equal(h(660.0), [120, 240])
    np.testing.assert_array_equal(s.requests[0], [2, 0])
    np.testing.assert_array_equal(s.rejected[0], [1, 0])


def test_extra_trains_increase_departures(make_sim):
    """Запрос дополнительных поездов увеличивает вывоз после задержки ввода и не раньше."""
    sim = make_sim()
    params = sim.fixed_params(1, c=0.0)
    a = sim.stations.ids.index("A")
    base = run(sim, params, 50)[2]
    req = run(sim, params, 50, actions=lambda k: Actions.from_dict(
        sim, 1, extra_trains=["A"] if k == 0 else []))[2]
    dep = lambda h: np.array([s.stations.departed[0, a] for s, _ in h])
    d0, d1 = dep(base), dep(req)
    np.testing.assert_array_equal(d0[:10], d1[:10])              # до ввода (10 мин) — одинаково
    assert d1[-1] > d0[-1] + 100


def test_typical_scenario_bottleneck_shifts(make_sim):
    """Калибровка (решение студента, DECISIONS.md, этап 2а): в типичном сценарии
    (средние параметры спроса) узкое место станции A меняется со временем:
    сначала турникеты (приём = 200 чел/мин, платформа не заполнена), затем вывоз
    поездами (вся цепочка заполнена, приём = 130 чел/мин); после ввода дополнительных
    поездов узким местом снова становятся турникеты."""
    sim = make_sim()
    st = sim.stations
    a = st.ids.index("A")
    turnstiles = per_min(sim, st.link_cap[st.first[a]])
    train = per_min(sim, st.veh_cap[a] / st.headways[a, 0])
    assert train < turnstiles
    params = sim.fixed_params(1, c=0.0)

    def phases(hist):
        adm = np.diff([0.0] + [s.stations.admitted[0, a] for s, _ in hist])
        plat_full = np.array([s.stations.x[0, st.last[a]] >= 0.95 * st.storage[st.last[a]]
                              for s, _ in hist])
        by_turnstiles = np.isclose(adm, turnstiles) & ~plat_full
        by_trains = np.isclose(adm, train) & plat_full
        return by_turnstiles, by_trains

    _, _, hist = run(sim, params, 70)
    t_turn, t_train = phases(hist)
    assert t_turn.sum() >= 10 and t_train.sum() >= 10
    assert np.flatnonzero(t_turn).max() < np.flatnonzero(t_train).min()   # сначала турникеты

    req = lambda k: Actions.from_dict(sim, 1, extra_trains=["A"] if k in (25, 26) else [])
    _, _, hist2 = run(sim, params, 70, actions=req)
    t_turn2, t_train2 = phases(hist2)
    later = slice(40, 70)                       # после ввода (через 10 мин) — снова турникеты
    assert t_turn2[later].sum() >= 3 and t_train2[later].sum() == 0


def test_station_config_validation(make_sim):
    def headway_below_min(d):
        d["stations"][0]["vehicle"]["headway_levels_s"] = [240, 60]
    with pytest.raises(ValueError, match="минимального"):
        make_sim(headway_below_min)

    def too_many_changes(d):
        d["stations"][0]["vehicle"]["max_level_changes"] = 3
    with pytest.raises(ValueError, match="max_level_changes"):
        make_sim(too_many_changes)

    def two_entrances(d):
        d["stations"][0]["entrances"] = ["M1", "M2"]
    with pytest.raises(ValueError, match="ровно один вход"):
        make_sim(two_entrances)

    def no_station(d):
        d["stations"] = d["stations"][:1]
    with pytest.raises(ValueError, match="принадлежать станции"):
        make_sim(no_station)
