"""Управление входами (закрыть / ограничить), перенаправление от закрытого входа,
сохранение людей со всеми механизмами, интерфейс действий, детерминизм."""
import numpy as np
import pytest

from sim.metrics import foreign_arrivals, uturns
from sim.simulator import Actions, Episode
from tests.stage2.conftest import run

N = 8000.0


def test_closed_entrance_passes_nobody(make_sim):
    """Закрытый на весь эпизод вход не пропускает ни одного человека; люди станции A
    уходят на станцию B (альтернатива), а не исчезают; разворот — только на ребре входа."""
    sim = make_sim()
    params = sim.fixed_params(1, c=0.0, total=N)
    act = Actions.from_dict(sim, 1, entrances={"M1": 0.0})
    s, _, _ = run(sim, params, 200, actions=lambda k: act)
    a, b = sim.stations.ids.index("A"), sim.stations.ids.index("B")
    k_m1 = sim.ctm.sink_nodes.index(sim.net.node_index["M1"])
    assert s.ctm.served[0, :, k_m1].sum() == 0.0
    assert s.stations.admitted[0, a] == 0.0 and s.stations.departed[0, a] == 0.0
    p = sim.net.dest_index
    bal = sim.balance(s)
    assert abs(bal["residual"][0]) < 1e-8 * N
    # всё, что шло в метро, уехало со станции B
    metro = (s.ctm.generated[0, p["M1"]].sum() + s.ctm.generated[0, p["M2"]].sum())
    assert s.stations.departed[0, b] == pytest.approx(metro, rel=1e-6)
    assert s.ctm.transferred[0, p["M1"]] == pytest.approx(-s.ctm.generated[0, p["M1"]].sum(), rel=1e-6)
    assert foreign_arrivals(sim.ctm, s.ctm).max() == 0.0
    turned = s.ctm.uturned[0].sum(axis=0)
    e2a = sim.net.edge_index["e2a"]
    assert turned[e2a] > 0 and np.delete(turned, e2a).max() == 0.0
    assert uturns(s.ctm)[0] == pytest.approx(turned[e2a])


def test_temporary_closure_people_go_elsewhere(make_sim):
    """Вход A закрыт на 10 мин в разгар выхода: за это время A никого не принимает,
    часть людей уходит на B, после открытия A снова принимает; баланс сохраняется."""
    sim = make_sim()
    params = sim.fixed_params(1, c=0.0, total=N)
    closed = range(8, 18)
    act = lambda k: Actions.from_dict(sim, 1, entrances={"M1": 0.0} if k in closed else {})
    s, _, hist = run(sim, params, 150, actions=act)
    base, _, _ = run(sim, params, 150)
    a, b = sim.stations.ids.index("A"), sim.stations.ids.index("B")
    adm = np.array([h.stations.admitted[0, a] for h, _ in hist])
    assert adm[closed[-1]] == adm[closed[0] - 1]              # закрыт — приёма нет
    assert adm[-1] > adm[closed[-1]]                          # после открытия — принимает
    assert s.stations.departed[0, b] > base.stations.departed[0, b] + 100
    assert s.stations.departed[0].sum() == pytest.approx(base.stations.departed[0].sum(), rel=1e-6)
    assert abs(sim.balance(s)["residual"][0]) < 1e-8 * N


def test_half_open_entrance(make_sim):
    """Уровень 0.5: приём дверей не больше половины их пропускной способности
    (320 → 160 чел/мин, ниже турникетов 200) и равен ей при очереди у двери;
    перенаправления нет."""
    sim = make_sim()
    params = sim.fixed_params(1, c=0.0)
    act = Actions.from_dict(sim, 1, entrances={"M1": 0.5})
    s, _, hist = run(sim, params, 60, actions=lambda k: act)
    a = sim.stations.ids.index("A")
    half = 0.5 * sim.stations.ent_cap[sim.stations.ent_ids.index("M1")] * sim.control_step
    rate = np.diff([0.0] + [h.stations.admitted[0, a] for h, _ in hist])
    assert rate.max() == pytest.approx(half, rel=1e-9)
    assert np.sum(np.isclose(rate, half)) >= 10
    assert np.all(s.ctm.transferred[0] == 0)


def test_conservation_all_mechanisms(make_sim):
    """Сохранение людей со всем сразу (точность truth: случайный выпуск с избыточной
    дисперсией и дискретные поезда): случайный спрос, c с различием по указателям
    и дрейфом, рекомендации, закрытие/ограничение входов, дополнительные поезда,
    обход затора, пакет из нескольких эпизодов."""
    def cfg(d):
        d["compliance"].update(node_sd_logit=0.8, drift_sd_logit_per_sqrt_h=1.0)
    sim = make_sim(cfg, fidelity="truth", exact=False)
    sim.detour_rate = 0.02
    rng = np.random.default_rng(11)
    B = 4
    params = sim.sample_params(rng, B)
    opts = rng.integers(-1, 2, (120, B, 1))
    levels = rng.choice(sim.entrance_levels, (120, B, 2))
    # закрыть обе станции направления «metro» нельзя — во втором входе тогда хотя бы 0.5
    levels[..., 1] = np.where(levels[..., 0] == 0, np.maximum(levels[..., 1], 0.5), levels[..., 1])

    def act(k):
        return Actions(levels[k], opts[k], (rng.random((B, 2)) < 0.05))
    s, _, hist = run(sim, params, 120, actions=act, seed=5)
    first = hist[0][0]
    pool0 = first.stadium.sum(axis=(1, 2)) + first.ctm.generated.sum(axis=(1, 2))
    np.testing.assert_allclose(pool0, np.round(pool0))      # целые люди
    for h, _ in hist[::10] + [hist[-1]]:
        bal = sim.balance(h)
        np.testing.assert_allclose(bal["residual"], 0.0, atol=1e-8 * pool0.max())
        np.testing.assert_allclose(bal["in_stadium"] + bal["released"], pool0, rtol=1e-12)
        assert h.stadium.min() >= 0 and h.ctm.n.min() >= -1e-9 and h.stations.x.min() >= -1e-9
        assert np.all(h.stations.x <= sim.stations.storage + 1e-9)
        assert sim.ctm.density(h.ctm).max() <= sim.ctm.fd.rho_jam * (1 + 1e-9)
    assert foreign_arrivals(sim.ctm, s.ctm).max() == 0.0


def test_no_open_alternative_people_wait(make_sim):
    """Если альтернатива закрытого входа — тоже закрытая станция, перенос не выполняется:
    люди ждут у двери; когда альтернатива открыта, ждавшие уходят на неё. Закрыть обе
    станции направления через действия нельзя, поэтому модуль осведомлённости
    проверяется напрямую (такое состояние возможно, если станции в разных направлениях)."""
    sim = make_sim()
    params = sim.fixed_params(1, c=0.0)
    closed_a = Actions.from_dict(sim, 1, entrances={"M1": 0.0})
    s, _, _ = run(sim, params, 15, actions=lambda k: closed_a)
    p = sim.net.dest_index
    e2a = sim.ctm.cells.edge == sim.net.edge_index["e2a"]
    # Накопим людей у закрытой двери A без переноса: уровни обеих станций — 0.
    both = np.zeros((1, 2))
    split = sim.signs.split(np.full((1, 1), -1), np.ones((1, 1)))
    cs = s.ctm
    sink_cap = sim.ctm.batch(sim.ctm.sink_capacity, 1)
    sink_cap[:, sim.stations.ent_sink] = 0.0
    for _ in range(60):
        before = cs.transferred.copy()
        cs = sim.closure.apply(cs, both, split)
        np.testing.assert_array_equal(cs.transferred, before)       # переноса нет
        cs = sim.ctm.step(cs, sink_capacity=sink_cap, split=split)
    waiting = cs.n[0, p["M1"], e2a].sum()
    assert waiting > 100                                           # ждут у двери A
    # Альтернатива (B) открыта — ждавшие у двери A переносятся на неё (частями: при
    # развороте входит столько, сколько помещается в первые ячейки рёбер площадки;
    # 40 шагов CTM = 200 с).
    cs2 = cs
    for _ in range(40):
        cs2 = sim.closure.apply(cs2, np.array([[0.0, 1.0]]), split)
        cs2 = sim.ctm.step(cs2, sink_capacity=sink_cap, split=split)
    moved = cs.transferred[0, p["M1"]] - cs2.transferred[0, p["M1"]]
    # ушли все ждавшие и подходящие после них; у двери снова копятся люди, потому что
    # обходное ребро e7 (3 м) пропускает меньше, чем подходит к двери A
    assert moved > waiting


def test_closing_whole_direction_forbidden(make_sim):
    sim = make_sim()
    s = sim.reset(sim.fixed_params(2))
    a = Actions.from_dict(sim, 2, entrances={"M1": 0.0})
    a.entrance_level[1, :] = 0.0                           # во втором эпизоде пакета закрыты обе
    with pytest.raises(ValueError, match="направления"):
        sim.step(s, a, np.random.default_rng(0))


def _slice_params(params, b):
    t = lambda x: x[b:b + 1]
    d, c = params.demand, params.compliance
    return type(params)(type(d)(t(d.total), t(d.sector_share), t(d.dest_share),
                                {k: t(v) for k, v in d.profile.items()}),
                        type(c)(t(c.base), t(c.offset)), t(params.first_departure))


def test_batch_equals_individual_runs(make_sim):
    """Пакетный путь включает станции, спрос, поведение и закрытие входов: прогон пакета
    из трёх эпизодов с разными параметрами и действиями совпадает с тремя одиночными."""
    def cfg(d):
        d["fidelity"]["model"]["vehicle_model"] = "discrete"
    sim = make_sim(cfg)
    B = 3
    params = sim.sample_params(np.random.default_rng(5), B)   # разные N, доли, c, фаза поездов
    acts = []
    for k in range(50):
        a = Actions.default(sim, B)
        a.recommend[:, 0] = [k % 3 - 1, 0, 1]
        a.entrance_level[1, 0] = 0.0 if 10 <= k < 20 else 1.0   # закрытие A во 2-м эпизоде
        a.entrance_level[2, 1] = 0.5
        a.extra_trains[0, 0] = k == 3
        acts.append(a)
    sb, _, _ = run(sim, params, 50, actions=lambda k: acts[k])
    for b in range(B):
        t = lambda x: x[b:b + 1]
        one = lambda k: Actions(t(acts[k].entrance_level), t(acts[k].recommend),
                                t(acts[k].extra_trains))
        s1, _, _ = run(sim, _slice_params(params, b), 50, actions=one)
        np.testing.assert_allclose(s1.ctm.n[0], sb.ctm.n[b], rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(s1.stations.x[0], sb.stations.x[b], rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(s1.stations.departed[0], sb.stations.departed[b], rtol=1e-10)
        np.testing.assert_allclose(s1.ctm.transferred[0], sb.ctm.transferred[b], atol=1e-8)
        np.testing.assert_allclose(s1.stadium[0], sb.stadium[b], rtol=1e-10, atol=1e-9)
    assert sb.ctm.transferred[1].min() < -10                  # закрытие действительно было
    assert sb.stations.requests[0, 0] == 1


def test_actions_validation(make_sim):
    sim = make_sim()
    s = sim.reset(sim.fixed_params(2))
    rng = np.random.default_rng(0)
    a = Actions.default(sim, 2)
    a.entrance_level[0, 0] = 0.3
    with pytest.raises(ValueError, match="entrance_level"):
        sim.step(s, a, rng)
    bad = Actions.default(sim, 1)                    # без пакетной оси нужного размера
    with pytest.raises(ValueError, match="форма"):
        sim.step(s, bad, rng)
    a = Actions.default(sim, 2)
    a.recommend[:] = 5
    with pytest.raises(ValueError, match="recommend"):
        sim.step(s, a, rng)
    with pytest.raises(ValueError, match="нет варианта"):
        Actions.from_dict(sim, 1, recommend={"J1": "west"})


def test_episode_determinism(make_sim):
    """Один seed — побитово одинаковые параметры, наблюдения и состояние; другой — другие."""
    def episode(seed):
        sim = make_sim(fidelity="truth", exact=False)
        rng = np.random.default_rng(seed)
        ep = Episode(sim, sim.sample_params(rng, 3), rng)
        obs = [ep.step(Actions.from_dict(sim, 3, recommend={"J1": "south"},
                                         entrances={"M1": 0.0} if 10 <= k < 15 else {})).values
               for k in range(40)]
        return np.stack(obs), ep.state
    o1, s1 = episode(42)
    o2, s2 = episode(42)
    o3, _ = episode(43)
    assert o1.tobytes() == o2.tobytes()
    assert s1.ctm.n.tobytes() == s2.ctm.n.tobytes()
    assert s1.stations.x.tobytes() == s2.stations.x.tobytes()
    assert o1.tobytes() != o3.tobytes()


def test_half_level_allowed_for_sole_station_in_direction(make_sim):
    """Ограничение 0.5 доступно любой станции, в том числе единственной в своём направлении
    (и одновременно на всех станциях направления); запрещено только закрытие (0) всех."""
    def cfg(d):
        d["stations"][0]["direction"] = "north"
        d["stations"][1]["direction"] = "south"
    sim = make_sim(cfg)
    s = sim.reset(sim.fixed_params(1))
    rng = np.random.default_rng(0)
    for levels in ({"M1": 0.5}, {"M2": 0.5}, {"M1": 0.5, "M2": 0.5}):
        sim.step(s, Actions.from_dict(sim, 1, entrances=levels), rng)   # без ошибки
    with pytest.raises(ValueError, match="'north'"):
        sim.step(s, Actions.from_dict(sim, 1, entrances={"M1": 0.0}), rng)

    same = make_sim()                                  # обе станции — направление metro
    s = same.reset(same.fixed_params(1))
    same.step(s, Actions.from_dict(same, 1, entrances={"M1": 0.5, "M2": 0.5}), rng)
    same.step(s, Actions.from_dict(same, 1, entrances={"M1": 0.0, "M2": 0.5}), rng)
