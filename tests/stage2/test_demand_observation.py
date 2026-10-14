"""Спрос (семплирование на эпизод, выпуск по профилю) и наблюдения (шумные счётчики)."""
import numpy as np
import pytest

from sim.demand import ReleaseNoise
from sim.simulator import Simulator
from tests.stage2.conftest import SCENARIO, run


def release_series(sim, p, noise, rng, T=4 * 3600.0, dt=5.0):
    """(T/dt, B, P, Ns) выпуск по шагам и конечный пул."""
    pool = sim.demand.initial_pool(p, noise)
    factor = np.ones(pool.shape[::2])
    out = []
    for k, t in enumerate(np.arange(0, T, dt)):
        if noise.has_factor and k % int(noise.correlation_time / dt) == 0:
            factor = noise.draw_factor(rng, factor.shape)
        r = sim.demand.release(p, pool, t, t + dt, noise, rng, factor)
        pool = pool - r
        out.append(r)
    return np.array(out), pool


def test_release_sums_to_total(make_sim):
    """Детерминированный выпуск: по шагам ровно N·(F(t1) − F(t0)), в сумме — N."""
    sim = make_sim()
    d = sim.demand
    p = d.sample(np.random.default_rng(0), 5)
    out, pool = release_series(sim, p, ReleaseNoise(), np.random.default_rng(1))
    cum = out.sum(axis=(2,)).cumsum(axis=0)                       # (T, B, Ns)
    np.testing.assert_allclose(cum[11], d.released(p, 60.0), rtol=1e-10)
    np.testing.assert_allclose(out.sum(axis=(0, 2, 3)), p.total, rtol=1e-9)
    assert pool.min() >= -1e-9


def test_binomial_release_integer_and_conserves_N(make_sim):
    """Случайный выпуск (люди выходят независимо): целые числа, не больше оставшихся,
    в сумме ровно начальный пул (округлённое N); дисперсия за минуту — пуассоновская
    (≈ среднему при малой доле выходящих за шаг)."""
    sim = make_sim()
    noise = ReleaseNoise("binomial")
    p = sim.demand.fixed(400)
    out, pool = release_series(sim, p, noise, np.random.default_rng(2))
    assert np.all(out == np.round(out)) and pool.min() >= 0
    np.testing.assert_array_equal(out.sum(axis=0) + pool, sim.demand.initial_pool(p, noise))
    per_min = out[:, :, :, 0].sum(axis=2).reshape(-1, 12, 400).sum(axis=1)   # сектор S1
    m, v = per_min[10:20].mean(axis=1), per_min[10:20].var(axis=1)
    np.testing.assert_allclose(v / m, 1.0, atol=0.25)


def test_overdispersed_release(make_sim):
    """Выпуск группами: тот же средний темп, но дисперсия за минуту заметно больше
    пуассоновской; N сохраняется."""
    sim = make_sim()
    p = sim.demand.fixed(400)
    bino, _ = release_series(sim, p, ReleaseNoise("binomial"), np.random.default_rng(3))
    over, pool = release_series(sim, p, ReleaseNoise("overdispersed", 0.3, 60.0),
                                np.random.default_rng(3))
    pm = lambda x: x[:, :, :, 0].sum(axis=2).reshape(-1, 12, 400).sum(axis=1)[10:20]
    b, o = pm(bino), pm(over)
    np.testing.assert_allclose(o.mean(axis=1), b.mean(axis=1), rtol=0.05)
    assert np.all(o.var(axis=1) > 3 * b.var(axis=1))       # m²·cv² ≫ m при m ~ 10³
    np.testing.assert_array_equal(over.sum(axis=0) + pool,
                                  sim.demand.initial_pool(p, ReleaseNoise("overdispersed", 0.3, 60.0)))


def test_multiplicative_release_is_process_noise(make_sim):
    """Шум процесса модели: непрерывный (не целый) выпуск, не больше оставшихся,
    в среднем — детерминированный."""
    sim = make_sim()
    p = sim.demand.fixed(400)
    noise = ReleaseNoise("multiplicative", 0.2, 60.0)
    out, pool = release_series(sim, p, noise, np.random.default_rng(4), T=3600.0)
    det, _ = release_series(sim, p, ReleaseNoise(), np.random.default_rng(4), T=3600.0)
    assert np.any(out != np.round(out)) and pool.min() >= 0
    np.testing.assert_allclose(out[:240].sum(axis=0).mean(axis=0), det[:240].sum(axis=0)[0], rtol=0.02)


def test_fidelity_blocks(make_sim, params, root):
    """truth — дискретные поезда и случайный выпуск, model — непрерывный отток и шум процесса;
    точность обязательна."""
    truth = make_sim(fidelity="truth", exact=False)
    model = make_sim(fidelity="model", exact=False)
    assert truth.stations.vehicle_model == "discrete" and truth.release_noise.integer
    assert model.stations.vehicle_model == "continuous" and not model.release_noise.integer
    assert model.release_noise.has_factor
    with pytest.raises(TypeError):
        Simulator.from_yaml(root / SCENARIO)
    with pytest.raises(ValueError, match="fidelity"):
        Simulator.from_yaml(root / SCENARIO, fidelity="oracle")


def test_sampled_parameters(make_sim):
    sim = make_sim()
    d = sim.demand
    p = d.sample(np.random.default_rng(1), 4000)
    assert p.total.mean() == pytest.approx(20000, rel=0.02)
    assert p.total.std() / p.total.mean() == pytest.approx(0.15, rel=0.1)
    np.testing.assert_allclose(p.sector_share.sum(axis=1), 1.0)
    np.testing.assert_allclose(p.dest_share.sum(axis=1), 1.0)
    np.testing.assert_allclose(p.sector_share.mean(axis=0), d.sector_mean, atol=0.01)
    np.testing.assert_allclose(p.dest_share.mean(axis=0), d.dest_mean, atol=0.01)
    # назначения с нулевой средней долей (S2 → M1) остаются нулевыми
    zero = d.dest_mean == 0
    assert zero.any() and np.all(p.dest_share[:, zero] == 0)
    m = p.profile["mean"] / 60
    assert 8 <= m.min() and m.max() <= 16 and m.std() > 1


def test_demand_seed_determinism(make_sim):
    d = make_sim().demand
    a, b, c = (d.sample(np.random.default_rng(s), 3) for s in (7, 7, 8))
    assert a.total.tobytes() == b.total.tobytes() and a.dest_share.tobytes() == b.dest_share.tobytes()
    assert a.total.tobytes() != c.total.tobytes()


def test_time_dependent_table_profile(make_sim):
    """Профиль, заданный таблицей интенсивности: выпуск идёт только на отрезках
    с ненулевой интенсивностью, пропорционально ей."""
    def cfg(d):
        d["demand"]["exit_profile"] = {"model": "table", "breakpoints_min": [0, 10, 20, 30],
                                       "relative_rate": [1.0, 0.0, 3.0]}
    sim = make_sim(cfg)
    p = sim.demand.fixed(1, total=4000)
    out, _ = release_series(sim, p, ReleaseNoise(), np.random.default_rng(0), T=4000.0)
    per = lambda t0, t1: out[int(t0 / 5):int(t1 / 5)].sum()
    assert per(0, 600) == pytest.approx(1000)
    assert per(600, 1200) == pytest.approx(0.0, abs=1e-9)
    assert per(1200, 1800) == pytest.approx(3000)
    assert per(1800, 4000) == pytest.approx(0.0, abs=1e-9)


def test_no_uncertainty_options(make_sim):
    def cfg(d):
        d["demand"].update(sector_concentration=None, destination_concentration=None)
        d["demand"]["attendance"]["cv"] = 0.0
    d = make_sim(cfg).demand
    p = d.sample(np.random.default_rng(0), 3)
    assert np.all(p.total == 20000)
    np.testing.assert_array_equal(p.dest_share[1], d.dest_mean)


def test_sector_shares_must_match_sources(make_sim):
    def cfg(d):
        d["demand"]["sector_shares"] = {"S1": 1.0}
    with pytest.raises(ValueError, match="sector_shares"):
        make_sim(cfg)


# --- наблюдения ----------------------------------------------------------------------

def test_counts_match_state_without_noise(make_sim):
    """Без шума показания count = людей на ребре, flow = прошло через конец ребра за интервал;
    рёбра без датчиков не наблюдаются."""
    def cfg(d):
        d["observation"] = {"sensors": [{"edge": "e1", "kind": "count"},
                                        {"edge": "e2", "kind": "flow"}],
                            "noise": {"model": "none"}}
    sim = make_sim(cfg)
    _, _, hist = run(sim, sim.fixed_params(2, total=6000), 20)
    e = sim.net.edge_index
    cells = sim.ctm.cells
    for s, obs in hist:
        on_e1 = s.ctm.n[:, :, cells.first[e["e1"]]:cells.last[e["e1"]] + 1].sum(axis=(1, 2))
        np.testing.assert_allclose(obs.values[:, 0], on_e1, rtol=1e-12)
        np.testing.assert_array_equal(obs.values, obs.true)
        np.testing.assert_allclose(obs.values[:, 1], s.edge_exit[:, e["e2"]], rtol=1e-12)
        assert obs.values.shape == (2, 2)
    assert hist[-1][1].values[:, 1].sum() > 0


@pytest.mark.parametrize("model", ["poisson", "lognormal"])
def test_noise_unbiased(make_sim, model):
    def cfg(d):
        d["observation"]["noise"] = {"model": model, "cv": 0.2} if model == "lognormal" \
            else {"model": model}
    sim = make_sim(cfg)
    x = np.full((20000, 4), 50.0)
    y = sim.observer.noisy(x, np.random.default_rng(0))
    assert y.mean() == pytest.approx(50.0, rel=0.01)
    expected_sd = np.sqrt(50.0) if model == "poisson" else 0.2 * 50.0
    assert y.std() == pytest.approx(expected_sd, rel=0.05)
    assert y.min() >= 0


def test_sensor_validation(make_sim):
    def cfg(d):
        d["observation"]["sensors"].append({"edge": "zz", "kind": "count"})
    with pytest.raises(ValueError, match="неизвестное ребро"):
        make_sim(cfg)


def test_stage2_figures_registered():
    from analysis import make_figures
    stage2 = {n for n, (st, _) in make_figures.REGISTRY.items() if st == "2"}
    assert stage2 == {"compliance_split", "station_chain"}   # план работ, этап 2
