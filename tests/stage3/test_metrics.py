"""Метрики (sim/metrics.py) на траекториях с известным ответом."""

import numpy as np
import pytest

from sim.metrics import (MetricsConfig, Trajectory, clearance_time, episode_metrics,
                         overloads, travel_times)


def make_traj(t, released, left, in_stadium=None, edge_max=None, total=None):
    """Траектория с заданными накопленными кривыми (T,) → B = 1."""
    t = np.asarray(t, float)
    T = t.size
    A = np.asarray(released, float)[:, None]
    D = np.asarray(left, float)[:, None]
    N = float(A[-1, 0]) if total is None else total
    stad = (N - A) if in_stadium is None else np.asarray(in_stadium, float)[:, None]
    em = np.zeros((T, 1, 2)) if edge_max is None else np.asarray(edge_max, float)
    z = np.zeros((T, 1))
    return Trajectory(t=t, edge_max=em, edge_mean=em, exposure=np.zeros_like(em),
                      over_time=np.zeros_like(em), any_over_time=z.copy(),
                      area_above=z.copy(), area_above_time=z.copy(),
                      exposure_congestion=np.zeros_like(em), area_congestion=z.copy(),
                      zone_full_time=np.zeros((T, 1, 2)),
                      in_stadium=stad, at_gates=z,
                      on_streets=A - D, in_stations=z.copy(), released=A, left=D,
                      departed=np.zeros((T, 1, 1)), station_links=np.zeros((T, 1, 1)),
                      queue_outside=np.zeros((T, 1, 1)), total=np.array([N]))


def test_constant_delay_gives_exact_mean_and_p95():
    """Выход с постоянной интенсивностью, каждый уходит ровно через τ: среднее и p95 = τ."""
    t = np.arange(0, 3601, 60.0)
    tau = 600.0
    A = np.clip(t / 1200, 0, 1) * 1000
    D = np.clip((t - tau) / 1200, 0, 1) * 1000
    mean, qs, cens = travel_times(make_traj(t, A, D))
    assert mean[0] == pytest.approx(tau, rel=1e-9)
    assert qs[0.95][0] == pytest.approx(tau, rel=1e-6)
    assert cens[0] == 0


def test_little_mean_independent_of_quantiles():
    """Среднее по Литтлу = площадь между кривыми / число вышедших."""
    t = np.arange(0, 7201, 60.0)
    A = 1000 * (1 - np.exp(-t / 600))
    A[-1] = A[-2]
    D = np.minimum(A, np.maximum(0, t - 300) * 0.5)
    m, _, _ = travel_times(make_traj(t, A, D))
    assert m[0] == pytest.approx(np.trapezoid(A - D, t) / A[-1])


def test_censored_people_counted_until_episode_end():
    """Половина не ушла к концу: они учитываются со временем до конца (оценка снизу)."""
    t = np.array([0.0, 100.0, 200.0])
    A = np.array([100.0, 100.0, 100.0])
    D = np.array([0.0, 50.0, 50.0])
    mean, qs, cens = travel_times(make_traj(t, A, D))
    assert cens[0] == 50
    assert qs[0.95][0] == pytest.approx(200.0)   # не ушедшие — 200 с (вышли в t = 0)


def test_clearance_time_interpolates_last_crossing():
    t = np.array([0.0, 60, 120, 180, 240])
    # в районе (стадион + улицы): 1000, 800, 5 (ниже 1 %), 50 (снова выше), 0
    in_area = np.array([1000.0, 800, 5, 50, 0])
    A = np.full(5, 1000.0)
    D = A - in_area
    tr = make_traj(t, A, D, in_stadium=np.zeros(5), total=1000.0)
    ct = clearance_time(tr, 0.01)
    # последний снимок выше порога 10 — t = 180 (50 чел.), пересечение 50 → 0 на 4/5 пути
    assert ct[0] == pytest.approx(180 + 60 * 40 / 50)


def test_clearance_inf_when_not_cleared():
    t = np.array([0.0, 60, 120])
    tr = make_traj(t, [100, 100, 100], [0, 10, 20], total=100.0)
    assert np.isinf(clearance_time(tr, 0.01)[0])


def test_overload_episodes_and_durations():
    """Ребро 0 перегружено в интервалах 1–2 и 4, ребро 1 — в интервале 2: три эпизода;
    длительности и экспозиция — суммы агрегатов интервалов."""
    t = np.arange(6) * 60.0
    tr = make_traj(t, np.zeros(6), np.zeros(6), total=1.0)
    tr.over_time[[1, 2, 4], 0, 0] = [60, 30, 15]
    tr.over_time[2, 0, 1] = 20
    tr.any_over_time[[1, 2, 4], 0] = [60, 40, 15]
    tr.exposure[1, 0, 0] = 600.0
    tr.area_above[:, 0] = [0, 50, 80, 0, 10, 0]
    o = overloads(tr)
    assert o["overload_count"][0] == 3
    assert o["overload_duration"][0] == pytest.approx(125)
    assert o["overload_time"][0] == pytest.approx(115)
    assert bool(o["overload_any"][0])
    np.testing.assert_allclose(o["overload_duration_edge"][0], [105, 20])
    assert o["exposure"][0] == 600.0 and o["area_above_max"][0] == 80


def test_zone_full_time_and_first_moment():
    """Зона 0 заполнена ≥ 95 % в интервалах 2 и 3 (по 30 и 60 с): длительность 90 с, первый
    момент — начало интервала 2; зона 1 не заполнялась — inf."""
    t = np.arange(5) * 60.0
    tr = make_traj(t, np.zeros(5), np.zeros(5), total=1.0)
    tr.zone_full_time[[2, 3], 0, 0] = [30, 60]
    tr.exposure_congestion[2, 0, 1] = 120.0
    tr.area_congestion[:, 0] = [0, 5, 9, 1, 0]
    o = overloads(tr)
    assert o["zone_full_time"][0, 0] == 90 and o["zone_full_time"][0, 1] == 0
    assert o["zone_full_first"][0, 0] == 60.0 and np.isinf(o["zone_full_first"][0, 1])
    assert o["congestion_exposure"][0] == 120.0 and o["congestion_area_max"][0] == 9


def test_metrics_config_from_yaml(root):
    cfg = MetricsConfig.from_yaml(root / "configs" / "metrics.yaml")
    # затор — FHWA-RD-98-107: улицы E/F табл. 2 (0,6 м²/чел), зоны D/E табл. 21 (0,27)
    assert cfg.congestion_street == pytest.approx(1 / 0.6, abs=1e-3)
    assert cfg.congestion_zone == pytest.approx(1 / 0.27, abs=1e-3)
    from sim.ctm import CTMParams
    assert cfg.congestion_zone > CTMParams.from_yaml(root / "configs" / "ctm.yaml").queue_density
    # опасная плотность — Still (движущаяся толпа), Green Guide (стоячая); выше затора
    assert cfg.rho_crit == (4.0, 4.7)
    assert cfg.rho_crit_street > cfg.congestion_street and cfg.rho_crit_zone > cfg.congestion_zone
    assert cfg.zone_full_fraction == 0.95
    assert cfg.clearance_fraction == 0.01      # постановка задачи: < 1 % людей


def test_metrics_config_requires_danger_above_congestion():
    d = {"congestion_street_ped_per_m2": 4.0, "congestion_zone_ped_per_m2": 3.7,
         "rho_crit_street_ped_per_m2": 4.0, "rho_crit_zone_ped_per_m2": 4.7,
         "zone_full_fraction": 0.95, "clearance_fraction": 0.01, "queue_density_ped_per_m2": None}
    with pytest.raises(ValueError, match="затора"):
        MetricsConfig.from_dict(d)


def test_metrics_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        MetricsConfig.from_dict({"congestion_street_ped_per_m2": 1.7,
                                 "congestion_zone_ped_per_m2": 3.7,
                                 "rho_crit_street_ped_per_m2": 4.0,
                                 "rho_crit_zone_ped_per_m2": 4.7, "zone_full_fraction": 0.95,
                                 "clearance_fraction": 0.01,
                                 "queue_density_ped_per_m2": None, "extra": 1})


def test_episode_metrics_keys():
    t = np.arange(0, 601, 60.0)
    tr = make_traj(t, np.minimum(t, 300), np.minimum(np.maximum(t - 60, 0), 300))
    m = episode_metrics(tr, MetricsConfig(1.7, 3.7, 4.0, 4.7, 0.95, 0.01, None))
    for k in ("clearance_time", "mean_time", "p95_time", "max_density", "overload_count",
              "overload_duration", "overload_any"):
        assert m[k].shape == (1,)
