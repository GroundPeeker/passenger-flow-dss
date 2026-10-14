"""Калибровка пропускной способности станций в модели планирования, фиксированная явка,
парные разности (доработка этапа 4)."""
import numpy as np
import pandas as pd
import pytest

from sim.simulator import Simulator


def saturated_departures(sim, minutes=90):
    p = sim.fixed_params(1, c=0.0)
    s = sim.reset(p)
    rng = np.random.default_rng(0)
    for _ in range(minutes):
        s, _ = sim.step(s, None, rng)
    return s.stations.departed[0]


def test_throughput_factor_scales_continuous_outflow(stadium):
    """κ уменьшает вывоз станции в насыщении пропорционально (непрерывная модель)."""
    sim1, simk = stadium(), stadium()
    simk.stations.throughput_factor[simk.stations.ids.index("A")] = 0.9
    d1, dk = saturated_departures(sim1, 60), saturated_departures(simk, 60)
    a = sim1.stations.ids.index("A")
    assert dk[a] == pytest.approx(0.9 * d1[a], rel=0.02)       # A в насыщении весь час
    assert dk[a] < d1[a]


def test_calibration_loaded_for_model_only(root):
    m = Simulator.from_yaml(root / "configs/scenario_stadium.yaml", fidelity="model")
    assert m.calibration == "configs/calibration_stadium.yaml"
    assert np.all(m.stations.throughput_factor < 1.0)
    u = Simulator.from_yaml(root / "configs/scenario_stadium.yaml", fidelity="model",
                            calibrated=False)
    assert u.calibration is None and np.all(u.stations.throughput_factor == 1.0)
    t = Simulator.from_yaml(root / "configs/scenario_stadium.yaml", fidelity="truth")
    assert t.calibration is None and np.all(t.stations.throughput_factor == 1.0)


def test_throughput_factor_rejected_for_discrete(stadium):
    def ch(cfg):
        cfg["fidelity"]["truth"]["calibration"] = "configs/calibration_stadium.yaml"
    with pytest.raises(ValueError, match="только для vehicle_model: continuous"):
        stadium(fidelity="truth", changes=ch, calibrated=True)


def test_stage4_planning_stays_uncalibrated():
    from experiments.stage4_compliance import load_config, make_sim
    cfg = load_config()
    assert cfg["planning"]["calibrated"] is False
    assert np.all(make_sim(cfg, "model", 20).stations.throughput_factor == 1.0)


def test_fixed_attendance_overrides_only_total():
    from experiments.stage4_compliance import episode_params, load_config, make_sim
    cfg = load_config()
    sim = make_sim(cfg, "truth", 20)
    p = episode_params(sim, [0, 1], 1.0)
    q = episode_params(sim, [0, 1], 1.0, attendance=35000)
    assert np.all(q.demand.total == 35000)
    assert np.array_equal(p.demand.sector_share, q.demand.sector_share)
    assert np.array_equal(p.first_departure, q.first_departure)
    s = sim.reset(q)
    assert sim.balance(s)["in_stadium"] == pytest.approx([35000, 35000], abs=10)


def test_paired_differences():
    from experiments.stage4_compliance import paired
    rows = []
    for seed, base in enumerate([10.0, 20.0, 30.0]):
        for name, delta in (("no_control", 0.0), ("fixed_c1", -1.0)):
            rows.append({"strategy": name, "lever": "switch", "c": 1.0, "seed": seed, "k": 0,
                         "mean_time_min": base + delta, "danger_exposure_ped_min": base,
                         "clearance_time_min": np.inf if name == "fixed_c1" and seed == 0 else 100.0,
                         "max_density": 1.0, "p95_time_min": base})
    pr = paired(pd.DataFrame(rows), ["k"], 150.0)
    r = pr.iloc[0]
    assert r.mean_time_min == pytest.approx(-1.0)
    assert r.mean_time_min_ci_lo == pytest.approx(-1.0)      # одинаковая разность — ДИ нулевой
    assert r.mean_time_min_n_better == 3 and r.danger_exposure_ped_min == 0.0
    assert r.clearance_time_min == pytest.approx(50.0 / 3)  # inf → 150 мин
