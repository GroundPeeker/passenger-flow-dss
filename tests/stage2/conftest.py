import json

import numpy as np
import pytest
import yaml

from sim.simulator import Simulator

SCENARIO = "configs/scenario_toy.yaml"


@pytest.fixture
def make_sim(params, root):
    """Симулятор тестового сценария с отладочным режимом CTM.

    ``changes`` — функция, правящая копию конфига; ``fidelity`` — truth / model.
    ``exact=True`` (по умолчанию) — детерминированный выпуск со стадиона (без шума
    процесса), чтобы проверять доли и баланс точно; тесты случайного выпуска передают
    ``exact=False``."""
    def make(changes=None, fidelity="model", exact=True):
        with open(root / SCENARIO, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg = json.loads(json.dumps(cfg))   # копия без общих YAML-якорей между станциями
        if exact:
            cfg["fidelity"][fidelity]["release_noise"] = {"model": "none"}
        if changes:
            changes(cfg)
        return Simulator(cfg, root, ctm_params=params, fidelity=fidelity)
    return make


def run(sim, params, n_steps, actions=None, seed=0):
    """Прогон на n_steps интервалов управления; ``actions(k)`` → Actions или None.
    Возвращает конечное состояние, накопленный выход с рёбер (B, E) и историю."""
    rng = np.random.default_rng(seed)
    s = sim.reset(params)
    exit_sum = np.zeros_like(s.edge_exit)
    hist = []
    for k in range(n_steps):
        s, obs = sim.step(s, actions(k) if actions else None, rng)
        exit_sum += s.edge_exit
        hist.append((s, obs))
    return s, exit_sum, hist
