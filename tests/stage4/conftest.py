import dataclasses
import json

import pytest
import yaml

from sim.simulator import Simulator

SCENARIO = "configs/scenario_stadium.yaml"


@pytest.fixture
def stadium(params, root):
    """Симулятор сценария «стадион» (с указателем смены станции на JN). ``exact=True`` —
    без шума выпуска; отладочный режим CTM — по фикстуре ``params``; ``calibrated`` —
    калибровка пропускной способности станций модели (по умолчанию выключена: тесты
    механизмов не зависят от её результата)."""
    def make(fidelity="model", exact=True, debug=True, changes=None, calibrated=False):
        with open(root / SCENARIO, encoding="utf-8") as f:
            cfg = json.loads(json.dumps(yaml.safe_load(f)))
        if exact:
            cfg["fidelity"][fidelity]["release_noise"] = {"model": "none"}
        if changes:
            changes(cfg)
        p = params if debug else dataclasses.replace(params, debug=False)
        return Simulator(cfg, root, ctm_params=p, fidelity=fidelity, calibrated=calibrated)
    return make
