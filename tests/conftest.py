import dataclasses
from pathlib import Path

import pytest

from sim.ctm import CTMParams

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def params():
    """Параметры CTM из конфига с включённым отладочным режимом: на каждом шаге
    проверяются матрица долей и инварианты."""
    return dataclasses.replace(CTMParams.from_yaml(ROOT / "configs" / "ctm.yaml"), debug=True)


@pytest.fixture
def root():
    return ROOT
