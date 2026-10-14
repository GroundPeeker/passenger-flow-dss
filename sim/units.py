"""Единицы измерения в конфигах.

Физическая величина в YAML задаётся ключом ``<имя>_<единица>``, например
``capacity_ped_per_min: 120`` или ``length_m: 200``. Загрузчик пересчитывает значение
в СИ (м, с, чел); внутри кода — только СИ. Ключ без единиц, неизвестная единица или
единица не той размерности — ошибка (лишний ключ не проходит проверку ``finish``).
"""

from __future__ import annotations

import math

# Размерность → {суффикс: множитель к СИ}.
UNITS: dict[str, dict[str, float]] = {
    "length": {"m": 1.0, "km": 1000.0},
    "time": {"s": 1.0, "min": 60.0, "h": 3600.0},
    "speed": {"m_per_s": 1.0, "km_per_h": 1.0 / 3.6},
    "rate": {"ped_per_s": 1.0, "ped_per_min": 1.0 / 60.0, "ped_per_h": 1.0 / 3600.0},
    "density": {"ped_per_m2": 1.0},
    "specific_flow": {"ped_per_m_per_s": 1.0, "ped_per_m_per_min": 1.0 / 60.0},
    "inv_area": {"per_m2": 1.0},
    "count": {"ped": 1.0},
    "inv_time": {"per_s": 1.0, "per_min": 1.0 / 60.0, "per_h": 1.0 / 3600.0},
}

_MISSING = object()


class Section:
    """Раздел конфига: читает ключи с пересчётом единиц и проверяет, что лишних нет."""

    def __init__(self, d: dict | None, where: str):
        self.d = dict(d or {})
        self.where = where
        self.used: set[str] = set()

    def get(self, key: str, default=_MISSING):
        if key in self.d:
            self.used.add(key)
            return self.d[key]
        if default is _MISSING:
            raise ValueError(f"{self.where}: нет обязательного ключа {key!r}")
        return default

    def quantity(self, name: str, kind: str, default=_MISSING):
        """Значение величины ``name`` размерности ``kind`` в СИ.
        ``null`` и ``inf``/``.inf`` допустимы (None / math.inf)."""
        if name in self.d:
            example = f"{name}_{next(iter(UNITS[kind]))}"
            raise ValueError(f"{self.where}: у {name!r} не указаны единицы, пишите например {example!r}")
        found = [(f"{name}_{u}", f) for u, f in UNITS[kind].items() if f"{name}_{u}" in self.d]
        if len(found) > 1:
            raise ValueError(f"{self.where}: {name!r} задан несколько раз: {[k for k, _ in found]}")
        if not found:
            if default is _MISSING:
                raise ValueError(f"{self.where}: нет обязательной величины {name!r} "
                                 f"(единицы: {list(UNITS[kind])})")
            return default
        key, factor = found[0]
        self.used.add(key)
        val = self.d[key]
        if val is None:
            return None
        if isinstance(val, str) and val.strip().lower() in ("inf", ".inf"):
            val = math.inf
        return float(val) * factor

    def finish(self) -> None:
        extra = sorted(set(self.d) - self.used)
        if extra:
            raise ValueError(f"{self.where}: неизвестные ключи {extra} "
                             f"(возможно, неверные единицы; допустимые — sim/units.py)")
