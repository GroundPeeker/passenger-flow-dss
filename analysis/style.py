"""Единый стиль рисунков (постановка задачи, раздел «Графики»).

- Шрифт с кириллицей: DejaVu Sans (входит в matplotlib).
- Размеры: под A4 (ширина 16 см) и под слайд 16:9.
- Цвета стратегий постоянные во всех рисунках: цвет следует за стратегией, а не за
  её местом в легенде. Категориальная палитра — проверенный порядок из 8 оттенков
  (дальтонико-безопасный для соседних серий); «без управления» — нейтральный серый
  как точка отсчёта. Девятой цветной серии нет: при необходимости — серый или фасеты.
- Сохранение: analysis/figures/stage<N>/<имя>.{png (300 dpi), pdf, csv, json}.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "analysis" / "figures"

CM = 1 / 2.54
A4_WIDTH = 16 * CM          # ширина рисунка в тексте диплома, дюймы
SLIDE = (13.33, 7.5)        # слайд 16:9, дюймы

# Категориальная палитра (порядок фиксирован, не циклить).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#898781"          # отсчётная серия (без управления, теория-ориентир)
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Постоянные цвета стратегий (постановка задачи, «Методы»).
STRATEGY_COLORS = {
    "proposed": SERIES[0],       # 6. фильтр частиц + сценарное MPC
    "least_loaded": SERIES[1],   # 3а. наименее загруженный путь — главный соперник
    "mpc_full": SERIES[2],       # 4. MPC при полном следовании
    "mpc_fixed_c": SERIES[3],    # 5. MPC с фиксированной c
    "threshold": SERIES[4],      # 3. пороговое правило
    "rl": SERIES[5],             # 8. PPO
    "oracle": SERIES[6],         # 7. оракул
    "fixed": SERIES[7],          # 2. фиксированная схема
    "no_control": NEUTRAL,       # 1. без управления
}

# Последовательная шкала (один оттенок, светлый → тёмный) для карт плотности.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])


def apply() -> None:
    """Глобальные настройки matplotlib: тонкие линии, ненавязчивая сетка."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })


def a4(height_cm: float) -> tuple[float, float]:
    """Размер рисунка под A4: ширина 16 см, высота в сантиметрах."""
    return A4_WIDTH, height_cm * CM


def _git_rev() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None  # репозиторий не под git


def save(fig, stage: str, name: str, data, meta: dict, out_dir: Path | None = None) -> list[Path]:
    """Сохраняет рисунок: .png (300 dpi), .pdf, .csv (данные), .json (конфиг, seed,
    версия кода, дата). ``data`` — pandas.DataFrame с данными рисунка."""
    d = (out_dir or FIG_DIR) / f"stage{stage}"
    d.mkdir(parents=True, exist_ok=True)
    base = d / name
    fig.savefig(base.with_suffix(".png"))
    fig.savefig(base.with_suffix(".pdf"))
    data.to_csv(base.with_suffix(".csv"), index=False)
    info = {"name": name, "stage": stage, "git": _git_rev(),
            "created": _dt.datetime.now().isoformat(timespec="seconds"), **meta}
    base.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=2,
                                                   default=str), encoding="utf-8")
    plt.close(fig)
    return [base.with_suffix(s) for s in (".png", ".pdf", ".csv", ".json")]


def save_animation(anim, stage: str, name: str, data, meta: dict, out_dir: Path | None = None,
                   fps: int = 8, dpi: int = 100) -> list[Path]:
    """Сохраняет анимацию: .gif всегда, .mp4 — если есть ffmpeg (постановка задачи, «Графики»);
    данные .csv и метаданные .json — как у рисунков."""
    import shutil
    from matplotlib.animation import FFMpegWriter, PillowWriter
    d = (out_dir or FIG_DIR) / f"stage{stage}"
    d.mkdir(parents=True, exist_ok=True)
    base = d / name
    paths = []
    if shutil.which("ffmpeg"):
        anim.save(base.with_suffix(".mp4"), writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
        paths.append(base.with_suffix(".mp4"))
    anim.save(base.with_suffix(".gif"), writer=PillowWriter(fps=fps), dpi=dpi)
    paths.append(base.with_suffix(".gif"))
    data.to_csv(base.with_suffix(".csv"), index=False)
    info = {"name": name, "stage": stage, "git": _git_rev(),
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "mp4": shutil.which("ffmpeg") is not None, **meta}
    base.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=2,
                                                   default=str), encoding="utf-8")
    plt.close(anim._fig)
    return paths + [base.with_suffix(".csv"), base.with_suffix(".json")]
