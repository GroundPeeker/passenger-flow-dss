"""Построение рисунков (постановка задачи, раздел «Графики»).

Запуск:
  uv run python -m analysis.make_figures --stage 1
  uv run python -m analysis.make_figures --all
  uv run python -m analysis.make_figures --name fundamental_diagram
Результат: analysis/figures/stage<N>/<имя>.{png,pdf,csv,json}.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from matplotlib.animation import Animation

from analysis import fig_stage1, fig_stage2, fig_stage3, fig_stage4, style

# Реестр: имя рисунка → (этап, функция). Функция возвращает (fig, DataFrame, meta);
# вместо fig может быть анимация matplotlib (сохраняется как .gif / .mp4).
REGISTRY = {
    "fundamental_diagram": ("1", fig_stage1.fundamental_diagram),
    "conservation": ("1", fig_stage1.conservation),
    "bottleneck_queue": ("1", fig_stage1.bottleneck_queue),
    "spacetime_spillback": ("1", fig_stage1.spacetime_spillback),
    "compliance_split": ("2", fig_stage2.compliance_split),
    "station_chain": ("2", fig_stage2.station_chain),
    "timeseries": ("3", fig_stage3.timeseries),
    "safety": ("3", fig_stage3.safety),
    "network_map": ("3", fig_stage3.network_map),
    "density_snapshots": ("3", fig_stage3.density_snapshots),
    "animation": ("3", fig_stage3.animation),
    "fixed_tradeoff": ("4", fig_stage4.fixed_tradeoff),
    "fixed_tradeoff_edges": ("4", fig_stage4.fixed_tradeoff_edges),
    "metrics_vs_compliance": ("4", fig_stage4.metrics_vs_compliance),
    "metrics_vs_compliance_seats": ("4", fig_stage4.metrics_vs_compliance_seats),
    "max_density_vs_compliance": ("4", fig_stage4.max_density_vs_compliance),
    "exposure_thresholds": ("4", fig_stage4.exposure_thresholds),
    "route_lever_effect": ("4", fig_stage4.route_lever_effect),
    "paired_difference": ("4", fig_stage4.paired_difference),
    "paired_by_attendance": ("4", fig_stage4.paired_by_attendance),
    "model_calibration": ("4", fig_stage4.model_calibration),
    "animation_compliance": ("4", fig_stage4.animation_compliance),
    "animation_frame_20": ("4", partial(fig_stage4.animation_frame, 20)),
    "animation_frame_35": ("4", partial(fig_stage4.animation_frame, 35)),
    "animation_frame_50": ("4", partial(fig_stage4.animation_frame, 50)),
}


def build(names: list[str], out_dir: Path | None = None) -> list[Path]:
    style.apply()
    paths = []
    for name in names:
        stage, func = REGISTRY[name]
        fig, data, meta = func()
        if isinstance(fig, Animation):
            saved = style.save_animation(fig, stage, name, data, meta, out_dir=out_dir,
                                         fps=meta.get("fps", 8))
        else:
            saved = style.save(fig, stage, name, data, meta, out_dir=out_dir)
        exts = ",".join(p.suffix[1:] for p in saved)
        print(f"этап {stage}: {name} → {saved[0].parent}/{name}.{{{exts}}}")
        paths += saved
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", help="номер этапа (1, 6a, …)")
    g.add_argument("--all", action="store_true")
    g.add_argument("--name", choices=sorted(REGISTRY))
    ap.add_argument("--out", type=Path, default=None, help="каталог вместо analysis/figures")
    args = ap.parse_args(argv)
    if args.all:
        names = list(REGISTRY)
    elif args.name:
        names = [args.name]
    else:
        names = [n for n, (st, _) in REGISTRY.items() if st == args.stage]
        if not names:
            ap.error(f"для этапа {args.stage} рисунков нет; есть этапы "
                     f"{sorted({st for st, _ in REGISTRY.values()})}")
    build(names, out_dir=args.out)


if __name__ == "__main__":
    main()
