"""Три постоянные команды проверки (постановка задачи, «Проверка этапов»): бенчмарк,
проверка детерминизма, построение рисунков. Здесь — быстрые проверки, что они работают."""
import json

from analysis import make_figures
from experiments import check_determinism
from sim import bench


def test_bench_config_reference_network():
    cfg = bench.load_bench()
    net = bench.reference_network(cfg)
    assert len(net.edges) == 50                      # постановка задачи: 50–80 рёбер
    assert cfg["duration"] == 2 * 3600


def test_bench_runs(capsys):
    bench.main(["--repeats", "1", "--batches", "1"])
    assert "мс на эпизод" in capsys.readouterr().out


def test_check_determinism_short():
    cfg = bench.load_bench()
    assert check_determinism.check(cfg, n_steps=60, verbose=False)


def test_make_figures_saves_all_formats(tmp_path):
    paths = make_figures.build(["bottleneck_queue"], out_dir=tmp_path)
    assert sorted(p.suffix for p in paths) == [".csv", ".json", ".pdf", ".png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    meta = json.loads((tmp_path / "stage1" / "bottleneck_queue.json").read_text(encoding="utf-8"))
    for key in ("name", "stage", "git", "created", "seed", "ctm_config"):
        assert key in meta


def test_every_stage1_figure_registered():
    stage1 = {n for n, (st, _) in make_figures.REGISTRY.items() if st == "1"}
    # план работ, этап 1: fundamental_diagram, conservation, bottleneck_queue (+ дополнительный)
    assert {"fundamental_diagram", "conservation", "bottleneck_queue"} <= stage1
    assert stage1 == {"fundamental_diagram", "conservation", "bottleneck_queue",
                      "spacetime_spillback"}
