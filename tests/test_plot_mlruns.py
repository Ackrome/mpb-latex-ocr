from pathlib import Path

from mpb_latex_ocr.plot_mlruns import discover_runs, read_metric_file, select_run, summarize_metrics


def test_read_metric_file_and_summary(tmp_path: Path):
    metric_path = tmp_path / "val_loss"
    metric_path.write_text("1000 0.8 1\n1001 0.4 2\n", encoding="utf-8")

    points = read_metric_file("run-a", "val_loss", metric_path)
    summary = summarize_metrics({"val_loss": points})

    assert [point.value for point in points] == [0.8, 0.4]
    assert summary["val_loss"]["final_step"] == 2.0
    assert summary["val_loss"]["best_value"] == 0.4


def test_discover_runs_and_select_latest(tmp_path: Path):
    first = tmp_path / "0" / "run-first"
    second = tmp_path / "0" / "run-second"
    for run, end_time in [(first, 10), (second, 20)]:
        (run / "metrics").mkdir(parents=True)
        (run / "metrics" / "val_loss").write_text("1000 1.0 1\n", encoding="utf-8")
        (run / "meta.yaml").write_text(
            f"run_id: {run.name}\nend_time: {end_time}\n",
            encoding="utf-8",
        )

    runs = discover_runs(tmp_path)

    assert [run.name for run in runs] == ["run-second", "run-first"]
    assert select_run(runs, "latest").name == "run-second"
