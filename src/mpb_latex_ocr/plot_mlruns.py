"""Plot training curves from local MLflow file-store metrics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MetricPoint:
    run_id: str
    metric: str
    timestamp: int
    value: float
    step: int


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot MLflow metric files without starting MLflow UI."
    )
    parser.add_argument("--mlruns-dir", default="mlruns", help="MLflow file-store root.")
    parser.add_argument("--run-id", default="latest", help="Run id to plot, or 'latest'.")
    parser.add_argument("--output-dir", default="outputs/training_curves")
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List discovered runs and exit.",
    )
    args = parser.parse_args(argv)

    mlruns_dir = Path(args.mlruns_dir)
    runs = discover_runs(mlruns_dir)
    if args.list_runs:
        print(json.dumps([run_summary(run) for run in runs], indent=2))
        return
    if not runs:
        raise FileNotFoundError(f"No MLflow runs with metrics found under {mlruns_dir}")

    run = select_run(runs, args.run_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = plot_run(run, output_dir)
    print(json.dumps(result, indent=2))


def discover_runs(mlruns_dir: Path) -> list[Path]:
    if not mlruns_dir.exists():
        return []
    runs = [
        metrics_dir.parent for metrics_dir in mlruns_dir.rglob("metrics") if metrics_dir.is_dir()
    ]
    runs = [run for run in runs if any((run / "metrics").iterdir())]
    runs.sort(key=run_sort_key, reverse=True)
    return runs


def run_sort_key(run: Path) -> tuple[int, float]:
    meta = read_meta(run / "meta.yaml")
    end_time = int(meta.get("end_time") or 0)
    mtime = max((path.stat().st_mtime for path in (run / "metrics").glob("*")), default=0.0)
    return end_time, mtime


def read_meta(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def select_run(runs: list[Path], run_id: str) -> Path:
    if run_id == "latest":
        return runs[0]
    for run in runs:
        if run.name == run_id:
            return run
    raise FileNotFoundError(f"Run id not found: {run_id}")


def run_summary(run: Path) -> dict[str, Any]:
    meta = read_meta(run / "meta.yaml")
    metrics = sorted(path.name for path in (run / "metrics").glob("*") if path.is_file())
    return {
        "run_id": run.name,
        "run_name": meta.get("run_name", ""),
        "experiment_id": meta.get("experiment_id", ""),
        "status": meta.get("status", ""),
        "start_time": meta.get("start_time", ""),
        "end_time": meta.get("end_time", ""),
        "metrics": metrics,
        "path": str(run),
    }


def read_run_metrics(run: Path) -> dict[str, list[MetricPoint]]:
    metrics: dict[str, list[MetricPoint]] = {}
    for path in sorted((run / "metrics").glob("*")):
        if not path.is_file():
            continue
        points = read_metric_file(run.name, path.name, path)
        if points:
            metrics[path.name] = points
    return metrics


def read_metric_file(run_id: str, metric: str, path: Path) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            points.append(
                MetricPoint(
                    run_id=run_id,
                    metric=metric,
                    timestamp=int(parts[0]),
                    value=float(parts[1]),
                    step=int(parts[2]),
                )
            )
        except ValueError:
            continue
    points.sort(key=lambda point: (point.step, point.timestamp))
    return points


def plot_run(run: Path, output_dir: Path) -> dict[str, Any]:
    metrics = read_run_metrics(run)
    if not metrics:
        raise ValueError(f"Run has no readable metrics: {run}")

    write_metrics_csv(metrics, output_dir / "metrics_long.csv")

    written: list[str] = []
    if any(
        name in metrics
        for name in ("train_loss_step", "train_loss_epoch", "train_loss", "val_loss")
    ):
        written.append(
            str(
                plot_metric_group(
                    metrics,
                    ["train_loss_step", "train_loss_epoch", "train_loss", "val_loss"],
                    output_dir / "loss_curves.png",
                    title="Loss Curves",
                    ylabel="loss",
                )
            )
        )

    if any(name in metrics for name in ("val_norm_edit_distance", "val_exact_match")):
        written.append(
            str(
                plot_metric_group(
                    metrics,
                    ["val_norm_edit_distance", "val_exact_match"],
                    output_dir / "validation_metrics.png",
                    title="Validation Metrics",
                    ylabel="metric value",
                )
            )
        )

    lr_metrics = [name for name in metrics if name.lower().startswith("lr")]
    if lr_metrics:
        written.append(
            str(
                plot_metric_group(
                    metrics,
                    lr_metrics,
                    output_dir / "learning_rate.png",
                    title="Learning Rate",
                    ylabel="learning rate",
                )
            )
        )

    written.append(str(plot_all_metrics(metrics, output_dir / "all_metrics.png")))
    summary = summarize_metrics(metrics)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "run": run_summary(run),
        "output_dir": str(output_dir),
        "plots": written,
        "metrics_csv": str(output_dir / "metrics_long.csv"),
        "summary_json": str(summary_path),
        "summary": summary,
    }


def write_metrics_csv(metrics: dict[str, list[MetricPoint]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "metric", "timestamp", "step", "value"],
        )
        writer.writeheader()
        for metric_name, points in metrics.items():
            for point in points:
                writer.writerow(
                    {
                        "run_id": point.run_id,
                        "metric": metric_name,
                        "timestamp": point.timestamp,
                        "step": point.step,
                        "value": point.value,
                    }
                )


def plot_metric_group(
    metrics: dict[str, list[MetricPoint]],
    names: list[str],
    path: Path,
    title: str,
    ylabel: str,
) -> Path:
    fig, axis = plt.subplots(figsize=(10, 5))
    for name in names:
        points = metrics.get(name)
        if not points:
            continue
        alpha = 0.35 if name.endswith("_step") else 1.0
        linewidth = 1.0 if name.endswith("_step") else 2.0
        axis.plot(
            [point.step for point in points],
            [point.value for point in points],
            label=name,
            alpha=alpha,
            linewidth=linewidth,
        )

    axis.set_title(title)
    axis.set_xlabel("step")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_all_metrics(metrics: dict[str, list[MetricPoint]], path: Path) -> Path:
    names = sorted(metrics)
    cols = 2
    rows = max(1, (len(names) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows), squeeze=False)
    for axis, name in zip(axes.ravel(), names, strict=False):
        points = metrics[name]
        axis.plot([point.step for point in points], [point.value for point in points])
        axis.set_title(name)
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.25)
    for axis in axes.ravel()[len(names) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def summarize_metrics(metrics: dict[str, list[MetricPoint]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, points in metrics.items():
        if not points:
            continue
        values = [point.value for point in points]
        final = points[-1]
        if "loss" in name or "edit_distance" in name or name.startswith("lr"):
            best_value = min(values)
        else:
            best_value = max(values)
        summary[name] = {
            "count": float(len(points)),
            "final_step": float(final.step),
            "final_value": final.value,
            "best_value": best_value,
        }
    return summary


if __name__ == "__main__":
    main()
