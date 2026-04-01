#!/usr/bin/env python3
"""Visualize one or more completed MiniHack runs using CORA metrics styling."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "continual_rl"))


if getattr(pio, "kaleido", None) is None:
    class _DummyScope:
        mathjax = None

    class _DummyKaleido:
        scope = _DummyScope()

    pio.kaleido = _DummyKaleido()
elif getattr(pio.kaleido, "scope", None) is None:
    class _DummyScope:
        mathjax = None

    pio.kaleido.scope = _DummyScope()

try:
    import kaleido  # noqa: F401
    STATIC_IMAGE_EXPORT_AVAILABLE = True
except Exception:
    STATIC_IMAGE_EXPORT_AVAILABLE = False

from continual_rl.utils.cora_metrics import MINIHACK, TO_PLOT  # noqa: E402
from continual_rl.utils.metrics import Metrics  # noqa: E402


DEFAULT_RUN_DIRS = [
    Path("/scratch/ece567w26_class_root/ece567w26_class/xjsong/clear-minihack_46369769"),
    Path("/scratch/ece567w26_class_root/ece567w26_class/xjsong/impala-minihack_46461949"),
]
DEFAULT_MAX_STEP = int(80e6)


def discover_run_dirs(run_root: Path) -> list[Path]:
    run_dirs = set()
    for event_file in run_root.rglob("events.out.tfevents.*"):
        run_dirs.add(event_file.parent)
    return sorted(run_dirs)


def read_scalar_tags(run_dir: Path) -> set[str]:
    scalar_tags: set[str] = set()
    for event_file in sorted(run_dir.glob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        scalar_tags.update(accumulator.Tags().get("scalars", []))
    return scalar_tags


def adapt_tasks_for_available_tags(base_tasks: dict, scalar_tags: set[str], tag_base: str) -> dict:
    tasks = {}
    for task_name, task_spec in base_tasks.items():
        task_copy = copy.deepcopy(task_spec)
        primary_tag = f"{tag_base}/{task_copy['i']}"
        if primary_tag in scalar_tags:
            task_copy.pop("eval_i", None)
            tasks[task_name] = task_copy

    return tasks


class HtmlMetrics(Metrics):
    def __init__(self, experiment_data: dict, output_dir: Path):
        super().__init__(experiment_data)
        self._output_dir = output_dir
        self._static_export_warning_printed = False

    def write_static_image(self, fig: go.Figure, output_path: Path) -> None:
        if not STATIC_IMAGE_EXPORT_AVAILABLE:
            if not self._static_export_warning_printed:
                print("Static image export skipped: install 'kaleido' in the current environment to enable PNG/PDF output.")
                self._static_export_warning_printed = True
            return

        try:
            fig.write_image(output_path)
        except Exception as exc:
            if not self._static_export_warning_printed:
                print(f"Static image export skipped due to write_image error: {exc}")
                self._static_export_warning_printed = True

    def plot_models(self, d):  # noqa: D401
        num_task_steps = self._experiment_data["num_task_steps"]
        num_cycles = self._experiment_data["num_cycles"]
        num_tasks = self._experiment_data.get("num_tasks", len(self._experiment_data["tasks"]))
        max_step = self._experiment_data.get("max_step")
        x_limit = num_task_steps * num_tasks * num_cycles
        if max_step is not None:
            x_limit = min(x_limit, max_step)
        x_range = [-10, x_limit]

        axis_size = self._experiment_data["axis_size"]
        axis_label_size = self._experiment_data["axis_label_size"]
        legend_size = self._experiment_data["legend_size"]
        title_size = self._experiment_data["title_size"]
        which_exp = self._experiment_data["which_exp"]

        figures = {}
        self._output_dir.mkdir(parents=True, exist_ok=True)

        for task_i, (task_k, task_v) in enumerate(self._experiment_data["tasks"].items()):
            fig = go.Figure()

            y_range = task_v.get("y_range", None)
            train_regions = task_v.get("train_regions", None)
            showlegend = True
            yaxis_dtick = task_v.get("yaxis_dtick", None)

            tag = f"{self._experiment_data['tag_base']}/{task_v['i']}"

            for model_k, model_v in self._experiment_data["models"].items():
                if tag not in d[model_k]:
                    continue
                data = d[model_k][tag]
                low_trace, trace, up_trace = self.create_scatters(data, model_k, model_v)
                fig.add_trace(low_trace)
                fig.add_trace(trace)
                fig.add_trace(up_trace)

            yaxis_range = [y_range[0], y_range[1] * 1.01]
            yaxis_label = self._experiment_data.get("yaxis_label", "Expected Return")
            fig.update_layout(
                yaxis=dict(
                    title=dict(text=yaxis_label, font=dict(size=axis_label_size)),
                    range=yaxis_range,
                    tick0=0,
                    dtick=yaxis_dtick,
                    tickfont=dict(size=axis_size),
                    gridcolor="rgb(230,236,245)",
                ),
                xaxis=dict(
                    title=dict(text="Step", font=dict(size=axis_label_size)),
                    range=x_range,
                    tickvals=self._experiment_data.get("xaxis_tickvals", None),
                    tickfont=dict(size=axis_size),
                ),
                title=dict(text=f"\n{task_k}", font=dict(size=title_size)),
                legend=dict(font=dict(size=legend_size, color="black"), x=1.15),
                showlegend=showlegend,
                title_x=0.15,
                plot_bgcolor="rgb(255,255,255)",
            )

            if train_regions is not None:
                for shaded_region in train_regions:
                    fig.add_shape(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=shaded_region[0],
                        y0=y_range[0],
                        x1=shaded_region[1],
                        y1=y_range[1],
                        line=dict(color="rgba(150, 150, 180, .3)", width=1),
                        fillcolor="rgba(230, 236, 245, 0.3)",
                    )

            base_name = f"{which_exp}_{task_i:02d}_{task_k.replace('/', '_').replace(' ', '_')}"
            fig.write_html(self._output_dir / f"{base_name}.html")
            figures[task_i] = fig

        self.write_overview_grid(d, x_range)
        return figures

    def write_overview_grid(self, d, x_range):
        tasks = list(self._experiment_data["tasks"].items())
        rows, cols = 3, 5
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[task_name for task_name, _ in tasks],
            horizontal_spacing=0.04,
            vertical_spacing=0.09,
        )

        axis_size = self._experiment_data["axis_size"]
        axis_label_size = self._experiment_data["axis_label_size"]
        legend_size = self._experiment_data["legend_size"]
        title_size = self._experiment_data["title_size"]
        yaxis_label = self._experiment_data.get("yaxis_label", "Expected Return")
        subplot_title_size = max(10, int(title_size * 0.58))
        shared_axis_label_size = max(12, int(axis_label_size * 0.75))
        xaxis_title = "Step"

        for task_i, (task_k, task_v) in enumerate(tasks):
            row = task_i // cols + 1
            col = task_i % cols + 1
            tag = f"{self._experiment_data['tag_base']}/{task_v['i']}"
            y_range = task_v.get("y_range", None)
            yaxis_dtick = task_v.get("yaxis_dtick", None)
            train_regions = task_v.get("train_regions", None)

            for model_k, model_v in self._experiment_data["models"].items():
                if tag not in d[model_k]:
                    continue
                low_trace, trace, up_trace = self.create_scatters(d[model_k][tag], model_k, model_v)
                trace.showlegend = task_i == 0
                fig.add_trace(low_trace, row=row, col=col)
                fig.add_trace(trace, row=row, col=col)
                fig.add_trace(up_trace, row=row, col=col)

            fig.update_xaxes(
                range=x_range,
                tickfont=dict(size=axis_size),
                showgrid=False,
                row=row,
                col=col,
            )
            fig.update_yaxes(
                range=[y_range[0], y_range[1] * 1.01],
                dtick=yaxis_dtick,
                tickfont=dict(size=axis_size),
                gridcolor="rgb(230,236,245)",
                row=row,
                col=col,
            )

            if train_regions is not None:
                for shaded_region in train_regions:
                    fig.add_shape(
                        type="rect",
                        x0=shaded_region[0],
                        y0=y_range[0],
                        x1=shaded_region[1],
                        y1=y_range[1],
                        line=dict(color="rgba(150, 150, 180, .3)", width=1),
                        fillcolor="rgba(230, 236, 245, 0.3)",
                        row=row,
                        col=col,
                    )

        fig.update_layout(
            height=980,
            width=1800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.04,
                xanchor="center",
                x=0.5,
                font=dict(size=max(legend_size, 15), color="black"),
            ),
            plot_bgcolor="rgb(255,255,255)",
            paper_bgcolor="rgb(255,255,255)",
            margin=dict(l=105, r=30, t=120, b=120),
        )

        for annotation in fig.layout.annotations:
            annotation.font = dict(size=subplot_title_size)

        fig.add_annotation(
            x=0.5,
            y=0,
            xref="paper",
            yref="paper",
            text=xaxis_title,
            showarrow=False,
            yshift=-82,
            font=dict(size=shared_axis_label_size),
        )
        fig.add_annotation(
            x=0,
            y=0.5,
            xref="paper",
            yref="paper",
            text=yaxis_label,
            showarrow=False,
            textangle=-90,
            xshift=-92,
            font=dict(size=shared_axis_label_size),
        )

        output_html = self._output_dir / f"{self._experiment_data['which_exp']}_overview_3x5.html"
        fig.write_html(output_html)
        self.write_static_image(fig, self._output_dir / f"{self._experiment_data['which_exp']}_overview_3x5.png")

    def visualize(self, plot_spec=None):  # noqa: D401
        if plot_spec is not None:
            self._experiment_data.update(plot_spec)

        tags = []
        for task_v in self._experiment_data["tasks"].values():
            tags.append(f"{self._experiment_data['tag_base']}/{task_v['i']}")
        print(f"tags: {tags}")

        combined = {}
        summary = {}
        for model_k, model_v in self._experiment_data["models"].items():
            print(f"loading data for model: {model_k}")
            data = self.read_experiment_data(model_v, tags)
            data = self.post_processing(data, tags)
            data = self.combine_experiment_data(data, tags)
            combined[model_k] = data
            summary[model_k] = {}

            for task_key, task_data in data.items():
                if len(task_data[0]) == 0:
                    continue
                final_index = len(task_data[0]) - 1
                summary[model_k][task_key] = {
                    "final_step": float(task_data[0][final_index]),
                    "final_mean": float(task_data[1][final_index]),
                    "final_sem": float(task_data[2][final_index]),
                }
                print(
                    f"{model_k}: task {task_key}: final performance: "
                    f"{task_data[1][final_index]:.2f} \\pm {task_data[2][final_index]:.2f}"
                )

        self.plot_models(combined)
        return summary


def clip_series_to_max_step(data: tuple[np.ndarray, np.ndarray, np.ndarray], max_step: int | None):
    if max_step is None:
        return data

    x, y_mean, y_std = data
    mask = x <= max_step
    if not np.any(mask):
        return data
    return x[mask], y_mean[mask], y_std[mask]


def _display_name_for_run(run_root: Path) -> str:
    run_name = run_root.name
    if run_name.startswith("clear-minihack_"):
        return "CLEAR"
    if run_name.startswith("impala-minihack_"):
        return "IMPALA"
    return run_name


def _model_style_for_run(run_root: Path, color_index: int) -> dict:
    clear_colors = [
        "rgba(210, 140, 217, 1)",
        "rgba(168, 85, 247, 1)",
    ]
    impala_colors = [
        "rgba(77, 102, 133, 1)",
        "rgba(37, 99, 235, 1)",
    ]
    if run_root.name.startswith("clear-minihack_"):
        color = clear_colors[color_index % len(clear_colors)]
    elif run_root.name.startswith("impala-minihack_"):
        color = impala_colors[color_index % len(impala_colors)]
    else:
        palette = [
            "rgba(77, 102, 133, 1)",
            "rgba(210, 140, 217, 1)",
            "rgba(214, 178, 84, 1)",
            "rgba(106, 166, 110, 1)",
        ]
        color = palette[color_index % len(palette)]
    return dict(name=run_root.name, color=color, color_alpha=0.18)


def build_experiment_data(run_roots: list[Path], tag_base: str, output_dir: Path, max_step: int | None) -> tuple[dict, list[str], dict]:
    run_info = []
    model_available_tags = []
    for run_root in run_roots:
        run_dirs = discover_run_dirs(run_root)
        if not run_dirs:
            raise RuntimeError(f"No TensorBoard event files found under {run_root}")
        available_tags = set.intersection(*(read_scalar_tags(run_dir) for run_dir in run_dirs))
        run_info.append((run_root, run_dirs, available_tags))
        model_available_tags.append(available_tags)

    common_available_tags = set.intersection(*model_available_tags)
    tasks = adapt_tasks_for_available_tags(MINIHACK["tasks"], common_available_tags, tag_base)
    if not tasks:
        raise RuntimeError(
            f"No MiniHack tasks matched tag base '{tag_base}' across runs. "
            f"Available scalar tags include: {sorted(list(common_available_tags))[:20]}"
        )

    experiment_data = copy.deepcopy(TO_PLOT)
    experiment_data.update(copy.deepcopy(MINIHACK))
    models = {}
    metadata_runs = []
    for idx, (run_root, run_dirs, available_tags) in enumerate(run_info):
        model_name = _display_name_for_run(run_root)
        model = _model_style_for_run(run_root, idx)
        model["runs"] = [str(run_dir) for run_dir in run_dirs]
        models[model_name] = model
        metadata_runs.append(
            {
                "run_root": str(run_root),
                "run_dirs": [str(run_dir) for run_dir in run_dirs],
                "available_scalar_tag_count": len(available_tags),
            }
        )

    experiment_data.update(
        exp_dir="",
        which_exp=f"minihack_overlay_{tag_base}",
        models=models,
        tasks=tasks,
        num_tasks=len(tasks),
        cache_dir=str(output_dir / "cache"),
        tag_base=tag_base,
        max_step=max_step,
    )

    metadata = {
        "run_roots": [str(run_root) for run_root in run_roots],
        "runs": metadata_runs,
        "tag_base": tag_base,
        "available_scalar_tag_count": len(common_available_tags),
        "selected_tasks": list(tasks.keys()),
        "model_names": list(models.keys()),
        "max_step": max_step,
    }
    return experiment_data, sorted(common_available_tags), metadata


def default_output_dir_for_runs(run_roots: list[Path], tag_base: str) -> Path:
    if len(run_roots) == 1:
        return run_roots[0] / "visualizations" / tag_base

    common_parent = Path(os.path.commonpath([str(run_root.parent) for run_root in run_roots]))
    return common_parent / "minihack-comparison" / tag_base


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    return f"{value:.2f}"


def write_index_html(output_dir: Path, metadata: dict) -> None:
    task_entries = []
    for idx, task_name in enumerate(metadata["selected_tasks"]):
        safe_name = task_name.replace("/", "_").replace(" ", "_")
        file_name = f"{metadata['which_exp']}_{idx:02d}_{safe_name}.html"
        task_entries.append((task_name, file_name))

    links_html = "\n".join(
        f'<li><a href="{file_name}">{task_name}</a></li>' for task_name, file_name in task_entries
    )
    summary_rows = []
    for idx, (task_name, file_name) in enumerate(task_entries):
        tag = metadata["task_tags"][idx]
        cells = [
            f'<td><a href="{file_name}">{task_name}</a></td>',
            f'<td><code>{tag}</code></td>',
        ]
        for model_name in metadata["model_names"]:
            result = metadata["final_summary"].get(model_name, {}).get(tag, {})
            final_mean = _format_metric(result.get("final_mean"))
            final_sem = _format_metric(result.get("final_sem"))
            cells.append(f"<td>{final_mean}</td>")
            cells.append(f"<td>{final_sem}</td>")
        summary_rows.append(f"<tr>{''.join(cells)}</tr>")

    header_cells = [
        "<th>Task</th>",
        "<th>Tag</th>",
    ]
    for model_name in metadata["model_names"]:
        header_cells.append(f"<th>{model_name} mean</th>")
        header_cells.append(f"<th>{model_name} sem</th>")
    summary_table_html = "\n".join(summary_rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MiniHack Visualization Index</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0f172a;
      --card: #111827;
      --fg: #e5e7eb;
      --muted: #94a3b8;
      --link: #93c5fd;
      --border: #334155;
    }}
    body {{
      margin: 0;
      font-family: sans-serif;
      background: var(--bg);
      color: var(--fg);
    }}
    main {{
      max-width: 960px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 20px;
    }}
    h1, h2 {{
      margin-top: 0;
    }}
    p, li {{
      line-height: 1.5;
    }}
    a {{
      color: var(--link);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    code {{
      color: var(--muted);
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
      columns: 2;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    @media (max-width: 700px) {{
      ul {{
        columns: 1;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="card">
      <h1>MiniHack Visualization Index</h1>
      <p><strong>Runs:</strong> {len(metadata['run_roots'])}</p>
      <p><strong>Tag base:</strong> <code>{metadata['tag_base']}</code></p>
      <p><strong>Tasks plotted:</strong> {len(task_entries)}</p>
    </div>
    <div class="card">
      <h2>Runs</h2>
      <ul>
        {"".join(f'<li><code>{run}</code></li>' for run in metadata["run_roots"])}
      </ul>
    </div>
    <div class="card">
      <h2>Task Plots</h2>
      <p><a href="{metadata['which_exp']}_overview_3x5.html">Open 3x5 overview grid</a></p>
      <ul>
        {links_html}
      </ul>
    </div>
    <div class="card">
      <h2>Final Return Summary</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              {''.join(header_cells)}
            </tr>
          </thead>
          <tbody>
            {summary_table_html}
          </tbody>
        </table>
      </div>
    </div>
  </main>
</body>
</html>
"""
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        default=DEFAULT_RUN_DIRS,
        help="One or more root directories containing completed MiniHack runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write visualization outputs. Defaults to a comparison directory when multiple runs are given.",
    )
    parser.add_argument(
        "--tag-base",
        type=str,
        default="eval_reward",
        help="TensorBoard scalar prefix to visualize, e.g. eval_reward or train_reward.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=DEFAULT_MAX_STEP,
        help="Only plot and summarize points up to this global step. Default: 80000000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_roots = [run_dir.resolve() for run_dir in args.run_dir]
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = default_output_dir_for_runs(run_roots, args.tag_base)

    experiment_data, available_tags, metadata = build_experiment_data(run_roots, args.tag_base, output_dir, args.max_step)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(experiment_data["cache_dir"]).mkdir(parents=True, exist_ok=True)

    metadata["available_scalar_tags_preview"] = sorted(available_tags)[:50]
    metadata["which_exp"] = experiment_data["which_exp"]
    metadata["task_tags"] = [
        f"{experiment_data['tag_base']}/{task_v['i']}" for task_v in experiment_data["tasks"].values()
    ]

    metrics = HtmlMetrics(experiment_data, output_dir)
    original_combine_experiment_data = metrics.combine_experiment_data

    def combine_and_clip(data, tags):
        available_tags = sorted(set.intersection(*(set(run_data.keys()) for run_data in data.values())))
        combined = original_combine_experiment_data(data, available_tags)
        return {tag: clip_series_to_max_step(tag_data, args.max_step) for tag, tag_data in combined.items()}

    metrics.combine_experiment_data = combine_and_clip
    metadata["final_summary"] = metrics.visualize()
    with open(output_dir / "visualization_manifest.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    write_index_html(output_dir, metadata)

    print(f"Wrote visualizations to: {output_dir}")
    print(f"Selected tasks: {metadata['selected_tasks']}")


if __name__ == "__main__":
    main()
