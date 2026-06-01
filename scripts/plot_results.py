# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false, reportIndexIssue=false
"""Paper-quality plotting for BioLogic State Machine simulation results."""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal, cast

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_DIR: Path = Path("results/csv")
FIGURE_DIR: Path = Path("results/figures")
TABLE_DIR: Path = Path("results/tables")

Format = Literal["pdf", "png", "both"]
Record = dict[str, Any]

LABELS: dict[str, str] = {
    "num_states": "Number of stored states",
    "state_dim": "Register dimension n",
    "transition_accuracy": "Transition accuracy",
    "recovery_accuracy": "Recovery accuracy",
    "flip_fraction": "Bit-flip fraction",
    "capacity_ratio": "Capacity ratio m/n",
    "write_fraction": "Target coordinates written",
    "sparse_transition_accuracy": "Sparse transition accuracy",
    "epochs": "Training epochs",
    "all_transition_accuracy": "Learned transition accuracy",
    "feature_capacity_ratio": "Pair-layer capacity ratio",
    "coverage_fraction": "Demonstration coverage",
    "grammar_name": "Grammar",
    "test_string_accuracy": "Test string accuracy",
    "length_generalization_accuracy": "Length generalization accuracy",
    "transition_coverage_train": "Training transition coverage",
    "masked_topk_accuracy": "Masked top-k accuracy",
    "topk_k": "Written coordinates k",
}

CASE_LABELS: dict[str, str] = {
    "same_payload_store": "STORE p1",
    "same_payload_compare": "COMPARE p1",
    "same_descriptor_store_p1": "STORE p1",
    "same_descriptor_store_p2": "STORE p2",
    "missing_descriptor": "payload only",
    "wrong_content_compare": "COMPARE p2",
}


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_csv(name: str) -> pd.DataFrame:
    path: Path = CSV_DIR / name
    if not path.exists():
        print(f"Warning: missing CSV {path}; skipping dependent figure(s).")
        return pd.DataFrame()
    return pd.read_csv(path)


def _records(df: pd.DataFrame) -> list[Record]:
    return cast(list[Record], df.to_dict("records"))


def _to_float(value: object) -> float | None:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _required(df: pd.DataFrame, cols: list[str], figure: str) -> bool:
    missing: list[str] = [col for col in cols if col not in df.columns]
    if missing:
        print(f"Warning: {figure} skipped; missing columns: {', '.join(missing)}")
        return False
    if df.empty:
        print(f"Warning: {figure} skipped; input data is empty.")
        return False
    return True


def _filter_rows(df: pd.DataFrame, predicate: Callable[[Record], bool]) -> pd.DataFrame:
    return pd.DataFrame([row for row in _records(df) if predicate(row)])


def _values(df: pd.DataFrame, col: str) -> list[object]:
    seen: set[object] = set()
    values: list[object] = []
    for row in _records(df):
        value = row.get(col)
        if value not in seen and value is not None:
            seen.add(value)
            values.append(value)
    return values


def _sorted_values(df: pd.DataFrame, col: str) -> list[object]:
    values: list[object] = _values(df, col)
    try:
        return sorted(values, key=lambda value: float(cast(Any, value)))
    except (TypeError, ValueError):
        return sorted(values, key=str)


def mean_sem(df: pd.DataFrame, group_cols: list[str], metric_col: str) -> pd.DataFrame:
    seed_groups: dict[tuple[tuple[object, ...], object], list[float]] = defaultdict(
        list
    )
    raw_groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    has_seed: bool = "seed" in df.columns

    for row in _records(df):
        metric = _to_float(row.get(metric_col))
        if metric is None:
            continue
        key = tuple(row.get(col) for col in group_cols)
        if has_seed:
            seed_groups[(key, row.get("seed"))].append(metric)
        else:
            raw_groups[key].append(metric)

    grouped_values: dict[tuple[object, ...], list[float]] = defaultdict(list)
    if has_seed:
        for (key, _seed), values in seed_groups.items():
            grouped_values[key].append(float(np.mean(values)))
    else:
        grouped_values = raw_groups

    rows: list[Record] = []
    for key, values in grouped_values.items():
        if not values:
            continue
        n = len(values)
        sem = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        out: Record = {col: key[idx] for idx, col in enumerate(group_cols)}
        out[metric_col] = float(np.mean(values))
        out["sem"] = sem
        out["n"] = n
        rows.append(out)
    return pd.DataFrame(rows)


def prettify_register_label(n: object) -> str:
    value = _to_float(n)
    if value is None:
        return str(n)
    return f"$n={int(value)}$"


def savefig(fig: Any, stem: str, fmt: str) -> list[Path]:
    ensure_dirs()
    formats: list[str] = ["pdf", "png"] if fmt == "both" else [fmt]
    paths: list[Path] = []
    for suffix in formats:
        path = FIGURE_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _rows_for(df: pd.DataFrame, col: str, value: object) -> pd.DataFrame:
    return _filter_rows(df, lambda row: row.get(col) == value)


def _metric_rows(df: pd.DataFrame, metric_col: str) -> list[float]:
    return [
        metric
        for metric in (_to_float(row.get(metric_col)) for row in _records(df))
        if metric is not None
    ]


def _mean_metric(df: pd.DataFrame, metric_col: str) -> float | None:
    values = _metric_rows(df, metric_col)
    return float(np.mean(values)) if values else None


def _median_metric(df: pd.DataFrame, metric_col: str) -> float | None:
    values = _metric_rows(df, metric_col)
    return float(np.median(values)) if values else None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _nearest_value(values: list[object], target: float) -> object | None:
    numeric: list[tuple[float, object]] = []
    for value in values:
        as_float = _to_float(value)
        if as_float is not None:
            numeric.append((as_float, value))
    if not numeric:
        return None
    return min(numeric, key=lambda item: abs(item[0] - target))[1]


def plot_line_with_sem(
    ax: Any,
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_legend: bool = True,
) -> None:
    summary = mean_sem(df, [x, hue], y)
    if summary.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    for hue_value in _sorted_values(summary, hue):
        line_df = _rows_for(summary, hue, hue_value)
        rows = sorted(
            _records(line_df),
            key=lambda row: float(cast(Any, row.get(x))),
        )
        xs = [float(row[x]) for row in rows]
        ys = [float(row[y]) for row in rows]
        sems = [float(row.get("sem", 0.0)) for row in rows]
        ax.errorbar(
            xs,
            ys,
            yerr=sems,
            marker="o",
            capsize=2.5,
            label=prettify_register_label(hue_value),
        )

    xticks = [float(cast(Any, value)) for value in _sorted_values(df, x)]
    ax.set_xticks(xticks)
    ax.set_title(title)
    ax.set_xlabel(xlabel or LABELS.get(x, x))
    ax.set_ylabel(ylabel or LABELS.get(y, y))
    if y in {
        "transition_accuracy",
        "recovery_accuracy",
        "sparse_transition_accuracy",
        "all_transition_accuracy",
        "accuracy",
        "success",
    }:
        ax.set_ylim(0.0, 1.03)
    if show_legend:
        ax.legend(title="Register dimension", frameon=False)


def _register_panel_title(register_type: str) -> str:
    if register_type == "nearest":
        return "Ideal nearest-attractor cleanup"
    if register_type == "hopfield":
        return "Hopfield-style cleanup"
    return register_type


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def plot_fig01(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig01_transition_accuracy"
    cols = ["register_type", "num_states", "state_dim", "seed", "transition_accuracy"]
    if not _required(df, cols, stem):
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for idx, register_type in enumerate(["nearest", "hopfield"]):
        ax = axes[idx]
        plot_line_with_sem(
            ax,
            _rows_for(df, "register_type", register_type),
            "num_states",
            "transition_accuracy",
            "state_dim",
            _register_panel_title(register_type),
            show_legend=idx == 1,
        )
        _add_panel_label(ax, f"({chr(ord('a') + idx)})")
    return savefig(fig, stem, fmt)


def plot_fig01b(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig01b_transition_accuracy_hopfield_only"
    cols = ["register_type", "num_states", "state_dim", "seed", "transition_accuracy"]
    if not _required(df, cols, stem):
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    plot_line_with_sem(
        ax,
        _rows_for(df, "register_type", "hopfield"),
        "num_states",
        "transition_accuracy",
        "state_dim",
        "Hopfield transition accuracy",
    )
    return savefig(fig, stem, fmt)


def plot_fig02(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig02_noise_recovery"
    cols = ["register_type", "flip_fraction", "state_dim", "seed", "recovery_accuracy"]
    if not _required(df, cols, stem):
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for idx, register_type in enumerate(["nearest", "hopfield"]):
        ax = axes[idx]
        plot_line_with_sem(
            ax,
            _rows_for(df, "register_type", register_type),
            "flip_fraction",
            "recovery_accuracy",
            "state_dim",
            _register_panel_title(register_type),
            show_legend=idx == 1,
        )
        _add_panel_label(ax, f"({chr(ord('a') + idx)})")
    return savefig(fig, stem, fmt)


def plot_fig02b(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig02b_noise_recovery_numstates32"
    cols = [
        "register_type",
        "num_states",
        "flip_fraction",
        "state_dim",
        "seed",
        "recovery_accuracy",
    ]
    if not _required(df, cols, stem):
        return []
    num_state_values = _sorted_values(df, "num_states")
    if 32 in num_state_values:
        target_state: object | None = 32
    else:
        below_64 = [
            value
            for value in num_state_values
            if (as_float := _to_float(value)) is not None and as_float < 64.0
        ]
        target_state = (
            below_64[-1] if below_64 else _nearest_value(num_state_values, 32.0)
        )
    if target_state is None:
        print(f"Warning: {stem} skipped; no num_states values available.")
        return []
    state_label = int(float(cast(Any, target_state)))
    filtered = _rows_for(df, "num_states", target_state)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for idx, register_type in enumerate(["nearest", "hopfield"]):
        ax = axes[idx]
        plot_line_with_sem(
            ax,
            _rows_for(filtered, "register_type", register_type),
            "flip_fraction",
            "recovery_accuracy",
            "state_dim",
            f"{_register_panel_title(register_type)} (m={state_label})",
            show_legend=idx == 1,
        )
        _add_panel_label(ax, f"({chr(ord('a') + idx)})")
    return savefig(fig, stem, fmt)


def _add_capacity_reference(ax: Any) -> None:
    ax.axvline(0.138, color="0.35", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.text(
        0.138,
        0.04,
        "classical Hopfield capacity ~0.138",
        rotation=90,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.25",
    )


def plot_fig03(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig03_capacity"
    cols = ["register_type", "capacity_ratio", "state_dim", "seed", "recovery_accuracy"]
    if not _required(df, cols, stem):
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for idx, register_type in enumerate(["nearest", "hopfield"]):
        ax = axes[idx]
        plot_line_with_sem(
            ax,
            _rows_for(df, "register_type", register_type),
            "capacity_ratio",
            "recovery_accuracy",
            "state_dim",
            _register_panel_title(register_type),
            show_legend=idx == 1,
        )
        _add_capacity_reference(ax)
        _add_panel_label(ax, f"({chr(ord('a') + idx)})")
    return savefig(fig, stem, fmt)


def plot_fig03b(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig03b_capacity_hopfield_only"
    cols = ["register_type", "capacity_ratio", "state_dim", "seed", "recovery_accuracy"]
    if not _required(df, cols, stem):
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    plot_line_with_sem(
        ax,
        _rows_for(df, "register_type", "hopfield"),
        "capacity_ratio",
        "recovery_accuracy",
        "state_dim",
        "Hopfield recovery decreases above capacity",
    )
    _add_capacity_reference(ax)
    return savefig(fig, stem, fmt)


def plot_fig04(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig04_sparse_transitions"
    cols = [
        "register_type",
        "mode",
        "write_fraction",
        "state_dim",
        "seed",
        "sparse_transition_accuracy",
    ]
    if not _required(df, cols, stem):
        return []
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), constrained_layout=True)
    configs = [
        ("keep_current", "nearest", "Nearest, keep-current"),
        ("keep_current", "hopfield", "Hopfield, keep-current"),
        ("random_noise", "nearest", "Nearest, random-noise"),
        ("random_noise", "hopfield", "Hopfield, random-noise"),
    ]
    for idx, (mode, register_type, title) in enumerate(configs):
        ax = axes[idx // 2][idx % 2]
        subset = _rows_for(_rows_for(df, "mode", mode), "register_type", register_type)
        plot_line_with_sem(
            ax,
            subset,
            "write_fraction",
            "sparse_transition_accuracy",
            "state_dim",
            title,
            show_legend=idx == 1,
        )
        _add_panel_label(ax, f"({chr(ord('a') + idx)})")
    return savefig(fig, stem, fmt)


def plot_fig04b(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig04b_sparse_transitions_random_noise"
    cols = [
        "register_type",
        "mode",
        "write_fraction",
        "state_dim",
        "seed",
        "sparse_transition_accuracy",
    ]
    if not _required(df, cols, stem):
        return []
    filtered = _rows_for(df, "mode", "random_noise")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for idx, register_type in enumerate(["nearest", "hopfield"]):
        ax = axes[idx]
        plot_line_with_sem(
            ax,
            _rows_for(filtered, "register_type", register_type),
            "write_fraction",
            "sparse_transition_accuracy",
            "state_dim",
            _register_panel_title(register_type),
            show_legend=idx == 1,
        )
        _add_panel_label(ax, f"({chr(ord('a') + idx)})")
    return savefig(fig, stem, fmt)


def plot_fig05(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig05_descriptor_payload"
    if not _required(df, ["case_name"], stem):
        return []

    case_rows = _filter_rows(
        df,
        lambda row: (
            row.get("case_name") != "descriptor_corruption"
            and _to_float(row.get("success")) is not None
        ),
    )
    corruption_rows = _filter_rows(
        df,
        lambda row: (
            row.get("case_name") == "descriptor_corruption"
            and _to_float(row.get("operation_success_rate")) is not None
            and _to_float(row.get("content_success_rate")) is not None
        ),
    )
    has_corruption = not corruption_rows.empty
    if case_rows.empty and not has_corruption:
        print(f"Warning: {stem} skipped; no plottable descriptor rows.")
        return []

    if has_corruption:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
        ax_cases = axes[0]
        ax_corruption = axes[1]
    else:
        print("Warning: descriptor corruption rates unavailable; plotting Part A only.")
        fig, ax_cases = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        ax_corruption = None

    if not case_rows.empty:
        summary = mean_sem(case_rows, ["case_name"], "success")
        labels = [
            CASE_LABELS.get(str(row["case_name"]), str(row["case_name"]))
            for row in _records(summary)
        ]
        heights = [float(row["success"]) for row in _records(summary)]
        ax_cases.bar(labels, heights, color="0.35")
        ax_cases.set_title("Protocol cases")
        ax_cases.set_ylabel("Success rate")
        ax_cases.set_ylim(0.0, 1.03)
        ax_cases.tick_params(axis="x", rotation=30)
        _add_panel_label(ax_cases, "(a)")

    if has_corruption and ax_corruption is not None:
        for metric, label in [
            ("operation_success_rate", "Operation success"),
            ("content_success_rate", "Content success"),
        ]:
            summary = mean_sem(corruption_rows, ["corruption_rate"], metric)
            rows = sorted(
                _records(summary),
                key=lambda row: float(row["corruption_rate"]),
            )
            xs = [float(row["corruption_rate"]) for row in rows]
            ys = [float(row[metric]) for row in rows]
            sems = [float(row.get("sem", 0.0)) for row in rows]
            ax_corruption.errorbar(
                xs, ys, yerr=sems, marker="o", capsize=2.5, label=label
            )
        ax_corruption.set_title("Descriptor corruption affects operation-level success")
        ax_corruption.set_xlabel("Descriptor corruption rate")
        ax_corruption.set_ylabel("Success rate")
        ax_corruption.set_ylim(0.0, 1.03)
        ax_corruption.set_xticks(
            [
                float(cast(Any, value))
                for value in _sorted_values(corruption_rows, "corruption_rate")
            ]
        )
        ax_corruption.legend(frameon=False)
        _add_panel_label(ax_corruption, "(b)")
    return savefig(fig, stem, fmt)


def _heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    metric_col: str,
    title: str,
    stem: str,
    fmt: str,
) -> list[Path]:
    if df.empty:
        print(f"Warning: {stem} skipped; no data.")
        return []
    rows = _sorted_values(df, row_col)
    cols = _sorted_values(df, col_col)
    matrix = np.full((len(rows), len(cols)), np.nan)
    summary = mean_sem(df, [row_col, col_col], metric_col)
    for record in _records(summary):
        r_idx = rows.index(record[row_col])
        c_idx = cols.index(record[col_col])
        metric = _to_float(record.get(metric_col))
        if metric is not None:
            matrix[r_idx, c_idx] = metric

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(LABELS.get(col_col, col_col))
    ax.set_ylabel(LABELS.get(row_col, row_col))
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(
        [f"{float(cast(Any, value)):g}" for value in cols], rotation=45, ha="right"
    )
    ax.set_yticklabels([f"{int(float(cast(Any, value)))}" for value in rows])
    for r_idx in range(len(rows)):
        for c_idx in range(len(cols)):
            value = matrix[r_idx, c_idx]
            if not np.isnan(value):
                ax.text(
                    c_idx,
                    r_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.55 else "black",
                    fontsize=8,
                )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(LABELS.get(metric_col, metric_col))
    return savefig(fig, stem, fmt)


def plot_fig06(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig06_hopfield_transition_heatmap"
    cols = ["register_type", "state_dim", "num_states", "transition_accuracy"]
    if not _required(df, cols, stem):
        return []
    return _heatmap(
        _rows_for(df, "register_type", "hopfield"),
        "state_dim",
        "num_states",
        "transition_accuracy",
        "Hopfield cleanup capacity dependence",
        stem,
        fmt,
    )


def plot_fig07(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig07_hopfield_capacity_heatmap"
    cols = ["register_type", "state_dim", "capacity_ratio", "recovery_accuracy"]
    if not _required(df, cols, stem):
        return []
    return _heatmap(
        _rows_for(df, "register_type", "hopfield"),
        "state_dim",
        "capacity_ratio",
        "recovery_accuracy",
        "Hopfield recovery decreases above capacity",
        stem,
        fmt,
    )


def _series_label(row: Record) -> str:
    feature_mode = str(row.get("feature_mode", ""))
    hidden_dim = _to_float(row.get("hidden_dim"))
    if feature_mode == "exact_pair":
        return "exact pair"
    if feature_mode == "hashed_pair":
        return f"hashed h={int(hidden_dim)}" if hidden_dim is not None else "hashed"
    if feature_mode == "random_conjunctive":
        return (
            f"random h={int(hidden_dim)}"
            if hidden_dim is not None
            else "random conjunctive"
        )
    return feature_mode


def _learning_records_with_series(
    df: pd.DataFrame,
    series_func: Callable[[Record], str],
) -> pd.DataFrame:
    rows: list[Record] = []
    for row in _records(df):
        out = dict(row)
        out["series"] = series_func(row)
        rows.append(out)
    return pd.DataFrame(rows)


def _with_feature_capacity_ratio(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[Record] = []
    for row in _records(df):
        num_states = _to_float(row.get("num_states"))
        num_inputs = _to_float(row.get("num_inputs"))
        feature_dim = _to_float(row.get("feature_dim"))
        if num_states is None or num_inputs is None or feature_dim is None:
            continue
        out = dict(row)
        out["feature_capacity_ratio"] = feature_dim / (num_states * num_inputs)
        rows.append(out)
    return pd.DataFrame(rows)


def _filter_learning_base(df: pd.DataFrame) -> pd.DataFrame:
    return _filter_rows(
        df,
        lambda row: (
            row.get("training_mode") == "full_table"
            and row.get("output_mode") == "dense"
            and _to_float(row.get("coverage_fraction")) == 1.0
            and str(row.get("update_rule")) == "delta"
        ),
    )


def _max_epoch_subset(df: pd.DataFrame) -> pd.DataFrame:
    epoch_values = [
        value for value in (_to_float(row.get("epochs")) for row in _records(df))
        if value is not None
    ]
    if not epoch_values:
        return pd.DataFrame()
    max_epoch = max(epoch_values)
    return _filter_rows(
        df,
        lambda row: (
            (epoch := _to_float(row.get("epochs"))) is not None
            and math.isclose(epoch, max_epoch)
        ),
    )


def _plot_metric_by_category(
    ax: Any,
    df: pd.DataFrame,
    x: str,
    y: str,
    category: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    summary = mean_sem(df, [x, category], y)
    if summary.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    for category_value in _sorted_values(summary, category):
        line_df = _rows_for(summary, category, category_value)
        rows = sorted(_records(line_df), key=lambda row: float(cast(Any, row[x])))
        xs = [float(row[x]) for row in rows]
        ys = [float(row[y]) for row in rows]
        sems = [float(row.get("sem", 0.0)) for row in rows]
        ax.errorbar(xs, ys, yerr=sems, marker="o", capsize=2.5, label=str(category_value))
    ax.set_xticks([float(cast(Any, value)) for value in _sorted_values(df, x)])
    ax.set_title(title)
    ax.set_xlabel(xlabel or LABELS.get(x, x))
    ax.set_ylabel(ylabel or LABELS.get(y, y))
    if y in {"all_transition_accuracy", "accuracy"}:
        ax.set_ylim(0.0, 1.03)
    ax.legend(frameon=False)


def plot_fig08(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig08_learned_transition_accuracy"
    cols = [
        "epochs",
        "all_transition_accuracy",
        "feature_mode",
        "hidden_dim",
        "training_mode",
        "output_mode",
        "coverage_fraction",
        "register_type",
        "update_rule",
    ]
    if not _required(df, cols, stem):
        return []
    base = _learning_records_with_series(_filter_learning_base(df), _series_label)
    if base.empty:
        print(f"Warning: {stem} skipped; no dense full-table delta learning rows.")
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for idx, register_type in enumerate(["nearest", "hopfield"]):
        ax = axes[idx]
        _plot_metric_by_category(
            ax,
            _rows_for(base, "register_type", register_type),
            "epochs",
            "all_transition_accuracy",
            "series",
            _register_panel_title(register_type),
        )
        _add_panel_label(ax, f"({chr(ord('a') + idx)})")
    return savefig(fig, stem, fmt)


def plot_fig09(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig09_pair_layer_capacity"
    cols = [
        "num_states",
        "num_inputs",
        "feature_dim",
        "feature_mode",
        "epochs",
        "all_transition_accuracy",
        "register_type",
        "training_mode",
        "output_mode",
    ]
    if not _required(df, cols, stem):
        return []
    base = _max_epoch_subset(_filter_learning_base(df))
    base = _filter_rows(
        base,
        lambda row: row.get("feature_mode") in {"exact_pair", "hashed_pair"},
    )
    base = _with_feature_capacity_ratio(base)
    if base.empty:
        print(f"Warning: {stem} skipped; no feature-capacity rows.")
        return []
    labelled = _learning_records_with_series(
        base,
        lambda row: _register_panel_title(str(row.get("register_type"))),
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    _plot_metric_by_category(
        ax,
        labelled,
        "feature_capacity_ratio",
        "all_transition_accuracy",
        "series",
        "Pair-layer capacity limits learned transitions",
    )
    return savefig(fig, stem, fmt)


def plot_fig10(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig10_learned_sparse_transitions"
    cols = [
        "feature_mode",
        "output_mode",
        "write_fraction",
        "unwritten_mode",
        "register_type",
        "epochs",
        "all_transition_accuracy",
    ]
    if not _required(df, cols, stem):
        return []
    base = _filter_rows(
        df,
        lambda row: (
            row.get("feature_mode") == "exact_pair"
            and row.get("output_mode") == "sparse_topk"
        ),
    )
    base = _max_epoch_subset(base)
    if base.empty:
        print(f"Warning: {stem} skipped; no sparse learned transition rows.")
        return []
    labelled = _learning_records_with_series(
        base,
        lambda row: (
            f"{str(row.get('register_type'))}, "
            f"{str(row.get('unwritten_mode')).replace('_', '-')}"
        ),
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    _plot_metric_by_category(
        ax,
        labelled,
        "write_fraction",
        "all_transition_accuracy",
        "series",
        "Sparse learned transition proposals",
    )
    return savefig(fig, stem, fmt)


def plot_fig11(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig11_coverage_seen_unseen"
    cols = [
        "training_mode",
        "coverage_fraction",
        "register_type",
        "seen_transition_accuracy",
        "unseen_transition_accuracy",
        "all_transition_accuracy",
        "epochs",
    ]
    if not _required(df, cols, stem):
        return []
    base = _filter_rows(
        df,
        lambda row: (
            row.get("training_mode") == "coverage_sweep"
            and row.get("register_type") == "nearest"
            and row.get("output_mode") == "dense"
        ),
    )
    base = _max_epoch_subset(base)
    rows: list[Record] = []
    for row in _records(base):
        for metric, label in [
            ("seen_transition_accuracy", "seen"),
            ("unseen_transition_accuracy", "unseen"),
            ("all_transition_accuracy", "all"),
        ]:
            value = _to_float(row.get(metric))
            coverage = _to_float(row.get("coverage_fraction"))
            if value is None or coverage is None:
                continue
            rows.append(
                {
                    "coverage_fraction": coverage,
                    "accuracy": value,
                    "series": label,
                }
            )
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        print(f"Warning: {stem} skipped; no coverage rows.")
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    _plot_metric_by_category(
        ax,
        long_df,
        "coverage_fraction",
        "accuracy",
        "series",
        "Seen and unseen arbitrary transitions",
        ylabel="Accuracy",
    )
    return savefig(fig, stem, fmt)


def _main_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "row_type" not in df.columns:
        return df
    return _rows_for(df, "row_type", "main")


def plot_fig12(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig12_tomita_generalization"
    cols = [
        "row_type",
        "grammar_name",
        "feature_mode",
        "register_type",
        "test_string_accuracy",
    ]
    if not _required(df, cols, stem):
        return []
    main = _filter_rows(
        _main_rows(df),
        lambda row: str(row.get("grammar_name", "")).startswith("tomita"),
    )
    if main.empty:
        print(f"Warning: {stem} skipped; no Tomita rows.")
        return []
    labelled = _learning_records_with_series(
        main,
        lambda row: f"{row.get('feature_mode')} {row.get('register_type')}",
    )
    summary = mean_sem(labelled, ["grammar_name", "series"], "test_string_accuracy")
    grammars = [str(value) for value in _sorted_values(summary, "grammar_name")]
    series = [str(value) for value in _sorted_values(summary, "series")]
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    x = np.arange(len(grammars))
    for idx, series_name in enumerate(series):
        rows = _records(_rows_for(summary, "series", series_name))
        values = {
            str(row["grammar_name"]): float(row["test_string_accuracy"])
            for row in rows
        }
        heights = [values.get(grammar, math.nan) for grammar in grammars]
        ax.bar(x + (idx - (len(series) - 1) / 2) * width, heights, width, label=series_name)
    ax.set_xticks(x)
    ax.set_xticklabels(grammars, rotation=30, ha="right")
    ax.set_ylabel("Test string accuracy")
    ax.set_title("Tomita grammar generalization")
    ax.set_ylim(0.0, 1.03)
    ax.legend(frameon=False)
    return savefig(fig, stem, fmt)


def plot_fig13(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig13_length_generalization"
    cols = ["row_type", "length", "test_string_accuracy", "grammar_name", "train_max_len"]
    if not _required(df, cols, stem):
        return []
    length_rows = _filter_rows(
        df,
        lambda row: (
            row.get("row_type") == "length"
            and row.get("feature_mode") == "exact_pair"
            and row.get("register_type") == "nearest"
        ),
    )
    if length_rows.empty:
        print(f"Warning: {stem} skipped; no length rows.")
        return []
    summary = mean_sem(length_rows, ["length"], "test_string_accuracy")
    rows = sorted(_records(summary), key=lambda row: float(row["length"]))
    xs = [float(row["length"]) for row in rows]
    ys = [float(row["test_string_accuracy"]) for row in rows]
    sems = [float(row.get("sem", 0.0)) for row in rows]
    train_max = _to_float(_records(length_rows)[0].get("train_max_len"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.errorbar(xs, ys, yerr=sems, marker="o", capsize=2.5)
    if train_max is not None:
        ax.axvline(train_max, color="0.35", linestyle="--", linewidth=1.1)
    ax.set_xticks(xs)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("String length")
    ax.set_ylabel("Accept/reject accuracy")
    ax.set_title("Length generalization")
    return savefig(fig, stem, fmt)


def plot_fig14(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig14_transition_coverage_vs_accuracy"
    cols = ["row_type", "transition_coverage_train", "test_string_accuracy", "grammar_name"]
    if not _required(df, cols, stem):
        return []
    main = _main_rows(df)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for register_type in _sorted_values(main, "register_type"):
        subset = _rows_for(main, "register_type", register_type)
        xs = [
            float(row["transition_coverage_train"])
            for row in _records(subset)
            if _to_float(row.get("test_string_accuracy")) is not None
        ]
        ys = [
            float(row["test_string_accuracy"])
            for row in _records(subset)
            if _to_float(row.get("test_string_accuracy")) is not None
        ]
        ax.scatter(xs, ys, s=24, alpha=0.75, label=str(register_type))
    ax.set_xlabel("Training transition coverage")
    ax.set_ylabel("Test string accuracy")
    ax.set_ylim(0.0, 1.03)
    ax.set_title("Transition coverage predicts grammar generalization")
    ax.legend(frameon=False)
    return savefig(fig, stem, fmt)


def plot_fig15(df: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig15_topk_bound"
    cols = [
        "row_type",
        "topk_k",
        "masked_topk_accuracy",
        "theoretical_k_delta_05",
        "theoretical_k_delta_01",
    ]
    if not _required(df, cols, stem):
        return []
    topk_rows = _rows_for(df, "row_type", "topk")
    if topk_rows.empty:
        print(f"Warning: {stem} skipped; no top-k rows.")
        return []
    summary = mean_sem(topk_rows, ["topk_k"], "masked_topk_accuracy")
    rows = sorted(_records(summary), key=lambda row: float(row["topk_k"]))
    xs = [float(row["topk_k"]) for row in rows]
    ys = [float(row["masked_topk_accuracy"]) for row in rows]
    sems = [float(row.get("sem", 0.0)) for row in rows]
    k05 = _to_float(_records(topk_rows)[0].get("theoretical_k_delta_05"))
    k01 = _to_float(_records(topk_rows)[0].get("theoretical_k_delta_01"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.errorbar(xs, ys, yerr=sems, marker="o", capsize=2.5)
    ax.axhline(0.95, color="0.35", linestyle="--", linewidth=1.0)
    ax.axhline(0.99, color="0.55", linestyle=":", linewidth=1.0)
    if k05 is not None:
        ax.axvline(k05, color="0.35", linestyle="--", linewidth=1.0)
    if k01 is not None:
        ax.axvline(k01, color="0.55", linestyle=":", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Written coordinates k")
    ax.set_ylabel("Masked top-k accuracy")
    ax.set_title("Sparse top-k basin targeting")
    return savefig(fig, stem, fmt)


def plot_fig16(exp07: pd.DataFrame, exp06: pd.DataFrame, fmt: str) -> list[Path]:
    stem = "fig16_seen_unseen_structured"
    if exp07.empty:
        print(f"Warning: {stem} skipped; no Exp07 data.")
        return []
    rows: list[Record] = []
    main = _main_rows(exp07)
    if not main.empty:
        rows.append(
            {
                "condition": "Exp07 structured seen",
                "accuracy": _mean_metric(main, "seen_transition_accuracy"),
            }
        )
        rows.append(
            {
                "condition": "Exp07 structured unseen",
                "accuracy": _mean_metric(main, "unseen_transition_accuracy"),
            }
        )
    if not exp06.empty:
        coverage = _filter_rows(
            exp06,
            lambda row: (
                row.get("training_mode") == "coverage_sweep"
                and row.get("register_type") == "nearest"
                and row.get("output_mode") == "dense"
            ),
        )
        if not coverage.empty:
            rows.append(
                {
                    "condition": "Exp06 random seen",
                    "accuracy": _mean_metric(coverage, "seen_transition_accuracy"),
                }
            )
            rows.append(
                {
                    "condition": "Exp06 random unseen",
                    "accuracy": _mean_metric(coverage, "unseen_transition_accuracy"),
                }
            )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    labels = [str(row["condition"]) for row in _records(plot_df)]
    heights = [float(row["accuracy"]) for row in _records(plot_df)]
    ax.bar(labels, heights, color="0.35")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("Transition accuracy")
    ax.set_title("Structured grammars vs random transition tables")
    return savefig(fig, stem, fmt)


def plot_transition_accuracy(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    paths.extend(plot_fig01(df, "both"))
    paths.extend(plot_fig01b(df, "both"))
    return paths


def plot_noise_recovery(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    paths.extend(plot_fig02(df, "both"))
    paths.extend(plot_fig02b(df, "both"))
    return paths


def plot_capacity(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    paths.extend(plot_fig03(df, "both"))
    paths.extend(plot_fig03b(df, "both"))
    return paths


def plot_sparse_transitions(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    paths.extend(plot_fig04(df, "both"))
    paths.extend(plot_fig04b(df, "both"))
    return paths


def plot_descriptor_payload(df: pd.DataFrame) -> list[Path]:
    return plot_fig05(df, "both")


def plot_learning_results(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    paths.extend(plot_fig08(df, "both"))
    paths.extend(plot_fig09(df, "both"))
    paths.extend(plot_fig10(df, "both"))
    paths.extend(plot_fig11(df, "both"))
    return paths


def plot_structured_grammar_results(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    paths.extend(plot_fig12(df, "both"))
    paths.extend(plot_fig13(df, "both"))
    paths.extend(plot_fig14(df, "both"))
    paths.extend(plot_fig15(df, "both"))
    paths.extend(plot_fig16(df, pd.DataFrame(), "both"))
    return paths


def _metric_at(
    df: pd.DataFrame,
    metric_col: str,
    filter_col: str,
    target: float,
) -> tuple[float | None, object | None]:
    value = _nearest_value(_sorted_values(df, filter_col), target)
    if value is None:
        return None, None
    return _mean_metric(_rows_for(df, filter_col, value), metric_col), value


def _target_label(target: float, actual: object | None) -> str:
    if actual is None:
        return f"{target:g}"
    actual_float = _to_float(actual)
    if actual_float is None:
        return str(actual)
    if math.isclose(actual_float, target, rel_tol=1e-9, abs_tol=1e-9):
        return f"{target:g}"
    return f"nearest to {target:g} ({actual_float:g})"


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _table_row(cols: list[str]) -> str:
    return " & ".join(_latex_escape(col) for col in cols) + r" \\"


def generate_summary_table(data: dict[str, pd.DataFrame]) -> str:
    rows: list[list[str]] = []
    exp01 = data.get("exp01_transition_accuracy.csv", pd.DataFrame())
    if not exp01.empty:
        nearest = _rows_for(exp01, "register_type", "nearest")
        hopfield = _rows_for(exp01, "register_type", "hopfield")
        rows.append(
            [
                "Exp01 Transition realization",
                "Transition accuracy",
                f"nearest mean = {_fmt(_mean_metric(nearest, 'transition_accuracy'))}",
                f"Hopfield mean = {_fmt(_mean_metric(hopfield, 'transition_accuracy'))}; median = {_fmt(_median_metric(hopfield, 'transition_accuracy'))}",
                "Exact transition works; recurrent cleanup is capacity-limited.",
            ]
        )

    exp02 = data.get("exp02_noise_recovery.csv", pd.DataFrame())
    if not exp02.empty:
        nearest = _rows_for(exp02, "register_type", "nearest")
        hopfield = _rows_for(exp02, "register_type", "hopfield")
        zero_hopfield = _rows_for(hopfield, "flip_fraction", 0.0)
        rows.append(
            [
                "Exp02 Noise recovery",
                "Recovery accuracy",
                f"nearest mean = {_fmt(_mean_metric(nearest, 'recovery_accuracy'))}",
                f"Hopfield mean = {_fmt(_mean_metric(hopfield, 'recovery_accuracy'))}; zero-noise mean = {_fmt(_mean_metric(zero_hopfield, 'recovery_accuracy'))}",
                "Basin recovery appears, but recurrent cleanup degrades under load/noise.",
            ]
        )

    exp03 = data.get("exp03_capacity.csv", pd.DataFrame())
    if not exp03.empty:
        nearest = _rows_for(exp03, "register_type", "nearest")
        hopfield = _rows_for(exp03, "register_type", "hopfield")
        cap0138, cap0138_value = _metric_at(
            hopfield, "recovery_accuracy", "capacity_ratio", 0.138
        )
        cap1, cap1_value = _metric_at(
            hopfield, "recovery_accuracy", "capacity_ratio", 1.0
        )
        cap0138_label = _target_label(0.138, cap0138_value)
        cap1_label = _target_label(1.0, cap1_value)
        rows.append(
            [
                "Exp03 Capacity",
                "Recovery accuracy",
                f"nearest mean = {_fmt(_mean_metric(nearest, 'recovery_accuracy'))}",
                f"Hopfield at m/n={cap0138_label}: {_fmt(cap0138)}; at m/n={cap1_label}: {_fmt(cap1)}",
                "Capacity-ratio limits motivate modular finite-state composition.",
            ]
        )

    exp04 = data.get("exp04_sparse_transitions.csv", pd.DataFrame())
    if not exp04.empty:
        random_noise = _rows_for(exp04, "mode", "random_noise")
        nearest = _rows_for(random_noise, "register_type", "nearest")
        hopfield = _rows_for(random_noise, "register_type", "hopfield")
        near03, near03_value = _metric_at(
            nearest, "sparse_transition_accuracy", "write_fraction", 0.30
        )
        near05, near05_value = _metric_at(
            nearest, "sparse_transition_accuracy", "write_fraction", 0.50
        )
        hop03, hop03_value = _metric_at(
            hopfield, "sparse_transition_accuracy", "write_fraction", 0.30
        )
        hop05, hop05_value = _metric_at(
            hopfield, "sparse_transition_accuracy", "write_fraction", 0.50
        )
        near03_label = _target_label(0.30, near03_value)
        near05_label = _target_label(0.50, near05_value)
        hop03_label = _target_label(0.30, hop03_value)
        hop05_label = _target_label(0.50, hop05_value)
        rows.append(
            [
                "Exp04 Sparse transitions",
                "Sparse transition accuracy",
                f"nearest random-noise at {near03_label}: {_fmt(near03)}; at {near05_label}: {_fmt(near05)}",
                f"Hopfield random-noise at {hop03_label}: {_fmt(hop03)}; at {hop05_label}: {_fmt(hop05)}",
                "Partial writes can target basins; keep-current exposes old-state inertia.",
            ]
        )

    exp05 = data.get("exp05_descriptor_payload.csv", pd.DataFrame())
    if not exp05.empty:
        cases = _filter_rows(
            exp05,
            lambda row: (
                row.get("case_name") != "descriptor_corruption"
                and _to_float(row.get("success")) is not None
            ),
        )
        rows.append(
            [
                "Exp05 Descriptor/payload",
                "Protocol-case success",
                f"case success = {_fmt(_mean_metric(cases, 'success'))}",
                "Executable witness, not a performance benchmark.",
                "Descriptors select operations independently of payload content.",
            ]
        )

    exp06 = data.get("exp06_learned_transitions.csv", pd.DataFrame())
    if not exp06.empty:
        exact = _filter_rows(
            exp06,
            lambda row: (
                row.get("feature_mode") == "exact_pair"
                and row.get("training_mode") == "full_table"
                and row.get("output_mode") == "dense"
                and _to_float(row.get("epochs")) == 1.0
            ),
        )
        nearest = _rows_for(exact, "register_type", "nearest")
        hopfield = _rows_for(exact, "register_type", "hopfield")
        rows.append(
            [
                "Exp06 Learned transitions",
                "Learned transition accuracy",
                f"nearest one-epoch mean = {_fmt(_mean_metric(nearest, 'all_transition_accuracy'))}",
                f"Hopfield one-epoch mean = {_fmt(_mean_metric(hopfield, 'all_transition_accuracy'))}",
                "Transition associations can be learned from demonstrations; cleanup still limits performance.",
            ]
        )

    exp07 = data.get("exp07_structured_grammar_learning.csv", pd.DataFrame())
    if not exp07.empty:
        main = _main_rows(exp07)
        exact = _filter_rows(
            main,
            lambda row: (
                row.get("feature_mode") == "exact_pair"
                and row.get("register_type") == "nearest"
            ),
        )
        rows.append(
            [
                "Exp07 Structured grammar learning",
                "String accuracy",
                f"nearest test mean = {_fmt(_mean_metric(exact, 'test_string_accuracy'))}",
                f"length-generalization mean = {_fmt(_mean_metric(exact, 'length_generalization_accuracy'))}",
                "Structured grammars test reusable transition acquisition beyond random FSM memorization.",
            ]
        )

    lines = [
        r"\begin{tabular}{p{0.18\linewidth} p{0.14\linewidth} p{0.18\linewidth} p{0.22\linewidth} p{0.22\linewidth}}",
        r"\hline",
        _table_row(
            [
                "Experiment",
                "Main metric",
                "Ideal cleanup result",
                "Hopfield/result",
                "Interpretation",
            ]
        ),
        r"\hline",
    ]
    lines.extend(_table_row(row) for row in rows)
    lines.extend([r"\hline", r"\end{tabular}"])
    table = "\n".join(lines) + "\n"
    ensure_dirs()
    path = TABLE_DIR / "simulation_summary.tex"
    path.write_text(table)
    print(table)
    print(f"Saved LaTeX table to {path}")
    return table


def plot_all_from_csv(fmt: str = "both") -> list[Path]:
    ensure_dirs()
    configure_style()
    data: dict[str, pd.DataFrame] = {
        name: load_csv(name)
        for name in [
            "exp01_transition_accuracy.csv",
            "exp02_noise_recovery.csv",
            "exp03_capacity.csv",
            "exp04_sparse_transitions.csv",
            "exp05_descriptor_payload.csv",
            "exp06_learned_transitions.csv",
            "exp07_structured_grammar_learning.csv",
        ]
    }

    paths: list[Path] = []
    exp01 = data["exp01_transition_accuracy.csv"]
    exp02 = data["exp02_noise_recovery.csv"]
    exp03 = data["exp03_capacity.csv"]
    exp04 = data["exp04_sparse_transitions.csv"]
    exp05 = data["exp05_descriptor_payload.csv"]
    exp06 = data["exp06_learned_transitions.csv"]
    exp07 = data["exp07_structured_grammar_learning.csv"]

    paths.extend(plot_fig01(exp01, fmt))
    paths.extend(plot_fig01b(exp01, fmt))
    paths.extend(plot_fig02(exp02, fmt))
    paths.extend(plot_fig02b(exp02, fmt))
    paths.extend(plot_fig03(exp03, fmt))
    paths.extend(plot_fig03b(exp03, fmt))
    paths.extend(plot_fig04(exp04, fmt))
    paths.extend(plot_fig04b(exp04, fmt))
    paths.extend(plot_fig05(exp05, fmt))
    paths.extend(plot_fig06(exp01, fmt))
    paths.extend(plot_fig07(exp03, fmt))
    paths.extend(plot_fig08(exp06, fmt))
    paths.extend(plot_fig09(exp06, fmt))
    paths.extend(plot_fig10(exp06, fmt))
    paths.extend(plot_fig11(exp06, fmt))
    paths.extend(plot_fig12(exp07, fmt))
    paths.extend(plot_fig13(exp07, fmt))
    paths.extend(plot_fig14(exp07, fmt))
    paths.extend(plot_fig15(exp07, fmt))
    paths.extend(plot_fig16(exp07, exp06, fmt))
    generate_summary_table(data)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "both"],
        default="both",
        help="Figure output format.",
    )
    args = parser.parse_args()
    paths = plot_all_from_csv(fmt=cast(str, args.format))
    print(f"Saved {len(paths)} figure files to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
