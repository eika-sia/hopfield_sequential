# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false, reportIndexIssue=false
"""Statistical summaries for BioLogic State Machine simulation CSVs."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import numpy as np
import pandas as pd

DEFAULT_CSV_DIR: Path = Path("results/csv")
DEFAULT_OUT_DIR: Path = Path("results/statistics")
TABLE_DIR: Path = Path("results/tables")
REPORT_MD: str = "statistics_report.md"
REPORT_TXT: str = "statistics_report.txt"
LATEX_TABLE: str = "simulation_statistics_table.tex"
ROUND_DIGITS: int = 3
CLASSICAL_CAPACITY: float = 0.138

Record = dict[str, Any]
LatexRow = tuple[str, str, str, str]

CSV_NAMES: dict[str, str] = {
    "exp01": "exp01_transition_accuracy.csv",
    "exp02": "exp02_noise_recovery.csv",
    "exp03": "exp03_capacity.csv",
    "exp04": "exp04_sparse_transitions.csv",
    "exp05": "exp05_descriptor_payload.csv",
    "exp06": "exp06_learned_transitions.csv",
}


@dataclass
class AnalysisResult:
    key: str
    title: str
    loaded: bool
    source_path: Path
    summary_path: Path | None = None
    markdown_lines: list[str] = field(default_factory=list)
    stdout_line: str | None = None
    paper_values: list[str] = field(default_factory=list)
    latex_rows: list[LatexRow] = field(default_factory=list)


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def ensure_dirs(
    out_dir: Path = DEFAULT_OUT_DIR,
    table_dir: Path = TABLE_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"missing CSV {path}; skipping dependent analysis.")
        return None
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


def _to_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _has_required(df: pd.DataFrame, columns: Sequence[str], label: str) -> bool:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        warn(f"{label} skipped; missing columns: {', '.join(missing)}")
        return False
    if df.empty:
        warn(f"{label} skipped; input data is empty.")
        return False
    return True


def _filter_rows(df: pd.DataFrame, predicate: Callable[[Record], bool]) -> pd.DataFrame:
    return pd.DataFrame([row for row in _records(df) if predicate(row)])


def _non_nan_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return _filter_rows(df, lambda row: _to_float(row.get(column)) is not None)


def _sort_key(value: object) -> tuple[int, float | str]:
    numeric = _to_float(value)
    if numeric is not None:
        return (0, numeric)
    return (1, str(value))


def _unique_sorted(df: pd.DataFrame, column: str) -> list[object]:
    values: list[object] = []
    seen: set[object] = set()
    for row in _records(df):
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
            continue
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return sorted(values, key=_sort_key)


def _metric_values(df: pd.DataFrame, metric_col: str) -> list[float]:
    values: list[float] = []
    for row in _records(df):
        value = _to_float(row.get(metric_col))
        if value is not None:
            values.append(value)
    return values


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _sem(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return _std(values) / math.sqrt(len(values))


def _summarize_values(values: Sequence[float], metric_col: str) -> Record:
    if not values:
        return {
            "count": 0,
            f"{metric_col}_mean": math.nan,
            f"{metric_col}_std": math.nan,
            f"{metric_col}_sem": math.nan,
            f"{metric_col}_min": math.nan,
            f"{metric_col}_median": math.nan,
            f"{metric_col}_max": math.nan,
        }
    return {
        "count": len(values),
        f"{metric_col}_mean": float(np.mean(values)),
        f"{metric_col}_std": _std(values),
        f"{metric_col}_sem": _sem(values),
        f"{metric_col}_min": float(np.min(values)),
        f"{metric_col}_median": float(np.median(values)),
        f"{metric_col}_max": float(np.max(values)),
    }


def mean_std_min_max(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_col: str,
) -> pd.DataFrame:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in _records(df):
        value = _to_float(row.get(metric_col))
        if value is None:
            continue
        key = tuple(row.get(column) for column in group_cols)
        groups[key].append(value)

    rows: list[Record] = []
    for key in sorted(groups, key=lambda item: tuple(_sort_key(part) for part in item)):
        out: Record = {column: key[index] for index, column in enumerate(group_cols)}
        out.update(_summarize_values(groups[key], metric_col))
        rows.append(out)
    return pd.DataFrame(rows)


def mean_sem(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_col: str,
) -> pd.DataFrame:
    summary = mean_std_min_max(df, group_cols, metric_col)
    mean_col = f"{metric_col}_mean"
    if mean_col in summary.columns:
        summary[metric_col] = summary[mean_col]
    return summary


def format_float(x: float) -> str:
    if math.isnan(x):
        return "n/a"
    return f"{x:.{ROUND_DIGITS}f}"


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else format_float(value)


def _fmt_with_actual(value: float | None, actual: float | None, target: float) -> str:
    if value is None:
        return "n/a"
    suffix = "" if actual is not None and math.isclose(actual, target) else f" at closest {format_float(actual or math.nan)}"
    return f"{format_float(value)}{suffix}"


def nearest_param_value(df: pd.DataFrame, column: str, target: float) -> float | None:
    values: list[float] = []
    for value in _unique_sorted(df, column):
        numeric = _to_float(value)
        if numeric is not None:
            values.append(numeric)
    if not values:
        return None
    return min(values, key=lambda value: abs(value - target))


def _filter_equal(df: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    target_float = _to_float(value)

    def predicate(row: Record) -> bool:
        current = row.get(column)
        current_float = _to_float(current)
        if target_float is not None and current_float is not None:
            return math.isclose(current_float, target_float)
        return current == value

    return _filter_rows(df, predicate)


def _filter_many(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    result = df
    for column, value in filters.items():
        result = _filter_equal(result, column, value)
    return result


def _mean_metric(df: pd.DataFrame, metric_col: str) -> float | None:
    return _mean(_metric_values(df, metric_col))


def _stats_for(df: pd.DataFrame, metric_col: str) -> Record:
    return _summarize_values(_metric_values(df, metric_col), metric_col)


def _mean_at(
    df: pd.DataFrame,
    metric_col: str,
    fixed_filters: dict[str, object],
    param_col: str,
    target: float,
) -> tuple[float | None, float | None]:
    filtered = _filter_many(df, fixed_filters)
    actual = nearest_param_value(filtered, param_col, target)
    if actual is None:
        return None, None
    value = _mean_metric(_filter_equal(filtered, param_col, actual), metric_col)
    return value, actual


def _mean_for_filters(
    df: pd.DataFrame,
    metric_col: str,
    filters: dict[str, object],
) -> float | None:
    return _mean_metric(_filter_many(df, filters), metric_col)


def _lowest_rows(df: pd.DataFrame, metric_col: str, limit: int) -> pd.DataFrame:
    rows = [
        row
        for row in _records(df)
        if _to_float(row.get(metric_col)) is not None
    ]
    rows.sort(key=lambda row: cast(float, _to_float(row.get(metric_col))))
    return pd.DataFrame(rows[:limit])


def _highest_rows(df: pd.DataFrame, metric_col: str, limit: int) -> pd.DataFrame:
    rows = [
        row
        for row in _records(df)
        if _to_float(row.get(metric_col)) is not None
    ]
    rows.sort(key=lambda row: cast(float, _to_float(row.get(metric_col))), reverse=True)
    return pd.DataFrame(rows[:limit])


def _pivot_mean(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    metric_col: str,
) -> pd.DataFrame:
    groups: dict[tuple[object, object], list[float]] = defaultdict(list)
    for row in _records(df):
        value = _to_float(row.get(metric_col))
        row_value = row.get(row_col)
        col_value = row.get(col_col)
        if value is None or row_value is None or col_value is None:
            continue
        groups[(row_value, col_value)].append(value)

    row_values = sorted({key[0] for key in groups}, key=_sort_key)
    col_values = sorted({key[1] for key in groups}, key=_sort_key)
    table_rows: list[Record] = []
    for row_value in row_values:
        out: Record = {row_col: row_value}
        for col_value in col_values:
            values = groups.get((row_value, col_value), [])
            out[str(col_value)] = float(np.mean(values)) if values else math.nan
        table_rows.append(out)
    return pd.DataFrame(table_rows)


def _metric_by_param(
    df: pd.DataFrame,
    fixed_filters: dict[str, object],
    param_col: str,
    metric_col: str,
) -> pd.DataFrame:
    rows: list[Record] = []
    filtered = _filter_many(df, fixed_filters)
    for value in _unique_sorted(filtered, param_col):
        mean = _mean_metric(_filter_equal(filtered, param_col, value), metric_col)
        numeric = _to_float(value)
        rows.append(
            {
                param_col: numeric if numeric is not None else value,
                metric_col: mean if mean is not None else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _threshold_below(
    df: pd.DataFrame,
    param_col: str,
    metric_col: str,
    threshold: float,
) -> float | None:
    table = _metric_by_param(df, {}, param_col, metric_col)
    rows = sorted(_records(table), key=lambda row: cast(float, _to_float(row[param_col])))
    for row in rows:
        metric = _to_float(row.get(metric_col))
        param = _to_float(row.get(param_col))
        if metric is not None and param is not None and metric < threshold:
            return param
    return None


def _threshold_reaches(
    df: pd.DataFrame,
    fixed_filters: dict[str, object],
    param_col: str,
    metric_col: str,
    threshold: float,
) -> float | None:
    table = _metric_by_param(df, fixed_filters, param_col, metric_col)
    rows = sorted(_records(table), key=lambda row: cast(float, _to_float(row[param_col])))
    for row in rows:
        metric = _to_float(row.get(metric_col))
        param = _to_float(row.get(param_col))
        if metric is not None and param is not None and metric >= threshold:
            return param
    return None


def _summary_frame(df: pd.DataFrame, summary_type: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.insert(0, "summary_type", summary_type)
    out.columns = [str(column) for column in out.columns]
    return out


def _write_summary(path: Path, frames: Sequence[pd.DataFrame]) -> Path:
    non_empty = [frame for frame in frames if not frame.empty]
    if non_empty:
        summary = pd.concat(non_empty, ignore_index=True, sort=False)
    else:
        summary = pd.DataFrame()
    summary.to_csv(path, index=False)
    return path


def _markdown_value(value: object) -> str:
    number = _to_float(value)
    if number is not None:
        return format_float(number)
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def _markdown_table(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    max_rows: int = 20,
) -> list[str]:
    if df.empty:
        return ["_No data._"]
    selected_columns = list(columns) if columns is not None else [str(col) for col in df.columns]
    available_columns = [column for column in selected_columns if column in df.columns]
    if not available_columns:
        return ["_No requested columns available._"]
    rows = _records(df.head(max_rows))
    output: list[str] = [
        "| " + " | ".join(available_columns) + " |",
        "| " + " | ".join(["---"] * len(available_columns)) + " |",
    ]
    for row in rows:
        output.append(
            "| "
            + " | ".join(_markdown_value(row.get(column)) for column in available_columns)
            + " |"
        )
    if len(df.index) > max_rows:
        output.append(f"\n_Showing {max_rows} of {len(df.index)} rows._")
    return output


def _add_table(
    lines: list[str],
    title: str,
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    max_rows: int = 20,
) -> None:
    lines.extend(["", f"### {title}", ""])
    lines.extend(_markdown_table(df, columns=columns, max_rows=max_rows))


def _latex_escape(text: str) -> str:
    replacements: dict[str, str] = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for original, replacement in replacements.items():
        escaped = escaped.replace(original, replacement)
    return escaped


def write_latex_table(path: Path, rows: Sequence[LatexRow]) -> None:
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Experiment & Key statistic & Result & Interpretation \\",
        r"\midrule",
    ]
    for experiment, statistic, result, interpretation in rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(experiment),
                    _latex_escape(statistic),
                    _latex_escape(result),
                    _latex_escape(interpretation),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_report(
    path: Path,
    loaded_files: Sequence[Path],
    missing_files: Sequence[Path],
    results: Sequence[AnalysisResult],
    paper_values: Sequence[str],
) -> None:
    lines: list[str] = [
        "# Simulation Statistics Report",
        "",
        "## Data files",
        "",
    ]
    if loaded_files:
        lines.append("Loaded CSVs:")
        lines.extend(f"- `{path_item}`" for path_item in loaded_files)
    if missing_files:
        lines.append("")
        lines.append("Missing CSVs:")
        lines.extend(f"- `{path_item}`" for path_item in missing_files)
    if not loaded_files and not missing_files:
        lines.append("_No CSV files were found._")

    for result in results:
        lines.extend(["", *result.markdown_lines])

    lines.extend(["", "## Cross-experiment interpretation", ""])
    lines.extend(results_cheat_sheet())

    lines.extend(["", "## Paper-ready numbers", ""])
    if paper_values:
        lines.extend(f"- {value}" for value in paper_values)
    else:
        lines.append("_No paper-ready values were available._")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _register_stats_line(
    label: str,
    stats: Record,
    metric_col: str,
    include_median: bool = False,
) -> str:
    parts = [
        f"mean={_markdown_value(stats.get(f'{metric_col}_mean'))}",
        f"std={_markdown_value(stats.get(f'{metric_col}_std'))}",
    ]
    if include_median:
        parts.append(f"median={_markdown_value(stats.get(f'{metric_col}_median'))}")
    parts.extend(
        [
            f"min={_markdown_value(stats.get(f'{metric_col}_min'))}",
            f"max={_markdown_value(stats.get(f'{metric_col}_max'))}",
        ]
    )
    return f"{label}: " + ", ".join(parts)


def _stats_record_for_register(
    overall: pd.DataFrame,
    register_type: str,
    metric_col: str,
) -> Record:
    filtered = _filter_equal(overall, "register_type", register_type)
    rows = _records(filtered)
    if rows:
        return rows[0]
    return _summarize_values([], metric_col)


def analyze_exp01(csv_dir: Path, out_dir: Path) -> AnalysisResult:
    source = csv_dir / CSV_NAMES["exp01"]
    df = load_csv(source)
    result = AnalysisResult("exp01", "Exp01: Transition realization", df is not None, source)
    if df is None:
        return result

    required = [
        "experiment",
        "seed",
        "num_states",
        "num_inputs",
        "state_dim",
        "input_dim",
        "register_type",
        "transition_accuracy",
        "mean_abs_overlap",
    ]
    if not _has_required(df, required, "Exp01"):
        return result

    metric = "transition_accuracy"
    nearest = _filter_equal(df, "register_type", "nearest")
    hopfield = _filter_equal(df, "register_type", "hopfield")
    overall = mean_std_min_max(df, ["register_type"], metric)
    by_dim_state = mean_sem(df, ["register_type", "state_dim", "num_states"], metric)
    hopfield_pivot = _pivot_mean(hopfield, "state_dim", "num_states", metric)
    nearest_failures = _filter_rows(nearest, lambda row: row.get(metric) != 1.0)
    worst = _lowest_rows(hopfield, metric, 20)

    summary_path = out_dir / "exp01_transition_accuracy_summary.csv"
    result.summary_path = _write_summary(
        summary_path,
        [
            _summary_frame(overall, "overall_by_register_type"),
            _summary_frame(by_dim_state, "by_register_type_state_dim_num_states"),
            _summary_frame(hopfield_pivot, "hopfield_state_dim_x_num_states_pivot"),
            _summary_frame(nearest_failures, "nearest_failures"),
            _summary_frame(worst, "worst_hopfield_rows"),
        ],
    )

    nearest_stats = _stats_record_for_register(overall, "nearest", metric)
    hopfield_stats = _stats_record_for_register(overall, "hopfield", metric)
    hop_64_64 = _mean_for_filters(
        df,
        metric,
        {"register_type": "hopfield", "state_dim": 64, "num_states": 64},
    )
    hop_512_64 = _mean_for_filters(
        df,
        metric,
        {"register_type": "hopfield", "state_dim": 512, "num_states": 64},
    )
    hop_64_4 = _mean_for_filters(
        df,
        metric,
        {"register_type": "hopfield", "state_dim": 64, "num_states": 4},
    )
    nearest_mean = _to_float(nearest_stats.get(f"{metric}_mean"))
    hopfield_mean = _to_float(hopfield_stats.get(f"{metric}_mean"))

    lines: list[str] = ["## Exp01: Transition realization", ""]
    lines.extend(
        [
            "- Nearest-attractor cleanup tests the exact transition construction under ideal basin decoding.",
            "- Hopfield cleanup tests a concrete recurrent implementation.",
            "- Hopfield failures concentrated in low-dimensional, high-load regimes.",
            f"- Nearest correctness check: {'passed' if nearest_failures.empty else 'failed'} ({len(nearest_failures.index)} failing rows).",
            f"- {_register_stats_line('Nearest transition accuracy', nearest_stats, metric)}.",
            f"- {_register_stats_line('Hopfield transition accuracy', hopfield_stats, metric, include_median=True)}.",
            f"- Hopfield mean accuracy at state_dim=64,num_states=64: {_fmt_optional(hop_64_64)}.",
            f"- Hopfield mean accuracy at state_dim=512,num_states=64: {_fmt_optional(hop_512_64)}.",
            f"- Hopfield mean accuracy at state_dim=64,num_states=4: {_fmt_optional(hop_64_4)}.",
        ]
    )
    _add_table(lines, "Overall summary by register type", overall)
    _add_table(lines, "Hopfield degradation pivot", hopfield_pivot, max_rows=30)
    _add_table(
        lines,
        "Worst Hopfield rows",
        worst,
        columns=["seed", "state_dim", "num_states", "num_inputs", metric, "mean_abs_overlap"],
    )
    if not nearest_failures.empty:
        _add_table(
            lines,
            "Nearest failures",
            nearest_failures,
            columns=["seed", "state_dim", "num_states", "num_inputs", metric],
        )
    result.markdown_lines = lines
    result.stdout_line = (
        f"Exp01: nearest mean={_fmt_optional(nearest_mean)}, "
        f"Hopfield mean={_fmt_optional(hopfield_mean)}"
    )
    result.paper_values = [
        f"Exp01 nearest transition accuracy mean/std/min/max: "
        f"{_register_stats_line('', nearest_stats, metric).lstrip(': ')}",
        f"Exp01 Hopfield transition accuracy mean/std/median/min/max: "
        f"{_register_stats_line('', hopfield_stats, metric, include_median=True).lstrip(': ')}",
        f"Exp01 Hopfield mean accuracy at state_dim=64,num_states=64: {_fmt_optional(hop_64_64)}",
        f"Exp01 Hopfield mean accuracy at state_dim=512,num_states=64: {_fmt_optional(hop_512_64)}",
    ]
    result.latex_rows = [
        (
            "Exp01",
            "Nearest transition accuracy",
            _fmt_optional(nearest_mean),
            "Exact transition construction is correct under ideal cleanup",
        ),
        (
            "Exp01",
            "Hopfield transition accuracy",
            _fmt_optional(hopfield_mean),
            "Recurrent cleanup is capacity-dependent",
        ),
    ]
    return result


def analyze_exp02(csv_dir: Path, out_dir: Path) -> AnalysisResult:
    source = csv_dir / CSV_NAMES["exp02"]
    df = load_csv(source)
    result = AnalysisResult("exp02", "Exp02: Noise recovery", df is not None, source)
    if df is None:
        return result

    required = [
        "experiment",
        "seed",
        "num_states",
        "state_dim",
        "register_type",
        "flip_fraction",
        "trials_per_state",
        "recovery_accuracy",
        "mean_abs_overlap",
    ]
    if not _has_required(df, required, "Exp02"):
        return result

    metric = "recovery_accuracy"
    overall = mean_std_min_max(df, ["register_type"], metric)
    by_flip = mean_sem(df, ["register_type", "flip_fraction"], metric)
    by_dim_state = mean_sem(df, ["register_type", "state_dim", "num_states"], metric)
    nearest_by_flip = _metric_by_param(df, {"register_type": "nearest"}, "flip_fraction", metric)
    hopfield_by_flip = _metric_by_param(df, {"register_type": "hopfield"}, "flip_fraction", metric)
    hopfield_dim_state_pivot = _pivot_mean(
        _filter_equal(df, "register_type", "hopfield"),
        "state_dim",
        "num_states",
        metric,
    )
    zero_noise = _filter_equal(df, "flip_fraction", 0.0)
    zero_by_register = mean_std_min_max(zero_noise, ["register_type"], metric)
    zero_by_dim_state = mean_sem(
        zero_noise,
        ["register_type", "state_dim", "num_states"],
        metric,
    )
    degradation = _noise_degradation_table(df, metric)
    worst = _lowest_rows(df, metric, 20)

    summary_path = out_dir / "exp02_noise_recovery_summary.csv"
    result.summary_path = _write_summary(
        summary_path,
        [
            _summary_frame(overall, "overall_by_register_type"),
            _summary_frame(by_flip, "by_register_type_flip_fraction"),
            _summary_frame(by_dim_state, "by_register_type_state_dim_num_states"),
            _summary_frame(nearest_by_flip, "nearest_recovery_by_flip_fraction"),
            _summary_frame(hopfield_by_flip, "hopfield_recovery_by_flip_fraction"),
            _summary_frame(hopfield_dim_state_pivot, "hopfield_state_dim_x_num_states_pivot"),
            _summary_frame(zero_by_register, "zero_noise_by_register_type"),
            _summary_frame(zero_by_dim_state, "zero_noise_by_register_type_state_dim_num_states"),
            _summary_frame(degradation, "noise_degradation"),
            _summary_frame(worst, "worst_rows"),
        ],
    )

    nearest_values = [
        _mean_at(df, metric, {"register_type": "nearest"}, "flip_fraction", target)
        for target in [0.0, 0.2, 0.3, 0.4]
    ]
    hopfield_values = [
        _mean_at(df, metric, {"register_type": "hopfield"}, "flip_fraction", target)
        for target in [0.0, 0.2, 0.3, 0.4]
    ]
    hop_64_64 = _mean_for_filters(
        df,
        metric,
        {"register_type": "hopfield", "state_dim": 64, "num_states": 64},
    )
    hop_512_64 = _mean_for_filters(
        df,
        metric,
        {"register_type": "hopfield", "state_dim": 512, "num_states": 64},
    )
    nearest_030, nearest_030_actual = _mean_at(
        df,
        metric,
        {"register_type": "nearest"},
        "flip_fraction",
        0.30,
    )
    hopfield_030, hopfield_030_actual = _mean_at(
        df,
        metric,
        {"register_type": "hopfield"},
        "flip_fraction",
        0.30,
    )

    lines: list[str] = ["## Exp02: Noise recovery", ""]
    lines.extend(
        [
            "- Nearest cleanup shows ideal basin robustness.",
            "- Hopfield cleanup combines baseline attractor stability and robustness to corruption.",
            "- Hopfield recovery below 1.0 at zero noise indicates that some stored patterns are unstable under recurrent Hebbian dynamics in high-load regimes.",
            "- The flip-fraction summaries keep nearest and Hopfield separate to avoid hiding overloaded regimes.",
        ]
    )
    lines.extend(_paper_param_bullets("Nearest recovery", [0.0, 0.2, 0.3, 0.4], nearest_values))
    lines.extend(_paper_param_bullets("Hopfield recovery", [0.0, 0.2, 0.3, 0.4], hopfield_values))
    lines.extend(
        [
            f"- Hopfield recovery for state_dim=64,num_states=64: {_fmt_optional(hop_64_64)}.",
            f"- Hopfield recovery for state_dim=512,num_states=64: {_fmt_optional(hop_512_64)}.",
        ]
    )
    _add_table(lines, "Overall summary by register type", overall)
    _add_table(lines, "Noise degradation", degradation)
    _add_table(lines, "Hopfield zero-noise recovery by state_dim,num_states", _filter_equal(zero_by_dim_state, "register_type", "hopfield"), max_rows=60)
    _add_table(lines, "Worst recovery rows", worst, columns=["seed", "register_type", "state_dim", "num_states", "flip_fraction", metric])

    result.markdown_lines = lines
    result.stdout_line = (
        f"Exp02: nearest@0.30={_fmt_with_actual(nearest_030, nearest_030_actual, 0.30)}, "
        f"Hopfield@0.30={_fmt_with_actual(hopfield_030, hopfield_030_actual, 0.30)}"
    )
    result.paper_values = [
        *[
            f"Exp02 nearest recovery at flip_fraction={target:.2f}: {_fmt_with_actual(value, actual, target)}"
            for target, (value, actual) in zip([0.0, 0.2, 0.3, 0.4], nearest_values)
        ],
        *[
            f"Exp02 Hopfield recovery at flip_fraction={target:.2f}: {_fmt_with_actual(value, actual, target)}"
            for target, (value, actual) in zip([0.0, 0.2, 0.3, 0.4], hopfield_values)
        ],
        f"Exp02 Hopfield zero-noise recovery by state_dim,num_states: {_zero_noise_text(zero_by_dim_state, 'hopfield', metric)}",
    ]
    result.latex_rows = [
        (
            "Exp02",
            "Nearest recovery at 30% corruption",
            _fmt_with_actual(nearest_030, nearest_030_actual, 0.30),
            "Ideal cleanup remains robust under corruption",
        ),
        (
            "Exp02",
            "Hopfield recovery at 30% corruption",
            _fmt_with_actual(hopfield_030, hopfield_030_actual, 0.30),
            "Recurrent cleanup has stability and noise limits",
        ),
    ]
    return result


def _noise_degradation_table(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows: list[Record] = []
    for register_type in _unique_sorted(df, "register_type"):
        out: Record = {"register_type": register_type}
        values: dict[float, float | None] = {}
        for target in [0.0, 0.2, 0.3, 0.4]:
            value, actual = _mean_at(
                df,
                metric_col,
                {"register_type": register_type},
                "flip_fraction",
                target,
            )
            values[target] = value
            out[f"recovery_at_{target:.2f}"] = value if value is not None else math.nan
            out[f"actual_flip_for_{target:.2f}"] = actual if actual is not None else math.nan
        if values[0.0] is not None and values[0.4] is not None:
            out["degradation_0_to_40"] = cast(float, values[0.0]) - cast(float, values[0.4])
        else:
            out["degradation_0_to_40"] = math.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _paper_param_bullets(
    label: str,
    targets: Sequence[float],
    values: Sequence[tuple[float | None, float | None]],
) -> list[str]:
    return [
        f"- {label} at {target:.2f}: {_fmt_with_actual(value, actual, target)}."
        for target, (value, actual) in zip(targets, values)
    ]


def _param_value_text(
    df: pd.DataFrame,
    fixed_filters: dict[str, object],
    param_col: str,
    metric_col: str,
) -> str:
    table = _metric_by_param(df, fixed_filters, param_col, metric_col)
    parts: list[str] = []
    for row in _records(table):
        param = _to_float(row.get(param_col))
        value = _to_float(row.get(metric_col))
        if param is not None and value is not None:
            parts.append(f"{param:.3f}={format_float(value)}")
    return ", ".join(parts) if parts else "n/a"


def _target_value_text(
    table: pd.DataFrame,
    fixed_filters: dict[str, object],
    target_col: str,
    metric_col: str,
) -> str:
    filtered = _filter_many(table, fixed_filters)
    rows = sorted(_records(filtered), key=lambda row: cast(float, _to_float(row.get(target_col))))
    parts: list[str] = []
    for row in rows:
        target = _to_float(row.get(target_col))
        value = _to_float(row.get(metric_col))
        if target is not None and value is not None:
            parts.append(f"{target:.2f}={format_float(value)}")
    return ", ".join(parts) if parts else "n/a"


def _zero_noise_text(table: pd.DataFrame, register_type: str, metric_col: str) -> str:
    filtered = _filter_equal(table, "register_type", register_type)
    rows = sorted(
        _records(filtered),
        key=lambda row: (
            cast(float, _to_float(row.get("state_dim"))),
            cast(float, _to_float(row.get("num_states"))),
        ),
    )
    parts: list[str] = []
    for row in rows:
        state_dim = _to_float(row.get("state_dim"))
        num_states = _to_float(row.get("num_states"))
        value = _to_float(row.get(f"{metric_col}_mean"))
        if state_dim is not None and num_states is not None and value is not None:
            parts.append(f"n={int(state_dim)},m={int(num_states)}:{format_float(value)}")
    return "; ".join(parts) if parts else "n/a"


def _mode_difference_text(table: pd.DataFrame, register_type: str) -> str:
    filtered = _filter_equal(table, "register_type", register_type)
    rows = sorted(_records(filtered), key=lambda row: cast(float, _to_float(row.get("write_fraction"))))
    parts: list[str] = []
    for row in rows:
        write_fraction = _to_float(row.get("write_fraction"))
        difference = _to_float(row.get("random_noise_minus_keep_current"))
        if write_fraction is not None and difference is not None:
            parts.append(f"{write_fraction:.2f}={format_float(difference)}")
    return ", ".join(parts) if parts else "n/a"


def _case_success_text(table: pd.DataFrame) -> str:
    parts: list[str] = []
    for row in _records(table):
        case_name = row.get("case_name")
        value = _to_float(row.get("success_mean"))
        if case_name is not None and value is not None:
            parts.append(f"{case_name}={format_float(value)}")
    return ", ".join(parts) if parts else "n/a"


def _max_epoch_df(df: pd.DataFrame) -> pd.DataFrame:
    epochs = [
        value
        for value in (_to_float(row.get("epochs")) for row in _records(df))
        if value is not None
    ]
    if not epochs:
        return pd.DataFrame()
    max_epoch = max(epochs)
    return _filter_rows(
        df,
        lambda row: (
            (epoch := _to_float(row.get("epochs"))) is not None
            and math.isclose(epoch, max_epoch)
        ),
    )


def analyze_exp03(csv_dir: Path, out_dir: Path) -> AnalysisResult:
    source = csv_dir / CSV_NAMES["exp03"]
    df = load_csv(source)
    result = AnalysisResult("exp03", "Exp03: Capacity limits", df is not None, source)
    if df is None:
        return result

    required = [
        "experiment",
        "seed",
        "state_dim",
        "num_states",
        "capacity_ratio",
        "register_type",
        "flip_fraction",
        "recovery_accuracy",
        "mean_abs_overlap",
    ]
    if not _has_required(df, required, "Exp03"):
        return result

    metric = "recovery_accuracy"
    nearest = _filter_equal(df, "register_type", "nearest")
    hopfield = _filter_equal(df, "register_type", "hopfield")
    overall = mean_std_min_max(df, ["register_type"], metric)
    by_ratio = mean_sem(df, ["register_type", "capacity_ratio"], metric)
    by_dim_ratio = mean_sem(df, ["register_type", "state_dim", "capacity_ratio"], metric)
    hopfield_by_ratio = _metric_by_param(df, {"register_type": "hopfield"}, "capacity_ratio", metric)
    nearest_by_ratio = _metric_by_param(df, {"register_type": "nearest"}, "capacity_ratio", metric)
    hopfield_heatmap = _pivot_mean(hopfield, "state_dim", "capacity_ratio", metric)
    capacity_comparison = _capacity_comparison_table(df, metric)
    thresholds = _capacity_threshold_table(hopfield, metric)
    worst = _lowest_rows(hopfield, metric, 20)

    summary_path = out_dir / "exp03_capacity_summary.csv"
    result.summary_path = _write_summary(
        summary_path,
        [
            _summary_frame(overall, "overall_by_register_type"),
            _summary_frame(by_ratio, "by_register_type_capacity_ratio"),
            _summary_frame(by_dim_ratio, "by_register_type_state_dim_capacity_ratio"),
            _summary_frame(hopfield_by_ratio, "hopfield_recovery_by_capacity_ratio"),
            _summary_frame(nearest_by_ratio, "nearest_recovery_by_capacity_ratio"),
            _summary_frame(hopfield_heatmap, "hopfield_state_dim_x_capacity_ratio_pivot"),
            _summary_frame(capacity_comparison, "classical_capacity_comparison"),
            _summary_frame(thresholds, "hopfield_threshold_estimates"),
            _summary_frame(worst, "worst_hopfield_rows"),
        ],
    )

    nearest_mean = _mean_metric(nearest, metric)
    hopfield_mean = _mean_metric(hopfield, metric)
    hopfield_0138, actual_0138 = _mean_at(
        df,
        metric,
        {"register_type": "hopfield"},
        "capacity_ratio",
        CLASSICAL_CAPACITY,
    )
    hopfield_1000, actual_1000 = _mean_at(
        df,
        metric,
        {"register_type": "hopfield"},
        "capacity_ratio",
        1.0,
    )
    below_090 = _threshold_below(hopfield, "capacity_ratio", metric, 0.90)
    below_075 = _threshold_below(hopfield, "capacity_ratio", metric, 0.75)

    lines: list[str] = ["## Exp03: Capacity limits", ""]
    lines.extend(
        [
            "- This is the clearest capacity-limit experiment.",
            "- Nearest cleanup is an ideal control and remains perfect.",
            "- Hopfield recovery is high below/near classical capacity and degrades above it.",
            "- This motivates modular state machines instead of monolithic registers.",
            f"- Nearest mean recovery: {_fmt_optional(nearest_mean)}.",
            f"- Hopfield mean recovery: {_fmt_optional(hopfield_mean)}.",
            f"- Hopfield recovery near capacity_ratio=0.138: {_fmt_with_actual(hopfield_0138, actual_0138, CLASSICAL_CAPACITY)}.",
            f"- Hopfield recovery at capacity_ratio=1.000: {_fmt_with_actual(hopfield_1000, actual_1000, 1.0)}.",
            f"- First capacity ratio where Hopfield mean drops below 0.90: {_fmt_threshold(below_090)}.",
            f"- First capacity ratio where Hopfield mean drops below 0.75: {_fmt_threshold(below_075)}.",
        ]
    )
    _add_table(lines, "Hopfield recovery by capacity ratio", hopfield_by_ratio, max_rows=30)
    _add_table(lines, "Nearest recovery by capacity ratio", nearest_by_ratio, max_rows=30)
    _add_table(lines, "Classical capacity comparison", capacity_comparison, max_rows=20)
    _add_table(lines, "Threshold estimates", thresholds, max_rows=20)
    _add_table(lines, "Worst Hopfield capacity rows", worst, columns=["seed", "state_dim", "num_states", "capacity_ratio", metric, "mean_abs_overlap"])

    result.markdown_lines = lines
    result.stdout_line = (
        f"Exp03: Hopfield@0.138={_fmt_with_actual(hopfield_0138, actual_0138, CLASSICAL_CAPACITY)}, "
        f"Hopfield@1.000={_fmt_with_actual(hopfield_1000, actual_1000, 1.0)}"
    )
    result.paper_values = [
        f"Exp03 Hopfield recovery by capacity_ratio: {_param_value_text(df, {'register_type': 'hopfield'}, 'capacity_ratio', metric)}",
        f"Exp03 nearest recovery by capacity_ratio: {_param_value_text(df, {'register_type': 'nearest'}, 'capacity_ratio', metric)}",
        f"Exp03 first capacity_ratio where Hopfield drops below 0.90: {_fmt_threshold(below_090)}",
        f"Exp03 first capacity_ratio where Hopfield drops below 0.75: {_fmt_threshold(below_075)}",
        f"Exp03 Hopfield recovery closest to capacity_ratio=0.138: {_fmt_with_actual(hopfield_0138, actual_0138, CLASSICAL_CAPACITY)}",
        f"Exp03 Hopfield recovery closest to capacity_ratio=1.000: {_fmt_with_actual(hopfield_1000, actual_1000, 1.0)}",
    ]
    result.latex_rows = [
        (
            "Exp03",
            "Hopfield recovery at capacity ratio 0.138",
            _fmt_with_actual(hopfield_0138, actual_0138, CLASSICAL_CAPACITY),
            "Recovery remains high near the classical capacity reference",
        ),
        (
            "Exp03",
            "Hopfield recovery at capacity ratio 1.000",
            _fmt_with_actual(hopfield_1000, actual_1000, 1.0),
            "Monolithic Hopfield registers degrade above capacity",
        ),
    ]
    return result


def _capacity_comparison_table(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows: list[Record] = []
    for target in [0.025, 0.050, 0.100, 0.138, 0.200, 0.300, 0.500, 0.750, 1.000]:
        value, actual = _mean_at(
            df,
            metric_col,
            {"register_type": "hopfield"},
            "capacity_ratio",
            target,
        )
        rows.append(
            {
                "target_capacity_ratio": target,
                "actual_capacity_ratio": actual if actual is not None else math.nan,
                "hopfield_recovery_accuracy": value if value is not None else math.nan,
                "used_nearest_available": False if actual is not None and math.isclose(actual, target) else True,
            }
        )
    return pd.DataFrame(rows)


def _capacity_threshold_table(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows: list[Record] = []
    for threshold in [0.99, 0.95, 0.90, 0.75, 0.50]:
        value = _threshold_below(df, "capacity_ratio", metric_col, threshold)
        rows.append(
            {
                "threshold": threshold,
                "first_capacity_ratio_below_threshold": value if value is not None else "not reached",
            }
        )
    return pd.DataFrame(rows)


def _fmt_threshold(value: float | None) -> str:
    return "not reached" if value is None else format_float(value)


def analyze_exp04(csv_dir: Path, out_dir: Path) -> AnalysisResult:
    source = csv_dir / CSV_NAMES["exp04"]
    df = load_csv(source)
    result = AnalysisResult("exp04", "Exp04: Sparse transitions", df is not None, source)
    if df is None:
        return result

    required = [
        "experiment",
        "seed",
        "num_states",
        "num_inputs",
        "state_dim",
        "input_dim",
        "register_type",
        "write_fraction",
        "mode",
        "sparse_transition_accuracy",
        "mean_abs_overlap",
    ]
    if not _has_required(df, required, "Exp04"):
        return result

    metric = "sparse_transition_accuracy"
    overall = mean_std_min_max(df, ["register_type"], metric)
    by_reg_mode = mean_std_min_max(df, ["register_type", "mode"], metric)
    by_write = mean_sem(df, ["register_type", "mode", "write_fraction"], metric)
    by_dim_write = mean_sem(df, ["register_type", "mode", "state_dim", "write_fraction"], metric)
    threshold_table = _sparse_threshold_table(df, metric)
    mode_diff = _mode_difference_table(df, metric)
    pivots = _sparse_pivot_frames(df, metric)
    worst = _lowest_rows(df, metric, 20)
    keep_current = _filter_equal(df, "mode", "keep_current")
    worst_keep_current = _lowest_rows(keep_current, metric, 20)
    subfull = _filter_rows(df, lambda row: (_to_float(row.get("write_fraction")) or 0.0) < 1.0)
    best_subfull = _highest_rows(subfull, metric, 20)

    summary_path = out_dir / "exp04_sparse_transitions_summary.csv"
    result.summary_path = _write_summary(
        summary_path,
        [
            _summary_frame(overall, "overall_by_register_type"),
            _summary_frame(by_reg_mode, "by_register_type_mode"),
            _summary_frame(by_write, "by_register_type_mode_write_fraction"),
            _summary_frame(by_dim_write, "by_register_type_mode_state_dim_write_fraction"),
            _summary_frame(threshold_table, "threshold_estimates"),
            _summary_frame(mode_diff, "random_noise_minus_keep_current"),
            *[_summary_frame(frame, name) for name, frame in pivots],
            _summary_frame(worst, "worst_rows"),
            _summary_frame(worst_keep_current, "worst_keep_current_rows"),
            _summary_frame(best_subfull, "best_subfull_write_fraction_rows"),
        ],
    )

    nearest_random_030, nearest_random_030_actual = _mean_at(
        df,
        metric,
        {"register_type": "nearest", "mode": "random_noise"},
        "write_fraction",
        0.30,
    )
    hopfield_random_030, hopfield_random_030_actual = _mean_at(
        df,
        metric,
        {"register_type": "hopfield", "mode": "random_noise"},
        "write_fraction",
        0.30,
    )
    random_noise_values = _sparse_values_for_targets(
        df,
        metric,
        "random_noise",
        ["nearest", "hopfield"],
        [0.10, 0.20, 0.30, 0.40, 0.50],
    )
    keep_current_values = _sparse_values_for_targets(
        df,
        metric,
        "keep_current",
        ["nearest", "hopfield"],
        [0.50, 0.75, 1.00],
    )

    lines: list[str] = ["## Exp04: Sparse transitions", ""]
    lines.extend(
        [
            "- Do not interpret aggregate mean alone.",
            "- keep_current and random_noise answer different questions.",
            "- keep_current measures sparse transition against old-state inertia.",
            "- random_noise measures whether partial target writes can bias the system into the target basin.",
            "- Sparse transitions work well in random_noise mode once enough target coordinates are written.",
            "- keep_current shows that reset/gating/inhibition may be needed before sparse writes.",
        ]
    )
    _add_table(lines, "Summary by register type and mode", by_reg_mode)
    _add_table(lines, "random_noise sparse accuracy by write fraction", random_noise_values, max_rows=20)
    _add_table(lines, "keep_current sparse accuracy at selected write fractions", keep_current_values, max_rows=20)
    _add_table(lines, "Threshold estimates", threshold_table, max_rows=30)
    _add_table(lines, "random_noise minus keep_current", mode_diff, max_rows=30)
    _add_table(lines, "Worst sparse-transition rows", worst, columns=["seed", "register_type", "mode", "state_dim", "num_states", "write_fraction", metric])
    _add_table(lines, "Best rows with write_fraction < 1.0", best_subfull, columns=["seed", "register_type", "mode", "state_dim", "num_states", "write_fraction", metric])

    result.markdown_lines = lines
    result.stdout_line = (
        f"Exp04: nearest random_noise@0.30={_fmt_with_actual(nearest_random_030, nearest_random_030_actual, 0.30)}, "
        f"Hopfield random_noise@0.30={_fmt_with_actual(hopfield_random_030, hopfield_random_030_actual, 0.30)}"
    )
    result.paper_values = [
        f"Exp04 nearest random_noise sparse accuracy by write_fraction: {_target_value_text(random_noise_values, {'register_type': 'nearest'}, 'target_write_fraction', metric)}",
        f"Exp04 Hopfield random_noise sparse accuracy by write_fraction: {_target_value_text(random_noise_values, {'register_type': 'hopfield'}, 'target_write_fraction', metric)}",
        f"Exp04 nearest keep_current sparse accuracy at write_fraction 0.50,0.75,1.00: {_target_value_text(keep_current_values, {'register_type': 'nearest'}, 'target_write_fraction', metric)}",
        f"Exp04 Hopfield keep_current sparse accuracy at write_fraction 0.50,0.75,1.00: {_target_value_text(keep_current_values, {'register_type': 'hopfield'}, 'target_write_fraction', metric)}",
        f"Exp04 nearest random_noise minus keep_current by write_fraction: {_mode_difference_text(mode_diff, 'nearest')}",
        f"Exp04 Hopfield random_noise minus keep_current by write_fraction: {_mode_difference_text(mode_diff, 'hopfield')}",
    ]
    for register_type in ["nearest", "hopfield"]:
        for threshold in [0.75, 0.90, 0.95]:
            reached = _threshold_reaches(
                df,
                {"register_type": register_type, "mode": "random_noise"},
                "write_fraction",
                metric,
                threshold,
            )
            result.paper_values.append(
                f"Exp04 {register_type} random_noise first reaches {threshold:.2f} at write_fraction={_fmt_threshold(reached)}"
            )
    result.latex_rows = [
        (
            "Exp04",
            "Nearest random_noise sparse accuracy at 0.30 write fraction",
            _fmt_with_actual(nearest_random_030, nearest_random_030_actual, 0.30),
            "Partial target writes can bias ideal cleanup into target basins",
        ),
        (
            "Exp04",
            "Hopfield random_noise sparse accuracy at 0.30 write fraction",
            _fmt_with_actual(hopfield_random_030, hopfield_random_030_actual, 0.30),
            "Sparse targeting is constrained by recurrent cleanup",
        ),
    ]
    return result


def _sparse_threshold_table(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows: list[Record] = []
    for register_type in _unique_sorted(df, "register_type"):
        for mode in _unique_sorted(_filter_equal(df, "register_type", register_type), "mode"):
            for threshold in [0.50, 0.75, 0.90, 0.95, 0.99]:
                value = _threshold_reaches(
                    df,
                    {"register_type": register_type, "mode": mode},
                    "write_fraction",
                    metric_col,
                    threshold,
                )
                rows.append(
                    {
                        "register_type": register_type,
                        "mode": mode,
                        "threshold": threshold,
                        "first_write_fraction_reaching_threshold": value if value is not None else "not reached",
                    }
                )
    return pd.DataFrame(rows)


def _mode_difference_table(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows: list[Record] = []
    for register_type in _unique_sorted(df, "register_type"):
        for write_fraction in _unique_sorted(_filter_equal(df, "register_type", register_type), "write_fraction"):
            random_mean = _mean_for_filters(
                df,
                metric_col,
                {
                    "register_type": register_type,
                    "mode": "random_noise",
                    "write_fraction": write_fraction,
                },
            )
            keep_mean = _mean_for_filters(
                df,
                metric_col,
                {
                    "register_type": register_type,
                    "mode": "keep_current",
                    "write_fraction": write_fraction,
                },
            )
            if random_mean is None or keep_mean is None:
                continue
            rows.append(
                {
                    "register_type": register_type,
                    "write_fraction": write_fraction,
                    "random_noise_mean": random_mean,
                    "keep_current_mean": keep_mean,
                    "random_noise_minus_keep_current": random_mean - keep_mean,
                }
            )
    return pd.DataFrame(rows)


def _sparse_pivot_frames(
    df: pd.DataFrame,
    metric_col: str,
) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    for mode in ["random_noise", "keep_current"]:
        for register_type in ["nearest", "hopfield"]:
            filtered = _filter_many(df, {"register_type": register_type, "mode": mode})
            frames.append(
                (
                    f"{register_type}_{mode}_by_write_fraction",
                    _metric_by_param(filtered, {}, "write_fraction", metric_col),
                )
            )
            frames.append(
                (
                    f"{register_type}_{mode}_state_dim_x_write_fraction_pivot",
                    _pivot_mean(filtered, "state_dim", "write_fraction", metric_col),
                )
            )
    return frames


def _sparse_values_for_targets(
    df: pd.DataFrame,
    metric_col: str,
    mode: str,
    register_types: Sequence[str],
    targets: Sequence[float],
) -> pd.DataFrame:
    rows: list[Record] = []
    for register_type in register_types:
        for target in targets:
            value, actual = _mean_at(
                df,
                metric_col,
                {"register_type": register_type, "mode": mode},
                "write_fraction",
                target,
            )
            rows.append(
                {
                    "register_type": register_type,
                    "mode": mode,
                    "target_write_fraction": target,
                    "actual_write_fraction": actual if actual is not None else math.nan,
                    metric_col: value if value is not None else math.nan,
                }
            )
    return pd.DataFrame(rows)


def analyze_exp05(csv_dir: Path, out_dir: Path) -> AnalysisResult:
    source = csv_dir / CSV_NAMES["exp05"]
    df = load_csv(source)
    result = AnalysisResult("exp05", "Exp05: Descriptor/payload separation", df is not None, source)
    if df is None:
        return result

    required = [
        "experiment",
        "seed",
        "case_name",
        "sequence",
        "final_state",
        "expected_state",
        "success",
        "payload_id",
        "descriptor",
        "corruption_rate",
        "num_trials",
        "operation_success_rate",
        "content_success_rate",
    ]
    if not _has_required(df, required, "Exp05"):
        return result

    protocol = _filter_rows(df, lambda row: _to_bool(row.get("success")) is not None)
    protocol_success = _protocol_success_table(protocol)
    protocol_overall = _protocol_overall(protocol)
    failures = _filter_rows(protocol, lambda row: _to_bool(row.get("success")) is False)
    corruption = _non_nan_rows(df, "corruption_rate")
    corruption_summary = _corruption_summary(corruption, ["corruption_rate"])
    descriptor_corruption = _corruption_summary(corruption, ["descriptor", "corruption_rate"])

    summary_path = out_dir / "exp05_descriptor_payload_summary.csv"
    result.summary_path = _write_summary(
        summary_path,
        [
            _summary_frame(protocol_success, "protocol_case_success"),
            _summary_frame(protocol_overall, "protocol_overall_success"),
            _summary_frame(failures, "protocol_failures"),
            _summary_frame(corruption_summary, "descriptor_corruption_by_rate"),
            _summary_frame(descriptor_corruption, "descriptor_corruption_by_descriptor_rate"),
        ],
    )

    overall_success = _to_float(protocol_overall.iloc[0]["success_mean"]) if not protocol_overall.empty else None
    op_0, op_0_actual = _mean_at(corruption, "operation_success_rate", {}, "corruption_rate", 0.0)
    op_05, op_05_actual = _mean_at(corruption, "operation_success_rate", {}, "corruption_rate", 0.5)
    op_10, op_10_actual = _mean_at(corruption, "operation_success_rate", {}, "corruption_rate", 1.0)
    content_0, content_0_actual = _mean_at(corruption, "content_success_rate", {}, "corruption_rate", 0.0)
    content_05, content_05_actual = _mean_at(corruption, "content_success_rate", {}, "corruption_rate", 0.5)
    content_10, content_10_actual = _mean_at(corruption, "content_success_rate", {}, "corruption_rate", 1.0)

    lines: list[str] = ["## Exp05: Descriptor/payload separation", ""]
    lines.extend(
        [
            "- This is an executable witness, not a performance benchmark.",
            "- It demonstrates that the same payload can produce different effects under different descriptors.",
            "- It demonstrates that payload without a descriptor is invalid/error-state producing.",
            "- Descriptor corruption affects operation-level behavior.",
            f"- Protocol success across non-NaN success rows: {_fmt_optional(overall_success)}.",
            f"- All non-NaN success values true: {'yes' if failures.empty else 'no'}.",
        ]
    )
    _add_table(lines, "Protocol case success", protocol_success, max_rows=30)
    _add_table(lines, "Descriptor corruption by rate", corruption_summary, max_rows=30)
    if not descriptor_corruption.empty:
        _add_table(lines, "Descriptor corruption by descriptor and rate", descriptor_corruption, max_rows=40)
    if not failures.empty:
        _add_table(lines, "Protocol failures", failures, max_rows=20)

    result.markdown_lines = lines
    result.stdout_line = f"Exp05: protocol success={_fmt_optional(overall_success)}"
    result.paper_values = [
        f"Exp05 protocol success rate across non-NaN success rows: {_fmt_optional(overall_success)}",
        f"Exp05 success by case_name: {_case_success_text(protocol_success)}",
        f"Exp05 operation success at corruption_rate=0.0: {_fmt_with_actual(op_0, op_0_actual, 0.0)}",
        f"Exp05 operation success at corruption_rate=0.5: {_fmt_with_actual(op_05, op_05_actual, 0.5)}",
        f"Exp05 operation success at corruption_rate=1.0: {_fmt_with_actual(op_10, op_10_actual, 1.0)}",
        f"Exp05 content success at corruption_rate=0.0: {_fmt_with_actual(content_0, content_0_actual, 0.0)}",
        f"Exp05 content success at corruption_rate=0.5: {_fmt_with_actual(content_05, content_05_actual, 0.5)}",
        f"Exp05 content success at corruption_rate=1.0: {_fmt_with_actual(content_10, content_10_actual, 1.0)}",
    ]
    result.latex_rows = [
        (
            "Exp05",
            "Protocol success rate",
            _fmt_optional(overall_success),
            "Descriptor/payload distinction is operational",
        )
    ]
    return result


def _protocol_success_table(df: pd.DataFrame) -> pd.DataFrame:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in _records(df):
        case_name = str(row.get("case_name"))
        success = _to_bool(row.get("success"))
        if success is None:
            continue
        groups[case_name].append(1.0 if success else 0.0)
    rows: list[Record] = []
    for case_name in sorted(groups):
        values = groups[case_name]
        rows.append(
            {
                "case_name": case_name,
                "count": len(values),
                "success_mean": float(np.mean(values)),
            }
        )
    return pd.DataFrame(rows)


def _protocol_overall(df: pd.DataFrame) -> pd.DataFrame:
    values: list[float] = []
    for row in _records(df):
        success = _to_bool(row.get("success"))
        if success is not None:
            values.append(1.0 if success else 0.0)
    return pd.DataFrame(
        [
            {
                "count": len(values),
                "success_mean": float(np.mean(values)) if values else math.nan,
            }
        ]
    )


def _corruption_summary(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metrics = ["operation_success_rate", "content_success_rate"]
    groups: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in metrics}
    )
    for row in _records(df):
        key = tuple(row.get(column) for column in group_cols)
        for metric in metrics:
            value = _to_float(row.get(metric))
            if value is not None:
                groups[key][metric].append(value)

    rows: list[Record] = []
    for key in sorted(groups, key=lambda item: tuple(_sort_key(part) for part in item)):
        out: Record = {column: key[index] for index, column in enumerate(group_cols)}
        for metric in metrics:
            values = groups[key][metric]
            out[f"{metric}_count"] = len(values)
            out[f"{metric}_mean"] = float(np.mean(values)) if values else math.nan
            out[f"{metric}_std"] = _std(values) if values else math.nan
            out[f"{metric}_sem"] = _sem(values) if values else math.nan
        rows.append(out)
    return pd.DataFrame(rows)


def analyze_exp06(csv_dir: Path, out_dir: Path) -> AnalysisResult:
    source = csv_dir / CSV_NAMES["exp06"]
    df = load_csv(source)
    result = AnalysisResult("exp06", "Exp06: Learned transitions", df is not None, source)
    if df is None:
        return result

    required = [
        "experiment",
        "seed",
        "num_states",
        "num_inputs",
        "state_dim",
        "input_dim",
        "feature_mode",
        "feature_dim",
        "hidden_dim",
        "training_mode",
        "coverage_fraction",
        "epochs",
        "update_rule",
        "output_mode",
        "write_fraction",
        "unwritten_mode",
        "register_type",
        "cleanup_type",
        "num_training_examples",
        "seen_transition_accuracy",
        "unseen_transition_accuracy",
        "all_transition_accuracy",
        "raw_bit_accuracy",
        "raw_overlap_with_target",
        "basin_margin_mean",
        "exact_upper_bound_accuracy",
        "untrained_baseline_accuracy",
        "mean_abs_overlap",
        "notes",
    ]
    if not _has_required(df, required, "Exp06"):
        return result

    metric = "all_transition_accuracy"
    overall = mean_std_min_max(df, ["feature_mode", "register_type"], metric)
    by_epoch = mean_sem(
        df,
        ["feature_mode", "hidden_dim", "register_type", "epochs"],
        metric,
    )
    dense_full = _filter_many(
        df,
        {
            "training_mode": "full_table",
            "output_mode": "dense",
            "coverage_fraction": 1.0,
            "update_rule": "delta",
        },
    )
    max_dense_full = _max_epoch_df(dense_full)
    hashed_max = _filter_equal(max_dense_full, "feature_mode", "hashed_pair")
    hashed_by_hidden = mean_sem(
        hashed_max,
        ["register_type", "hidden_dim"],
        metric,
    )
    sparse = _filter_many(
        df,
        {
            "feature_mode": "exact_pair",
            "output_mode": "sparse_topk",
        },
    )
    sparse_max = _max_epoch_df(sparse)
    sparse_by_write = mean_sem(
        sparse_max,
        ["register_type", "unwritten_mode", "write_fraction"],
        metric,
    )
    coverage = _filter_many(
        df,
        {
            "feature_mode": "exact_pair",
            "training_mode": "coverage_sweep",
            "output_mode": "dense",
        },
    )
    coverage_max = _max_epoch_df(coverage)
    coverage_all = mean_sem(
        coverage_max,
        ["register_type", "coverage_fraction"],
        metric,
    )
    coverage_seen = mean_sem(
        coverage_max,
        ["register_type", "coverage_fraction"],
        "seen_transition_accuracy",
    )
    coverage_unseen = mean_sem(
        coverage_max,
        ["register_type", "coverage_fraction"],
        "unseen_transition_accuracy",
    )

    summary_path = out_dir / "exp06_learned_transitions_summary.csv"
    result.summary_path = _write_summary(
        summary_path,
        [
            _summary_frame(overall, "overall_by_feature_mode_register_type"),
            _summary_frame(by_epoch, "learning_curve_by_feature_mode_hidden_dim_register_type_epoch"),
            _summary_frame(hashed_by_hidden, "hashed_pair_by_hidden_dim_at_max_epoch"),
            _summary_frame(sparse_by_write, "sparse_topk_by_write_fraction_at_max_epoch"),
            _summary_frame(coverage_all, "coverage_all_accuracy_at_max_epoch"),
            _summary_frame(coverage_seen, "coverage_seen_accuracy_at_max_epoch"),
            _summary_frame(coverage_unseen, "coverage_unseen_accuracy_at_max_epoch"),
        ],
    )

    exact_one = _filter_many(
        df,
        {
            "feature_mode": "exact_pair",
            "training_mode": "full_table",
            "output_mode": "dense",
            "coverage_fraction": 1.0,
            "epochs": 1,
            "update_rule": "delta",
        },
    )
    exact_nearest = _mean_for_filters(
        exact_one,
        metric,
        {"register_type": "nearest"},
    )
    exact_hopfield = _mean_for_filters(
        exact_one,
        metric,
        {"register_type": "hopfield"},
    )
    hidden_nearest_text = _param_value_text(
        hashed_max,
        {"register_type": "nearest"},
        "hidden_dim",
        metric,
    )
    hidden_hopfield_text = _param_value_text(
        hashed_max,
        {"register_type": "hopfield"},
        "hidden_dim",
        metric,
    )
    first_hidden_095 = _first_param_reaching(
        hashed_max,
        {"register_type": "nearest"},
        "hidden_dim",
        metric,
        0.95,
    )
    sparse_near_030, sparse_near_030_actual = _mean_at(
        sparse_max,
        metric,
        {"register_type": "nearest", "unwritten_mode": "random_noise"},
        "write_fraction",
        0.30,
    )
    sparse_hop_030, sparse_hop_030_actual = _mean_at(
        sparse_max,
        metric,
        {"register_type": "hopfield", "unwritten_mode": "random_noise"},
        "write_fraction",
        0.30,
    )
    sparse_near_050, sparse_near_050_actual = _mean_at(
        sparse_max,
        metric,
        {"register_type": "nearest", "unwritten_mode": "random_noise"},
        "write_fraction",
        0.50,
    )
    sparse_hop_050, sparse_hop_050_actual = _mean_at(
        sparse_max,
        metric,
        {"register_type": "hopfield", "unwritten_mode": "random_noise"},
        "write_fraction",
        0.50,
    )
    cov_nearest_seen_050, cov_nearest_seen_actual = _mean_at(
        coverage_max,
        "seen_transition_accuracy",
        {"register_type": "nearest"},
        "coverage_fraction",
        0.50,
    )
    cov_nearest_unseen_050, cov_nearest_unseen_actual = _mean_at(
        coverage_max,
        "unseen_transition_accuracy",
        {"register_type": "nearest"},
        "coverage_fraction",
        0.50,
    )

    lines: list[str] = ["## Exp06: Learned transitions", ""]
    lines.extend(
        [
            "- exact_pair/full_table shows that transition associations can be acquired from demonstrations.",
            "- hashed_pair exposes capacity limits in the state-input conjunctive interface layer.",
            "- sparse learned writer rows test whether learned transitions can target basins partially.",
            "- coverage split rows show that arbitrary FSMs do not generalize without structural regularity.",
            f"- exact_pair nearest accuracy after one epoch/full table: {_fmt_optional(exact_nearest)}.",
            f"- exact_pair Hopfield accuracy after one epoch/full table: {_fmt_optional(exact_hopfield)}.",
            f"- First hashed hidden_dim reaching 0.95 nearest accuracy: {_fmt_threshold(first_hidden_095)}.",
            f"- nearest sparse random-noise accuracy at write_fraction=0.30: {_fmt_with_actual(sparse_near_030, sparse_near_030_actual, 0.30)}.",
            f"- Hopfield sparse random-noise accuracy at write_fraction=0.30: {_fmt_with_actual(sparse_hop_030, sparse_hop_030_actual, 0.30)}.",
            f"- nearest seen/unseen at coverage_fraction=0.50: seen={_fmt_with_actual(cov_nearest_seen_050, cov_nearest_seen_actual, 0.50)}, unseen={_fmt_with_actual(cov_nearest_unseen_050, cov_nearest_unseen_actual, 0.50)}.",
        ]
    )
    _add_table(lines, "Overall summary by feature mode and register type", overall, max_rows=40)
    _add_table(lines, "Hashed-pair capacity at max epoch", hashed_by_hidden, max_rows=40)
    _add_table(lines, "Sparse learned transitions at max epoch", sparse_by_write, max_rows=40)
    _add_table(lines, "Coverage all accuracy", coverage_all, max_rows=30)
    _add_table(lines, "Coverage seen accuracy", coverage_seen, max_rows=30)
    _add_table(lines, "Coverage unseen accuracy", coverage_unseen, max_rows=30)

    result.markdown_lines = lines
    result.stdout_line = (
        f"Exp06: exact_pair nearest@1 epoch={_fmt_optional(exact_nearest)}, "
        f"Hopfield@1 epoch={_fmt_optional(exact_hopfield)}"
    )
    result.paper_values = [
        f"Exp06 exact_pair nearest accuracy after one epoch/full table: {_fmt_optional(exact_nearest)}",
        f"Exp06 exact_pair Hopfield accuracy after one epoch/full table: {_fmt_optional(exact_hopfield)}",
        f"Exp06 hashed_pair nearest accuracy by hidden_dim at max epoch: {hidden_nearest_text}",
        f"Exp06 hashed_pair Hopfield accuracy by hidden_dim at max epoch: {hidden_hopfield_text}",
        f"Exp06 first hidden_dim reaching 0.95 nearest accuracy: {_fmt_threshold(first_hidden_095)}",
        f"Exp06 learned sparse nearest accuracy at write_fraction=0.30: {_fmt_with_actual(sparse_near_030, sparse_near_030_actual, 0.30)}",
        f"Exp06 learned sparse Hopfield accuracy at write_fraction=0.30: {_fmt_with_actual(sparse_hop_030, sparse_hop_030_actual, 0.30)}",
        f"Exp06 learned sparse nearest accuracy at write_fraction=0.50: {_fmt_with_actual(sparse_near_050, sparse_near_050_actual, 0.50)}",
        f"Exp06 learned sparse Hopfield accuracy at write_fraction=0.50: {_fmt_with_actual(sparse_hop_050, sparse_hop_050_actual, 0.50)}",
        f"Exp06 nearest seen accuracy at coverage_fraction=0.50: {_fmt_with_actual(cov_nearest_seen_050, cov_nearest_seen_actual, 0.50)}",
        f"Exp06 nearest unseen accuracy at coverage_fraction=0.50: {_fmt_with_actual(cov_nearest_unseen_050, cov_nearest_unseen_actual, 0.50)}",
    ]
    result.latex_rows = [
        (
            "Exp06",
            "Learned exact_pair nearest accuracy after one epoch",
            _fmt_optional(exact_nearest),
            "Transition acquisition from demonstrations succeeds under ideal cleanup",
        ),
        (
            "Exp06",
            "Learned exact_pair Hopfield accuracy after one epoch",
            _fmt_optional(exact_hopfield),
            "Learned transitions inherit recurrent cleanup limits",
        ),
    ]
    return result


def _first_param_reaching(
    df: pd.DataFrame,
    fixed_filters: dict[str, object],
    param_col: str,
    metric_col: str,
    threshold: float,
) -> float | None:
    table = _metric_by_param(df, fixed_filters, param_col, metric_col)
    rows = sorted(_records(table), key=lambda row: cast(float, _to_float(row.get(param_col))))
    for row in rows:
        param = _to_float(row.get(param_col))
        metric = _to_float(row.get(metric_col))
        if param is not None and metric is not None and metric >= threshold:
            return param
    return None


def results_cheat_sheet() -> list[str]:
    return [
        "| Experiment | Main result | Interpretation | Best figure/table |",
        "| --- | --- | --- | --- |",
        "| Exp01 | Nearest transition accuracy = 1.000; Hopfield is capacity-dependent. | Exact transition construction works; recurrent cleanup imposes capacity limits. | fig01/fig06 and Exp01 summary |",
        "| Exp02 | Nearest robust under corruption; Hopfield lower with monotonic degradation. | State vectors behave as basins; Hopfield implementation has stability/noise limits. | fig02 and Exp02 summary |",
        "| Exp03 | Hopfield near-perfect up to around classical capacity, then degrades. | Capacity limits motivate modularity. | fig03/fig07 and statistics table |",
        "| Exp04 | random_noise sparse transitions improve strongly with write fraction; keep_current is harsher. | Sparse basin targeting is possible but may require reset/gating. | fig04b and Exp04 mode-difference table |",
        "| Exp05 | Protocol cases succeed. | Descriptor/payload distinction is operational. | fig05 and Exp05 summary |",
        "| Exp06 | Learned exact-pair transitions acquire demonstrated FSM associations. | Transition maps can be learned from demonstrations but inherit interface and cleanup capacity limits. | fig08-fig11 and Exp06 summary |",
    ]


def collect_results(csv_dir: Path, out_dir: Path) -> list[AnalysisResult]:
    analyses: list[Callable[[Path, Path], AnalysisResult]] = [
        analyze_exp01,
        analyze_exp02,
        analyze_exp03,
        analyze_exp04,
        analyze_exp05,
        analyze_exp06,
    ]
    return [analysis(csv_dir, out_dir) for analysis in analyses]


def _loaded_missing_paths(csv_dir: Path, results: Sequence[AnalysisResult]) -> tuple[list[Path], list[Path]]:
    loaded = [result.source_path for result in results if result.loaded and result.source_path.exists()]
    missing = [
        csv_dir / filename
        for filename in CSV_NAMES.values()
        if not (csv_dir / filename).exists()
    ]
    return loaded, missing


def _paper_values(results: Sequence[AnalysisResult]) -> list[str]:
    values: list[str] = []
    for result in results:
        values.extend(result.paper_values)
    return values


def _latex_rows(results: Sequence[AnalysisResult]) -> list[LatexRow]:
    rows: list[LatexRow] = []
    for result in results:
        rows.extend(result.latex_rows)
    return rows


def _write_reports(
    out_dir: Path,
    results: Sequence[AnalysisResult],
    loaded_files: Sequence[Path],
    missing_files: Sequence[Path],
) -> tuple[Path, Path, Path]:
    md_path = out_dir / REPORT_MD
    txt_path = out_dir / REPORT_TXT
    latex_path = TABLE_DIR / LATEX_TABLE
    paper_values = _paper_values(results)
    write_markdown_report(md_path, loaded_files, missing_files, results, paper_values)
    txt_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    write_latex_table(latex_path, _latex_rows(results))
    return md_path, txt_path, latex_path


def _print_compact_summary(
    results: Sequence[AnalysisResult],
    loaded_files: Sequence[Path],
    md_path: Path,
    latex_path: Path,
) -> None:
    print("Loaded files:")
    for path in loaded_files:
        print(f"- {path}")
    for result in results:
        if result.stdout_line is not None:
            print(result.stdout_line)
    print(f"Markdown report: {md_path}")
    print(f"LaTeX table: {latex_path}")


def _print_paper_only(results: Sequence[AnalysisResult]) -> None:
    for value in _paper_values(results):
        print(f"- {value}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate statistical summaries from existing BioLogic experiment CSVs.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help="Directory containing experiment CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for statistics reports and summary CSVs.",
    )
    parser.add_argument(
        "--paper-only",
        action="store_true",
        help="Print only paper-ready values to stdout after writing outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_dir = cast(Path, args.csv_dir)
    out_dir = cast(Path, args.out_dir)
    ensure_dirs(out_dir, TABLE_DIR)
    results = collect_results(csv_dir, out_dir)
    loaded_files, missing_files = _loaded_missing_paths(csv_dir, results)
    md_path, _txt_path, latex_path = _write_reports(out_dir, results, loaded_files, missing_files)
    if cast(bool, args.paper_only):
        _print_paper_only(results)
    else:
        _print_compact_summary(results, loaded_files, md_path, latex_path)


if __name__ == "__main__":
    main()
