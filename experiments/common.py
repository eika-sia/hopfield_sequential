"""Shared helpers for experiment scripts."""

from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.context import BaseContext
from pathlib import Path
from time import monotonic
from typing import Callable, Iterable, TypeVar

import numpy as np
import pandas as pd

from biologic.register import HopfieldRegister, NearestAttractorRegister

CSV_DIR: Path = Path("results/csv")
FIGURE_DIR: Path = Path("results/figures")
T = TypeVar("T")
U = TypeVar("U")


def ensure_output_dirs() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def make_register(
    register_type: str, codebook: np.ndarray
) -> NearestAttractorRegister | HopfieldRegister:
    if register_type == "nearest":
        return NearestAttractorRegister(codebook)
    if register_type == "hopfield":
        return HopfieldRegister.from_codebook(codebook)
    raise ValueError(f"unknown register_type: {register_type}")


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    ensure_output_dirs()
    path = CSV_DIR / filename
    df.to_csv(path, index=False)
    return path


def quick_suffix(quick: bool) -> str:
    return "quick" if quick else "full"


def resolve_jobs(jobs: int | None) -> int:
    """Resolve CLI jobs value; 0 means all available CPUs."""
    if jobs is None:
        return 1
    if jobs < 0:
        raise ValueError("jobs must be >= 0")
    if jobs == 0:
        return os.cpu_count() or 1
    return jobs


def print_progress(
    label: str,
    completed: int,
    total: int,
    start_time: float,
) -> None:
    """Print compact progress with throughput and ETA."""
    if total <= 0:
        return
    elapsed = monotonic() - start_time
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    remaining = total - completed
    eta = remaining / rate if rate > 0.0 else float("inf")
    eta_text = "unknown" if eta == float("inf") else f"{eta / 3600:.2f}h"
    print(
        f"{label}: {completed}/{total} ({completed / total:.1%}), "
        f"{rate * 3600:.1f} conditions/hour, ETA {eta_text}",
        flush=True,
    )


def progress_interval(total: int) -> int:
    """Return an update interval that gives about 100 progress lines per run."""
    return max(1, total // 100)


def parallel_map(
    func: Callable[[T], U],
    items: Iterable[T],
    jobs: int | None = 1,
    progress_label: str = "Progress",
) -> list[U]:
    """Map ``func`` over items, using worker processes when jobs > 1."""
    item_list: list[T] = list(items)
    total = len(item_list)
    start_time = monotonic()
    interval = progress_interval(total)
    resolved_jobs: int = resolve_jobs(jobs)
    if resolved_jobs == 1 or total <= 1:
        rows: list[U] = []
        for index, item in enumerate(item_list, start=1):
            rows.append(func(item))
            if index == total or index % interval == 0:
                print_progress(progress_label, index, total, start_time)
        return rows

    context: BaseContext | None = (
        mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    )
    results: dict[int, U] = {}
    with ProcessPoolExecutor(max_workers=resolved_jobs, mp_context=context) as executor:
        futures = {
            executor.submit(func, item): index
            for index, item in enumerate(item_list)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if completed == total or completed % interval == 0:
                print_progress(progress_label, completed, total, start_time)
    return [results[index] for index in range(total)]
