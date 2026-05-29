"""Shared helpers for experiment scripts."""

from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import BaseContext
from pathlib import Path
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


def parallel_map(
    func: Callable[[T], U],
    items: Iterable[T],
    jobs: int | None = 1,
) -> list[U]:
    """Map ``func`` over items, using worker processes when jobs > 1."""
    item_list: list[T] = list(items)
    resolved_jobs: int = resolve_jobs(jobs)
    if resolved_jobs == 1 or len(item_list) <= 1:
        return [func(item) for item in item_list]

    context: BaseContext | None = (
        mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    )
    chunksize: int = max(1, len(item_list) // (resolved_jobs * 4))
    with ProcessPoolExecutor(max_workers=resolved_jobs, mp_context=context) as executor:
        return list(executor.map(func, item_list, chunksize=chunksize))
