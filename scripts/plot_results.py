"""Generate matplotlib figures from experiment CSV outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

FIGURE_DIR = Path("results/figures")
CSV_DIR = Path("results/csv")


def _save_current(name: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIGURE_DIR / f"{name}.png", FIGURE_DIR / f"{name}.pdf"]
    for path in paths:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return paths


def _line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    name: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> list[Path]:
    plt.figure(figsize=(7, 4.5))
    for value, group in sorted(df.groupby(hue), key=lambda item: item[0]):
        summary = group.groupby(x, as_index=False)[y].mean().sort_values(x)
        plt.plot(summary[x], summary[y], marker="o", label=f"{hue}={value}")
    plt.title(title)
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    return _save_current(name)


def plot_transition_accuracy(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for register_type, group in df.groupby("register_type"):
        paths.extend(
            _line_plot(
                group,
                x="num_states",
                y="transition_accuracy",
                hue="state_dim",
                title=f"Transition accuracy ({register_type})",
                name=f"fig_transition_accuracy_{register_type}",
                xlabel="Number of states",
                ylabel="Transition accuracy",
            )
        )
    return paths


def plot_noise_recovery(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for register_type, group in df.groupby("register_type"):
        paths.extend(
            _line_plot(
                group,
                x="flip_fraction",
                y="recovery_accuracy",
                hue="state_dim",
                title=f"Noise recovery ({register_type})",
                name=f"fig_noise_recovery_{register_type}",
                xlabel="Flip fraction",
                ylabel="Recovery accuracy",
            )
        )
    return paths


def plot_capacity(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for register_type, group in df.groupby("register_type"):
        paths.extend(
            _line_plot(
                group,
                x="capacity_ratio",
                y="recovery_accuracy",
                hue="state_dim",
                title=f"Capacity sweep ({register_type})",
                name=f"fig_capacity_{register_type}",
                xlabel="Capacity ratio",
                ylabel="Recovery accuracy",
            )
        )
    return paths


def plot_sparse_transitions(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for mode, mode_group in df.groupby("mode"):
        plt.figure(figsize=(7, 4.5))
        for (register_type, state_dim), group in sorted(
            mode_group.groupby(["register_type", "state_dim"]), key=lambda item: item[0]
        ):
            summary = (
                group.groupby("write_fraction", as_index=False)["sparse_transition_accuracy"]
                .mean()
                .sort_values("write_fraction")
            )
            plt.plot(
                summary["write_fraction"],
                summary["sparse_transition_accuracy"],
                marker="o",
                label=f"{register_type}, dim={state_dim}",
            )
        plt.title(f"Sparse transitions ({mode})")
        plt.xlabel("Write fraction")
        plt.ylabel("Sparse transition accuracy")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, alpha=0.3)
        plt.legend()
        paths.extend(_save_current(f"fig_sparse_transition_{mode}"))
    return paths


def plot_descriptor_payload(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    cases = df[df["case_name"] != "descriptor_corruption"].copy()
    if not cases.empty:
        summary = cases.groupby("case_name", as_index=False)["success"].mean()
        plt.figure(figsize=(7, 4.5))
        plt.bar(summary["case_name"], summary["success"])
        plt.title("Descriptor/payload cases")
        plt.xlabel("Case")
        plt.ylabel("Success rate")
        plt.ylim(-0.02, 1.02)
        plt.xticks(rotation=25, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        paths.extend(_save_current("fig_descriptor_payload_cases"))

    corruption = df[df["case_name"] == "descriptor_corruption"].copy()
    if not corruption.empty:
        summary = corruption.groupby("corruption_rate", as_index=False)[
            ["operation_success_rate", "content_success_rate"]
        ].mean()
        plt.figure(figsize=(7, 4.5))
        plt.plot(
            summary["corruption_rate"],
            summary["operation_success_rate"],
            marker="o",
            label="Operation success",
        )
        plt.plot(
            summary["corruption_rate"],
            summary["content_success_rate"],
            marker="o",
            label="Content success",
        )
        plt.title("Descriptor corruption")
        plt.xlabel("Corruption rate")
        plt.ylabel("Success rate")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, alpha=0.3)
        plt.legend()
        paths.extend(_save_current("fig_descriptor_corruption"))
    return paths


def plot_all_from_csv(csv_dir: Path = CSV_DIR) -> list[Path]:
    paths: list[Path] = []
    mapping = [
        ("exp01_transition_accuracy.csv", plot_transition_accuracy),
        ("exp02_noise_recovery.csv", plot_noise_recovery),
        ("exp03_capacity.csv", plot_capacity),
        ("exp04_sparse_transitions.csv", plot_sparse_transitions),
        ("exp05_descriptor_payload.csv", plot_descriptor_payload),
    ]
    for filename, plotter in mapping:
        path = csv_dir / filename
        if path.exists():
            paths.extend(plotter(pd.read_csv(path)))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    args = parser.parse_args()
    paths = plot_all_from_csv(args.csv_dir)
    print(f"Saved {len(paths)} figure files to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
