"""Plot COMSOL photonic-crystal bands along Gamma-X-M-Gamma.

By default this reads comsol_csv/2608101808.csv next to this script.  Pass a
different path as the first argument to plot another COMSOL export instead.

Example
-------
python comsol.py
python comsol.py "C:\\Users\\cojyi\\Desktop\\BIC\\comsol_csv\\2608101808.csv"

Optional comparison file
------------------------
python comsol.py comsol_csv\\2608101808.csv --python-bands python_bands.csv

``python_bands.csv`` must have one row per k-point.  Its first column is the
k-path coordinate and the following columns are the Python-calculated bands.
An optional header row is allowed.  The k-path coordinate can be normalized
(0 to 1) or use the same 0 to 105 coordinates as the COMSOL export.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HIGH_SYMMETRY_INDICES = np.array([0, 35, 70, 105])
HIGH_SYMMETRY_LABELS = (r"$\Gamma$", "X", "M", r"$\Gamma$")

DEFAULT_COMSOL_CSV = Path(__file__).parent / "comsol_csv" / "2608101808.csv"


def parse_comsol_bands(filename: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return k-point coordinates and normalized frequencies from a COMSOL CSV."""
    rows: list[tuple[int, float]] = []
    with filename.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.reader(file):
            if not row or row[0].lstrip().startswith("%"):
                continue
            # COMSOL uses i rather than Python's j for complex numbers.
            try:
                k_index = int(row[0])
                frequency = complex(row[2].strip().replace("i", "j")).real
            except (IndexError, ValueError):
                continue
            rows.append((k_index, frequency))

    if not rows:
        raise ValueError("No numeric COMSOL data rows were found.")

    k_values = np.array(sorted({k for k, _ in rows}))
    modes_per_point = len(rows) // len(k_values)
    if len(rows) != len(k_values) * modes_per_point:
        raise ValueError("Each k-point must contain the same number of eigenmodes.")

    bands = np.full((modes_per_point, len(k_values)), np.nan)
    for col, k_index in enumerate(k_values):
        # Sort ascending: COMSOL doesn't guarantee a consistent eigenmode
        # order across k-points, which otherwise shows up as spurious
        # vertical spikes where two modes swap positions.
        values = sorted(value for k, value in rows if k == k_index)
        bands[:, col] = values
    return k_values, bands


def read_python_bands(filename: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a wide CSV: x, band_1, band_2, ...; skip a non-numeric header."""
    numeric_rows: list[list[float]] = []
    with filename.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.reader(file):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                numeric_rows.append([float(item) for item in row])
            except ValueError:  # e.g. x, band1, band2, ...
                continue
    data = np.asarray(numeric_rows, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Python CSV needs x plus at least one band column.")
    return data[:, 0], data[:, 1:].T


def decorate_k_path(ax: plt.Axes, k_values: np.ndarray) -> None:
    """Add Gamma-X-M-Gamma labels at 0, 35, 70, and 105 (or scaled positions)."""
    # Scaling keeps the labels correct if a comparison uses x from 0 to 1.
    ticks = np.interp(HIGH_SYMMETRY_INDICES, [0, 105], [k_values.min(), k_values.max()])
    ax.set_xticks(ticks, HIGH_SYMMETRY_LABELS)
    for x in ticks:
        ax.axvline(x, color="0.75", lw=0.8, zorder=0)
    ax.set_xlim(k_values.min(), k_values.max())
    ax.set_xlabel("Wave vector")
    ax.set_ylabel(r"Normalized frequency $fa/c$")
    ax.grid(axis="y", color="0.9", lw=0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comsol_csv",
        type=Path,
        nargs="?",
        default=DEFAULT_COMSOL_CSV,
        help=f"COMSOL Global Evaluation CSV (default: {DEFAULT_COMSOL_CSV.name})",
    )
    parser.add_argument("--python-bands", type=Path, help="optional wide-format Python band CSV")
    parser.add_argument("--output", type=Path, default=Path("png") / "band_comparison.png")
    parser.add_argument("--ymax", type=float, default=None, help="optional y-axis upper limit")
    args = parser.parse_args()

    comsol_x, comsol_bands = parse_comsol_bands(args.comsol_csv)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)

    for band in comsol_bands:
        ax.plot(comsol_x, band, "o", color="#1565C0", ms=2.5, mew=0, label="COMSOL")

    if args.python_bands:
        python_x, python_bands = read_python_bands(args.python_bands)
        # A normalized Python path (0--1) needs the COMSOL coordinate scale
        # before it can be overlaid with U=0--105 data.
        if not np.isclose(python_x.min(), comsol_x.min()) or not np.isclose(
            python_x.max(), comsol_x.max()
        ):
            python_x = np.interp(
                python_x,
                [python_x.min(), python_x.max()],
                [comsol_x.min(), comsol_x.max()],
            )
        for band in python_bands:
            ax.plot(python_x, band, "o", color="#E65100", ms=2.5, mew=0, label="Python")

    # Use proxy artists so the legend contains each source only once.
    handles = [
        plt.Line2D([], [], color="#1565C0", marker="o", ls="", label="COMSOL"),
    ]
    if args.python_bands:
        handles.append(plt.Line2D([], [], color="#E65100", marker="o", ls="", label="Python"))
    ax.legend(handles=handles, frameon=False)
    decorate_k_path(ax, comsol_x)
    ax.set_ylim(bottom=0, top=args.ymax)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
