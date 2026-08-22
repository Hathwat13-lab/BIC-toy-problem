"""Plot TE03-like branches for bilayer COMSOL sweep across all gaps.

Creates one combined figure per gap with all shifts arranged in a subplot grid.
- x-axis: incidence angle (deg)
- y-axis: eigenfrequency real part (THz)
- dot color: log10(Q)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = Path("comsol_csv") / "bilayer.csv"
OUT_DIR = Path("png") / "bilayer"

FREQ_MIN_THz = 1.95
FREQ_MAX_THz = 2.075


def load_rows(path: Path) -> list[tuple[float, float, float, float, float]]:
    """Return rows as (U_rad, shift_m, gap_m, freq_real_THz, Q)."""
    rows: list[tuple[float, float, float, float, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.reader(file):
            if not row or row[0].lstrip().startswith("%"):
                continue
            try:
                u_rad = float(row[0])
                shift_m = float(row[1])
                gap_m = float(row[2])
                freq_complex = complex(row[3].strip().replace("i", "j"))
                q = float(row[7])
            except (IndexError, ValueError):
                continue
            rows.append((u_rad, shift_m, gap_m, freq_complex.real, q))
    if not rows:
        raise ValueError("No numeric rows found in CSV.")
    return rows


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def select_points(
    rows: list[tuple[float, float, float, float, float]],
    gap_m: float,
    shift_m: float,
) -> list[tuple[float, float, float]]:
    return [
        (u, f, q)
        for (u, s, g, f, q) in rows
        if close(g, gap_m)
        and close(s, shift_m)
        and (FREQ_MIN_THz <= f <= FREQ_MAX_THz)
        and q > 0.0
    ]


def make_symmetric_points(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Mirror +U points to -U so the view spans -10 to 10 deg."""
    mirrored: list[tuple[float, float, float]] = []
    for u, f, q in points:
        mirrored.append((u, f, q))
        if u > 0.0:
            mirrored.append((-u, f, q))
    return mirrored


def plot_all_shifts_for_gap(
    rows: list[tuple[float, float, float, float, float]],
    gap_m: float,
) -> Path | None:
    shifts = sorted({s for _, s, g, _, _ in rows if close(g, gap_m)})
    if not shifts:
        return None

    selected_by_shift: dict[float, list[tuple[float, float, float]]] = {}
    all_logq: list[float] = []
    for shift_m in shifts:
        pts = make_symmetric_points(select_points(rows, gap_m, shift_m))
        if not pts:
            continue
        selected_by_shift[shift_m] = pts
        all_logq.extend(math.log10(q) for _, _, q in pts)

    if not selected_by_shift:
        return None

    n = len(selected_by_shift)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.4 * nrows),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes_arr = np.atleast_1d(axes).ravel()

    vmin = min(all_logq)
    vmax = max(all_logq)
    mappable = None

    for ax, (shift_m, pts) in zip(axes_arr, sorted(selected_by_shift.items())):
        deg = np.array([math.degrees(u) for u, _, _ in pts])
        freq = np.array([f for _, f, _ in pts])
        logq = np.log10(np.array([q for _, _, q in pts]))
        mappable = ax.scatter(
            deg,
            freq,
            c=logq,
            s=12,
            marker="o",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_title(f"shift={shift_m * 1e6:.0f} um", fontsize=10)
        ax.set_xlim(-10.0, 10.0)
        ax.set_ylim(FREQ_MIN_THz, FREQ_MAX_THz)
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.axvline(0.0, color="0.7", lw=0.8)

    for ax in axes_arr[n:]:
        ax.axis("off")

    for i, ax in enumerate(axes_arr[:n]):
        if i // ncols == nrows - 1:
            ax.set_xlabel("Incidence angle U (deg)")
        if i % ncols == 0:
            ax.set_ylabel("Eigenfrequency (THz)")

    fig.suptitle(
        f"gap={gap_m * 1e6:.0f} um, TE03 window (1.95-2.075 THz), mirrored to +/-U",
        fontsize=12,
    )
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes_arr[:n], fraction=0.02, pad=0.02)
        cbar.set_label("log10(Q)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (
        f"bilayer_gap{gap_m * 1e6:.0f}um_all_shifts_te03_logq_symm_pm10deg_1p95_2p075.png"
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    rows = load_rows(CSV_PATH)
    gaps = sorted({g for _, _, g, _, _ in rows})
    outputs: list[Path] = []
    for gap_m in gaps:
        output = plot_all_shifts_for_gap(rows, gap_m)
        if output is not None:
            outputs.append(output)

    if not outputs:
        raise ValueError("No figures generated. Check frequency window and input CSV.")

    print("Saved:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
