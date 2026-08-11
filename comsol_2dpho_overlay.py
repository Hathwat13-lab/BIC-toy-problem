"""Overlay a 2dpho.py band-structure result (lines) with a COMSOL export (points).

Combines the CSV parsing from comsol.py (COMSOL Global Evaluation export) and
the CSV parsing from show.py (2dpho.py solver output) onto one Gamma-X-M-Gamma
plot, capped at nu_max, so the two calculations can be compared directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from parameters import a, Nx, c
from comsol import parse_comsol_bands

nu_max = 0.8
num_python_bands = 8

python_csv = Path(
    r"C:\Users\cojyi\Desktop\BIC\2dpho_a=10_d=1.65_eps1=1_eps2=8.9_Nx=35_Gmf=15_20260807_130452.csv"
)
comsol_csv = Path(r"C:\Users\cojyi\Desktop\BIC\20260811.csv")
output_png = Path(r"C:\Users\cojyi\Desktop\BIC\comsol_2dpho_overlay.png")


def read_2dpho_bands(filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (k_path, TE_nu, TM_nu) from a 2dpho.py solver CSV."""
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    kvecs = data[:, 1:3]
    total_bands = (data.shape[1] - 3) // 2
    TE_omega = data[:, 3:3 + total_bands]
    TM_omega = data[:, 3 + total_bands:3 + 2 * total_bands]
    TE_nu = TE_omega * a / (2 * np.pi * c)
    TM_nu = TM_omega * a / (2 * np.pi * c)

    step_lengths = np.linalg.norm(np.diff(kvecs, axis=0), axis=1)
    k_path = np.concatenate(([0.0], np.cumsum(step_lengths)))
    return k_path, TE_nu, TM_nu


def main() -> None:
    k_path, TE_nu, TM_nu = read_2dpho_bands(python_csv)
    comsol_index, comsol_bands = parse_comsol_bands(comsol_csv)
    if len(comsol_index) != len(k_path):
        raise ValueError(
            f"COMSOL k-point count ({len(comsol_index)}) does not match the "
            f"2dpho k-path length ({len(k_path)}); check Nx matches for both."
        )

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)

    for band in range(num_python_bands):
        # solver 풀 때 두 모드의 label을 반대로 해서 plot 시 legend를 거꾸로 잡아줌 (show.py와 동일)
        ax.plot(k_path, TM_nu[:, band], color="red", lw=1.2, label="TE" if band == 0 else None)
        ax.plot(k_path, TE_nu[:, band], color="blue", lw=1.2, label="TM" if band == 0 else None)

    for i, band in enumerate(comsol_bands):
        ax.plot(k_path, band, "o", color="black", ms=2.5, mew=0, label="COMSOL" if i == 0 else None)

    symmetry_indices = [0, Nx, 2 * Nx, 3 * Nx]
    symmetry_positions = k_path[symmetry_indices]
    for position in symmetry_positions[1:-1]:
        ax.axvline(position, color="black", linewidth=0.8, alpha=0.5)

    ax.set_xticks(symmetry_positions, [r"$\Gamma$", "X", "M", r"$\Gamma$"])
    ax.set_xlim(k_path[0], k_path[-1])
    ax.set_ylim(0, nu_max)
    ax.set_xlabel("Wave vector")
    ax.set_ylabel(r"Normalized frequency $\nu = \omega a/(2\pi c)$")
    ax.legend()

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
