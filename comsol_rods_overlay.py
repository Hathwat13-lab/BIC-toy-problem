"""Overlay a 2dpho_rods.py band-structure result (lines) with a COMSOL rods export (points).

Same idea as comsol_2dpho_overlay.py, but for the circular-rod (Bessel J1)
solver variant instead of the vein/slit variant.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from parameters import a, Nx, c
from comsol import parse_comsol_bands
from comsol_2dpho_overlay import read_2dpho_bands, comsol_index_to_k_path

nu_max = 1.5
num_python_bands = 8

python_csv = Path(
    r"C:\Users\cojyi\Desktop\BIC\python_csv\2dpho_rods_a=10_r=2_eps1=1_eps2=8.9_Nx=50_Gmf=40_20260813_115749.csv"
)
comsol_csv = Path(r"C:\Users\cojyi\Desktop\BIC\comsol_csv\rods.csv")
output_png = Path(r"C:\Users\cojyi\Desktop\BIC\png\comsol_2dpho_overlay_rods.png")


def main() -> None:
    k_path, TE_nu, TM_nu = read_2dpho_bands(python_csv)
    comsol_index, comsol_bands = parse_comsol_bands(comsol_csv)
    comsol_x = comsol_index_to_k_path(comsol_index, k_path, Nx)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)

    for band in range(num_python_bands):
        # solver 풀 때 두 모드의 label을 반대로 해서 plot 시 legend를 거꾸로 잡아줌 (show.py와 동일)
        ax.plot(k_path, TM_nu[:, band], color="red", lw=1.2, label="TE" if band == 0 else None)
        ax.plot(k_path, TE_nu[:, band], color="blue", lw=1.2, label="TM" if band == 0 else None)

    for i, band in enumerate(comsol_bands):
        ax.plot(comsol_x, band, "o", color="black", ms=2.5, mew=0, label="COMSOL" if i == 0 else None)

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
