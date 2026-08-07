"""Plot TE and TM dispersion bands from the CSV produced by 2dpho.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from parameters import a, d, eps1, eps2, Gmf, Nx, c

num_bands = 8
nu_max = 0.8

# Choose the exact result file to plot.
csv_filename = Path(
    "2dpho_a=10_d=1.65_eps1=1_eps2=8.9_Nx=35_Gmf=15_20260807_130452.csv"
)
if not csv_filename.is_file():
    raise FileNotFoundError(f"CSV file not found: {csv_filename}")

data = np.loadtxt(csv_filename, delimiter=",", skiprows=1)

kvecs = data[:, 1:3]
total_bands = (data.shape[1] - 3) // 2
TE_omega = data[:, 3:3 + total_bands]
TM_omega = data[:, 3 + total_bands:3 + 2 * total_bands]

TE_nu = TE_omega * a / (2 * np.pi * c)
TM_nu = TM_omega * a / (2 * np.pi * c)

# Cumulative distance along the Gamma-X-M-Gamma path.
step_lengths = np.linalg.norm(np.diff(kvecs, axis=0), axis=1)
k_path = np.concatenate(([0.0], np.cumsum(step_lengths)))

for band in range(num_bands):
    plt.plot(k_path, TM_nu[:, band], color="blue", label="TM" if band == 0 else None)
    plt.plot(k_path, TE_nu[:, band], color="red", label="TE" if band == 0 else None)

symmetry_indices = [0, Nx, 2 * Nx, 3 * Nx]
symmetry_positions = k_path[symmetry_indices]
for position in symmetry_positions[1:-1]:
    plt.axvline(position, color="black", linewidth=0.8, alpha=0.5)

plt.xticks(symmetry_positions, [r"$\Gamma$", "X", "M", r"$\Gamma$"])
plt.xlim(k_path[0], k_path[-1])
plt.ylim(0, nu_max)
plt.xlabel("Wave vector")
plt.ylabel(r"Normalized frequency $\nu = \omega a/(2\pi c)$")
plt.legend()
plt.tight_layout()
plt.show()
