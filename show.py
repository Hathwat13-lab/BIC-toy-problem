"""Plot TE and TM dispersion bands from the CSV produced by 2dpho.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from parameters import a, d, eps1, eps2, Gmf, Nx, c

num_bands = 8
nu_max = 0.8

# Select the most recently written CSV for the shared parameter set.
csv_pattern = (
    f"2dpho_a={a:g}_d={d:g}_eps1={eps1:g}_eps2={eps2:g}_"
    f"Nx={Nx}_Gmf={Gmf:g}_*.csv"
)
csv_files = sorted(Path(".").glob(csv_pattern))
if not csv_files:
    raise FileNotFoundError(f"No result CSV matches: {csv_pattern}")

csv_filename = csv_files[-1]
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
