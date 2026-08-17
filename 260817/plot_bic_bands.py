"""Plot the U=0-10deg COMSOL eigenfrequency sweep in 260817(80_10deg).csv.

Produces two figures, both saved next to this script:

1. band_structure_all_modes.png
   Every eigenfrequency (real part) at every angle, plotted as a scatter
   "dot" band structure vs U converted from radians to degrees.

2. TE03_Q_vs_deg.png
   Q factor vs angle for the two TE03 branches that meet at U=0:
   - TE03 BIC   (high-Q, starts at 2.0324 THz)   -> red
   - TE03 lossy (starts at 2.1836+0.017492i THz) -> blue
   Each branch is tracked step-to-step by nearest real-frequency neighbor
   so the two curves don't swap onto neighboring modes.

Usage
-----
python plot_bic_bands.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path(__file__).parent / "260817(80_10deg).csv"

# Seed eigenfrequencies at U=0 (rad) that identify the two branches to track.
BIC_SEED = 2.0324
LOSSY_SEED = complex(2.1836, 0.017492)


def load_sweep(path: Path) -> dict[float, list[tuple[complex, float]]]:
    """Return {U_rad: [(eigenfrequency_THz, Q), ...]} sorted by U."""
    by_u: dict[float, list[tuple[complex, float]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.reader(file):
            if not row or row[0].lstrip().startswith("%"):
                continue
            try:
                u = float(row[0])
                freq = complex(row[1].strip().replace("i", "j"))
                q = float(row[3])
            except (IndexError, ValueError):
                continue
            by_u.setdefault(u, []).append((freq, q))
    return dict(sorted(by_u.items()))


def track_branch(by_u: dict[float, list[tuple[complex, float]]], seed: complex):
    """Follow the mode nearest `seed` at U=0, then nearest real-freq neighbor
    at each subsequent angle. Returns (deg_array, Q_array, freq_array)."""
    us = list(by_u)
    degs, qs, freqs = [], [], []
    prev_freq = seed
    for u in us:
        modes = by_u[u]
        freq, q = min(modes, key=lambda fq: abs(fq[0].real - prev_freq.real))
        degs.append(u * 180.0 / math.pi)
        qs.append(q)
        freqs.append(freq)
        prev_freq = freq
    return np.array(degs), np.array(qs), np.array(freqs)


def plot_band_structure(by_u: dict[float, list[tuple[complex, float]]], out_path: Path) -> None:
    degs, res = [], []
    for u, modes in by_u.items():
        for freq, _q in modes:
            degs.append(u * 180.0 / math.pi)
            res.append(freq.real)

    fig, ax = plt.subplots(figsize=(3.6, 5.0), constrained_layout=True)
    ax.plot(degs, res, "o", color="#1565C0", ms=2.0, mew=0)
    ax.set_xlabel(r"Angle $\theta$ (deg)")
    ax.set_ylabel("Eigenfrequency (THz)")
    ax.set_xlim(0, max(degs))
    ax.set_ylim(1.8, 2.6)
    ax.grid(axis="y", color="0.9", lw=0.8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_branch_q(
    bic: tuple[np.ndarray, np.ndarray, np.ndarray],
    lossy: tuple[np.ndarray, np.ndarray, np.ndarray],
    out_path: Path,
    yscale: str = "log",
    xmax: float | None = None,
) -> None:
    bic_deg, bic_q, _ = bic
    lossy_deg, lossy_q, _ = lossy

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.plot(lossy_deg, lossy_q, "o-", color="#1565C0", ms=3.5, lw=1.2, label="TE03 lossy")
    ax.plot(bic_deg, bic_q, "o-", color="#C62828", ms=3.5, lw=1.2, label="TE03 BIC")
    ax.set_xlabel(r"Angle $\theta$ (deg)")
    ax.set_ylabel("Q factor")
    ax.set_yscale(yscale)
    ax.set_xlim(0, xmax if xmax is not None else max(bic_deg.max(), lossy_deg.max()))
    ax.grid(axis="y", which="both", color="0.9", lw=0.8)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    by_u = load_sweep(CSV_PATH)

    plot_band_structure(by_u, Path(__file__).parent / "band_structure_all_modes.png")

    bic = track_branch(by_u, complex(BIC_SEED, 0.0))
    lossy = track_branch(by_u, LOSSY_SEED)

    # Sanity check: flag any step-to-step jump that suggests branch mixing.
    for name, (degs, qs, freqs) in (("BIC", bic), ("lossy", lossy)):
        d_freq = np.abs(np.diff(freqs.real))
        jump = np.argmax(d_freq)
        if d_freq[jump] > 5e-3:
            print(
                f"WARNING: possible branch mix in {name} branch near "
                f"{degs[jump]:.2f} deg (Delta_f={d_freq[jump]:.4g} THz)"
            )

    plot_branch_q(bic, lossy, Path(__file__).parent / "TE03_Q_vs_deg.png", yscale="log")
    plot_branch_q(bic, lossy, Path(__file__).parent / "TE03_Q_vs_deg_linear.png", yscale="linear")
    plot_branch_q(
        bic,
        lossy,
        Path(__file__).parent / "TE03_Q_vs_deg_linear_zoom0-0.5deg.png",
        yscale="linear",
        xmax=0.5,
    )

    print(
        "Saved band_structure_all_modes.png, TE03_Q_vs_deg.png, "
        "TE03_Q_vs_deg_linear.png, and TE03_Q_vs_deg_linear_zoom0-0.5deg.png"
    )


if __name__ == "__main__":
    main()
