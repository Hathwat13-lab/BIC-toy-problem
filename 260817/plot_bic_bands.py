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

3. TE03_Q_vs_deg_per_degree_log.png
   Same two branches, but sampled at one point per whole degree (0..10),
   log-scale y-axis, with the Q value labeled at each point.

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
    xmin: float = 0.0,
    xmax: float | None = None,
) -> None:
    bic_deg, bic_q, _ = bic
    lossy_deg, lossy_q, _ = lossy
    xmax_eff = xmax if xmax is not None else max(bic_deg.max(), lossy_deg.max())

    # Filter to the visible x-range before plotting so autoscale computes the
    # y-limits from what's actually shown, not from points outside the crop.
    bic_mask = (bic_deg >= xmin) & (bic_deg <= xmax_eff)
    lossy_mask = (lossy_deg >= xmin) & (lossy_deg <= xmax_eff)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.plot(
        lossy_deg[lossy_mask], lossy_q[lossy_mask],
        "o-", color="#1565C0", ms=3.5, lw=1.2, label="TE03 lossy",
    )
    ax.plot(
        bic_deg[bic_mask], bic_q[bic_mask],
        "o-", color="#C62828", ms=3.5, lw=1.2, label="TE03 BIC",
    )
    ax.set_xlabel(r"Angle $\theta$ (deg)")
    ax.set_ylabel("Q factor")
    ax.set_yscale(yscale)
    ax.set_xlim(xmin, xmax_eff)
    ax.grid(axis="y", which="both", color="0.9", lw=0.8)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_q(q: float) -> str:
    return f"{q:.2e}" if q >= 1e4 else f"{q:.1f}"


def plot_branch_q_per_degree(
    bic: tuple[np.ndarray, np.ndarray, np.ndarray],
    lossy: tuple[np.ndarray, np.ndarray, np.ndarray],
    out_path: Path,
) -> None:
    """One Q point per whole degree (0..10), log scale, value labeled at each point."""
    bic_deg, bic_q, _ = bic
    lossy_deg, lossy_q, _ = lossy
    deg_ticks = np.arange(0, round(bic_deg.max()) + 1)

    def sample(degs: np.ndarray, qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = [int(np.argmin(np.abs(degs - d))) for d in deg_ticks]
        return degs[idx], qs[idx]

    bic_x, bic_y = sample(bic_deg, bic_q)
    lossy_x, lossy_y = sample(lossy_deg, lossy_q)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.plot(lossy_x, lossy_y, "o-", color="#1565C0", ms=5, lw=1.2, label="TE03 lossy")
    ax.plot(bic_x, bic_y, "o-", color="#C62828", ms=5, lw=1.2, label="TE03 BIC")

    for x, y in zip(bic_x, bic_y):
        ax.annotate(
            _format_q(y), (x, y), textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=8, color="#C62828",
        )
    for x, y in zip(lossy_x, lossy_y):
        ax.annotate(
            _format_q(y), (x, y), textcoords="offset points", xytext=(0, -12),
            ha="center", fontsize=8, color="#1565C0",
        )

    ax.set_xlabel(r"Angle $\theta$ (deg)")
    ax.set_ylabel("Q factor")
    ax.set_yscale("log")
    ax.set_xticks(deg_ticks)
    ax.set_xlim(deg_ticks.min() - 0.3, deg_ticks.max() + 0.3)
    ax.margins(y=0.15)
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
        Path(__file__).parent / "TE03_Q_vs_deg_linear_zoom0.1-0.5deg.png",
        yscale="linear",
        xmin=0.1,
        xmax=0.5,
    )
    plot_branch_q_per_degree(bic, lossy, Path(__file__).parent / "TE03_Q_vs_deg_per_degree_log.png")

    print(
        "Saved band_structure_all_modes.png, TE03_Q_vs_deg.png, "
        "TE03_Q_vs_deg_linear.png, TE03_Q_vs_deg_linear_zoom0.1-0.5deg.png, "
        "and TE03_Q_vs_deg_per_degree_log.png"
    )


if __name__ == "__main__":
    main()
