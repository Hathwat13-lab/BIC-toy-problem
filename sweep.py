"""Run the requested eps2 and d/a sweeps sequentially.

Resumable: a case is skipped if its output CSV already exists, so a
re-run after an interruption picks up where it left off.
"""

import os
from pathlib import Path
import subprocess
import sys

from parameters import a, eps1, Gmf, Nx


EPS2_SWEEP = [3.0, 5.0, 7.0, 11.0, 13.0, 15.0]
D_OVER_A_SWEEP = [0.075, 0.105, 0.175, 0.225, 0.400, 0.750]

CASES = [("eps2 sweep", 2.5, eps2) for eps2 in EPS2_SWEEP]
CASES += [("d/a sweep", a * d_over_a, 15.0) for d_over_a in D_OVER_A_SWEEP]


def output_exists(workspace, d_value, eps2_value):
    pattern = (
        f"2dpho_a={a:g}_d={d_value:g}_eps1={eps1:g}_eps2={eps2_value:g}_"
        f"Nx={Nx}_Gmf={Gmf:g}_*.csv"
    )
    return next(workspace.glob(pattern), None) is not None


def run_sweep():
    workspace = Path(__file__).resolve().parent

    for case_index, (sweep_name, d_value, eps2_value) in enumerate(CASES, start=1):
        if output_exists(workspace, d_value, eps2_value):
            print(
                f"[{case_index}/{len(CASES)}] {sweep_name}: "
                f"d={d_value:g}, eps2={eps2_value:g} -> output already exists, skipping",
                flush=True,
            )
            continue

        environment = os.environ.copy()
        environment["BIC_D"] = f"{d_value:g}"
        environment["BIC_EPS2"] = f"{eps2_value:g}"

        print(
            f"[{case_index}/{len(CASES)}] {sweep_name}: "
            f"d={d_value:g} nm, d/a={d_value / a:g}, eps2={eps2_value:g}",
            flush=True,
        )
        subprocess.run(
            [sys.executable, "2dpho.py"],
            cwd=workspace,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    run_sweep()
