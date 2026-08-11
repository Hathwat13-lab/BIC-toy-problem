"""Shared parameters for the 2D photonic-crystal solver and plotter."""

import os


def _env_float(name, default):
    """Read a float override for a sweep, otherwise use the default."""
    return float(os.environ.get(name, default))

# Geometry and material parameters (nm)
a = 10.0
d = 1.65
eps1 = 1.0
eps2 = 8.9

# Numerical parameters
Gmf = 40.0
Nx = 50

# Speed of light in nm / ps
c = 3 * 10**5
