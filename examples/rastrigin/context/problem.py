"""
Rastrigin function — a classic multimodal optimization benchmark.

The Rastrigin function has many local minima arranged in a regular lattice,
making it a challenging test for optimization. The global minimum is 0.0
at the origin (all zeros).

    f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))

Each variable is typically bounded to [-5.12, 5.12].
"""

import math

N_DIMS = 10
BOUNDS = (-5.12, 5.12)


def rastrigin(x):
    """Evaluate the Rastrigin function. Lower is better. Minimum is 0.0."""
    if len(x) != N_DIMS:
        raise ValueError(f"Expected {N_DIMS} dimensions, got {len(x)}")
    A = 10
    return A * len(x) + sum(xi**2 - A * math.cos(2 * math.pi * xi) for xi in x)
