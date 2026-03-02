# Template from slide 30, lecture 01
"""
Mandelbrot Set Generator

Author: Mikkel Korsgaard Sørensen
Course: Numerical Scientific Computing 2026
"""
import numpy as np
# This is a comment

def f(x):
    """
    Example function.

    Parameters
    ----------
    x : float
        Input value

    Returns
    -------
    float
        Output value
    """
    # TODO: Implement the algorithm
    pass


def mandelbrot_point(c: complex, max_iter: int) -> int:
    z = 0j
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z**2 + c

    return max_iter


def compute_mandelbrot(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    max_iter: int
) -> list[list[int]]:
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)

    img = []

    for i in range(height):
        img_row = []
        for j in range(width):
            img_row.append(mandelbrot_point(x[j] + 1j * y[i], max_iter))
        img.append(img_row)

    return img


def compute_mandelbrot_numpy(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    max_iter: int
):
    # Inspired by slide 30, lecture 02
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)

    X, Y = np.meshgrid(x, y)

    C = X + 1j*Y

    Z = np.zeros_like(C)
    M = np.zeros_like(C, dtype=int)
    for _ in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1

    return M
