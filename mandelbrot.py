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
    """Computes the escape iteration count for a single point in the mandelbrot set.
    
    Parameters
    ----------
    c : complex 
        The complex coordinate to test in the Mandelbrot iteration.
    max_iter : int
        Maximum number of iterations to perform.
    
    Returns
    -------
    int
        Number of iterations computed. 
    """
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
    """Computes the mandelbrot set for a rectangular defined region.
    
    Parameters
    ----------
    x_min : float
        Minimum x-value (left boundary) of the complex plane.
    x_max : float
        Maximum x-value (right boundary) of the complex plane.
    y_min : float
        Minimum y-value (bottom boundary) of the complex plane.
    y_max : float
        Maximum y-value (top boundary) of the complex plane.
    width : int
        Number of pixels to compute width wise.
    height : int
        Number of pixels to compute height wise.
    max_iter : int
        Maximum number of iterations per pixel.
    
    Returns
    -------
    list[list[int]]
        A Python matrix (height x width) with iteration counts per pixel.
    """
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
    """Computes the mandelbrot set for a rectangular defined region.
    
    Parameters
    ----------
    x_min : float
        Minimum x-value (left boundary) of the complex plane.
    x_max : float
        Maximum x-value (right boundary) of the complex plane.
    y_min : float
        Minimum y-value (bottom boundary) of the complex plane.
    y_max : float
        Maximum y-value (top boundary) of the complex plane.
    width : int
        Number of pixels to compute width wise.
    height : int
        Number of pixels to compute height wise.
    max_iter : int
        Maximum number of iterations per pixel.
    
    Returns
    -------
    NDArray[int]
        A numpy matrix (height x width) with iteration counts per pixel.
    """
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
