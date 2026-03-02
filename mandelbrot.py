# Template from slide 30, lecture 01
"""
Mandelbrot Set Generator

Author: Mikkel Korsgaard Sørensen
Course: Numerical Scientific Computing 2026
"""
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
