import numpy as np
import numpy.typing as npt
from numba import njit
from mandelbrot_pixel import mandelbrot_pixel

@njit
def mandelbrot_chunk(
    row_start: int, 
    row_end: int, 
    N: int,
    x_min: float, 
    x_max: float, 
    y_min: float, 
    y_max: float, 
    max_iter: int
) -> npt.NDArray[np.int32]:
    """Computes the mandelbrot set based on a single chunk.
    
    Parameters
    ----------
    row_start : int
        The starting index at which the chunk begins.
    row_end : int
        The end index at which the chunk ends.
    N : int
        Number of columns (image width). Assumes squared result image.
    x_min : float
        Minimum x-value (left boundary) of the complex plain.
    x_max : float
        Maximum x-value (right boundary) of the complex plain.
    y_min : float
        Minimum y-value (bottom boundary) of the complex plain.
    y_max : float
        Maximum y-value (top boundary) of the complex plain.
    max_iter : int
        Maximum number of iterations per pixel.
    
    Returns
    -------
    npt.NDArray[np.int32]:
        2D array with iteration counts per pixel.
    """

    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out