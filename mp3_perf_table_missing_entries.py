import numpy as np
from numba import njit
from timeit import timeit

# From lecture 3

@njit
def mandelbrot_point_numba(c, max_iter=100):
    z = 0j
    for n in range(max_iter):
        if z.real*z.real+z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter

@njit
def mandelbrot_numba_typed(xmin, xmax, ymin, ymax,
                            width, height, max_iter=100, dtype=np.float64):
    x = np.linspace(xmin, xmax, width).astype(dtype)
    y = np.linspace(ymin, ymax, height).astype(dtype)
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            result[i, j] = mandelbrot_point_numba(c, max_iter)
    return result

# Add a row for each resolution you tested. Include a Numba f32 result at the same resolution as each GPU entry so speedups are fair comparisons.

# warmup
mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float32)

print(timeit(lambda: mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float32), number=3)/3)
print(timeit(lambda: mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float64), number=3)/3)
print(timeit(lambda: mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 2048, 2048, dtype=np.float32), number=3)/3)
print(timeit(lambda: mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 2048, 2048, dtype=np.float64), number=3)/3)