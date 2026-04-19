from multiprocessing import Pool
import pytest
import numpy as np
from numba import njit

## Function to test
def mandelbrot_pixel(c: complex, max_iter: int) -> int:
    # From slide 20
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter

## >= 3 test functions
# Cases based on the ones from lecture 9 code_examples.md
def test_origin():
    assert mandelbrot_pixel(0+0j, 100) == 100

def test_far_outside():
    # escapes on iteration 1
    assert mandelbrot_pixel(5.0+0j, 100) == 1

def test_left_tip():
    assert mandelbrot_pixel(-2.5+0j, 100) == 1

## Using pytest.mark.parametrize (across multiple inputs)
KNOWN_CASES = [
    (0+0j,    100, 100),   # origin: never escapes
    (5.0+0j,  100,   1),   # far outside, escapes on iteration 1
    (-2.5+0j, 100,   1),   # left tip of set
]

@pytest.mark.parametrize("c, max_iter, expected", KNOWN_CASES)
def test_pixel_all(c, max_iter, expected):
    assert mandelbrot_pixel(c, max_iter) == expected


# from lecture 3 milestone 3
@njit                       # <-- entire function compiled !
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    """Fully JIT - compiled Mandelbrot --- structure identical to naive."""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):             # compiled loop
        for j in range(width):          # compiled loop
            c = x[j] + 1j * y[i]
            z = 0j                      # complex literal: type inference works!
            n = 0
            while n < max_iter and (z.real*z.real + z.imag*z.imag) <= 4.0:
                z = z*z + c
                n += 1
            result[i, j] = n
    return result

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


# Used below
# From slide 27, with the implementation filled in by me
# based on lecture 4.

# From L04 (unchanged — add `cache=True`` if not already there):
@njit(cache=True)
def mandelbrot_pixel_numba(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4.0: return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

# Used below
@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / (N - 1) # Bug: numpy and numba assumes N -1 
    dy = (y_max - y_min) / (N - 1) # so we must do the same then.
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel_numba(x_min + col*dx, c_imag, max_iter)
    return out

# Used below
def _worker(args): 
    return mandelbrot_chunk(*args) # plain Python, must be module-level

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max,
    # From lecture 05
    # Extend your L04 mandelbrot parallel to (include n chunks):
    # From slide 27
                        max_iter=100, n_workers=4, n_chunks=None):
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, tiny)         # warm-up: load JIT cache in workers
        parts = pool.map(_worker, chunks)
    return np.vstack(parts)

## Test Numpy/Numba/multiprocessing on a small grid (32x32)
def test_numpy_numba_parallel():
    N = 32
    x_min = -2.0
    x_max = 1.0
    y_min = -1.5
    y_max = 1.5
    max_iter = 100

    res1 = compute_mandelbrot_numpy(x_min, x_max, y_min, y_max, N, N, max_iter)
    res2 = mandelbrot_naive_numba(x_min, x_max, y_min, y_max, N, N, max_iter)
    res3 = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter)
    
    assert np.array_equal(res1, res2)
    assert np.array_equal(res1, res3)