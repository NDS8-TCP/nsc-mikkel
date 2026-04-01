#!/usr/bin/env python
# coding: utf-8


import numpy as np
from numba import njit
# From lecture 4

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4.0: return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out

from dask import delayed
import dask
# From lecture 6

def mandelbrot_dask(N, x_min, x_max, y_min, y_max,
                    max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(
            row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)

import statistics
import time
from dask.distributed import Client

# From lexture 6

N, max_iter = 4096, 100 # Changed from 1024 to 4096
X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
client = Client("tcp://10.92.1.106:8786") # Use remote cluster at strato
client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX,  # warm up all workers
                                    Y_MIN, Y_MAX, 10))

n_chunks_arr = [1, 2, 4, 8, 16, 32]
times_res = []
lifs = []
vs1x = []
speedups = []

for n_chunks in n_chunks_arr:
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = mandelbrot_dask(
            N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks)
        times.append(time.perf_counter() - t0)

    T_p = statistics.median(times)
    times_res.append(T_p)
    T_1 = times_res[0]
    vs1x.append(T_p / T_1)
    speedups.append(T_1 / T_p)
    lifs.append(8 * T_p / T_1 - 1)

client.close()

print("-"*10)
print(f"Number of chunks: {n_chunks_arr}")
print(f"Times: {times_res}")
print(f"Lifs: {lifs}")
print(f"vs 1x {vs1x}")
print(f"Speedups {speedups}")
print("-"*10)
