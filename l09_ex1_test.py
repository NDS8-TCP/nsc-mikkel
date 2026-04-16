# From slide 20, adapted to be used in a function

from numba import njit

# From slide 20


def mandelbrot_pixel(c: complex, max_iter: int) -> int:
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter

# From lecture 3 milestone 3

# From slide 36. Approach A (where I have filled in the gaps in the hybrid)


@njit
def mandelbrot_point_numba(c, max_iter=100):
    z = 0j
    for n in range(max_iter):
        if z.real*z.real+z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter


def test_ex1_step1():
    # origin: never escapes
    assert mandelbrot_pixel(0+0j, 100) == 100

    # far outside: escapes on iter 1
    assert mandelbrot_pixel(5.0+0j, 100) == 1


def test_ex1_step2():
    assert mandelbrot_point_numba(0+0j, 100) == 100
    assert mandelbrot_point_numba(5.0+0j, 100) == 1
