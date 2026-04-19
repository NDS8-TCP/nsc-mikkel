# From lecture 9, slide 20

def mandelbrot_pixel(c: complex, max_iter: int) -> int:
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter