from hypothesis import given, settings
from hypothesis.strategies import integers
from hypothesis.strategies import complex_numbers

def mandelbrot_pixel(c: complex, max_iter: int) -> int:
    # From lecture 9, slide 20
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter

@given(integers())          # generates random integers
def test_abs_non_negative(x):
    assert abs(x) >= 0      # must hold for ANY integer

# Draw random points with |c| <= 3 (covers both inside and outside the set)
@given(complex_numbers(max_magnitude=3.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_result_in_range(c):
    assert 0 <= mandelbrot_pixel(c, 100) <= 100

# Draw random points far outside the set (|c| between 3 and 10)
@given(complex_numbers(min_magnitude=3.0, max_magnitude=10.0,
                       allow_nan=False, allow_infinity=False))
def test_outside_set_escapes(c):
    assert mandelbrot_pixel(c, 100) < 100

assert mandelbrot_pixel(0+2j, 100) == 2   # catches it: mutant returns 1, assertion fails
assert mandelbrot_pixel(0+2j, 100) < 100  # misses it:  mutant returns 1, assertion passes