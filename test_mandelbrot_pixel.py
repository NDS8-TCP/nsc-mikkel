# From lecture 9, code_example.md (moodle)

import pytest
from mandelbrot_pixel import mandelbrot_pixel

KNOWN_CASES = [
    (0+0j,    100, 100),   # origin: never escapes
    (5.0+0j,  100,   1),   # far outside, escapes on iteration 1
    (-2.5+0j, 100,   1),   # left tip of set
]

@pytest.mark.parametrize("c, max_iter, expected", KNOWN_CASES)
def test_mandelbrot_pixel(c, max_iter, expected):
    assert mandelbrot_pixel(c, max_iter) == expected
