import pytest
from calculator import add, subtract


def test_add_positive():
    assert add(3, 5) == 8


def test_add_negative():
    assert add(-2, -3) == -5


def test_add_mixed():
    assert add(-1, 4) == 3


def test_add_floats():
    assert add(1.5, 2.5) == 4.0


def test_subtract_positive():
    assert subtract(10, 4) == 6


def test_subtract_negative():
    assert subtract(-3, -2) == -1


def test_subtract_mixed():
    assert subtract(5, -3) == 8


def test_subtract_floats():
    assert subtract(3.5, 1.5) == 2.0


def test_subtract_to_zero():
    assert subtract(7, 7) == 0
