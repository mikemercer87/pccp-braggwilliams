from mysoftware import *
import pytest

def test_square_integers():
    assert square(2) == 4
    assert square(0) == 0
    assert square(-1) == 1
    assert square(3) == 9

def test_1overr():
    assert oneoverr(1) == 1.0
    assert oneoverr(2) == 0.5
    with pytest.raises(ZeroDivisionError):
        oneoverr(0) == 0.0
    assert oneoverr(-3) == (-1.0/3.0)
