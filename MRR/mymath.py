from functools import reduce

import numpy as np
import numpy.typing as npt


def is_zero(x: np.float_, y: np.float_) -> np.bool_:
    """
    x > y > 0
    """
    result: np.float_ = x / y - np.floor(x / y, dtype=np.float_)
    return result < 0.1


def lcm(xs: npt.NDArray[np.float_]) -> np.float_:
    return reduce(_lcm, xs)


def mod(x: np.float_, y: np.float_) -> np.float_:
    result: np.float_ = x - y * np.floor(x / y)
    return result


def _gcd(x: np.float_, y: np.float_) -> np.float_:
    """
    x > y
    """
    n = 0
    while y != 0 and not is_zero(x, y) and n < 10:
        x, y = y, mod(x, y)
        n += 1

    if is_zero(x, y):
        return y
    else:
        return x


def _lcm(x: np.float_, y: np.float_) -> np.float_:
    if x > y:
        return x * y / _gcd(x, y)
    else:
        return x * y / _gcd(y, x)
