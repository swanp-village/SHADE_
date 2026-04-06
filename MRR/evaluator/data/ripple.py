from ..model import Model
import numpy as np

def generate_model(ripple, rank):
    m = Model('ripple{}'.format(ripple))
    d1 = np.arange(-60, 0, 0.5)
    d2 = np.arange(ripple, 0, 0.5)
    f1 = np.repeat(0, m.length_of_3db_band / 5 / 1e-12 - d2.size) - 1e-12
    f2 = np.repeat(ripple, m.length_of_3db_band / 5 / 1e-12 - d2.size) + 1e-12
    f3 = np.repeat(0, m.length_of_3db_band / 5 / 1e-12) - 1e-12
    f4 = np.repeat(-60, (m.FSR - m.length_of_3db_band) / 2 / 1e-12 - d1.size)
    f1[1] = f1[1] + 1e-12
    f2[1] = f2[1] - 1e-12
    f3[1] = f3[1] + 1e-12
    y = np.hstack((
        f4,
        d1,
        f1,
        d2[::-1],
        f2,
        d2,
        f3,
        d2[::-1],
        f2,
        d2,
        f1,
        d1[::-1],
        f4
    ))
    m.set_y(y)
    m.set_rank(rank)

    return m

data = [
    *[
        generate_model(i / 10, 1)
        for i in np.arange(0, -10, -1)
    ],
    *[
        generate_model(i / 10, 2)
        for i in np.arange(-10, -20, -1)
    ],
    *[
        generate_model(i / 10, 3)
        for i in np.arange(-20, -50, -1)
    ]
]
