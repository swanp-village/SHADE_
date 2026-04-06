from ..model import Model
import numpy as np

def generate_model(crosstalk, rank):
    m = Model('crosstalk_pulse{}'.format(crosstalk))
    d1 = np.arange(-60, 0, 0.5)
    d2 = np.arange(-60, crosstalk, 0.5)
    f1 = np.repeat(-60, (m.FSR - m.length_of_3db_band) / 4 / 1e-12 - d2.size)
    f2 = np.repeat(-60, (m.FSR - m.length_of_3db_band) / 4 / 1e-12 - d2.size - d1.size)
    f3 = np.repeat(0, m.length_of_3db_band / 1e-12) - 1e-12
    f3[1] = f3[1] + 1e-12
    y = np.hstack((
        f1,
        d2,
        d2[::-1],
        f2,
        d1,
        f3,
        d1[::-1],
        f2,
        d2,
        d2[::-1],
        f1
    ))
    m.set_y(y)
    m.set_rank(rank)

    return m

data = [
    *[
        generate_model(i, 3)
        for i in np.arange(-5, -30, -1)
    ],
    *[
        generate_model(i, 2)
        for i in np.arange(-30, -40, -1)
    ],
    *[
        generate_model(i, 1)
        for i in np.arange(-40, -60, -1)
    ]
]
