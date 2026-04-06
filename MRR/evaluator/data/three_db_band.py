from ..model import Model
import numpy as np

def generate_model(three_db_band, rank):
    m = Model('three_db_band{}'.format(three_db_band))
    d = np.arange(-60, 0, 0.5)
    length_of_3db_band = m.length_of_3db_band * three_db_band
    f1 = np.repeat(-60, (m.FSR - length_of_3db_band) / 2 / 1e-12 - d.size)
    f2 = np.repeat(0, length_of_3db_band / 1e-12) - 1e-12
    f2[1] = f2[1] + 1e-12
    y = np.hstack((
        f1,
        d,
        f2,
        d[::-1],
        f1
    ))
    m.set_y(y)
    m.set_rank(rank)

    return m

data = [
    *[
        generate_model(i, 3)
        for i in np.arange(0.80, 0.95, 0.01)
    ],
    *[
        generate_model(i, 2)
        for i in np.arange(0.95, 0.99, 0.002)
    ],
    *[
        generate_model(i, 1)
        for i in np.arange(0.99, 1.02, 0.002)
    ],
    *[
        generate_model(i, 2)
        for i in np.arange(1.02, 1.06, 0.002)
    ],
    *[
        generate_model(i, 3)
        for i in np.arange(1.06, 1.20, 0.01)
    ]
]
