from ..model import Model
import numpy as np

def generate_model(insertion_loss, rank):
    m = Model('insertion_loss{}'.format(insertion_loss))
    d = np.arange(-60, insertion_loss, 0.5)
    f1 = np.repeat(-60, (m.FSR - m.length_of_3db_band) / 2 / 1e-12 - d.size)
    f2 = np.repeat(insertion_loss, m.length_of_3db_band / 1e-12) - 1e-12
    f2[1] = f2[1] + 1e-12
    y = np.hstack((
        f1,
        d,
        f2,
        d[::-1],
        f1
    ))
    m.set_y(y)
    m.set_rank(1)

    return m

data = [
    *[
        generate_model(i, 1)
        for i in np.arange(-1, -4, -0.2)
    ],
    *[
        generate_model(i, 2)
        for i in np.arange(-4, -20, -1)
    ],
    *[
        generate_model(i, 3)
        for i in np.arange(-20, -30, -1)
    ]
]
