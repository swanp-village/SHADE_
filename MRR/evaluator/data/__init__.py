from random import sample
from math import floor
from .crosstalk_pulse import data as crosstalk_pulse
from .crosstalk import data as crosstalk
from .insertion_loss import data as insertion_loss
from .ripple import data as ripple
from .stopband import data as stopband
from .three_db_band import data as three_db_band


all_data = [
    *crosstalk_pulse,
    *crosstalk,
    *insertion_loss,
    *ripple,
    *stopband,
    *three_db_band,
]


def split(data, test_size):
    length = len(data)
    test_length = floor(length * test_size)
    shuffled_data = sample(data, length)
    test = shuffled_data[:test_length]
    train = shuffled_data[test_length:]

    return train, test


def get_splitted_data(test_size=0.25):
    data = [
        crosstalk_pulse,
        crosstalk,
        insertion_loss,
        ripple,
        stopband,
        three_db_band,
    ]
    train = []
    test = []

    for d in data:
        d_train, d_test = split(d, test_size)
        train = [*train, *d_train]
        test = [*test, *d_test]

    return train, test
