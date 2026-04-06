import json
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from MRR.evaluator import evaluate_band

from .config import config
from .data import all_data, get_splitted_data

weight_of_binary_evaluation = np.array([0.5])
weight_list = list(
    product(
        *[
            np.arange(1, 6, 0.5),
            np.arange(1, 6, 0.5),
            np.arange(1, 6, 0.5),
            np.arange(1, 6, 0.5),
            np.arange(1, 6, 0.5),
            np.array([1]),
            np.arange(1, 6, 0.5),
            weight_of_binary_evaluation,
            weight_of_binary_evaluation,
            weight_of_binary_evaluation,
            weight_of_binary_evaluation,
            weight_of_binary_evaluation,
            weight_of_binary_evaluation,
            weight_of_binary_evaluation,
        ]
    )
)
train, test = get_splitted_data(0.3)
train_rank_list = np.array([d.rank for d in train])
test_rank_list = np.array([d.rank for d in test])


def train_data(weight):
    train_evaluate_result = np.array(
        [
            evaluate_band(
                x=d.x,
                y=d.y,
                center_wavelength=config.center_wavelength,
                length_of_3db_band=config.length_of_3db_band,
                max_crosstalk=config.max_crosstalk,
                H_i=config.H_i,
                H_p=config.H_p,
                H_s=config.H_s,
                r_max=config.r_max,
                weight=weight,
                ignore_binary_evaluation=False,
            )
            for d in train
        ]
    )
    test_evaluate_result = np.array(
        [
            evaluate_band(
                x=d.x,
                y=d.y,
                center_wavelength=config.center_wavelength,
                length_of_3db_band=config.length_of_3db_band,
                max_crosstalk=config.max_crosstalk,
                H_i=config.H_i,
                H_p=config.H_p,
                H_s=config.H_s,
                r_max=config.r_max,
                weight=weight,
                ignore_binary_evaluation=False,
            )
            for d in test
        ]
    )
    train_evaluate_result = np.array(train_evaluate_result).reshape(-1, 1)
    test_evaluate_result = np.array(test_evaluate_result).reshape(-1, 1)
    X_train = train_evaluate_result
    y_train = train_rank_list
    X_test = test_evaluate_result
    y_test = test_rank_list
    clf = SVC()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score, weight)

    return score


def train_evaluator():
    max_score = 0

    with Pool() as p:
        scores = p.map(train_data, weight_list)

    scores = np.array(scores)
    weights = np.array(weight_list)
    max_score = np.max(scores)
    optimized_weights = weights[scores == max_score]
    print(max_score)
    save_result(max_score, optimized_weights.tolist())


def show_data(skip_plot):
    for data_i in all_data:
        result = evaluate_band(
            x=data_i.x,
            y=data_i.y,
            center_wavelength=config.center_wavelength,
            length_of_3db_band=config.length_of_3db_band,
            max_crosstalk=config.max_crosstalk,
            H_i=config.H_i,
            H_p=config.H_p,
            H_s=config.H_s,
            r_max=config.r_max,
            weight=config.weight,
            ignore_binary_evaluation=False,
        )
        print(result)
        data_i.export_graph(skip_plot)


def save_result(score, weight):
    p = Path.cwd().joinpath("MRR", "Evaluator", "Model", "result.json")
    result = {"score": score, "weights": weight}
    src = json.dumps(result, indent=4)
    p.write_text(src)
