import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from config.model import OptimizationConfig, SimulationConfig


class Logger:
    def __init__(self) -> None:
        format = "%Y-%m-%d-%H-%M-%S"
        self.result_path = Path.cwd() / "result"
        self.result_path.mkdir(exist_ok=True)
        self.target = self.result_path / datetime.now().strftime(format)
        #self.target.mkdir()
        self.target.mkdir(parents=True, exist_ok=True)    #250813に追加。同じ名前のディレクトリが存在しても強制的にファイルを作成する様に

    def save_optimization_config(self, config: OptimizationConfig) -> None:
        self.config = config
        src = json.dumps(asdict(config), indent=4)
        path = self.target / "config.json"
        path.write_text(src)

    def save_simulation_config(self, config: SimulationConfig) -> None:
        self.config = config
        src = json.dumps(
            {**asdict(config), "K": config.K.tolist(), "L": config.L.tolist(), "lambda_limit": []},
            indent=4,
        )
        path = self.target / "config.json"
        path.write_text(src)

    def save_config(
        self,
        L: npt.NDArray[np.float_],
        K: npt.NDArray[np.float_],
        lambda_limit: npt.NDArray[np.float_],
        **kwargs: dict[str, Any],
    ) -> None:
        src = json.dumps(
            {**kwargs, "K": K.tolist(), "L": L.tolist(), "lambda_limit": lambda_limit.tolist()},
            indent=4,
        )
        path = self.target / "config.json"
        path.write_text(src)

    def generate_image_path(self, name: str = "out") -> Path:
        return self.target / f"{name}.pdf"

    def save_data_as_csv(self, x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], name: str = "out") -> None:
        path = self.target / f"{name}.tsv"
        with open(path, "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(x.tolist(), y.tolist()))

    def save_evaluation_value(
        self, E_list: list[float], method_list: list[int], name: str = "evaluation_value"
    ) -> None:
        path = self.target / f"{name}.tsv"
        with open(path, "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(E_list, method_list))

    def save_result(self, L: npt.NDArray[np.float_], K: npt.NDArray[np.float_]) -> None:
        result = {
            "eta": self.config.eta,
            "alpha": self.config.alpha,
            "K": K.tolist(),
            "L": L.tolist(),
            "n_eff": self.config.n_eff,
            "n_g": self.config.n_g,
            "center_wavelength": self.config.center_wavelength,
        }
        src = json.dumps(result, indent=4)
        path = self.target / "result.json"
        path.write_text(src)

    def save_DE_data(
        self,
        N_list: list[npt.NDArray[np.int_]],
        L_list: list[npt.NDArray[np.float_]],
        K_list: list[npt.NDArray[np.float_]],
        FSR_list: npt.NDArray[np.float_],
        E_list: list[float],
        method_list: list[int],
        best_E_list: list[float],
        analyze_score_list: list[float],
    ):
        n = range(len(N_list))
        with open(self.target / "N_list.tsv", "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(n, N_list))
        with open(self.target / "L_list.tsv", "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(n, L_list))
        with open(self.target / "K_list.tsv", "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(n, K_list))
        with open(self.target / "FSR_list.tsv", "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(n, FSR_list.tolist()))
        with open(self.target / "E_list.tsv", "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(n, E_list))
        with open(self.target / "method_list.tsv", "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(n, method_list))
        with open(self.target / "best_E_list.tsv", "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(n, best_E_list))
        with open(self.target / "analyze_score_list.tsv", "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(n, analyze_score_list))

    def typeset_pgfplots_graph(self, tsv_name: Optional[str] = "out", label: str = "") -> None:
        import subprocess
        from importlib.resources import read_text

        from MRR import templates

        tsv_path = f"{tsv_name}.tsv"
        template = read_text(templates, "pgfplots.tex").replace("data.txt", tsv_path).replace("Legend Text", label)
        with open(self.target.joinpath("pgfplots.tex"), "w") as fp:
            fp.write(template)
        subprocess.run(["lualatex", "pgfplots"], cwd=self.target, stdout=subprocess.DEVNULL)

    def print_parameters(
        self,
        K: npt.NDArray[np.float_],
        L: npt.NDArray[np.float_],
        N: npt.NDArray[np.int_],
        FSR: np.float_,
        E: np.float_,
        analyze_score: Optional[np.float_] = None,
        format: bool = False,
    ) -> None:
        if format:
            print("K  : {}".format(K.round(2).tolist()))
            print("L  : {}".format((L * 1e6).round(1).tolist()))
        else:
            print("K  : {}".format(K.tolist()))
            print("L  : {}".format(L.tolist()))
        #print("N  : {}".format(N.tolist()))
        print("FSR: {}".format(FSR))
        print("E  : {}".format(E))
        if analyze_score is not None:
            print("score: {}".format(analyze_score))
