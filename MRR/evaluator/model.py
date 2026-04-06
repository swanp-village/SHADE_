from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.ticker import FuncFormatter

from MRR.evaluator.config import config


@FuncFormatter  # type: ignore
def formatter(x: float, pos: int) -> str:
    return "%.1f" % x


class Model:
    def __init__(
        self,
        name: str,
    ) -> None:
        self.FSR = config.FSR
        self.center_wavelength = config.center_wavelength
        self.length_of_3db_band = config.length_of_3db_band
        self.x = np.hstack(
            (
                np.arange(self.center_wavelength - self.FSR / 2, self.center_wavelength, 1e-12),
                np.arange(self.center_wavelength, self.center_wavelength + self.FSR / 2, 1e-12),
            )
        )
        self.y = np.repeat(0, self.x.size)
        self.img_path = Path("MRR/Evaluator/Model/figure") / f"{name}.pdf"
        self.set_rank(1)
        self.validate()

    def export_graph(self, skip_plot: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel("Wavelength (nm)", fontsize=24)
        ax.set_ylabel("Drop Port Power (dB)", fontsize=24)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xticks([1500, 1550, 1600], minor=False)
        ax.set_ylim([-70, 0])

        ax.semilogx(self.x * 1e9, self.y)
        fig.savefig(self.img_path)
        if not skip_plot:
            plt.show()

    def validate(self) -> None:
        if self.x.size != self.FSR / 1e-12:
            print("The size of x is wrong.")
        if self.x.size != self.y.size:
            print("The size of x does not equal to the size of y.")
        if self.x[self.x.size // 2] != self.center_wavelength:
            print("The center value of x is not the center wavelength.")

    def set_y(self, y: npt.NDArray[np.float_]) -> None:
        self.y = y
        self.validate()

    def set_rank(self, rank: int) -> None:
        self.rank = rank
