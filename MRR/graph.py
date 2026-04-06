from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import rc
from matplotlib.ticker import AutoLocator, FormatStrFormatter, MultipleLocator

rc("text", usetex=True)
rc("font", size=16)


class Graph:
    def __init__(self, is_focus: bool = False):
        self.is_focus = is_focus

    def create(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

    def plot(
        self,
        x: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_],
        label: Optional[str] = None,
    ) -> None:
        self.ax.semilogx(x * 1e9, y, label=label)

    def show(
        self,
        img_path: Path = Path("img/out.pdf"),
    ) -> None:
        self.ax.set_xlabel(r"Wavelength \(\lambda\) (nm)", fontsize=24)
        self.ax.set_ylabel("Transmittance (dB)", fontsize=24)
        self.ax.axis([None, None, None, 5])

        if self.is_focus:
            self.ax.set_xlim([1549, 1551])
            self.ax.set_ylim([-12, 0])
            self.ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            self.ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
            self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            self.ax.yaxis.set_major_locator(MultipleLocator(2))
        else:
            self.ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
            self.ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
            self.ax.xaxis.set_major_locator(AutoLocator())
            self.ax.set_ylim([-60, 0])
        plt.legend(loc="upper center", fontsize=12, frameon=False)
        #
        # plt.legend(loc="lower right", ncol=2)
        self.fig.savefig(img_path)
        plt.show()
