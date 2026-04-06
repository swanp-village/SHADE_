import csv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

fig, ax = plt.subplots(figsize=(8, 6))
rc("text", usetex=True)
rc("font", size=16)


@dataclass
class AnalyzerResult:
    dir: str
    label: str


data_list = [
    AnalyzerResult(
        dir="N10L1K1shift", label="\(L\pm1\%,K\pm1\% (\sigma_L=\\frac{0.01}{3}, \sigma_K=\\frac{0.01}{3})\)"
    ),
    AnalyzerResult(
        dir="N10L1K10shift", label="\(L\pm1\%,K\pm10\% (\sigma_L=\\frac{0.01}{3}, \sigma_K=\\frac{0.1}{3})\)"
    ),
    AnalyzerResult(
        dir="N10L1K50shift", label="\(L\pm1\%,K\pm50\% (\sigma_L=\\frac{0.01}{3}, \sigma_K=\\frac{0.5}{3})\)"
    ),
    AnalyzerResult(
        dir="N10L1K100shift", label="\(L\pm1\%,K\pm100\% (\sigma_L=\\frac{0.01}{3}, \sigma_K=\\frac{1}{3})\)"
    ),
]

for d in data_list:
    with open(f"snippet/data/{d.dir}/analyzer_result.txt") as f:
        reader = csv.reader(f)
        ring = [float(row[0]) / 14.23348546406131 for row in reader]
    ring_hist, bin_edges = np.histogram(ring, range=(0, 1), bins=15 * 4)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    plt.plot(bin_centers, ring_hist, label=d.label, lw=1)

plt.ylabel("Frequency", size=20)
plt.xlabel("Normarized Evaluation Function Value", size=20)
plt.legend(loc="best", frameon=False)
fig.savefig("result/error10.pdf")
plt.show()
