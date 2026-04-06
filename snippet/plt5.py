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


# data_list = [
#     AnalyzerResult(dir="N4L001K1", label="K\(\pm\)1\% L\(\pm\)0.01\%"),
#     AnalyzerResult(dir="N4L001K10", label="K\(\pm\)10\% L\(\pm\)0.01\%"),
#     AnalyzerResult(dir="N4L001K50", label="K\(\pm\)50\% L\(\pm\)0.01\%"),
#     AnalyzerResult(dir="N4L001K100", label="K\(\pm\)100\% L\(\pm\)0.01\%"),
# ]
# data_list = [
#     AnalyzerResult(dir="N10L001K1", label="K\(\pm\)1\% L\(\pm\)0.01\%"),
#     AnalyzerResult(dir="N10L001K10", label="K\(\pm\)10\% L\(\pm\)0.01\%"),
#     AnalyzerResult(dir="N10L001K50", label="K\(\pm\)50\% L\(\pm\)0.01\%"),
#     AnalyzerResult(dir="N10L001K100", label="K\(\pm\)100\% L\(\pm\)0.01\%"),
# ]
# data_list = [
#     AnalyzerResult(dir="N12L001K1", label="K\(\pm\)1\% L\(\pm\)0.01\%"),
#     AnalyzerResult(dir="N12L001K10", label="K\(\pm\)10\% L\(\pm\)0.01\%"),
#     AnalyzerResult(dir="N12L001K50", label="K\(\pm\)50\% L\(\pm\)0.01\%"),
#     AnalyzerResult(dir="N12L001K100", label="K\(\pm\)100\% L\(\pm\)0.01\%"),
# ]
# data_list = [
#     AnalyzerResult(dir="N4L01K1", label="K\(\pm\)1\% L\(\pm\)0.1\%"),
#     AnalyzerResult(dir="N4L01K10", label="K\(\pm\)10\% L\(\pm\)0.1\%"),
#     AnalyzerResult(dir="N4L01K50", label="K\(\pm\)50\% L\(\pm\)0.1\%"),
#     AnalyzerResult(dir="N4L01K100", label="K\(\pm\)100\% L\(\pm\)0.1\%"),
# ]
# data_list = [
#     AnalyzerResult(dir="N4L1K1", label="K\(\pm\)1\% L\(\pm\)1\%"),
#     AnalyzerResult(dir="N4L1K10", label="K\(\pm\)10\% L\(\pm\)1\%"),
#     AnalyzerResult(dir="N4L1K50", label="K\(\pm\)50\% L\(\pm\)1\%"),
#     AnalyzerResult(dir="N4L1K100", label="K\(\pm\)100\% L\(\pm\)1\%"),
# ]
data_list = [
    AnalyzerResult(dir="N4L1K1shift", label="\(L\pm1\%,K\pm1\% (r_L=\\frac{0.01}{3}, r_K=\\frac{0.01}{3})\)"),
    AnalyzerResult(dir="N4L1K10shift", label="\(L\pm1\%,K\pm10\% (r_L=\\frac{0.01}{3}, r_K=\\frac{0.1}{3})\)"),
    AnalyzerResult(dir="N4L1K50shift", label="\(L\pm1\%,K\pm50\% (r_L=\\frac{0.01}{3}, r_K=\\frac{0.5}{3})\)"),
    AnalyzerResult(dir="N4L1K100shift", label="\(L\pm1\%,K\pm100\% (r_L=\\frac{0.01}{3}, r_K=\\frac{1}{3})\)"),
]

for d in data_list:
    with open(f"snippet/data/{d.dir}/analyzer_result.txt") as f:
        reader = csv.reader(f)
        ring = [float(row[0]) for row in reader]
    ring_hist, bin_edges = np.histogram(ring, range=(0, 15), bins=15 * 4)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    plt.plot(bin_centers, ring_hist, label=d.label)

plt.ylabel("Frequency", size=20)
plt.xlabel("Evaluation Function Value", size=20)
plt.legend(loc="best", frameon=False)
fig.savefig("result/error4.pdf")
plt.show()
