header = r"""
\begin{table}[t]
\caption{Optimization result of 12th-order MRR filter designed to meet target filter characteristics.}\label{tab:result}
\centering
\begin{tabular}{cc|cc}
\toprule
\multicolumn{2}{c|}{Round-trip length} & \multicolumn{2}{c}{Coupling efficiency} \\
\toprule"""
# python simulator.py -f --format 4_DE22 6_DE9 8_DE29 10_DE15 12_DE6 14_DE6
# K = [0.42, 0.06, 0.05, 0.05, 0.31]
# L = [82.4, 82.4, 55.0, 55.0]

# K = [0.36, 0.04, 0.03, 0.04, 0.06, 0.12, 0.62]
# L = [55.0, 55.0, 55.0, 82.4, 82.4, 82.4]

# K = [0.44, 0.05, 0.03, 0.18, 0.69, 0.21, 0.07, 0.10, 0.53]
# L = [55.0, 55.0, 55.0, 329.7, 329.7, 82.4, 82.4, 82.4]

# K = [0.77, 0.23, 0.08, 0.08, 0.11, 0.10, 0.09, 0.06, 0.09, 0.20, 0.75]
# L = [82.4, 82.4, 82.4, 109.9, 109.9, 109.9, 82.4, 82.4, 109.9, 109.9]

# K = [0.73, 0.16, 0.06, 0.06, 0.24, 0.71, 0.65, 0.18, 0.06, 0.11, 0.21, 0.30, 0.8]
# L = [82.4, 82.4, 82.4, 82.4, 329.7, 329.7, 329.7, 55.0, 55.0, 55.0, 55.0, 55.0]

K = [0.66, 0.14, 0.08, 0.07, 0.26, 0.67, 0.68, 0.68, 0.24, 0.12, 0.52, 0.50, 0.11, 0.10, 0.58]
L = [82.4, 82.4, 82.4, 82.4, 329.7, 329.7, 329.7, 329.7, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0]

lst = [(i + 1, k, l) for (i, (k, l)) in enumerate(zip(K[1:], L))]
s = [
    r"\(L_{" + str(i) + r"}\) & \SI{" + str(k) + r"}{\micro \metre} & \(K_{" + str(i) + r"}\) & " + str(l) + r" \\"
    for i, k, l in lst
]

footer = r"""
\bottomrule
\end{tabular}
\end{table}"""
print(header)
print(r"& & \(K_{0}\) & " + str(K[0]) + r" \\")
print("\n".join(s))
print(footer)
