import matplotlib.pyplot as plt
from matplotlib import rc

rc("text", usetex=True)
rc("font", size=16)
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

order = [4, 6, 8, 10, 12, 14]
shape_factor = [
    0.445822994210091,
    0.761353517364203,
    0.7525581395348837,
    0.7680074836295603,
    0.7711627906976745,
    0.7753424657534247,
]
crosstalk = [
    -30.129842121605055,
    -44.32602607339349,
    -43.84737323107735,
    -53.239044127713726,
    -54.82174019,
    -48.150563751802665,
]
ax1.plot(
    order,
    shape_factor,
    color=colors[0],
    label=r"Shape factor \(\frac{\Delta\lambda_\mathrm{1dB}}{\Delta\lambda_\mathrm{10dB}}\)",
    marker="o",
)
ax2.plot(order, crosstalk, color=colors[1], label="Crosstalk", marker="o")
ax1.set_ylim([0, 1])
ax2.set_ylim([-60, -20])
ax1.set_xlabel("Order", fontsize=24)
ax1.set_ylabel(r"Shape factor \(\frac{\Delta\lambda_\mathrm{1dB}}{\Delta\lambda_\mathrm{10dB}}\)", fontsize=24)
ax2.set_ylabel(r"Crosstalk (dB)", fontsize=24)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left", frameon=False)
fig.savefig("result/crosstalk_and_shape_factor.pdf")
plt.show()
