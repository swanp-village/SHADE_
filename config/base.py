from config.model import OptimizationConfig

config = OptimizationConfig(
    eta=0.996,
    #alpha=52.96,
    alpha=11.51, #伝搬損失係数=1.0dB/cm
    n_eff=2.2,
    n_g=4.4,
    number_of_rings=6,
    center_wavelength=1550e-9,
    FSR=35e-9,
    length_of_3db_band=1e-9,
    max_crosstalk=-30,    #今までのクロストークの計算
    #max_crosstalk=30,    #クロストークの相対評価を行う場合。絶対値
    H_p=-20,
    H_s=-60,
    H_i=-10,
    r_max=1,
    weight=[1.0, 3.5, 1.0, 3.5, 5.0, 3.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    min_ring_length=50e-6,
    number_of_generations=1,
    strategy=[0.03, 0.07, 0.2, 0.7],
)

# weight = [通過域, 阻止域, 挿入損失, 3dB波長帯域, リプル, クロストーク, 形状]
# weight=[1.0, 3.5, 1.0, 5.0, 3.5, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] 2025/5/11
# 2025/8/13に変更
# 挿入損失 3.5
#2025/08/13 クロストーク 3.5➡1.0
#2025/08/13 リプル 5.0
