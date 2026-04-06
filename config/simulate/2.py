import numpy as np

config = {
    "eta": 0.99,  # 結合損
    "alpha": 353,  # 伝搬損失係数
    "K": np.array([0.632, 0.064, 0.632]),  # 結合率
    "L": np.array([201.4e-06, 201.8e-06]),  # リング周長
    "n_eff": 2.3,  # 実行屈折率
    "n_g": 3.2,  # 群屈折率
    "center_wavelength": 1550e-9,
    "lambda_limit": np.arange(1540e-9, 1560e-9, 1e-12),
}
