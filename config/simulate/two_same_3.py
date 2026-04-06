import numpy as np

config = {
    "eta": 1,  # 結合損
    "alpha": 0,  # 伝搬損失係数
    "K": np.array([0.65, 0.17, 0.43]),  # 結合効率
    "L": np.array([3e-4, 3e-4]),  # リング周長
    "n_eff": 3.3938,  # 実行屈折率
    "n_g": 4.2,  # 群屈折率
    "lambda_limit": np.arange(1545e-9, 1555e-9, 1e-12),  # 波長レンジ
}
