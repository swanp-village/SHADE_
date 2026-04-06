import os
import numpy as np
import pandas as pd  # Excel保存用
import numpy.typing as npt
from datetime import datetime
from dataclasses import dataclass

from MRR.evaluator import evaluate_band
from MRR.simulator import (
    calculate_practical_FSR,
    calculate_ring_length,
    calculate_x,
    optimize_N,
)
from MRR.transfer_function import simulate_transfer_function

@dataclass
class OptimizeKParams:
    L: npt.NDArray[np.float_]
    n_g: float
    n_eff: float
    eta: float
    alpha: float
    center_wavelength: float
    length_of_3db_band: float
    FSR: np.float_
    max_crosstalk: float
    H_p: float
    H_s: float
    H_i: float
    r_max: float
    weight: list[float]

def get_save_path():
    """
    実行日時ごとのフォルダを作成し、Excel保存用のファイルパスを生成。
    
    Returns:
    - 保存先のファイルパス
    """
    # 実行時間のフォーマット YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # `results/YYYY-MM-DD_HH-MM-SS/` というフォルダを作成
    folder_path = f"results/{timestamp}"
    os.makedirs(folder_path, exist_ok=True)  # フォルダが無ければ作成

    # Excelの保存先パス
    return f"{folder_path}/evaluation_results_transposed.csv"  # CSVの保存先

def evaluate_with_error(
    K: npt.NDArray[np.float_],
    params: OptimizeKParams,
    error_value: float = 0.005
) -> None:
    """
    既存の結合率に誤差を加え、その評価値を再計算し、Excelに保存する。

    Parameters:
    - K: 理論的に最適化された結合率の配列
    - params: 最適化パラメータ
    - error_value: 結合率に加える誤差（デフォルトは 0.005）
    """
    save_path = get_save_path()  # 実行ごとにフォルダを作成

    # 誤差を加えた結合率
    perturbed_K = np.clip(K + error_value, 0, params.eta)

    # 波長と透過特性の計算
    x = calculate_x(center_wavelength=params.center_wavelength, FSR=params.FSR)
    y_original = simulate_transfer_function(
        wavelength=x,
        L=params.L,
        K=K,
        alpha=params.alpha,
        eta=params.eta,
        n_eff=params.n_eff,
        n_g=params.n_g,
        center_wavelength=params.center_wavelength,
    )
    y_perturbed = simulate_transfer_function(
        wavelength=x,
        L=params.L,
        K=perturbed_K,
        alpha=params.alpha,
        eta=params.eta,
        n_eff=params.n_eff,
        n_g=params.n_g,
        center_wavelength=params.center_wavelength,
    )

    # 評価値の計算
    E_original = evaluate_band(
        x=x, y=y_original, center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band, max_crosstalk=params.max_crosstalk,
        H_p=params.H_p, H_s=params.H_s, H_i=params.H_i, r_max=params.r_max, weight=params.weight,
        ignore_binary_evaluation=False,
    )
    E_perturbed = evaluate_band(
        x=x, y=y_perturbed, center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band, max_crosstalk=params.max_crosstalk,
        H_p=params.H_p, H_s=params.H_s, H_i=params.H_i, r_max=params.r_max, weight=params.weight,
        ignore_binary_evaluation=False,
    )

    # 結果の表示
    print("\n=== 結合率の評価結果 ===")
    for i in range(len(K)):
        print(f"K[{i}]: 理論値={K[i]:.5f}, 誤差値={perturbed_K[i]:.5f}")
    print(f"\n理論値の評価値: {E_original:.5f}")
    print(f"誤差を加えた評価値: {E_perturbed:.5f}")
    print(f"評価値の変動: {abs(E_original - E_perturbed):.5f}")

    # **行と列を入れ替えたデータフレームを作成**
    df = pd.DataFrame({
        "K_Original": K,
        "K_Perturbed": perturbed_K,
        "Evaluation_Original": [E_original] * len(K),
        "Evaluation_Perturbed": [E_perturbed] * len(K),
        "Evaluation_Diff": [abs(E_original - E_perturbed)] * len(K)
    }).T  # **転置（行と列を入れ替え）**
    
    # CSVに保存
    df.to_csv(save_path, header=False)

    # 保存メッセージ
    print(f"結果を '{save_path}' に保存しました。")


# 例としての結合率データとパラメータ
K_theoretical = np.array([0.10706391, 0.03830442, 0.03604803, 0.05259838, 0.08201191, 0.24465265, 0.81698317])  # 理論的な結合率
params = OptimizeKParams(
    L=np.array([6.20e-5, 6.20e-5,  7.75e-5, 7.75e-5, 7.75e-5, 7.75e-5 ]),
    n_g=4.4,
    n_eff=2.2,
    eta=0.996,
    alpha=11.51,
    center_wavelength=1550e-9,
    length_of_3db_band=1e-9,
    FSR=35e-9,
    max_crosstalk=-30,
    H_p=-20,
    H_s=-60,
    H_i=-10,
    r_max=1,
    weight=[1.0, 3.5, 1.0, 3.5, 5.0, 3.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
)

# 誤差を加えた評価値の計算
evaluate_with_error(K=K_theoretical, params=params, error_value = -0.005)






