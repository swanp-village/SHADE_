from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution    #Scipyを使うときはこちらのコメントアウトを削除
#from MRR.DE_myself import differential_evolution
from MRR.SHADE import SHADE        #20231219 に追加。　未完成のSHADEの導入
from MRR.SHADE_myself import shade

from config.random import get_differential_evolution_rng
from MRR.analyzer import analyze
from MRR.evaluator import evaluate_band
from MRR.graph import Graph
from MRR.logger import Logger
from MRR.simulator import (
    calculate_practical_FSR,
    calculate_ring_length,
    calculate_x,
    optimize_N,
)
from MRR.transfer_function import simulate_transfer_function


def optimize_L(
    n_g: float,
    n_eff: float,
    FSR: float,
    center_wavelength: float,
    min_ring_length: float,
    number_of_rings: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float_], np.float_]:
    for i in range(100):
        N = optimize_N(
            center_wavelength=center_wavelength,
            min_ring_length=min_ring_length,
            n_eff=n_eff,
            n_g=n_g,
            number_of_rings=number_of_rings,
            FSR=FSR,
            rng=rng,
        )
        L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
        practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)
        if practical_FSR > FSR * 0.99 and practical_FSR < FSR * 1.01 and np.all(L < 0.1):
            break
    if i == 99:
        raise Exception("FSR is too strict")

    return N, L, practical_FSR


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


normal_evaluations = []  # 通常の評価値を記録
perturbed_evaluations = []  # 誤差を加えた評価値を記録


"""
def combined_evaluation(K: npt.NDArray[np.float_], params: OptimizeKParams) -> float:
    global normal_evaluations, perturbed_evaluations

    
    #誤差を表示する評価関数

    
    # 通常の評価値
    E_optimal = optimize_K_func(K, params)

    # 誤差を加えた評価値
    E_perturbed = optimize_perturbed_K_func(K, params)

    # 評価値を記録
    normal_evaluations.append(E_optimal)
    perturbed_evaluations.append(E_perturbed)

    # 評価値をその場で表示
    #print(f"Normal Evaluation: {E_optimal}, Perturbed Evaluation: {E_perturbed}")

    # 評価値の変動量
    delta_E = abs(E_optimal - E_perturbed)

    # 総合評価値（小さいほど良い）
    total_score = E_optimal + delta_E
    return total_score
"""

"""
def combined_evaluation(K: npt.NDArray[np.float_], params: OptimizeKParams) -> float:
    #誤差の正負両方を考慮した総合評価値を計算。

    #Parameters:
    #- K: 結合率の配列
    #- params: 最適化パラメータ

    #Returns:
    #- total_score: 総合評価値
    
    global normal_evaluations, perturbed_evaluations

    # 通常の評価値
    E_optimal = optimize_K_func(K, params)

    # 正負の誤差による評価値
    E_positive, E_negative = optimize_perturbed_K_func(K, params)

    # 評価値を記録
    normal_evaluations.append(E_optimal)
    perturbed_evaluations.append((E_positive, E_negative))

    # 評価値の統合
    delta_E_positive = abs(E_optimal - E_positive)
    delta_E_negative = abs(E_optimal - E_negative)

    # 総合評価値 (例: 平均変動量をペナルティとして加算)
    total_score = E_optimal + (delta_E_positive + delta_E_negative) / 2

    return total_score
"""


def combined_evaluation(
    K: npt.NDArray[np.float_], params: OptimizeKParams
) -> float:
    # 通常の評価値と誤差を考慮した評価値を組み合わせた総合評価関数。 これは1つの誤差を考慮したもの。
    # 通常の評価値
    E_optimal = optimize_K_func(K, params)

    # 誤差を加えた評価値
    E_perturbed = optimize_perturbed_K_func(K, params)

    # 評価値の変動量
    delta_E = abs(E_optimal - E_perturbed)

    # 総合評価値（小さいほど良い）
    total_score = E_optimal + delta_E  # ペナルティとして変動量を加算
    return total_score



"""
def evaluation_callback(population: npt.NDArray[np.float_], convergence: float) -> None:
    #各世代終了時に通常の評価値、誤差を加えた評価値を出力。
    
    global normal_evaluations, perturbed_evaluations

    # 現在の世代の個体数
    population_size = len(population)

    # 評価値が不足している場合はスキップ
    if len(normal_evaluations) < population_size or len(perturbed_evaluations) < population_size:
        print(f"Generation {len(normal_evaluations) // population_size}: Evaluations not complete yet.")
        return

    # 現在の世代の評価値を取得
    start_idx = len(normal_evaluations) - population_size
    normal_values = normal_evaluations[start_idx:]
    perturbed_values = perturbed_evaluations[start_idx:]

    print(f"Generation {len(normal_evaluations) // population_size}:")
    print(f"  Normal Evaluations (last): {normal_values}")
    print(f"  Perturbed Evaluations (last): {perturbed_values}")
"""



"""
    #誤差を含んだ評価値を出力する場合
def optimize_K(
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
) -> tuple[npt.NDArray[np.float_], float]:
    bounds = [(1e-12, eta) for _ in range(number_of_rings + 1)]

    result = differential_evolution(
        func=combined_evaluation,
        bounds=bounds,
        args=(params,),
        strategy="currenttobest1bin",
        popsize=35,
        maxiter=1000,
        seed=rng,
        disp=True,
        updating="immediate",
        workers=-1
    )

    E: float = -result.fun
    K: npt.NDArray[np.float_] = result.x

    return K, E
"""


"""
def optimize_K(
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
) -> tuple[npt.NDArray[np.float_], float]:
    bounds = [(1e-12, eta) for _ in range(number_of_rings + 1)]

    result = differential_evolution(
        optimize_K_func,
        bounds,
        args=(params,),
        disp = True,
        strategy="currenttobest1bin",
        #strategy="rand1bin",
        #strategy="randtobest1bin",
        workers=-1,
        updating="immediate",
        popsize=35,
        maxiter=500,
        seed=rng,
    )
    E: float = -result.fun
    K: npt.NDArray[np.float_] = result.x

    return K, E
"""




"""
    #誤差を含めて結合率を最適化する用
def optimize_K_with_perturbation(
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
) -> tuple[npt.NDArray[np.float_], float]:
    # 1. 初期設定
    bounds = [(1e-12, eta) for _ in range(number_of_rings + 1)]  # Kの範囲

    # 2. 最適化実行
    result = differential_evolution(
        func=combined_evaluation,  # 総合評価関数を最適化
        bounds=bounds,
        args=(params,),  # paramsを引数として渡す
        strategy="currenttobest1bin",  # 差分進化戦略
        popsize=15,  # 集団サイズ
        maxiter=500,  # 最大世代数
        tol=1e-6,  # 収束許容誤差
        seed=rng,  # 乱数生成器
        disp=True,  # 最適化過程を表示
        workers=-1,  # 並列化
    )

    # 3. 結果の出力
    E: float = -result.fun  # 最小化問題として解かれるため符号を反転
    K: npt.NDArray[np.float_] = result.x  # 最適化された結合率

    return K, E

"""



"""
def optimize_K(             #通常のSHADE用
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
) -> tuple[npt.NDArray[np.float_], float]:
    bounds = np.array([(1e-12, eta) for _ in range(number_of_rings + 1)])

    result = SHADE(optimize_K_func, 
                   bounds, 
                   params, 
                   pop_size=15, 
                   max_iter = 300,
                   H = 50,
                   tol = 0.01, 
                   callback = None, 
                   rng = None
                  )
    
    E: float = -result[1]
    K: npt.NDArray[np.float_] = result[0]

    return K, E
"""




def optimize_K(             #通常のSHADE用,誤差を考慮
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
) -> tuple[npt.NDArray[np.float_], float]:
    bounds = np.array([(1e-12, eta) for _ in range(number_of_rings + 1)])

    result = SHADE(combined_evaluation, 
                   bounds, 
                   params, 
                   pop_size=15, 
                   max_iter = 1500,
                   H = 50,
                   tol = 0.01, 
                   callback = None, 
                   rng = None
                  )
    
    E: float = -result[1]
    K: npt.NDArray[np.float_] = result[0]

    return K, E



"""

def optimize_K(             #SHADE_old用
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
) -> tuple[npt.NDArray[np.float_], float]:
    bounds = np.array([(1e-12, eta) for _ in range(number_of_rings + 1)])

    result = SHADE(optimize_K_func, 
                   bounds, 
                   params, 
                   pop_size=50, 
                   max_iter = 5000,
                   F = 0.5,
                   cr = 0.7,
                   ftol = 10**-8, 
                   callback = None, 
                   rng = None
                  )
    
    E: float = -result[1]
    K: npt.NDArray[np.float_] = result[0]

    return K, E

"""

"""

def optimize_K(             #DE_myself 用
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
) -> tuple[npt.NDArray[np.float_], float]:
    #bounds = np.array([(1e-12, eta) for _ in range(number_of_rings + 1)])
    number_of_rings = 6
    eta = 0.996
    result = differential_evolution(optimize_K_func, 
                                    number_of_rings,
                                    eta,
                                    pop_size = 20,
                                    gen = 3000,
                                    tol = 1e-6,
                                    seed = 43,
                                    workers = 12,
                                    params=params
                                   )
    
    E: float = -result[1]
    K: npt.NDArray[np.float_] = result[0]

    return K, E

"""

"""

def optimize_K(             #SHADE_myself 用
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
) -> tuple[npt.NDArray[np.float_], float]:
    #bounds = np.array([(1e-12, eta) for _ in range(number_of_rings + 1)])
    number_of_rings = 6
    eta = 0.996
    result = shade(optimize_K_func, 
                                    number_of_rings,
                                    eta,
                                    pop_size = 20,
                                    gen = 10000,
                                    tol = 1e-6,
                                    memory_size=20,
                                    workers = 12,
                                    params=params
                                   )
    
    E: float = -result[1]
    K: npt.NDArray[np.float_] = result[0]

    return K, E

"""

def optimize(
    n_g: float,
    n_eff: float,
    eta: float,
    alpha: float,
    center_wavelength: float,
    length_of_3db_band: float,
    FSR: float,
    max_crosstalk: float,
    H_p: float,
    H_s: float,
    H_i: float,
    r_max: float,
    weight: list[float],
    min_ring_length: float,
    number_of_rings: int,
    number_of_generations: int,
    strategy: list[float],
    logger: Logger,
    skip_plot: bool = False,
    seedsequence: np.random.SeedSequence = np.random.SeedSequence(),
    fixed_N: Optional[npt.NDArray[np.int_]] = None,
) -> None:
    rng = get_differential_evolution_rng(seedsequence=seedsequence)
    N_list: list[npt.NDArray[np.int_]] = [np.array([]) for _ in range(number_of_generations)]
    L_list: list[npt.NDArray[np.float_]] = [np.array([]) for _ in range(number_of_generations)]
    K_list: list[npt.NDArray[np.float_]] = [np.array([]) for _ in range(number_of_generations)]
    FSR_list: npt.NDArray[np.float_] = np.zeros(number_of_generations, dtype=np.float_)
    E_list: list[float] = [0 for _ in range(number_of_generations)]
    method_list: list[int] = [0 for _ in range(number_of_generations)]
    best_E_list: list[float] = [0 for _ in range(number_of_generations)]
    analyze_score_list: list[float] = [0 for _ in range(number_of_generations)]
    for m in range(number_of_generations):
        N: npt.NDArray[np.int_]
        L: npt.NDArray[np.float_]
        practical_FSR: np.float_

        kind: npt.NDArray[np.int_]
        counts: npt.NDArray[np.int_]

        # もし固定配置(fixed_N)が指定されていたら、Methodの抽選は一切せずにそれを使う
        if fixed_N is not None:
            N = fixed_N
            # Nが決まったので、それに基づいてLとFSRを計算
            L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
            practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)

            # Methodは使っていないのでダミーの値(例えば0)を入れておくか、
            # ログに残したいなら適当な値を入れる
            method = 4

        # 固定配置がない場合（通常モード）は、従来どおりMethod 1〜4で決める
        else:
            # 世代数によるMethodの選択
            if m < 10:
                method: int = 4
            else:
                method = rng.choice([1, 2, 3, 4], p=strategy)

            # Methodごとの分岐処理
            if method == 1:
                max_index = np.argmax(E_list)
                max_N = N_list[max_index]

                kind, counts = rng.permutation(np.unique(max_N, return_counts=True), axis=1)  # type: ignore
                N = np.repeat(kind, counts)
                L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
                practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)
            elif method == 2:
                max_index = np.argmax(E_list)
                max_N = N_list[max_index]
                N = rng.permutation(max_N)
                L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
                practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)
            elif method == 3:
                max_index = np.argmax(E_list)
                max_N = N_list[max_index]
                kind = np.unique(max_N)  # type: ignore
                N = rng.choice(kind, number_of_rings)
                while not set(kind) == set(N):
                    N = rng.choice(kind, number_of_rings)
                L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
                practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)
            else:
                N, L, practical_FSR = optimize_L(
                    n_g=n_g,
                    n_eff=n_eff,
                    FSR=FSR,
                    center_wavelength=center_wavelength,
                    min_ring_length=min_ring_length,
                    number_of_rings=number_of_rings,
                    rng=rng,
                )
        """
        if m < 10:
            method: int = 4
        else:
            method = rng.choice([1, 2, 3, 4], p=strategy)

        if method == 1:
            max_index = np.argmax(E_list)
            max_N = N_list[max_index]

            kind, counts = rng.permutation(np.unique(max_N, return_counts=True), axis=1)  # type: ignore
            N = np.repeat(kind, counts)
            L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
            practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)
        elif method == 2:
            max_index = np.argmax(E_list)
            max_N = N_list[max_index]
            N = rng.permutation(max_N)
            L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
            practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)
        elif method == 3:
            max_index = np.argmax(E_list)
            max_N = N_list[max_index]
            kind = np.unique(max_N)  # type: ignore
            N = rng.choice(kind, number_of_rings)
            while not set(kind) == set(N):
                N = rng.choice(kind, number_of_rings)
            L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
            practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)
        else:
            N, L, practical_FSR = optimize_L(
                n_g=n_g,
                n_eff=n_eff,
                FSR=FSR,
                center_wavelength=center_wavelength,
                min_ring_length=min_ring_length,
                number_of_rings=number_of_rings,
                rng=rng,
            )
        """

        """
        N = [88, 110, 88, 110, 110, 88]
        L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
        """
        
        print("L確認",L)
        print("N確認",N)


        normal_evaluations = []  # 通常の評価値を記録
        perturbed_evaluations = []  # 誤差を加えた評価値を記録


        #K, E = optimize_K_with_perturbation(
        K, E = optimize_K(
            eta=eta,
            number_of_rings=number_of_rings,
            rng=rng,
            params=OptimizeKParams(
                L=L,
                n_g=n_g,
                n_eff=n_eff,
                eta=eta,
                alpha=alpha,
                center_wavelength=center_wavelength,
                length_of_3db_band=length_of_3db_band,
                FSR=practical_FSR,
                max_crosstalk=max_crosstalk,
                H_p=H_p,
                H_s=H_s,
                H_i=H_i,
                r_max=r_max,
                weight=weight,
            ),
        )

        N_list[m] = N
        L_list[m] = L
        FSR_list[m] = practical_FSR
        K_list[m] = K
        E_list[m] = E
        analyze_score = 0.0
        
            
        if E > 100:
            for L_error_rate, K_error_rate in zip([0.01, 0.1, 1, 10], [1, 10, 100]):
                analyze_result = analyze(
                    n=1000,
                    L_error_rate=L_error_rate,
                    K_error_rate=K_error_rate,
                    L=L,
                    K=K,
                    n_g=n_g,
                    n_eff=n_eff,
                    eta=eta,
                    alpha=alpha,
                    center_wavelength=center_wavelength,
                    length_of_3db_band=length_of_3db_band,
                    FSR=FSR,
                    max_crosstalk=max_crosstalk,
                    H_p=H_p,
                    H_s=H_s,
                    H_i=H_i,
                    r_max=r_max,
                    weight=weight,
                    min_ring_length=min_ring_length,
                    seedsequence=seedsequence,
                    skip_plot=True,
                    logger=logger,
                )
                if analyze_result > 0.5:
                    analyze_score += 1
            analyze_score_list[m] = analyze_score
        best_index = np.argmax(E_list)
        best_N = N_list[best_index]
        best_L = L_list[best_index]
        best_K = K_list[best_index]
        best_FSR = FSR_list[best_index]
        best_E = E_list[best_index]
        best_analyze_score = analyze_score_list[best_index]
        print(m + 1)
        logger.print_parameters(K=K, L=L, N=N, FSR=practical_FSR, E=E, analyze_score=analyze_score, format=True)
        print("==best==")
        logger.print_parameters(
            K=best_K, L=best_L, N=best_N, FSR=best_FSR, E=best_E, analyze_score=best_analyze_score, format=True
        )
        print("================")

        method_list[m] = method
        best_E_list[m] = best_E

        """
        if best_E_list[m] < 10 and m == 30:                #20231205 additon by naganuma
            print("It ends because E is low")
            break
        else:
            pass
        """

    max_index = np.argmax(E_list)
    result_N = N_list[max_index]
    result_L = L_list[max_index]
    result_K = K_list[max_index]
    result_FSR = FSR_list[max_index]
    result_E = E_list[max_index]
    result_analyze_score = analyze_score_list[max_index]
    x = calculate_x(center_wavelength=center_wavelength, FSR=result_FSR)
    y = simulate_transfer_function(
        wavelength=x,
        L=result_L,
        K=result_K,
        alpha=alpha,
        eta=eta,
        n_eff=n_eff,
        n_g=n_g,
        center_wavelength=center_wavelength,
    )
    print("result")
    logger.print_parameters(
        K=result_K, L=result_L, N=result_N, FSR=result_FSR, E=result_E, analyze_score=result_analyze_score
    )
    logger.save_result(L=result_L, K=result_K)
    print("save data")
    logger.save_DE_data(
        N_list=N_list,
        L_list=L_list,
        K_list=K_list,
        FSR_list=FSR_list,
        E_list=E_list,
        method_list=method_list,
        best_E_list=best_E_list,
        analyze_score_list=analyze_score_list,
    )
    print("end")
    if E > 0 and not skip_plot:
        graph = Graph()
        graph.create()
        graph.plot(x, y)
        graph.show(logger.generate_image_path())
       


def optimize_K_func(K: npt.NDArray[np.float_], params: OptimizeKParams) -> np.float_:
    x = calculate_x(center_wavelength=params.center_wavelength, FSR=params.FSR)
    y = simulate_transfer_function(
        wavelength=x,
        L=params.L,
        K=K,
        alpha=params.alpha,
        eta=params.eta,
        n_eff=params.n_eff,
        n_g=params.n_g,
        center_wavelength=params.center_wavelength,
    )

    return -evaluate_band(
        x=x,
        y=y,
        center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band,
        max_crosstalk=params.max_crosstalk,
        H_p=params.H_p,
        H_s=params.H_s,
        H_i=params.H_i,
        r_max=params.r_max,
        weight=params.weight,
        ignore_binary_evaluation=False,
    )




"""
#誤差を割合で掛け算するやつ
def optimize_perturbed_K_func(K: npt.NDArray[np.float_], params: OptimizeKParams) -> np.float_:
    #-----------------------------------
    #誤差として結合率Kに一定の「割合」を適用した場合の評価値を計算します。
    #（修正版：加算から乗算に変更）
    #-----------------------------------
    # 誤差の割合を設定 (例: 0.5% の誤差)
    # 論文パワポの K_e_rate に相当します。
    # 例えば error_rate = 0.005 とすると、結合率は 1.005倍になります。
    error_rate = 0.20
    
    # 誤差を加える（ここを変更）
    # perturbed_K = K + error_rate              <- 変更前 (単純な足し算)
    perturbed_K = K * (1 + error_rate)      # <- 変更後 (割合での掛け算)

    # 範囲外をクリップ (上限値を超えないように調整)
    perturbed_K = np.clip(perturbed_K, 1e-12, params.eta)

    # 波長と透過特性を計算
    x = calculate_x(center_wavelength=params.center_wavelength, FSR=params.FSR)
    try:
        y = simulate_transfer_function(
            wavelength=x,
            L=params.L,
            K=perturbed_K,
            alpha=params.alpha,
            eta=params.eta,
            n_eff=params.n_eff,
            n_g=params.n_g,
            center_wavelength=params.center_wavelength,
        )
    except Exception as e:
        print(f"Error in simulate_transfer_function: {e}")
        return np.inf

    # 評価値を計算
    return -evaluate_band(
        x=x,
        y=y,
        center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band,
        max_crosstalk=params.max_crosstalk,
        H_p=params.H_p,
        H_s=params.H_s,
        H_i=params.H_i,
        r_max=params.r_max,
        weight=params.weight,
        ignore_binary_evaluation=False,
    )
"""








def optimize_perturbed_K_func(K: npt.NDArray[np.float_], params: OptimizeKParams) -> np.float_:
    
    #誤差として全ての結合率を 0.005 増加させる。
    #範囲外 (eta) を超えた場合は eta に制限する。
    
    error_rate = 0.005
    
    # 誤差を加える
    perturbed_K = K + error_rate

    # 範囲外をクリップ
    perturbed_K = np.clip(perturbed_K, 1e-12, params.eta)

    # 波長と透過特性を計算
    x = calculate_x(center_wavelength=params.center_wavelength, FSR=params.FSR)
    try:
        y = simulate_transfer_function(
            wavelength=x,
            L=params.L,
            K=perturbed_K,
            alpha=params.alpha,
            eta=params.eta,
            n_eff=params.n_eff,
            n_g=params.n_g,
            center_wavelength=params.center_wavelength,
        )
    except Exception as e:
        print(f"Error in simulate_transfer_function: {e}")
        return np.inf

    # 評価値を計算
    return -evaluate_band(
        x=x,
        y=y,
        center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band,
        max_crosstalk=params.max_crosstalk,
        H_p=params.H_p,
        H_s=params.H_s,
        H_i=params.H_i,
        r_max=params.r_max,
        weight=params.weight,
        ignore_binary_evaluation=False,
    )




"""
def optimize_perturbed_K_func(K: npt.NDArray[np.float_], params: OptimizeKParams) -> tuple[float, float]:
    #誤差として結合率 K に +0.005 および -0.005 を適用した場合の評価値を計算。

    #Parameters:
    #- K: 結合率の配列
    #- params: 最適化パラメータ

    #Returns:
    #- E_positive: +0.005 の誤差を加えた場合の評価値
    #- E_negative: -0.005 の誤差を加えた場合の評価値
    
    # 正の誤差を加える
    perturbed_K_positive = np.clip(K + 0.005, 1e-12, params.eta)

    # 負の誤差を加える
    perturbed_K_negative = np.clip(K - 0.005, 1e-12, params.eta)

    # 波長を計算
    x = calculate_x(center_wavelength=params.center_wavelength, FSR=params.FSR)

    # 正の誤差での評価値
    y_positive = simulate_transfer_function(
        wavelength=x,
        L=params.L,
        K=perturbed_K_positive,
        alpha=params.alpha,
        eta=params.eta,
        n_eff=params.n_eff,
        n_g=params.n_g,
        center_wavelength=params.center_wavelength,
    )
    E_positive = -evaluate_band(
        x=x,
        y=y_positive,
        center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band,
        max_crosstalk=params.max_crosstalk,
        H_p=params.H_p,
        H_s=params.H_s,
        H_i=params.H_i,
        r_max=params.r_max,
        weight=params.weight,
        ignore_binary_evaluation=False,
    )

    # 負の誤差での評価値
    y_negative = simulate_transfer_function(
        wavelength=x,
        L=params.L,
        K=perturbed_K_negative,
        alpha=params.alpha,
        eta=params.eta,
        n_eff=params.n_eff,
        n_g=params.n_g,
        center_wavelength=params.center_wavelength,
    )
    E_negative = -evaluate_band(
        x=x,
        y=y_negative,
        center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band,
        max_crosstalk=params.max_crosstalk,
        H_p=params.H_p,
        H_s=params.H_s,
        H_i=params.H_i,
        r_max=params.r_max,
        weight=params.weight,
        ignore_binary_evaluation=False,
    )

    return E_positive, E_negative
"""
