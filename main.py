import argparse
import numpy as np

from config.base import config
from MRR.logger import Logger
from MRR.model.DE import optimize

# ★追加関数: 文字列(AABBBB)を数値配列([88, 88, 110...])に変換する
def pattern_to_N(pattern_str):
    if not pattern_str:
        return None
    
    # A=88, B=110 の固定値設定
    val_A = 88
    val_B = 110
    
    N_list = []
    for char in pattern_str:
        if char == 'A':
            N_list.append(val_A)
        elif char == 'B':
            N_list.append(val_B)
        else:
            raise ValueError(f"Unknown character '{char}' in pattern. Use 'A' or 'B'.")
            
    return np.array(N_list) # numpy配列として返す



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-plot", action="store_true")
    # ★追加: パターンを受け取る引数
    parser.add_argument("--pattern", type=str, default=None, help="Ring config pattern e.g. 'AABBBB'")
    
    args = vars(parser.parse_args())
    skip_plot = args["skip_plot"]
    pattern_str = args["pattern"] # パターン文字列を取得
    
    # パターンがあれば配列に変換、なければ None
    fixed_N = pattern_to_N(pattern_str)
    
    # ログ出力（確認用）
    if fixed_N is not None:
        print(f"========== Running with Fixed Pattern: {pattern_str} ==========")
        print(f"Values: {fixed_N}")
    else:
        print("========== Running with Standard Optimization (No fixed pattern) ==========")

    logger = Logger()
    logger.save_optimization_config(config)
    
    optimize(
        n_g=config.n_g,
        n_eff=config.n_eff,
        eta=config.eta,
        alpha=config.alpha,
        center_wavelength=config.center_wavelength,
        length_of_3db_band=config.length_of_3db_band,
        FSR=config.FSR,
        max_crosstalk=config.max_crosstalk,
        H_p=config.H_p,
        H_s=config.H_s,
        H_i=config.H_i,
        r_max=config.r_max,
        weight=config.weight,
        min_ring_length=config.min_ring_length,
        number_of_rings=config.number_of_rings,
        number_of_generations=config.number_of_generations,
        strategy=config.strategy,
        seedsequence=config.seedsequence,
        logger=logger,
        skip_plot=skip_plot,
        fixed_N=fixed_N, # ★ここでDE.pyに渡す！
    )







""" #変更前のmain関数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-plot", action="store_true")
    args = vars(parser.parse_args())
    skip_plot = args["skip_plot"]
    logger = Logger()
    logger.save_optimization_config(config)
    optimize(
        n_g=config.n_g,
        n_eff=config.n_eff,
        eta=config.eta,
        alpha=config.alpha,
        center_wavelength=config.center_wavelength,
        length_of_3db_band=config.length_of_3db_band,
        FSR=config.FSR,
        max_crosstalk=config.max_crosstalk,
        H_p=config.H_p,
        H_s=config.H_s,
        H_i=config.H_i,
        r_max=config.r_max,
        weight=config.weight,
        min_ring_length=config.min_ring_length,
        number_of_rings=config.number_of_rings,
        number_of_generations=config.number_of_generations,
        strategy=config.strategy,
        seedsequence=config.seedsequence,
        logger=logger,
        skip_plot=skip_plot,
    )
"""
