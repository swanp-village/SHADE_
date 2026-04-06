import argparse
import csv
import os
import datetime
from glob import glob
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.model import SimulationConfig
from MRR.simulator import Accumulator, SimulatorResult, simulate_MRR

def plot_combined_results(results: list[SimulatorResult], output_folder: Path, base_name: str, x_limits=None) -> None:
    """複数のシミュレーション結果を1つのグラフに重ねてプロットする（誤差解析モード用）"""
    fig, ax = plt.subplots()
    for result in results:
        ax.plot(result.x * 1e9, result.y, label=result.label)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmittance (dB)")
    ax.set_ylim(-60, 0)
    ax.legend()
    ax.tick_params(direction="in", length=6, width=1, which="both")
    fig.savefig(output_folder / f"{base_name}_original_combined.png")
    plt.close(fig)

    if x_limits:
        fig, ax = plt.subplots()
        for result in results:
            ax.plot(result.x * 1e9, result.y, label=result.label)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmittance (dB)")
        ax.set_xlim(x_limits)
        ax.set_ylim(-60, 0)
        ax.legend()
        ax.tick_params(direction="in", length=6, width=1, which="both")
        fig.savefig(output_folder / f"{base_name}_modified_combined.png")
        plt.close(fig)

def plot_results(results: list[SimulatorResult], output_folder: Path, x_limits=None, y_limits=None) -> None:
    """単体のシミュレーション結果をプロットする（通常モード用）"""
    for result in results:
        fig, ax = plt.subplots()
        ax.plot(result.x * 1e9, result.y, label=result.label)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmittance (dB)")
        ax.set_ylim(-60, 0)
        ax.legend()
        ax.tick_params(direction="in", length=6, width=1, which="both")
        fig.savefig(output_folder / f"{result.name}_original.png")
        plt.close(fig)

        if x_limits:
            fig, ax = plt.subplots()
            ax.plot(result.x * 1e9, result.y, label=result.label)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Transmittance (dB)")
            ax.set_xlim(x_limits)
            ax.set_ylim(-60, 0)
            ax.legend()
            ax.tick_params(direction="in", length=6, width=1, which="both")
            fig.savefig(output_folder / f"{result.name}_modified.png")
            plt.close(fig)

def save_excel_file(basedir: Path, results: list[SimulatorResult], base_name: str, x_limits=None, range_specified: bool = False) -> None:
    """
    シミュレーション結果をExcelファイルに保存。
    - 誤差解析モードでは3つのデータを並べて出力。
    - 通常モードでは1つのデータを出力。
    """
    excel_path = basedir / f"{base_name}_analysis.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        max_points = 2500
        
        # --- 全体範囲シート ---
        x_full = results[0].x
        step = 1 if x_full.size < max_points else x_full.size // max_points
        data_full = {'Wavelength (nm)': x_full[::step] * 1e9}
        for result in results:
            data_full[f'Transmittance_{result.label} (dB)'] = result.y[::step]
        df_full = pd.DataFrame(data_full)
        df_full.to_excel(writer, sheet_name='Full_Range_Analysis', index=False)

        # --- 指定範囲シート（指定がある場合のみ） ---
        if range_specified:
            x_nm_unfiltered = results[0].x * 1e9
            indices = (x_nm_unfiltered >= x_limits[0]) & (x_nm_unfiltered <= x_limits[1])
            data_filtered = {'Wavelength (nm)': x_nm_unfiltered[indices]}
            for result in results:
                data_filtered[f'Transmittance_{result.label} (dB)'] = result.y[indices]
            df_filtered = pd.DataFrame(data_filtered)
            df_filtered.to_excel(writer, sheet_name='Filtered_Range_Analysis', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRR Simulator")
    parser.add_argument("NAME", help="from config.simulate import NAME", nargs="*")
    parser.add_argument("--error-analysis", action="store_true", help="Run simulation with K errors and plot on one graph.")
    
    # ▼▼▼ 追加 ▼▼▼
    parser.add_argument("--error-rate", type=float, default=None, help="Use multiplicative error model with the given rate (e.g., 0.03 for 3%%). If not specified, additive model is used.")
    # ▲▲▲ 追加 ▲▲▲

    parser.add_argument("--x-min", type=float, default=None, help="X-axis minimum value (nm)")
    parser.add_argument("--x-max", type=float, default=None, help="X-axis maximum value (nm)")
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    
    args = vars(parser.parse_args())
    
    error_analysis_mode = args["error_analysis"]
    error_rate_val = args["error_rate"] # 追加
    range_specified = args["x_min"] is not None and args["x_max"] is not None
    x_limits = (args["x_min"], args["x_max"]) if range_specified else None

    if args["list"]:
        print("\t".join([os.path.splitext(os.path.basename(p))[0] for p in sorted(glob("config/simulate/*.py"))]))
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = Path(f"graphs/{now}")
        
        for name in args["NAME"]:
            if name.endswith(".py"): name = name[:-3]
            try:
                imported_module = import_module(f"config.simulate.{name}")
                imported_config = getattr(imported_module, "config")
                base_config = SimulationConfig(**imported_config)
                
                if range_specified:
                    base_config.lambda_limit = (args["x_min"] * 1e-9, args["x_max"] * 1e-9)
                else:
                    base_config.lambda_limit = None

                if error_analysis_mode:
                    print(f"--- Running Error Analysis for {name} ---")
                    results_for_analysis = []
                    
                    k_original = np.array(base_config.K)
                    
                    # ▼▼▼ 変更点: error-rate引数の有無で処理を分岐 ▼▼▼
                    if error_rate_val is not None:
                        # --- 掛け算モデル ---
                        print(f"Using Multiplicative Error Model with rate: {error_rate_val}")
                        k_plus = np.clip(k_original * (1 + error_rate_val), 0, 1)
                        k_minus = np.clip(k_original * (1 - error_rate_val), 0, 1)
                        label_plus = f"K*(1+{error_rate_val})"
                        label_minus = f"K*(1-{error_rate_val})"
                    else:
                        # --- 従来の足し算モデル ---
                        error_val = 0.01 # 従来の値
                        print(f"Using Additive Error Model with value: {error_val}")
                        k_plus = np.clip(k_original + error_val, 0, 1)
                        k_minus = np.clip(k_original - error_val, 0, 1)
                        label_plus = f"K+{error_val}"
                        label_minus = f"K-{error_val}"
                    # ▲▲▲ 変更点 ▲▲▲
                        
                    k_configs = [
                        {"k": k_original, "label": "Original", "name": f"{name}_original"},
                        {"k": k_plus, "label": label_plus, "name": f"{name}_plus_err"},
                        {"k": k_minus, "label": label_minus, "name": f"{name}_minus_err"},
                    ]
                    
                    for k_config in k_configs:
                        print(f"Simulating with: {k_config['label']}")
                        result = simulate_MRR( K=k_config["k"], label=k_config["label"], name=k_config["name"], L=base_config.L, n_eff=base_config.n_eff, n_g=base_config.n_g, eta=base_config.eta, alpha=base_config.alpha, center_wavelength=base_config.center_wavelength, lambda_limit=base_config.lambda_limit, accumulator=Accumulator(init_graph=False), skip_graph=True, skip_evaluation=True, length_of_3db_band=0, max_crosstalk=0, H_p=0, H_i=0, H_s=0, r_max=0, weight=[] )
                        results_for_analysis.append(result)

                    if not args["skip_plot"]:
                        output_folder.mkdir(parents=True, exist_ok=True)
                        plot_combined_results(results_for_analysis, output_folder, base_name=name, x_limits=x_limits)
                        save_excel_file(output_folder, results_for_analysis, base_name=name, x_limits=x_limits, range_specified=range_specified)
                        print(f"Combined graph and Excel file saved to {output_folder}")

                else: # 通常モード
                    print(f"--- Running Standard Simulation for {name} ---")
                    result = simulate_MRR( K=base_config.K, label=name, name=name, L=base_config.L, n_eff=base_config.n_eff, n_g=base_config.n_g, eta=base_config.eta, alpha=base_config.alpha, center_wavelength=base_config.center_wavelength, lambda_limit=base_config.lambda_limit, accumulator=Accumulator(init_graph=False), skip_graph=True, skip_evaluation=False, length_of_3db_band=0, max_crosstalk=0, H_p=0, H_i=0, H_s=0, r_max=0, weight=[] )
                    
                    if not args["skip_plot"]:
                        output_folder.mkdir(parents=True, exist_ok=True)
                        plot_results([result], output_folder, x_limits=x_limits)
                        save_excel_file(output_folder, [result], base_name=name, x_limits=x_limits, range_specified=range_specified)
                        print(f"Graph and Excel file saved to {output_folder}")

            except (ModuleNotFoundError, AttributeError) as e:
                print(e)
