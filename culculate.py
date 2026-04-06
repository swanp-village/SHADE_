import numpy as np
from scipy.signal import argrelmin, argrelmax
from MRR.transfer_function import simulate_transfer_function
from MRR.simulator import calculate_x

# ---------------------------------------------------------------- #
# ▼▼▼ ハイブリッド版の評価関数群 ▼▼▼
# ---------------------------------------------------------------- #

def evaluate_graph_hybrid(x_nm, graph_db):
    """
    主要な評価項目をまとめて計算・表示する最終版。
    Shape Factorの計算を追加。
    """
    insertion_loss = evaluate_insertion_loss(graph_db)
    
    # 3dB帯域幅を計算
    bandwidth_3db = calculate_bandwidth_nm(x_nm, graph_db, 3.0)
    
    # リプルとクロストークを計算
    ripple = evaluate_ripple_original_logic(graph_db)
    crosstalk = evaluate_crosstalk_corrected(graph_db)
    
    # --- ▼▼▼ ここからが追加部分 ▼▼▼ ---
    
    # Shape Factorのために1dBと10dBの帯域幅も計算
    bandwidth_1db = calculate_bandwidth_nm(x_nm, graph_db, 1.0)
    bandwidth_10db = calculate_bandwidth_nm(x_nm, graph_db, 10.0)
    
    # ゼロ除算を避けてShape Factorを計算
    shape_factor = bandwidth_1db / bandwidth_10db if bandwidth_10db > 0 else 0.0
    
    # --- ▲▲▲ ここまでが追加部分 ▲▲▲ ---

    print("--- Hybrid Evaluation Results (Final) ---")
    print(f"Insertion Loss = {insertion_loss:.6f} dB")
    print(f"3dB Bandwidth  = {bandwidth_3db:.6f} nm")
    print(f"Ripple         = {ripple:.6f} dB")
    print(f"Crosstalk      = {crosstalk:.6f} dB")
    print(f"Shape Factor   = {shape_factor:.6f}") # 表示を追加
    return



# --- 帯域を正確に特定するための新しいヘルパー関数 ---
def _find_band_indices(y_transmittance, db_down):
    peak_value = np.max(y_transmittance)
    threshold = peak_value - db_down
    indices = np.where(y_transmittance >= threshold)[0]
    if len(indices) == 0:
        return None
    # 帯域の左端と右端のインデックスを返す
    return indices[0], indices[-1] 

# --- 新しい正確な帯域幅計算 ---
def calculate_bandwidth_nm(x_wavelength, y_transmittance, db_down):
    band_indices = _find_band_indices(y_transmittance, db_down)
    if band_indices is None:
        return 0.0
    left_idx, right_idx = band_indices
    return x_wavelength[right_idx] - x_wavelength[left_idx]

# --- 元々の評価関数（一部修正） ---
def evaluate_insertion_loss(graph_db):
    return np.max(graph_db)

def evaluate_ripple_original_logic(graph_db):
    # ★正確な帯域のインデックスを取得
    band_indices_3db = _find_band_indices(graph_db, 3.0)
    if band_indices_3db is None:
        return 0.0
    left_idx, right_idx = band_indices_3db
    
    # 元々のロジック：特定した帯域スライスに対して極大・極小を探す
    pass_band_slice = graph_db[left_idx : right_idx+1]
    point = _local_maximum_and_minimum(pass_band_slice)
    
    # point[1][0]は極大値のリスト, point[1][1]は極小値のリスト
    if not point[1][0] or not point[1][1]:
        return 0.0
    
    ripple = max(point[1][0]) - min(point[1][1])
    return ripple


# ★★★★★ ここが修正されたクロストーク関数 ★★★★★
def evaluate_crosstalk_corrected(graph_db):
    """元々のロジックを正しく再現したクロストーク評価関数"""
    band_indices_3db = _find_band_indices(graph_db, 3.0)
    if band_indices_3db is None:
        return 999.0
    _, right_idx = band_indices_3db

    # 帯域の右端から、グラフが初めて上昇に転じる点を探す
    rising_edge_start_index = -1
    if right_idx + 1 >= len(graph_db):
        return 999.0 # 阻止域がない

    for i in range(right_idx + 1, len(graph_db)):
        if graph_db[i] > graph_db[i-1]:
            rising_edge_start_index = i
            break
    
    # もし上昇点が最後まで見つからなければ（単調減少）、クロストークは非常に良い
    if rising_edge_start_index == -1:
        return 999.0

    # 上昇点から先で最も高い値をサイドローブのピークとする
    sidelobe_peak = np.max(graph_db[rising_edge_start_index:])
    
    return np.max(graph_db) - sidelobe_peak
    

# --- 元々のコードで使われていたヘルパー関数 ---
def _local_maximum_and_minimum(graph):
    max_point_y, min_point_y = [], []
    # 簡易的な実装（連続する2点間の比較）
    for i in range(1, len(graph) - 1):
        if graph[i-1] < graph[i] and graph[i] > graph[i+1]:
            max_point_y.append(graph[i])
        elif graph[i-1] > graph[i] and graph[i] < graph[i+1]:
            min_point_y.append(graph[i])
    # point_y[0]が極大値リスト, point_y[1]が極小値リスト
    return [[], [max_point_y, min_point_y]]


# ---------------------------------------------------------------- #
# ▼▼▼ メインの実行部分 ▼▼▼
# ---------------------------------------------------------------- #

x_m = calculate_x(center_wavelength=1550e-9, FSR=35e-9)
graph_db = simulate_transfer_function(
    wavelength=x_m,
    L=np.array([
        6.2e-05, 6.2e-05, 7.75e-05, 7.75e-05, 7.75e-05, 7.75e-05,
    ]),
    K=np.array([0.12691515, 0.03465236, 0.0510429, 0.0541586, 0.06728448, 0.12441187, 0.60229625]),
    alpha=11.51, eta=0.996, n_eff=2.2, n_g=4.4, center_wavelength=1550e-9
)

x_nm = x_m * 1e9
# ハイブリッド版の評価関数を呼び出す
evaluate_graph_hybrid(x_nm, graph_db)





