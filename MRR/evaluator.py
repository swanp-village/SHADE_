
import numpy as np
import numpy.typing as npt
from scipy.signal import argrelmax, argrelmin



def evaluate_band(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    center_wavelength: float,
    length_of_3db_band: float,
    max_crosstalk: float,
    H_p: float,
    H_s: float,
    H_i: float,
    r_max: float,
    weight: list[float],
    ignore_binary_evaluation: bool = False,
) -> np.float_:
    pass_band, cross_talk = _get_pass_band(x=x, y=y, H_p=H_p, center_wavelength=center_wavelength)
    if pass_band.shape[0] == 1:
        start = pass_band[0][0]
        end = pass_band[0][1]

    else:
        return np.float_(0)

    result = [
        _evaluate_pass_band(x=x, y=y, H_p=H_p, start=start, end=end),
        _evaluate_stop_band(x=x, y=y, H_p=H_p, H_s=H_s, start=start, end=end),
        #_evaluate_insertion_loss(x=x, y=y, H_i=H_i, center_wavelength=center_wavelength),    #昔の、挿入損失
        _evaluate_insertion_loss(x=x, y=y, H_i=H_i, center_wavelength=center_wavelength, steepness=2.0),    #新しい挿入損失、引数が増えた
        _evaluate_3db_band(x=x, y=y, length_of_3db_band=length_of_3db_band, start=start, end=end),
        _evaluate_ripple(x=x, y=y, r_max=r_max, start=start, end=end),
        _evaluate_cross_talk(y=y, max_crosstalk=max_crosstalk, pass_band_start=start, pass_band_end=end),
        _evaluate_shape_factor(x=x, y=y, start=start, end=end),
    ]
    n_eval = len(result)
    W_c = weight[:n_eval]
    W_b = weight[n_eval:]
    E_c = np.float_(0)
    E_b = np.float_(1)
    for i in range(n_eval):
        E_c += result[i][0] * W_c[i]
        if not result[i][1]:
            E_b *= W_b[i]
    if ignore_binary_evaluation:
        return E_c
    E = E_c * E_b
    # 各評価関数の結果を表示
    
    #print(E_b)
    #print(E_c)
    #for i, res in enumerate(result):
        #print(f"評価関数 {i+1}: 値 = {res[0]}, バイナリ評価 = {res[1]}")
    
    
    return E





"""
#⓹　クロストークのペナルティを調整したやつ
def evaluate_band(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    center_wavelength: float,
    length_of_3db_band: float,
    max_crosstalk: float,
    H_p: float,
    H_s: float,
    H_i: float,
    r_max: float,
    weight: list[float],
    ignore_binary_evaluation: bool = False,
) -> np.float_:
    pass_band, cross_talk = _get_pass_band(x=x, y=y, H_p=H_p, center_wavelength=center_wavelength)
    if pass_band.shape[0] == 1:
        start = pass_band[0][0]
        end = pass_band[0][1]

    else:
        return np.float_(0)

    result = [
        _evaluate_pass_band(x=x, y=y, H_p=H_p, start=start, end=end),
        _evaluate_stop_band(x=x, y=y, H_p=H_p, H_s=H_s, start=start, end=end),
        #_evaluate_insertion_loss(x=x, y=y, H_i=H_i, center_wavelength=center_wavelength),    #昔の、挿入損失
        _evaluate_insertion_loss(x=x, y=y, H_i=H_i, center_wavelength=center_wavelength, steepness=2.0),    #新しい挿入損失、引数が増えた
        _evaluate_3db_band(x=x, y=y, length_of_3db_band=length_of_3db_band, start=start, end=end),
        _evaluate_ripple(x=x, y=y, r_max=r_max, start=start, end=end),
        _evaluate_cross_talk_with_dynamic_penalty(y=y, max_crosstalk=max_crosstalk, pass_band_start=start, pass_band_end=end),
        _evaluate_shape_factor(x=x, y=y, start=start, end=end),
    ]
    n_eval = len(result)
    W_c = weight[:n_eval]
    W_b = weight[n_eval:]
    E_c = np.float_(0)
    E_b = np.float_(1)
    for i in range(n_eval):
        E_c += result[i][0] * W_c[i]
        
        # --- ▼ここからが変更点▼ ---
        # 評価結果の2番目の値の「型」で処理を分岐する
        
        # bool型なら、従来のバイナリ評価
        if isinstance(result[i][1], (bool, np.bool_)):
            if not result[i][1]:
                E_b *= W_b[i]
        
        # float型なら、新しい動的ペナルティ評価
        else:
            # 関数から返されたペナルティ係数を直接掛け合わせる
            E_b *= result[i][1]
            
    if ignore_binary_evaluation:
        return E_c
        
    E = E_c * E_b
    
    
    return E
"""







def _calculate_pass_band_range(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    H_p: float,
) -> npt.NDArray[np.int_]:
    start = 0
    end = x.size - 1
    a: npt.NDArray[np.bool_] = np.where(y <= H_p, True, False)
    b: npt.NDArray[np.bool_] = np.append(a[1:], a[-1])
    pass_band_range: npt.NDArray[np.int_] = np.where(np.logical_xor(a, b))[0]
    if pass_band_range.size == 0:
        return pass_band_range
    if not a[pass_band_range][0]:
        pass_band_range = np.append(start, pass_band_range)
    if a[pass_band_range][-1]:
        pass_band_range = np.append(pass_band_range, end)
    pass_band_range = np.reshape(pass_band_range, [pass_band_range.size // 2, 2])

    return pass_band_range


def _get_pass_band(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    H_p: float,
    center_wavelength: float,
) -> (npt.ArrayLike, npt.ArrayLike):
    pass_band = []
    cross_talk = []
    for start, end in _calculate_pass_band_range(x=x, y=y, H_p=H_p):
        if center_wavelength >= x[start] and center_wavelength <= x[end]:
            pass_band.extend([start, end])
        else:
            cross_talk.extend([start, end])

    pass_band = np.reshape(pass_band, [len(pass_band) // 2, 2])
    cross_talk = np.reshape(cross_talk, [len(cross_talk) // 2, 2])

    return pass_band, cross_talk


def _get_3db_band1(x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], start: int, end: int) -> npt.ArrayLike:
    border: np.float_ = y.max() - 3
    a = np.where(y[start:end] <= border, True, False)
    b = np.append(a[1:], a[-1])
    index = np.where(np.logical_xor(a, b))[0]

    return index


def _get_3db_band(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    start: int,
    end: int
) -> npt.NDArray[np.int_]:
    # dBスケール前提（最大値から3dB以内）
    y_db = y[start:end]
    threshold = y_db.max() - 3

    # 条件を満たすインデックス
    mask = y_db >= threshold
    idx = np.where(mask)[0]

    if idx.size == 0:
        return np.array([], dtype=int)

    # 通過帯域の範囲（インデックスそのものを返す）
    return idx

def _evaluate_pass_band(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], H_p: float, start: int, end: int
) -> tuple[np.float_, bool]:
    distance: np.float_ = x[1] - x[0]
    a = abs(H_p * (x[end] - x[start]))
    if a < 1e-9:
        penalty = np.exp(-a/1e-9)
        return (np.float_(penalty),False)
    b = abs(np.sum(H_p - y[start:end]) * distance)
    E = b / a
    return (E, True)


def _evaluate_stop_band(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], H_p: float, H_s: float, start: int, end: int
) -> tuple[np.float_, bool]:
    distance: np.float_ = x[1] - x[0]
    c = abs((H_s - H_p) * ((x[start] - x[0]) + (x[-1] - x[end])))

    if c < 1e-9:
        penalty = np.exp(-c/1e-9)
        return (np.float_(penalty),False)

    y1 = np.where(y[0:start] > H_s, H_p - y[0:start], H_p - H_s)
    y1 = np.where(y1 > 0, y1, 0)
    y2 = np.where(y[end:-1] > H_s, H_p - y[end:-1], H_p - H_s)
    y2 = np.where(y2 > 0, y2, 0)
    d = abs((np.sum(y1) + np.sum(y2)) * distance)
    E = d / c

    return (E, True)


"""
def _evaluate_insertion_loss(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    H_i: float,
    center_wavelength: float,
) -> tuple[np.float_, bool]:
    # center_wavelength に最も近い x のインデックスを探す
    idx = np.argmin(np.abs(x - center_wavelength))
    insertion_loss_at_center = y[idx] # これで単一の数値が得られる
    loss_abs = np.abs(insertion_loss_at_center)
    H_i_abs = np.abs(H_i)
    E = 1 / (1 + np.exp(loss_abs - H_i_abs))
    if insertion_loss_at_center < H_i:
        return(E, False)
    else:
        return(E, True)
 """            



#挿入損失が0dBに近づくほど急激にスコアが上昇する評価関数。新たな引数があるので注意
def _evaluate_insertion_loss(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    H_i: float,
    center_wavelength: float,
    steepness: float = 2.0  # ★スコアカーブの急峻さを調整する新しいパラメータ
) -> tuple[np.float_, bool]:
    # 1. 挿入損失を特定（ここは変更なし）
    idx = np.argmin(np.abs(x - center_wavelength))
    insertion_loss_at_center = y[idx]

    # 2. バイナリ評価（合格/不合格）は変更なし
    is_ok = insertion_loss_at_center >= H_i

    # 3. 新しい連続スコアの計算ロジック
    loss_abs = np.abs(insertion_loss_at_center)
    H_i_abs = np.abs(H_i)

    # 損失を目標値H_iを基準に正規化（0が良い、1が目標値）
    # 損失が0dBより良い（利得がある）場合は0にクリップ
    normalized_loss = np.maximum(0, loss_abs / H_i_abs)

    # べき乗関数でスコアを計算
    # (1 - 正規化損失) の値を steepness 乗する
    # 損失が目標値を超えた場合はスコアが0になるようにクリップ
    score = (np.maximum(0, 1 - normalized_loss)) ** steepness
    
    return (np.float_(score), is_ok)







def _evaluate_3db_band(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], length_of_3db_band: float, start: int, end: int
) -> tuple[np.float_, bool]:
    distance: np.float_ = x[1] - x[0]
    index = _get_3db_band(x=x, y=y, start=start, end=end)
    if index.size <= 1:
        # ペナルティ計算を追加
        return (np.float_(0), False)

    practical_length_of_3db_band = distance * (index[-1] - index[0])
    if practical_length_of_3db_band > length_of_3db_band:
        E = (2 * length_of_3db_band - practical_length_of_3db_band) / length_of_3db_band
    else:
        E = practical_length_of_3db_band / length_of_3db_band
    E = E ** 3
    return (E, True)
    


#標準偏差型
def _evaluate_ripple(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], r_max: float, start: int, end: int
) -> tuple[np.float_, bool]:
    pass_band = y[start:end]
    index = _get_3db_band(x=x, y=y, start=start, end=end)
    if index.size <= 3:
        print("すくない",index)
        return (np.float_(0), False)
        # ペナルティ計算

    n = index.size
    if n > 10:  # 十分な点数あるとき
        central_index = index[int(0.1 * n) : int(0.9 * n)]
        three_db_band = pass_band[central_index]
    else:
        three_db_band = pass_band[index]  # 点数少ない時はそのまま使う
        
    std_ripple = np.std(three_db_band)
    range_ripple = three_db_band.max() - three_db_band.min()

 
    r_max1 = 1.0

    if std_ripple > r_max1 or range_ripple > r_max:
        E = 0
    else:
        E = 1 - (std_ripple + range_ripple) / (r_max1 * r_max)

    return (np.float_(E), True)


"""
def _evaluate_cross_talk(
    y: npt.NDArray[np.float_], max_crosstalk: float, pass_band_start: int, pass_band_end: int
) -> tuple[np.float_, bool]:
    start = y[:pass_band_start]
    end = y[pass_band_end:]
    maxid_start = np.append(0, argrelmax(start))
    maxid_end = np.append(argrelmax(end), -1)
    start_peak = start[maxid_start]
    end_peak = end[maxid_end]
    a = np.any(start_peak > max_crosstalk)
    b = np.any(end_peak > max_crosstalk)
    if a or b:
        return (1 / (1 + penalty), False)
    return (np.float_(1), True)
"""


#これを用いればうまくクロストークを維持しながら最適化できるが、挿入損失が低い。挿入損失を調整すればあるいは
"""
#トップとサイドの差をしきい値の判定に用いる
def _evaluate_cross_talk(
    y: npt.NDArray[np.float_], 
    max_crosstalk: float,  # ★ここに正の数(例: 30)を渡すことを想定
    pass_band_start: int, 
    pass_band_end: int
) -> tuple[np.float_, bool]:
    
    # --- ステップ1: ピーク検出 ---
    overall_peak = np.max(y)
    start_region = y[:pass_band_start]
    end_region = y[pass_band_end:]
    
    maxid_start = np.append(0, argrelmax(start_region))
    maxid_end = np.append(argrelmax(end_region), -1)
    all_start_peaks = start_region[maxid_start]
    all_end_peaks = end_region[maxid_end]

    # --- ステップ2: 新しい合否判定ロジック ---
    # 全てのサイドローブについて、主信号との差（クロストーク）を計算
    suppression_start = overall_peak - all_start_peaks
    suppression_end = overall_peak - all_end_peaks
    
    # ★ 計算したクロストークが、引数で指定した値(max_crosstalk)を下回るかチェック
    is_fail_start = np.any(suppression_start < max_crosstalk)
    is_fail_end = np.any(suppression_end < max_crosstalk)
    
    # --- ステップ3: スコア計算と返り値 ---
    worst_start_peak = np.max(all_start_peaks)
    worst_end_peak = np.max(all_end_peaks)
    worst_suppression_start = overall_peak - worst_start_peak
    worst_suppression_end = overall_peak - worst_end_peak
    
    score_val = worst_suppression_start + worst_suppression_end

    if is_fail_start or is_fail_end:
        # 不合格の場合
        if score_val <= 0:
            E = 0.0
        else:
            normalized_score = np.log(score_val + 1)
            E = 1 - np.exp(-normalized_score / 4)
        return (np.float_(E), False)
    else:
        # 合格の場合
        return (np.float_(1), True)
"""



"""
#⓹
def _evaluate_cross_talk_with_dynamic_penalty(
    y: npt.NDArray[np.float_],
    max_crosstalk: float,       # 確保したいクロストーク量 [dB] (例: 30.0)
    pass_band_start: int,
    pass_band_end: int,
    # --- ★新しい制御パラメータ ---
    min_penalty_multiplier: float = 0.5, # ペナルティ係数の最小値
    penalty_steepness: float = 0.6       # ペナルティが急になる度合い
) -> tuple[np.float_, np.float_]: # 返り値の2番目が bool から float に変更
    
    # --- 連続スコアの計算（ここは変更なし） ---
    overall_peak = np.max(y)
    start_region = y[:pass_band_start]
    end_region = y[pass_band_end:]
    
    maxid_start = np.append(0, argrelmax(start_region))
    maxid_end = np.append(argrelmax(end_region), -1)
    all_start_peaks = start_region[maxid_start]
    all_end_peaks = end_region[maxid_end]

    worst_start_peak = np.max(all_start_peaks)
    worst_end_peak = np.max(all_end_peaks)
    worst_suppression_start = overall_peak - worst_start_peak
    worst_suppression_end = overall_peak - worst_end_peak
    
    score_val = worst_suppression_start + worst_suppression_end
    
    if score_val <= 0:
        continuous_score = 0.0
    else:
        normalized_score = np.log(score_val + 1)
        continuous_score = 1 - np.exp(-normalized_score / 4)

    # --- ★新しい動的ペナルティ係数の計算 ---
    # 全てのサイドローブの抑制量を計算
    suppression_start = overall_peak - all_start_peaks
    suppression_end = overall_peak - all_end_peaks
    
    # 最も抑制できていない値（ワーストケース）を特定
    worst_suppression = min(np.min(suppression_start), np.min(suppression_end))
    
    # 閾値からの違反量を計算（違反していなければ0以下になる）
    violation_db = max_crosstalk - worst_suppression
    
    if violation_db <= 0:
        # 違反がない場合、ペナルティは無し
        penalty_multiplier = 1.0
    else:
        # 違反がある場合、違反量に応じてペナルティを計算
        # 指数関数的に1.0からmin_penalty_multiplierへと減衰させる
        penalty_multiplier = min_penalty_multiplier + \
            (1.0 - min_penalty_multiplier) * np.exp(-penalty_steepness * violation_db)
            
    return (np.float_(continuous_score), np.float_(penalty_multiplier))
"""






#これが一番うまくいく。(過去いっただけ、二つ↑のやつが現状良い)

def _evaluate_cross_talk(  y: npt.NDArray[np.float_], max_crosstalk: float, pass_band_start: int, pass_band_end: int
) -> tuple[np.float_, bool]:
    overall_peak = np.max(y)
    start = y[:pass_band_start]
    end = y[pass_band_end:]
    maxid_start = np.append(0, argrelmax(start))
    maxid_end = np.append(argrelmax(end), -1)
    start_peak = start[maxid_start]
    end_peak = end[maxid_end]
    start_peak_db = np.max(start_peak)
    end_peak_db = np.max(end_peak)
    excess_start = overall_peak - start_peak_db
    excess_end = overall_peak - end_peak_db
    
    #print("start_peak:" , start_peak_db)
    #print("end_peak" , end_peak_db)
    #print(excess_start)
    #print(excess_end)
    
    score = np.sum(excess_start) + np.sum(excess_end)
    #print(score)
    a = np.any(start_peak > max_crosstalk)
    b = np.any(end_peak > max_crosstalk)
    if a or b :
        if score <= 0 :
            E = 0
        else:
            normalized_score = np.log(score + 1)
            E = 1 - np.exp(-normalized_score/4)
        return(E,False)
    else:
        return(1,True)





"""
def _evaluate_cross_talk(  y: npt.NDArray[np.float_], max_crosstalk: float, pass_band_start: int, pass_band_end: int
) -> tuple[np.float_, bool]:
    overall_peak = np.max(y)
    start = y[:pass_band_start]
    end = y[pass_band_end:]
    maxid_start = np.append(0, argrelmax(start))
    maxid_end = np.append(argrelmax(end), -1)
    start_peak = start[maxid_start]
    end_peak = end[maxid_end]
    start_peak_db = np.max(start_peak)
    end_peak_db = np.max(end_peak)
    excess_start = overall_peak - start_peak_db
    excess_end = overall_peak - end_peak_db
    score = np.sum(excess_start) + np.sum(excess_end)
    a = np.any(start_peak > max_crosstalk)
    b = np.any(end_peak > max_crosstalk)
    normalized_score = np.log(score + 1)
    E = 1 - np.exp(-normalized_score/4)
    if a or b:
        return(E, False)
    else:
        return(E, True)
"""    



"""
def _evaluate_cross_talk(
    y: npt.NDArray[np.float_], max_crosstalk: float, pass_band_start: int, pass_band_end: int
) -> tuple[np.float_, bool]:
    start = y[:pass_band_start]
    end = y[pass_band_end:]
    maxid_start = np.append(0, argrelmax(start))
    maxid_end = np.append(argrelmax(end), -1)
    start_peak = start[maxid_start]
    end_peak = end[maxid_end]
    excess_start = np.maximum(start_peak - max_crosstalk, 0)
    excess_end = np.maximum(end_peak - max_crosstalk, 0)
    penalty = np.sum(excess_start) + np.sum(excess_end)
    a = np.any(start_peak > max_crosstalk)
    b = np.any(end_peak > max_crosstalk)
    if a or b:
        return (1 / (1 + penalty), False)
    else:
        E = 1
        return(E,True)
"""


    

def _evaluate_shape_factor(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], start: int, end: int
) -> tuple[np.float_, bool]:
    index = _get_3db_band1(x=x, y=y, start=start, end=end)
    if index.size <= 1:
        return (np.float_(1e-6), False)

    practical_length = x[index[-1]] - x[index[0]]
    total_length = x[end] - x[start]
    E = practical_length / total_length
    if E < 0.5:
        return (E ** 2, False)  # ペナルティを連続化
    return (E, True)
