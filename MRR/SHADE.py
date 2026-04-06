import numpy as np
import random
import math
import time
from multiprocessing import Pool	#宮崎で追加
from scipy import stats
from pyDOE2 import lhs

def SHADE(func, bounds, params, pop_size, max_iter, H,  tol, callback=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    xdim = len(bounds)      #最適化する変数の個数(結合率の数を入力することになる)
    dimbounds = np.ravel(bounds)
    #populations = rng.uniform(low=np.amin(dimbounds), high=np.amax(dimbounds), size=(pop_size, xdim))       #解候補の初期配置、boundsの最小値~最大値の中からランダムで数値を決定し、初期解の個数(pop_size)分だけ生成

    lhs_samples = lhs(xdim, samples=pop_size, criterion="maximin")  # LHSで初期配置
    populations = np.amin(dimbounds) + lhs_samples * (np.amax(dimbounds) - np.amin(dimbounds))

    populations_G = populations     #各世代Gの解を記録。世代毎のGを記録しておき、各解候補の更新は別のものに記録する。
    obj_list = [func(pop, params) for pop in populations]       #生成した初期解を関数に代入し評価値を返したリストを作成
    obj_list_G = obj_list       #各世代Gの評価値を記録。扱いはpopulations_Gと同様
    best_x = populations[np.argmin(obj_list)]       #最もよい評価を得た際の解を記録,np.argminは最小の番号を返す
    best_obj = min(obj_list)        #最もよい評価を得た際の評価を記録する
    prev_obj = best_obj     #最もよい評価を今後比較のために記録する
    CR_para = 0.5
    F_para = 0.5
    MCR_para_H = [CR_para] * H     #各解候補のCR,メモリHの数だけ過去のパラメータを保存できる。
    MF_para_H = [F_para] * H       #各解候補のF
    k = 0       #メモリに更新を行った回数を数えるための変数
    Archive = [ [0.0] * xdim ] *  pop_size    #外部アーカイブ。解候補の数だけ存在する。
    Archivetimes = 0        #外部アーカイブに更新が入った回数を記録する。アーカイブがあふれるまでカウントを続けさせる。
    time_sta = time.perf_counter()      #時間計測開始


    for i in range(max_iter):
        print("これは",i,"世代を表している")
        
        r = [random.randint(0,H-1) for i in range(pop_size)]       #ランダムにメモリHの中から一つ番号を選ぶ。それがその世代が参照する制御パラメータになる。
        P_i = pop_size * random.uniform((2/pop_size), 0.2)     #カレントトゥピーベストのためのP、これで上位いくつまでかを小数で表す。おそらく2~3になる。
        P_i_int = math.floor(P_i)       #上記のPを整数に変換。小数を切り捨てることにより上位何位までを指定できるように。
        S_F = np.array([])
        S_CR = np.array([])
        delta_fk = np.array([])


        
        
        mut_cross_paras = [[MF_para_H[r[j]], MCR_para_H[r[j]], bounds, j, pop_size, obj_list_G, populations_G, P_i_int, Archive, xdim, np.random.randint(0, 2 ** 32 -1)] for j in range(pop_size)]
        p = Pool(processes = pop_size)
                  
        tmp = list( p.map(wrapper_mut_cross, mut_cross_paras) ) #一時的な答え、この後スライスし、必要なところだけ切り取る
        
        all_trial = np.zeros((pop_size, xdim))
        all_Fi = np.zeros(pop_size)
        all_CRi = np.zeros(pop_size)
        for I in range(pop_size):
            all_trial[I] = tmp[I][0]
            all_Fi[I] = tmp[I][1]
            all_CRi[I] = tmp[I][2]

        

        for j in range(pop_size):
            obj_list[j], populations[j], S_F, S_CR, delta_fk, Archive, Archivetimes = selection(func, params, j, obj_list_G, populations_G, all_trial[j], all_Fi[j], all_CRi[j], S_F, S_CR, delta_fk, Archive, Archivetimes)
        
        if S_F.size !=0 and S_CR.size !=0:
            #MF_para_H[k] = ( sum( ( delta_fk * (S_F ** S_F) ) / sum(delta_fk) ) ) / ( sum( ( delta_fk * S_F ) / sum(delta_fk) ) )
            MF_para_H[k] = np.average(S_F * S_F, weights = delta_fk) / np.average(S_F, weights = delta_fk)
            MCR_para_H[k] = np.average(S_CR, weights = delta_fk)
            k = k + 1
            if k > (H-1):
                k = 0
        print("記録メモリ F = ",MF_para_H)
        print("記録メモリ CR = ",MCR_para_H)
        print("scipy Fi = ",all_Fi)
        print("scipy CRi = ",all_CRi)




        populations_G = populations     #世代Gの最適化が終了したため、世代Gの記録を更新する。
        obj_list_G = obj_list
        best_obj = min(obj_list)        #解候補を更新し、そのたびに最高の評価値がある場合は更新
        best_x = populations[np.argmin(obj_list)]       #最高の評価値が更新された場合用に記述、その解を記録
        print("現在の評価値 = ",obj_list_G)


        if best_obj < prev_obj:     #一周ごとに更新後の最高評価と更新前の最高評価を比べる
            print("std(標準偏差？) = ", np.std(obj_list))
            print("mean(平均) = ", np.mean(obj_list))
            print("評価値確認",obj_list)
            if np.std(obj_list) <= tol * np.abs(np.mean(obj_list)):     #用検討。収束方法を検討
                break       #収束した
            prev_obj = best_obj     #この記述は収束していない場合に行われる。今までの最高評価を更新する。
        
        if callback is not None:
            callback(i, best_x, best_obj, populations)
        
        

        time_end = time.perf_counter()
        tim = time_end - time_sta
        print("現在の経過時間は",tim)
    
    return best_x, best_obj     #best_x➡最適化が終わったK, best_obj➡最適化が終わったE
    

def mut_cross(MF_para_H, MCR_para_H, bounds, j, pop_size, obj_list_G, populations_G, P_i_int, Archive, dims, seed):
    rng = np.random.default_rng(seed)
    select_populations = Archive + populations_G
    Fi = -1.0
    while Fi <= 0.0:
        Fi = stats.cauchy.rvs(loc = MF_para_H, scale = math.sqrt(0.1), size = 1, random_state = rng)
        #Fi = rng.normal(MF_para_H,math.sqrt(0.1))
        if Fi > 1.0:
            Fi = 1.0
    #Fi = 0.5
    A = np.array(obj_list_G)
    A_sort_index = np.argsort(A)        #ここ二行でobj_list_Gのソートを行っている。評価の良い順に並べ、そのインデックスがリストになっている。
    xpbest_group = [populations_G[A_sort_index[i]] for i in range(P_i_int)]      #G世代の解候補の中から、評価が高いものをP_i_intの数だけ選んだ集合を作る。
    xpbest = random.choice(xpbest_group)        #G世代の解候補の中から上位N×P番目までの候補から一つを選んだ。
    indexes = [i for i in range(pop_size) if i != j]        #jは現在選んでいる解。それ以外の番号を指定しているインデックスを作成
    a = populations_G[rng.choice(indexes, 1, replace = False)]        #現在選んでいる解以外から1つを選ぶ。
    b = random.choice(select_populations)       #アーカイブも含めて解候補の中から解を一つ選ぶ。➡「要改変」アーカイブは問題ないが、G世代の解候補から選ぶ際に、jを除いていない。これにより注目しているものと同じ要素を選ぶ可能性がある。
    while np.all(b == 0.0):           #bが0のときは一生新たに選択をする。bがゼロとなるのは埋まっていないアーカイブを選択した場合である。
        b = random.choice(select_populations)
    mutated = populations_G[j] + Fi * (xpbest - populations_G[j]) + Fi * (a - b)       #突然変異を表す式。「要改変」➡現在はカレントトゥベスト➡最終的にはカレントトゥピーベストにする
    #変異によって生まれたベクトルが異常な場合、値を範囲内に収める。
    for i in range(dims):
        if mutated[0][i] <= 0:
            mutated[0][i] = populations_G[j][i] / 2
        elif mutated[0][i] > 0.996:
            mutated[0][i] = (populations_G[j][i] + 0.996) / 2

    trial = np.zeros(dims)
    CRi = stats.norm.rvs(loc = MCR_para_H, scale = math.sqrt(0.1), size = 1, random_state = rng)
    #CRi = rng.normal(MCR_para_H,math.sqrt(0.1))
    if CRi > 1:
        CRi = 1
    elif CRi < 0:
        CRi = 0
    #CRi = 0.7
    p = rng.random(dims)        #0~1の値をランダムにxdimの数だけ生成する
    p[rng.choice([i for i in np.arange(len(p))], 1)] = 0      #pの中で一つだけ確定で0にする。こうすることによってcrよりpが小さくなるのが一つ以上できるので、確定で一つはmutatedになる。
    
    for i in range(dims):        #crよりpが小さい場合はmutated,そうでなければ変更しないようにする。
        if p[i] <= CRi:
            trial[i] = mutated[0][i]
        else:
            trial[i] = populations_G[j][i]

    return trial, Fi, CRi      
    

def wrapper_mut_cross(args):
    return mut_cross(*args)


def selection(func, params, j, obj_list, populations, trial, Fi, CRi, S_F, S_CR, delta_fk, Archive, Archivetimes):
    obj_trial = func(trial, params)     #交叉によって生成された解候補(pop_size分だけある)の評価値を計算する
    if obj_trial < obj_list[j]:     #交叉によって生成された解候補が現在のものより優れていた場合、更新する。
        
        if Archivetimes == (len(Archive)-1):        #populations[j]が更新される前にアーカイブに保存する。
            Archive[random.randint(0, Archivetimes)] = populations[j]
        else:
            Archive[Archivetimes] = populations[j]
            Archivetimes = Archivetimes + 1

        delta_fk_cal = abs(obj_list[j] - obj_trial)
        delta_fk = np.append(delta_fk, delta_fk_cal)
        
        S_F = np.append(S_F, Fi)
        S_CR = np.append(S_CR, CRi)

        populations[j] = trial
        obj_list[j] = obj_trial
        
        
    else:       #優れていない場合でも、アーカイブに悪いものを保存する。
        if Archivetimes == (len(Archive)-1):
            Archive[random.randint(0, Archivetimes)] = trial
        else:
            Archive[Archivetimes] = trial
            Archivetimes = Archivetimes + 1


    return obj_list[j], populations[j], S_F, S_CR, delta_fk, Archive, Archivetimes
