import numpy as np

def SHADE(func, bounds, params, pop_size=50, max_iter=500, F=0.5, cr=0.7,  ftol=10**-8, callback=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    print("パラメータ確認",params)

    xdim = len(bounds)      #最適化する変数の個数(結合率の数を入力することになる)
    dimbounds = np.ravel(bounds)
    populations = rng.uniform(low=np.amin(dimbounds), high=np.amax(dimbounds), size=(pop_size, xdim))       #解候補の初期配置、boundsの最小値~最大値の中からランダムで数値を決定し、初期解の個数(pop_size)分だけ生成
    print("結合率である解候補の初期を示す。")
    print(populations)
    obj_list = [func(pop, params) for pop in populations]       #生成した初期解を関数に代入し評価値を返したリストを作成
    best_x = populations[np.argmin(obj_list)]       #最もよい評価を得た際の解を記録
    best_obj = min(obj_list)        #最もよい評価を得た際の評価を記録する
    prev_obj = best_obj     #最もよい評価を今後比較のために記録する

    for i in range(max_iter):
        for j in range(pop_size):

            mutated = mutation(F, bounds, j, pop_size, best_x, populations, rng)

            trial = crossover(mutated, populations[j], xdim, cr, rng)

            obj_list, populations = selection(func, params, j, obj_list, populations, trial)
        
        best_obj = min(obj_list)        #解候補を更新し、そのたびに最高の評価値がある場合は更新
        best_x = populations[np.argmin(obj_list)]       #最高の評価値が更新された場合用に記述、その解を記録
        
        print("評価値 = ",obj_list)
        print("結合率は以下の通りです。")
        print(populations)

        if best_obj < prev_obj:     #一周ごとに更新後の最高評価と更新前の最高評価を比べる
            if abs(prev_obj - best_obj) <= ftol:
                break       #収束した
            prev_obj = best_obj     #この記述は収束していない場合に行われる。今までの最高評価を更新する。
        
        if callback is not None:
            callback(i, best_x, best_obj, populations)

    
    return best_x, best_obj     #best_x➡最適化が終わったK, best_obj➡最適化が終わったE
    


    


    

def mutation(F, bounds, j, pop_size, best_x, populations, rng):
    count = 0
    dimbounds = np.ravel(bounds)

    indexes = [i for i in range(pop_size) if i != j]        #jは現在選んでいる解。それ以外の番号を指定しているインデックスを作成
    a,b = populations[rng.choice(indexes, 2, replace = False)]        #現在選んでいる解以外から2つを選ぶ。
    mutated = populations[j] + F * (best_x - populations[j]) + F * (a - b)       #突然変異を表す式。「要改変」➡現在はカレントトゥベスト➡最終的にはカレントトゥピーベストにする
    #print("初めに生成した変異ベクトル = ",mutated)
    while any([k >= np.amax(dimbounds) for k in mutated]) or any([k <= np.amin(dimbounds) for k in mutated]):
        #print("変異ベクトルが範囲外です。ループします。現在の回数 =",count)
        count += 1
        a,b = populations[rng.choice(indexes, 2, replace = False)]        #現在選んでいる解以外から2つを選ぶ。
        mutated = populations[j] + F * (best_x - populations[j]) + F * (a - b)
        #print("これは再度生成した変異ベクトルです。値が正常か確認できます。値は = ",a)
        if count > 50:
            mutated = np.clip(mutated, bounds[:, 0], bounds[:, 1])      #変異によって生まれたベクトルが異常な場合、値を範囲内に収める。
    

    return mutated


def crossover(mutated , target, dims, cr, rng):
    p = rng.random(dims)        #0~1の値をランダムにxdimの数だけ生成する
    p[rng.choice([i for i in np.arange(len(p))], 1)] = 0.0      #pの中で一つだけ確定で0にする。こうすることによってcrよりpが小さくなるのが一つ以上できるので、確定で一つはmutatedになる。
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]       #crよりpが小さい場合はmutated,そうでなければ変更しないようにする。

    return trial
    

def selection(func, params, j, obj_list, populations, trial):
    obj_trial = func(trial, params)     #交叉によって生成された解候補(pop_size分だけある)の評価値を計算する
    if obj_trial < obj_list[j]:     #交叉によって生成された解候補が現在のものより優れていた場合、更新する。
        populations[j] = trial
        obj_list[j] = obj_trial

    return obj_list, populations
