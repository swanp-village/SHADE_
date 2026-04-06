import numpy as np
from pyDOE2 import lhs
import random
from concurrent.futures import ThreadPoolExecutor

def shade(objective_function, number_of_rings, eta=0.996, pop_size=20, gen=500, tol=1e-6, memory_size=5, workers=4, seed=None, params=None):
    """
    SHADE (Success-History based Adaptive Differential Evolution) with detailed logging.
    """
    np.random.seed(seed)
    random.seed(seed)

    # 次元数と探索範囲
    dim = number_of_rings + 1
    min_val = 1e-12
    max_val = eta

    # 初期化
    lhs_samples = lhs(dim, samples=pop_size, criterion="maximin")
    population = min_val + (max_val - min_val) * lhs_samples
    fitness_values = [objective_function(ind, params) for ind in population]

    archive = []  # 外部アーカイブ
    memory_cr = np.full(memory_size, 0.5)  # 初期CRメモリ
    memory_f = np.full(memory_size, 0.5)   # 初期Fメモリ

    best_idx = np.argmin(fitness_values)
    best_individual = population[best_idx]
    best_fitness = fitness_values[best_idx]

    print(f"Initial Population:\n{population}")
    print(f"Initial Fitness:\n{fitness_values}")
    print(f"Initial Memory CR: {memory_cr}")
    print(f"Initial Memory F: {memory_f}")

    for g in range(gen):
        new_population = []
        new_fitness_values = []
        s_cr, s_f = [], []  # 成功したCRとFを記録
        delta_fitness = []  # 成功した適応度差

        # 並列処理で評価
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            generation_cr = []  # 現世代のCR
            generation_f = []   # 現世代のF
            for i, target in enumerate(population):
                # 1. CRとFを適応的に生成
                idx = random.randint(0, memory_size - 1)
                cr = np.clip(np.random.normal(memory_cr[idx], 0.1), 0, 1)
                f = -1
                while f <= 0 or f > 1:
                    f = np.random.normal(memory_f[idx], 0.1)

                generation_cr.append(cr)
                generation_f.append(f)

                # 2. current-to-pbest/1 変異
                p = max(2, int(pop_size * 0.2))  # 上位20%
                pbest_idx = random.choice(np.argsort(fitness_values)[:p])
                pbest = population[pbest_idx]
                
                candidates = [ind for j, ind in enumerate(population) if j != i]
                if archive:
                    candidates += archive
                a, b = random.sample(candidates, 2)

                mutant = target + f * (pbest - target) + f * (a - b)
                mutant = np.clip(mutant, min_val, max_val)

                # 3. 交叉
                trial = np.where(np.random.rand(dim) < cr, mutant, target)
                trial = np.clip(trial, min_val, max_val)

                # 並列で評価
                futures.append(executor.submit(objective_function, trial, params))

            # 4. 選択操作
            for i, future in enumerate(futures):
                trial_fitness = future.result()
                if trial_fitness < fitness_values[i]:
                    new_population.append(trial)
                    new_fitness_values.append(trial_fitness)

                    # 成功履歴の記録
                    s_cr.append(generation_cr[i])
                    s_f.append(generation_f[i])
                    delta_fitness.append(fitness_values[i] - trial_fitness)

                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness
                else:
                    new_population.append(population[i])
                    new_fitness_values.append(fitness_values[i])

        # 5. メモリ更新
        if s_cr:
            weights = np.array(delta_fitness) / np.sum(delta_fitness)
            memory_f[idx] = np.sum(weights * (np.array(s_f) ** 2)) / np.sum(weights * np.array(s_f))
            memory_cr[idx] = np.sum(weights * np.array(s_cr)) / np.sum(weights)

        # 集団と適応度の更新
        population = new_population
        fitness_values = new_fitness_values

        # 外部アーカイブの更新
        archive = archive + [population[i] for i in range(pop_size) if fitness_values[i] != best_fitness]
        if len(archive) > pop_size:
            archive = random.sample(archive, pop_size)

        # ログ出力
        fitness_std = np.std(fitness_values)
        print(f"\nGeneration {g}:")
        print(f"Best Fitness = {best_fitness}, Std = {fitness_std}")
        #print(f"Current Population:\n{population}")
        #print(f"Memory CR: {memory_cr}")
        #print(f"Memory F: {memory_f}")
        #print(f"Generated CR: {generation_cr}")
        #print(f"Generated F: {generation_f}")

        # 収束判定
        if fitness_std < tol:
            print(f"Converged at Generation {g}: Best Fitness = {best_fitness}")
            break

    return best_individual, best_fitness
