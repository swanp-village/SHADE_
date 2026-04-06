import math
import numpy as np
import numpy.typing as npt
import pprint

from scipy.optimize import differential_evolution
from MRR.SHADE import SHADE
from MRR.benchmark_function import BenchmarkFunction as BF

bf = BF()

#Sphere
def sphere(input_array): 
    return (input_array**2).sum(axis=-1)





#Sphere
a = 5.12
number_of_x = 2 #解の個数(次元の数ともいえる)
bounds = np.array([[-a, a] for _ in range(number_of_x)])
#A_bounds = np.array([(-a, a) for _ in range(number_of_x)])
#bounds = np.array([[-a, a],[-a, a]])
#bounds = A_bounds.reshape(-1,1)
params = 0
pop_size = 10
max_iter = 6000
H = 50
tol = 0.01
rng = np.random.Generator

print(bounds)
print(len(bounds.shape))
print(len(bounds))


#print( SHADE(bf.rastrigin, bounds, params, pop_size, max_iter, H, tol, callback = None, rng = None) )

result = differential_evolution(sphere, 
                            bounds, 
                            strategy="currenttobest1bin", 
                            disp = True,
                            workers=-1, 
                            updating="deferred", 
                            popsize=15,
                            maxiter=3000
                            )
                            

pprint.pprint(result)
