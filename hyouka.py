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
    input_array = input_array.reshape(1,-1)
    return (input_array**2).sum(axis=-1)

#ellipsoid
def ellipsoid(input_array): 
    input_array = input_array.reshape(1,-1)
    #print("input_array = ",input_array)
    dim = input_array.shape[1]
    
    coef = [1000**(i/(dim-1)) for i in range(dim)]
    coef = np.array(coef)
    
    return ((input_array*coef)**2).sum(axis=1)

#k_tablet
def k_tablet(input_array):
    input_array = input_array.reshape(1,-1)
    dim = input_array.shape[1]
    k = math.ceil(dim/4)
            
    return (input_array[:,:k]**2).sum(axis=1) \
            +((100*input_array[:,k:])**2).sum(axis=1)


#rosenbrock_star
def rosenbrock_star(input_array):
    input_array = input_array.reshape(1,-1)
    input_array_1st = input_array[:,:1]
    input_array_rest = input_array[:,1:]
    
    return (100*(input_array_1st-input_array_rest**2)**2 \
            +(1-input_array_rest)**2).sum(axis=1)

#rosenbrock_chain
def rosenbrock_chain(input_array):
    input_array = input_array.reshape(1,-1)
    
    return (100*(input_array[:,1:]-input_array[:,:-1]**2)**2 \
            +(1-input_array[:,:-1])**2).sum(axis=1)

#bohachevsky
def bohachevsky(input_array):
    input_array = input_array.reshape(1,-1)
    
    return (input_array[:,:-1]**2 \
            +2*input_array[:,1:]**2 \
            -0.3*np.cos(3*np.pi*input_array[:,:-1]) \
            -0.4*np.cos(4*np.pi*input_array[:,1:]) \
            +0.7).sum(axis=1)

#ackley
def ackley(input_array):
    input_array = input_array.reshape(1,-1)
    dim = input_array.shape[1]
    
    return 20 \
            -20*np.exp(-0.2*((input_array**2).sum(axis=1)/dim)**0.5) \
            +np.e \
            -np.exp((np.cos(2*np.pi*input_array)).sum(axis=1)/dim)

#schaffer
def schaffer(input_array):
    input_array = input_array.reshape(1,-1)
    
    return ((input_array[:,:-1]**2+input_array[:,1:]**2)**0.25 \
                *(np.sin(50*(input_array[:,:-1]**2+input_array[:,1:]**2)**0.1)**2 \
            +1.0)).sum(axis=1)

#rastrigin
def rastrigin(input_array):
    input_array = input_array.reshape(1,-1)
    dim = input_array.shape[1]
    
    return 10*dim + \
            ((input_array-1)**2-10*np.cos(2*np.pi*(input_array-1))).sum(axis=1)

#





a = 5.12
#a = 32.768
#a = 100
number_of_x = 7 #解の個数(次元の数ともいえる)
bounds = np.array([[-a, a] for _ in range(number_of_x)])
params = 0
pop_size = 15
max_iter = 6000
H = 50
tol = 0.01
rng = np.random.Generator

print("bounds = ",bounds)
print("bounds.shape[1] = ",bounds.shape[1])
print("len(bounds) = ",len(bounds))
print("len(bounds.shape) = ",len(bounds.shape))


print( SHADE(rastrigin, bounds, params, pop_size, max_iter, H, tol, callback = None, rng = None) )


"""
result = differential_evolution(rastrigin, 
                            bounds, 
                            strategy="currenttobest1bin",
                            #disp = True,
                            workers=-1, 
                            updating="deferred", 
                            popsize=15,
                            maxiter=500
                            )

pprint.pprint(result)
"""
