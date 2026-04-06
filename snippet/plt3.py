import matplotlib.pyplot as plt
from random import seed, random

fig, ax = plt.subplots(figsize=(8, 6))
seed(1)

def evaluate(x):
    if x > 0.5:
        return x
    else:
        return x * 0.5

x = [i for i in range(1000)]
y = [
    evaluate(random())
    for _ in x
]

ax.scatter(x, y)
ax.set_ylabel('score')
plt.show()
