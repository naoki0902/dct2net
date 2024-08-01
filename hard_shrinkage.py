import numpy as np
import matplotlib.pyplot as plt


def hard_shrinkage(x, lamd):
    
    y = np.zeros_like(x)
    idx = np.where(np.abs(x) > lamd)
    y[idx] = x[idx]

    return y
    

def differentiable_shrinkage(x, lamd, m):

    return x * (x ** (2*m) / (x ** (2*m) + lamd ** (2*m)))


lamd = 1.0
m = 32
x = np.arange(-2.0, 2.0, 0.01)
y_hard_shrinkage = hard_shrinkage(x, lamd)
y_differentiable_shrinkage = differentiable_shrinkage(x, lamd, m)

plt.figure()
plt.plot(x, y_hard_shrinkage, label='phi')
plt.plot(x, y_differentiable_shrinkage, label='zeta (m=32)')
plt.legend()
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('shrinkage.png')