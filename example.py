import numpy as np
from neldermead import NelderMead

if __name__ == '__main__':
    def rosenbrock(x):
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

    nelder = NelderMead()
    x0 = np.array([0., 0.])
    res = nelder.minimize(rosenbrock, x0, 100)
    print(res)