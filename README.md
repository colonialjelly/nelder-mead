# Nelder-Mead Method

A python implementation of the Nelder-Mead method for gradient free optimization.  

## Example Usage:

```python
import numpy as np
from neldermead import NelderMead

def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

nelder = NelderMead()
x0 = np.array([0., 0.])
res = nelder.minimize(rosenbrock, x0, 100)
print(res) # {'x': array([1., 1.]), 'iteration': 96}
```

## References

- Original paper by J. A. Nelder and R. Mead  [A simplex method for function minimization](http://people.duke.edu/~hpgavin/cee201/Nelder+Mead-ComputerJournal-1965.pdf)

