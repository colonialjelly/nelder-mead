import numpy as np
from simplex import Simplex


class NelderMead:
    """
        Nelder-Mead method is a gradient free optimization heuristic. It takes four scalar coefficients:

        alpha: reflection coefficient
        beta: expansion coefficient
        gamma: expansion coefficient
        delta: expansion coefficient

        According to the original paper the parameters above should satisfy the following inequalities:
        - alpha > 0
        - beta > 1
        - beta > alpha
        - 0 < gamma < 1
        - 0 < delta < 1

        Nelder, J A, and R Mead. 1965. A Simplex Method for Function
        Minimization. The Computer Journal 7: 308-13.
    """
    def __init__(self, params=None):
        if params is not None:
            self.alpha = params['alpha']  # reflection
            self.beta = params['beta']  # expansion
            self.gamma = params['gamma']  # contraction
            self.delta = params['delta']  # shrink
        else:
            self.alpha = 1.
            self.beta = 2.
            self.gamma = 0.5
            self.delta = 0.5

    def minimize(self, func, x0, max_iter=100, tol=1e-8):
        """
        :param func: The function to be optimized (callable)
        :param x0: The starting point (ndarray)
        :param max_iter: Maximum number of allowed iterations (int)
        :param tol: Tolerance for stopping criterion (float)
        :return: The minimum point found after either the iterations run out
        """
        simplex = Simplex(func, x0)
        current_iter = 0
        while True:
            simplex.order()
            x_best, f_best = simplex.best()
            x_second_worst, f_second_worst = simplex.second_worst()
            x_worst, f_worst = simplex.worst()
            if current_iter >= max_iter or self._stop_criterion(simplex, tol):
                return {'x': x_best, 'iteration': current_iter}
            current_iter += 1
            x_centroid = simplex.calculate_centroid()
            x_reflected, f_reflected = self._reflect(x_centroid, x_worst, func)

            if f_best <= f_reflected < f_second_worst:
                simplex.accept(x_reflected)
                continue

            if f_reflected < f_best:
                x_expansion, f_expansion = self._expand(x_centroid, x_reflected, func)
                if f_expansion < f_reflected:
                    simplex.accept(x_expansion)
                else:
                    simplex.accept(x_reflected)
                continue

            if f_reflected >= f_second_worst:
                if f_second_worst <= f_reflected < f_worst:
                    x_contracted, f_contracted = self._outside_contract(x_centroid, x_reflected, func)
                    if f_contracted <= f_reflected:
                        simplex.accept(x_contracted)
                        continue
                    else:
                        simplex.update(self._shrink(simplex))
                        continue

            if f_reflected >= f_worst:
                x_contracted, f_contracted = self._inside_contract(x_centroid, x_worst, func)
                if f_contracted < f_worst:
                    simplex.accept(x_contracted)
                    continue
                else:
                    simplex.update(self._shrink(simplex))
                    continue

    def _reflect(self, x_centroid, x_worst, func):
        x_reflected = x_centroid + self.alpha*(x_centroid - x_worst)
        return x_reflected, func(x_reflected)

    def _expand(self, x_centroid, x_reflection, func):
        x_expanded = x_centroid + self.beta*(x_reflection - x_centroid)
        return x_expanded, func(x_expanded)

    def _outside_contract(self, x_centroid, x_reflection, func):
        x_contracted = x_centroid + self.gamma*(x_reflection - x_centroid)
        return x_contracted, func(x_contracted)

    def _inside_contract(self, x_centroid, x_worst, func):
        x_contracted = x_centroid - self.gamma*(x_centroid - x_worst)
        return x_contracted, func(x_contracted)

    def _shrink(self, simplex):
        vertices = simplex.vertices
        ordered_simplex_idx = simplex.vertices_order
        shrinked_simplex = np.zeros_like(simplex.vertices)
        x_best, _ = simplex.best()
        for i, idx in enumerate(ordered_simplex_idx):
            shrinked_simplex[i] = x_best + self.delta*(vertices[idx] - x_best)
        return shrinked_simplex

    def _stop_criterion(self, simplex, tol):
        vertices = simplex.vertices
        x_best, _ = simplex.best()
        d = np.maximum(1, np.linalg.norm(x_best))
        return (1./d) * np.max(np.linalg.norm(vertices - x_best)) <= tol
