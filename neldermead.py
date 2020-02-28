import numpy as np


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
        x_reflected = x_centroid + self.alpha * (x_centroid - x_worst)
        return x_reflected, func(x_reflected)

    def _expand(self, x_centroid, x_reflection, func):
        x_expanded = x_centroid + self.beta * (x_reflection - x_centroid)
        return x_expanded, func(x_expanded)

    def _outside_contract(self, x_centroid, x_reflection, func):
        x_contracted = x_centroid + self.gamma * (x_reflection - x_centroid)
        return x_contracted, func(x_contracted)

    def _inside_contract(self, x_centroid, x_worst, func):
        x_contracted = x_centroid - self.gamma * (x_centroid - x_worst)
        return x_contracted, func(x_contracted)

    def _shrink(self, simplex):
        vertices = simplex.vertices
        ordered_simplex_idx = simplex.vertices_order
        shrinked_simplex = np.zeros_like(simplex.vertices)
        x_best, _ = simplex.best()
        for i, idx in enumerate(ordered_simplex_idx):
            shrinked_simplex[i] = x_best + self.delta * (vertices[idx] - x_best)
        return shrinked_simplex

    def _stop_criterion(self, simplex, tol):
        vertices = simplex.vertices
        x_best, _ = simplex.best()
        d = np.maximum(1, np.linalg.norm(x_best))
        return (1. / d) * np.max(np.linalg.norm(vertices - x_best)) <= tol


class Simplex:
    def __init__(self, func, x0, init_type='pfeffer'):
        if init_type == 'pfeffer':
            self.vertices = self.pfeffer_init_simplex(x0)
        elif init_type == 'spendley':
            self.vertices = self.spendley_init_simplex(x0)
        else:
            raise Exception("Unknown initialization type")
        self.func = func
        self.vertices_order = None
        self.centroid = None

    def pfeffer_init_simplex(self, x0):
        """
        Ellen Fan.
        Global optimization of lennard-jones atomic clusters.
        Technical report, McMaster University, February 2002.
        """
        du = 0.05
        dz = 0.0075
        N = len(x0)
        vertices = np.zeros((N + 1, N), dtype=x0.dtype)
        vertices[0] = x0
        for i in range(N):
            y = np.array(x0, copy=True)
            if y[i] != 0:
                y[i] = (1 + du) * y[i]
            else:
                y[i] = dz
            vertices[i + 1] = y
        return vertices

    def spendley_init_simplex(self, x0, c=1):
        """
        W. Spendley, G. R. Hext, and F. R. Himsworth.
        Sequential application of simplex designs in optimisation and evolutionary operation.
        Technometrics, 4(4):441–461, 1962
        """
        N = len(x0)
        vertices = np.zeros((N + 1, N), dtype=x0.dtype)
        vertices[N] = x0
        for i in range(N):
            b = (c / (N * np.sqrt(2))) * (np.sqrt(N + 1) - 1)
            a = b + (c / np.sqrt(2))
            x = np.repeat(b, N)
            x[i] = a
            vertices[i] = x

        return vertices

    def update(self, new_vertices):
        """Updates the simplex with the new vertices"""
        self.vertices = new_vertices
        self.order()
        self.calculate_centroid()

    def accept(self, x):
        """Replace the worst point with te given point"""
        self.vertices[self.vertices_order[-1]] = x

    def order(self):
        func_values = np.apply_along_axis(func1d=self.func, axis=1, arr=self.vertices)
        self.vertices_order = np.argsort(func_values, axis=0)

    def calculate_centroid(self):
        """Calculates the centroid of the simplex. The centroid does not include the worst point"""
        idx = self.vertices_order[:-1]
        self.centroid = np.mean(self.vertices[idx], axis=0)
        return self.centroid

    def best(self):
        x_best = self.vertices[self.vertices_order[0]]
        return x_best, self.func(x_best)

    def second_worst(self):
        x_second_worst = self.vertices[self.vertices_order[-2]]
        return x_second_worst, self.func(x_second_worst)

    def worst(self):
        x_worst = self.vertices[self.vertices_order[-1]]
        return x_worst, self.func(x_worst)
