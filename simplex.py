import numpy as np


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
            b = (c / (N * np.sqrt(2))) * (np.sqrt(N+1) - 1)
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


