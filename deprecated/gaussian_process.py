import numpy as np

class SquaredDistanceKernel():
    def __init__(self, kernel_param=0.1):
        self.kernel_parameter = kernel_param

    def compute(self, a, b):
        sq_dist = np.sum(a**2, 1).reshape(-1,1) + np.sum(b**2, 1) - 2 * np.dot(a, b.T)
        return np.exp(-0.5 * (1/self.kernel_parameter) * sq_dist)


class Matern52Kernel():
    def __init__(self, kernel_param=0.1):
        self.kernel_parameter = kernel_param

    def compute(self, a, b):
        sq_dist = np.sum(a ** 2, 1).reshape(-1,1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
        sq_dist *= 5
        return (1 + np.sqrt(sq_dist) + sq_dist/3) * np.exp(-np.sqrt(sq_dist))