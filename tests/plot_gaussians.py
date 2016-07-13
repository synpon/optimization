import numpy as np
import matplotlib.pyplot as plt

from gaussian_process import SquaredDistanceKernel, Matern52Kernel

#C:\Python27\python "C:\Users\Christopher\Google Drive\Attalus\plot_gaussians.py"

def plot_func(data, mu, cov):
    l = np.linalg.cholesky(cov)
    f_post = mu + np.dot(l, np.random.normal(size=(mu.shape[0], 1))) # Change the second argument to generate more lines
    plt.plot(data, f_post)

bounds = [0, 10, -10, 10]
plot_point_count = 1000

noise_variance = 0.001
kernel = SquaredDistanceKernel(kernel_param=0.01)

x = np.linspace(bounds[0], bounds[1], plot_point_count).reshape(-1,1)
mu = np.zeros_like(x)
cov = kernel.compute(x, x) + noise_variance * np.eye(x.shape[0])

plot_func(x, mu, cov)
plt.show()
