import numpy as np
import matplotlib.pyplot as plt

from gaussian_process import SquaredDistanceKernel, Matern52Kernel

#C:\Python27\python "C:\Users\Christopher\Google Drive\Attalus\plot_gaussians.py"
# Could add structure at different scales

def plot_func(data, mu, cov):
	l = np.linalg.cholesky(cov) # Lower triangular nxn matrix
	v1 = np.dot(l, np.random.normal(size=(mu.shape[0], 1000)))
	v2 = np.dot(np.random.normal(size=(mu.shape[0], 1000)), l)
	f_post = mu + np.dot(v1,v2)
	plt.imshow(f_post)
	
bounds = [0, 10, -10, 10]
n = 1000 # number of points to be plotted

noise_variance = 0.001
kernel = SquaredDistanceKernel(kernel_param=0.01)

x = np.linspace(bounds[0], bounds[1], n).reshape(-1,1)
y = np.linspace(bounds[0], bounds[1], n).reshape(-1,1)
mu = np.zeros_like(x)
cov = np.dot(kernel.compute(x,x),kernel.compute(y,y)) + noise_variance * np.eye(x.shape[0]) # nxn matrix

plot_func(x, mu, cov)
plt.show()
