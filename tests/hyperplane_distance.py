import numpy as np
m = 2 # dimensionality

def hyperplane_point_dist(point,hyperplane_points):
	X = np.matrix(hyperplane_points)
	k = np.ones((m,1))
	a = np.dot(np.linalg.inv(X), k) # vector
	D = np.dot(point,a) - 1 # scalar
	D /= np.linalg.norm(a)
	return np.abs(D)
	
print hyperplane_point_dist(np.array([0,3]),[np.array([4,2]),np.array([-1,6])])
