import numpy as np

from constants import entropy_beta, m, num_gaussians, cov_range, weight_gaussians
from gmm import GMM

# Percentage of zero losses 
def gmm_zeros(gmm):
	n = 1000
	points = gmm.gen_points(n)
	
	losses = []
	for i in range(num_gaussians):
		d = points - gmm.mean_vectors[i]
		d = np.reshape(d,(n,m))
		loss = np.dot(d, gmm.inv_cov_matrices[i])
		loss = np.square(loss)
		loss = -np.exp(-0.5*loss)
		losses.append(loss)
		
	losses = np.array(losses)
	z = np.zeros_like(losses)
	z = np.equal(z,losses).astype(int)
	print "Zeros: ", np.mean(z)
