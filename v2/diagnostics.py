import tensorflow as tf
import numpy as np

from constants import entropy_beta, m, num_gaussians, cov_range, weight_gaussians, grad_noise
from gmm import GMM, StateOps

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
	
	
def test_gmm_sgd():
	gmm = GMM()
	gmm_zeros(gmm)
	
	point = np.random.rand(m)
	state_ops = StateOps()
	
	sess = tf.Session()
	
	print "\nLoss \t\t Grad sizes"
	
	for i in range(1000):
		losses = []
		grad_sizes = []
	
		grads = sess.run([state_ops.grads],
								feed_dict={	state_ops.point:point, 
											state_ops.mean_vectors:gmm.mean_vectors, 
											state_ops.inv_cov_matrices:gmm.inv_cov_matrices})
		grads = np.reshape(grads[0],[m,])
		
		if grad_noise > 0:
			grads += np.abs(grads)*grad_noise*np.random.random((m))
		
		grad_sizes.append(np.mean(abs(grads)))
		
		point += -0.1*grads
		### Try an Adam version, which should work in all cases, given a sufficiently high dimensionality and no plateaus
		
		loss = gmm.gmm_loss(np.reshape(point,(m,1)))
		losses.append(loss)
		
		if i % 100 == 0:
			print "{:4f} \t {:4f}".format(np.mean(losses), np.mean(grad_sizes))
			losses = []
			grad_sizes = []


if __name__ == "__main__":
	test_gmm_sgd()
