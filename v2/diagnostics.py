import tensorflow as tf
import numpy as np

from constants import entropy_beta, m, num_gaussians, cov_range, weight_gaussians, grad_noise
from gmm import GMM, StateOps

class SGD(object):
	def __init__(self):
		self.lr = 0.1
		
		
class Adam(object):
	def __init__(self):
		self.lr = 0.001
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.epsilon = 1e-8
		
sgd = SGD()
adam = Adam()

sess = tf.Session()

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
	
	
def optimize(point, gmm, optimizer):
	print "\nLoss \t\t Grad sizes"	
	M = np.zeros_like(point)
	V = np.zeros_like(point)
	
	state_ops = StateOps()
	start_loss = None
	
	for i in range(1,1000):
		losses = []
		grad_sizes = []
		
		feed_dict = {state_ops.point: point, 
					state_ops.mean_vectors: gmm.mean_vectors, 
					state_ops.inv_cov_matrices: gmm.inv_cov_matrices}
					
		if weight_gaussians:
			feed_dict[state_ops.gaussian_weights] = gmm.gaussian_weights
	
		grads = sess.run([state_ops.grads],feed_dict=feed_dict)
		grads = np.reshape(grads[0],[m,])
		
		if grad_noise > 0:
			grads += np.abs(grads)*grad_noise*np.random.random((m))
		
		grad_sizes.append(np.mean(abs(grads)))
		
		if optimizer == 'sgd':
			point += -sgd.lr*grads
			
		elif optimizer == 'adam':
			t = i
			lr_t = adam.lr
			lr_t *= np.sqrt(1 - np.power(adam.beta2,t))
			lr_t /= (1 - np.power(adam.beta1,t))

			# Copied from the Chainer implementation
			M += (1 - adam.beta1) * (grads - M)
			V += (1 - adam.beta2) * (grads * grads - V)
			point -= lr_t * M / (np.sqrt(V) + adam.epsilon)			
		
		loss = gmm.gmm_loss(np.reshape(point,(m,1)))
		losses.append(loss)
		
		if i == 1:
			start_loss = loss
		
		if i % 100 == 0:
			print "{:4g} \t {:4g}".format(np.mean(losses), np.mean(grad_sizes))
			losses = []
			grad_sizes = []
			
	loss_change = loss - start_loss
	print "Total change in loss: ", loss_change
	print "Relative change magnitude: ", np.abs(loss_change)/np.abs(start_loss)
			
			
def main():
	gmm = GMM()
	gmm_zeros(gmm)
	
	point = np.random.rand(m)

	print "\nOptimizing with SGD"
	optimize(point,gmm,'sgd')
	
	print "\nOptimizing with Adam"
	optimize(point,gmm,'adam')


if __name__ == "__main__":
	main()
