import tensorflow as tf
import numpy as np

from constants import m, num_gaussians, cov_range, weight_gaussians
from ac_network import inv_scale_grads

class GMM(object):
	def __init__(self):
		
		self.mean_vectors = []
		self.inv_cov_matrices = []
		
		for i in range(num_gaussians):
			self.mean_vectors.append(np.random.rand(m,1))

			# Covariance matrices must be positive-definite
			Q = np.random.rand(m,m)*cov_range[1]
			Q_T = np.transpose(Q)
			
			D = np.abs(np.random.rand(m)*cov_range[1])
			D = np.diagflat(D)
			
			C = np.dot(np.dot(Q_T,D),Q)
			C = np.power(C,0.33)/m # Re-scale
			C = np.linalg.inv(C)
			self.inv_cov_matrices.append(C)
		
		
	def gmm_loss(self, points):
		losses = []
		for i in range(num_gaussians):
			d = points - self.mean_vectors[i]
			d = np.reshape(d,(1,m))
			loss = np.dot(d,self.inv_cov_matrices[i])
			loss = np.square(loss)
			loss = -np.exp(-0.5*loss)
			losses.append(loss)
		return np.mean(losses)
		
		
	def gen_points(self,num_points):
		### Use points near the means?
		point = np.random.rand(m*num_points)
		point = np.reshape(point,[1,m,num_points])
		return point
		
		
	def choose_action(self,mean,variance):
		for i,v in enumerate(variance):
			mean[i] += np.random.normal(0,v)
		
		mean = inv_scale_grads(mean)	
		return mean
	
	
	def act(self, state, action):	
		action = np.reshape(action,[m])
		state.point += action
		loss = self.gmm_loss(np.reshape(state.point,[1,m,1]))
		reward = -loss ### check sign
		return reward, state
		
		
class StateOps(object):
	def __init__(self):
	
		##### Graph to compute the gradients #####
		self.point = tf.placeholder(tf.float32, [m])
		self.mean_vectors = tf.placeholder(tf.float32, [num_gaussians,m,1])
		self.inv_cov_matrices = tf.placeholder(tf.float32, [num_gaussians,m,m])
		
		point = tf.reshape(self.point, [1,m,1])
		point = tf.tile(point, multiples=[1,1,num_gaussians])
		mean_vectors = tf.reshape(self.mean_vectors, [1,m,num_gaussians])
		d = point - mean_vectors # 1,m,num_gaussians

		losses = tf.batch_matmul(tf.transpose(d,[2,0,1]),self.inv_cov_matrices)
		# Follows the code in SciPy's multivariate_normal
		losses = tf.square(losses) # element-wise (num_gaussians,1,m)
		losses = tf.reduce_sum(losses,[2]) # Sum over the dimensions (num_gaussians,1)
		
		if weight_gaussians:
			raise NotImplementedError
		
		# The pdfs of the Gaussians are negative in order to create a minimization problem.
		losses = -tf.exp(-0.5*losses)
		
		grads = tf.gradients(losses,point)[0]
		self.grads = tf.reduce_mean(grads,[2]) # Average over the Gaussians

		
class State(object):
	def __init__(self,gmm,state_ops,sess):

		### Generate a point near the means?
		self.point = np.random.rand(m)
		
		self.grads = sess.run([state_ops.grads],
								feed_dict={	state_ops.point:self.point, 
											state_ops.mean_vectors:gmm.mean_vectors, 
											state_ops.inv_cov_matrices:gmm.inv_cov_matrices})
		self.grads = self.grads[0]