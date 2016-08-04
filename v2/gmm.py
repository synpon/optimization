import tensorflow as tf
import numpy as np

from constants import m, n, num_gaussians, cov_range, weight_gaussians
ratio = int(np.ceil(n/num_gaussians))

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
			C = np.linalg.inv(C)
			self.inv_cov_matrices.append(C)
			
			### Gradients?

		
	def gmm_loss(self, point):
		losses = []
		### Check equations
		for i in range(num_gaussians):
			d = point - self.mean_vectors[i]
			d = np.reshape(d,(1,m))
			loss = np.dot(d,self.inv_cov_matrices[i])
			loss = loss*loss
			loss = -np.exp(-0.5*loss)
			losses.append(loss)

		return np.mean(losses)
		
		
	def gen_point(self):
		### Use points near the means
		point = np.random.rand(m)
		point = np.reshape(point,[1,m,1])
		return point
		
		
	def choose_action(self,mean,variance):
		for i,v in enumerate(variance):
			mean[i] += np.random.normal(0,v)
		return mean
	
	
	def act(self, state, action):
		# There are no terminal states
		state += action
		loss = self.gmm_loss(state)
		reward = -loss ### check sign
		return reward, state
		
		