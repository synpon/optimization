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

		### Output percentage of zeros for 1000 random points when the first GMM is created
		#points = self.gen_points(1000)
		#losses = self.gmm_loss(points)
		#percent_zeros = np.mean(losses <= e-30) ###
		#print "Percentage of zeros: %f", percent_zeros
		
		
	def gmm_loss(self, points): ### Doesn't work with gen_points
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
		### Use points near the means
		point = np.random.rand(m*num_points)
		point = np.reshape(point,[1,m,num_points])
		return point
		
		
	def choose_action(self,mean,variance):
		for i,v in enumerate(variance):
			mean[i] += np.random.normal(0,v)
		
		mean = inv_scale_grads(mean)	
		return mean
	
	
	def act(self, state, action):
		state += action
		loss = self.gmm_loss(state)
		reward = -loss ### check sign
		return reward, state
		

class State(object):
	def __init__(self):
		pass
		
		