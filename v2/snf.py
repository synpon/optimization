from __future__ import division

import tensorflow as tf
import numpy as np

from constants import m, alpha, var_size, grad_noise
from ac_network import inv_scale_grads


class SNF(object):
	def __init__(self):
		self.means = np.random.rand(m)
		self.means = np.reshape(self.means,[m,1])
		
		self.variances = np.random.rand(m)*var_size
		self.variances = np.reshape(self.variances,[m,1])
		
		self.weights = np.random.rand(m)
		self.weights = np.reshape(self.weights,[m,1])
		
		
	def snf_loss(self, points):
		### Use tensorflow code instead, shared with StateOps.__init__()
		num_points = points.shape[1]
		losses = points - np.tile(self.means,[1,num_points])
		losses = np.square(losses)
		tiled_variances = np.tile(self.variances,[1,num_points])
		losses /= -2*tiled_variances
		losses = -np.exp(losses) # Negative so we can minimize the loss
		var_coeffs = 1/np.sqrt(2*tiled_variances*3.14)
		losses *= var_coeffs
		losses *= np.tile(self.weights,[1,num_points]) # element-wise
		
		mean = np.mean(losses,axis=0)/m # Average over the dimensions
		# If a product version is used here it must be computed in log-form to avoid NaNs
		losses = mean
		return np.mean(losses)
		
		
	def gen_points(self,num_points):
		points = np.random.rand(m*num_points)
		points = np.reshape(points,[m,num_points])
		return points
		
		
	def choose_action(self,mean,variance):
		for i,v in enumerate(variance):
			mean[i] += np.random.normal(0,v)*mean[i]
		mean = inv_scale_grads(mean)	
		return mean
	
	
	def act(self, state, action):
		state.point += action
		loss = self.snf_loss(state.point)
		reward = -loss
		return reward, state
		
		
class StateOps(object):
	def __init__(self):
	
		#===# Graph to compute the gradients #===#
		self.point = tf.placeholder(tf.float32, [m,1])
		self.means = tf.placeholder(tf.float32, [m,1])
		self.variances = tf.placeholder(tf.float32, [m,1])
		self.weights = tf.placeholder(tf.float32, [m,1])
		
		losses = self.point - self.means
		losses = tf.square(losses)
		losses /= -2*self.variances
		losses = -tf.exp(losses)
		var_coeffs = 1/tf.sqrt(2*self.variances*3.14)
		losses *= var_coeffs
		losses *= self.weights # element-wise
		
		mean = tf.reduce_mean(losses,reduction_indices=[0])/m # Mean over the dimensions
		# The NaN error is probably born here
		#geom_mean = tf.pow(tf.reduce_prod(losses,reduction_indices=[0]),1/m) # Geometric mean over the dimensions
		loss = mean# + (1-alpha)*geom_mean		
		
		self.grads = tf.gradients(loss,self.point)[0]
		self.grads = tf.reshape(self.grads,[m,1])
		
		
class State(object):
	def __init__(self, snf, state_ops, sess):

		self.point = snf.gen_points(1)
		
		self.update_grads(snf, state_ops, sess)
		
		if grad_noise > 0:
			self.grads += np.abs(self.grads)*grad_noise*np.random.random((m,1))
		
		
	def update_grads(self, snf, state_ops, sess):
		self.grads = sess.run([state_ops.grads],
								feed_dict={	state_ops.point: self.point,
											state_ops.means: snf.means,
											state_ops.variances: snf.variances,
											state_ops.weights: snf.weights})
		self.grads = self.grads[0]	

		if np.any(np.isnan(self.grads)):
			print np.concatenate([self.grads,snf.means,snf.variances,snf.weights,self.point],axis=1)
			raise ValueError		
			
			