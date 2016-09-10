from __future__ import division

import tensorflow as tf
import numpy as np

from constants import m, var_size, grad_noise
from ac_network import inv_scale_grads


class SNF(object):
	def __init__(self):
		self.means = np.random.rand(m)
		self.means = np.reshape(self.means,[m,1])
		
		self.variances = np.random.rand(m)*var_size
		self.variances = np.reshape(self.variances,[m,1])
		
		self.weights = np.random.rand(m)
		self.weights = np.reshape(self.weights,[m,1])
	
	
	def calc_loss(self, point, state_ops, sess):
		loss = sess.run([state_ops.loss], 
						feed_dict={	state_ops.point: point, 
									state_ops.means: self.means, 
									state_ops.variances: self.variances, 
									state_ops.weights: self.weights})
		return loss[0][0]
		
		
	def calc_grads(self, point, state_ops, sess):
		grads = sess.run([state_ops.grads], 
						feed_dict={	state_ops.point: point, 
									state_ops.means: self.means, 
									state_ops.variances: self.variances, 
									state_ops.weights: self.weights})
		return np.reshape(grads[0],[1,m,1])
		
		
	def gen_points(self,num_points):
		points = np.random.rand(m*num_points)
		points = np.reshape(points,[m,num_points])
		return points
		
		
	def choose_action(self,mean,variance):
		for i,v in enumerate(variance):
			mean[i] += np.random.normal(0,v)*mean[i]
		mean = inv_scale_grads(mean)
		mean = np.reshape(mean,[1,m,1])
		return mean
	
	
	def act(self, state, action, state_ops, sess):
		action = np.reshape(action,[m,1])
		state.point += action		
		loss = self.calc_loss(state.point, state_ops, sess)
		reward = -loss
		state.calc_and_set_grads(self, state_ops, sess)
		return reward, state
		
		
class StateOps:
	# StateOps is needed as a separate class because TensorFlow
	# objects are not thread-safe
	def __init__(self):
		#===# Graph to compute the loss and gradients #===#
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
		
		self.loss = tf.reduce_mean(losses,reduction_indices=[0]) # Mean over the dimensions
		
		self.grads = tf.gradients(self.loss,self.point)[0]
		self.grads = tf.reshape(self.grads,[1,m,1])
		
		
class State(object):
	def __init__(self, snf, state_ops, sess):
		self.point = snf.gen_points(1)
		self.calc_and_set_grads(snf, state_ops, sess)
		
		
	def calc_and_set_grads(self, snf, state_ops, sess):
		self.grads = snf.calc_grads(self.point, state_ops, sess) # [1,m,1]
		if grad_noise > 0:
			self.grads += np.abs(self.grads)*grad_noise*np.random.random((1,m,1))
