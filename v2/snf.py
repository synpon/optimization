from __future__ import division

import tensorflow as tf
import numpy as np

from constants import k, m, var_size, grad_noise
from nn_utils import inv_scale_grads


class SNF(object):
	def __init__(self):
		self.hyperplanes = np.random.rand(m*m*k)
		self.hyperplanes = np.reshape(self.hyperplanes,[m,m,k])
		
		self.variances = np.random.rand(k)*var_size
		self.variances = np.reshape(self.variances,[k,1])
		
		self.weights = np.random.rand(k)
		self.weights = np.reshape(self.weights,[k,1])
	
	
	def calc_loss(self, point, state_ops, sess):
		loss = sess.run([state_ops.loss], 
						feed_dict={	state_ops.point: point, 
									state_ops.hyperplanes: self.hyperplanes, 
									state_ops.variances: self.variances, 
									state_ops.weights: self.weights})
		return loss[0][0]
		
		
	def calc_grads(self, point, state_ops, sess):
		grads = sess.run([state_ops.grads], 
						feed_dict={	state_ops.point: point, 
									state_ops.hyperplanes: self.hyperplanes, 
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
		#mean = inv_scale_grads(mean)
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
		#===# Graph to compute the loss and gradients for a single point #===#
		self.point = tf.placeholder(tf.float32, [m,1])
		self.variances = tf.placeholder(tf.float32, [k,1])
		self.weights = tf.placeholder(tf.float32, [k,1])
		self.hyperplanes = tf.placeholder(tf.float32, [m,m,k]) # Points which define the hyperplanes
		
		hyperplanes = tf.reshape(self.hyperplanes, [k,m,m])
		hp_inv = tf.batch_matrix_inverse(hyperplanes) # [k,m,m]
		x = tf.ones((k,m,1))
		a = tf.batch_matmul(hp_inv, x) # [k,m,1]
		point = tf.transpose(self.point,[1,0]) # [1,m]
		point = tf.reshape(point,[1,1,m])
		point = tf.tile(point,[k,1,1]) # [k,1,m]
		D = tf.batch_matmul(point,a) - 1 # [k,1,1]
		D = tf.reshape(D,[k])
		norm = tf.sqrt(tf.reduce_sum(tf.square(a),reduction_indices=[1])) # [k]
		D /= norm # [k]
		losses = tf.abs(D) # [k]
		
		losses = tf.square(losses) # [k]
		losses /= -2*self.variances # [k]
		losses = -tf.exp(losses) # [k]
		var_coeffs = 1/tf.sqrt(2*self.variances*3.14) # [k]
		losses *= var_coeffs # [k]
		losses *= self.weights # element-wise [k]
		
		self.loss = tf.reduce_mean(losses,reduction_indices=[0]) # Average over the hyperplanes
		
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