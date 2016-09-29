from __future__ import division

import tensorflow as tf
import numpy as np

from constants import k, m, var_size, rnn_size, num_rnn_layers, rnn_type
from nn_utils import scale_grads, np_inv_scale_grads


class SNF(object):
	def __init__(self):
		self.hyperplanes = np.random.rand(m*m*k)
		self.hyperplanes = np.reshape(self.hyperplanes,[m,m,k])
		
		self.variances = np.random.rand(k)*var_size
		self.variances = np.reshape(self.variances,[k,1])
		
		self.weights = np.random.rand(k)
		self.weights = np.reshape(self.weights,[k,1])
	
		
	def calc_loss_and_grads(self, point, state_ops, sess):
		loss, grads = sess.run([state_ops.loss, state_ops.grads], 
						feed_dict={	state_ops.point: point, 
									state_ops.hyperplanes: self.hyperplanes, 
									state_ops.variances: self.variances, 
									state_ops.weights: self.weights})
		return loss, grads
		
		
	def gen_points(self,num_points):
		points = np.random.rand(m*num_points)
		points = np.reshape(points,[m,num_points])
		return points
	
		
def calc_snf_loss_tf(point,hyperplanes,variances,weights):
	#variances = tf.maximum(variances,1e-6) # Avoid NaN errors
	hyperplanes = tf.reshape(hyperplanes, [k,m,m])
	hp_inv = tf.batch_matrix_inverse(hyperplanes) # [k,m,m]
	x = tf.ones((k,m,1))
	a = tf.batch_matmul(hp_inv, x) # [k,m,1]
	point = tf.reshape(point,[1,1,m])
	point = tf.tile(point,[k,1,1]) # [k,1,m]
	D = tf.batch_matmul(point,a) - 1 # [k,1,1]
	D = tf.reshape(D,[k])
	norm = tf.sqrt(tf.reduce_sum(tf.square(a),reduction_indices=[1])) # [k]
	D /= norm#tf.maximum(norm,1e-6) # [k]
	losses = tf.abs(D) # [k]
	
	losses = tf.square(losses) # [k]
	losses /= -2*variances # [k]
	losses = -tf.exp(losses) # [k]
	var_coeffs = 1/tf.sqrt(2*variances*3.14) # [k]
	losses *= var_coeffs # [k]
	losses *= weights # element-wise [k]
	
	loss = tf.reduce_mean(losses) # Average over the hyperplanes 
	return loss
	
	
def calc_grads_tf(loss,point):
	grads = tf.gradients(loss,point)[0]
	grads = tf.reshape(grads,[1,m,1])
	grads = scale_grads(grads)
	return grads
		
		
class StateOps:
	"""
	The definition of this graph could be included within State but making it separate means 
	the graph only has to be created once, as opposed to once every time an instance of State 
	is created. This results in a 100-1000x speed-up.
	"""

	def __init__(self): 
		#===# Graph to compute the loss and gradients for a single point #===#
		self.point = tf.placeholder(tf.float32, [m,1])
		self.variances = tf.placeholder(tf.float32, [k,1])
		self.weights = tf.placeholder(tf.float32, [k,1])
		self.hyperplanes = tf.placeholder(tf.float32, [m,m,k]) # Points which define the hyperplanes
		
		self.loss = calc_snf_loss_tf(self.point,self.hyperplanes,self.variances,self.weights)
		self.grads = calc_grads_tf(self.loss,self.point)

		
class State(object):

	def __init__(self, snf, state_ops, sess):
		self.snf = snf
		self.point = snf.gen_points(1)
		self.counter = 1
		self.loss_and_grads(snf, state_ops, sess) # calc and set
		
		if rnn_type == 'lstm':
			self.rnn_state = np.zeros([m,2*rnn_size*num_rnn_layers])
		else:
			self.rnn_state = np.zeros([m,rnn_size*num_rnn_layers])
		
		
	def loss_and_grads(self, snf, state_ops, sess):
		[self.loss,self.grads] = snf.calc_loss_and_grads(self.point, state_ops, sess)
		
		