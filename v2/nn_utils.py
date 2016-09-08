from __future__ import division

import numpy as np
import tensorflow as tf

from constants import grad_scaling_method, grad_scaling_factor, p


def weight_matrix(num_in, num_out):
	with tf.variable_scope("weight"):
		d = 1.0/np.sqrt(num_in)
		return tf.Variable(tf.random_uniform(shape=[num_in, num_out], minval=-d, maxval=d))

		
def bias_vector(num_in, num_out):
	with tf.variable_scope("bias"):
		d = 1.0/np.sqrt(num_in)
		return tf.Variable(tf.random_uniform(shape=[num_out], minval=-d, maxval=d))

		
def fc_layer(layer_in, num_in, num_out, activation_fn):
	# [batch_size x m x num_in] x [num_in x num_out] = [batch_size x m x num_out]
	batch_size = tf.shape(layer_in)[0]
	
	W = weight_matrix(num_in, num_out)
	W = tf.reshape(W,[1,num_in,num_out])
	W = tf.tile(W,tf.pack([batch_size,1,1]))
	
	b = bias_vector(num_in, num_out)
	out = tf.batch_matmul(layer_in, W) + b
	if activation_fn != None:
		out = activation_fn(out)
		
	return out
		
		
def scale_grads(input):
	if grad_scaling_method == 'scalar':
		return input*tf.constant(grad_scaling_factor)	
		
	elif grad_scaling_method == 'full':	
		p_ = tf.constant(p)
		grad_threshold = tf.exp(-p_)
	
		# Operations are element-wise
		mask = tf.greater(tf.abs(input),grad_threshold)
		mask = tf.to_float(mask) # Convert from boolean
		inv_mask = 1 - mask
		
		x1_cond1 = tf.log(tf.abs(input))/p_
		x2_cond1 = tf.sign(input)
		x1_cond2 = -tf.ones(tf.shape(input))
		x2_cond2 = tf.exp(p_)*input
		
		x1 = x1_cond1*mask + x1_cond2*inv_mask
		x2 = x2_cond1*mask + x2_cond2*inv_mask
		
		return tf.concat(2,[x1,x2])	
	return input
	
	
def inv_scale_grads(input): ### Doesn't work
	if grad_scaling_method == 'scalar':
		return input/tf.constant(grad_scaling_factor)	
		
	elif grad_scaling_method == 'full':	
		p_ = tf.constant(p)
	
		# Operations are element-wise
		a,b = tf.split(2,2,input)
		mask = tf.equal(tf.abs(b),1.0)
		mask = tf.to_float(mask) # Convert from boolean
		inv_mask = 1 - mask
		
		x_cond1 = tf.sign(b)*tf.exp(a*p_)
		x_cond2 = b/tf.exp(p_)
		
		return x_cond1*mask + x_cond2*inv_mask		
	return input