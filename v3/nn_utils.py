from __future__ import division

import numpy as np
import tensorflow as tf

from constants import grad_scaling_method


def xavier_initializer(dims):
	d = np.sqrt(6.0)/np.sqrt(sum(dims))
	return tf.random_uniform_initializer(minval=-d, maxval=d)


def weight_matrix(num_in, num_out):
	with tf.variable_scope("weight"):
		# Uses 1 rather than 6 as the fc_layer3 does not have an activation function
		d = np.sqrt(1.0)/np.sqrt(num_in+num_out)
		return tf.Variable(tf.random_uniform(shape=[num_in, num_out], minval=-d, maxval=d))

		
def bias_vector(num_in, num_out):
	with tf.variable_scope("bias"):
		d = np.sqrt(1.0)/np.sqrt(num_out)
		return tf.Variable(tf.random_uniform(shape=[num_out], minval=-d, maxval=d))

		
def fc_layer3(layer_in, num_in, num_out, activation_fn):
	# [batch_size x m x num_in] x [num_in x num_out] = [batch_size x m x num_out]
	batch_size = tf.shape(layer_in)[0]
	
	W = weight_matrix(num_in, num_out)
	W = tf.reshape(W,[1,num_in,num_out])
	W = tf.tile(W,tf.pack([batch_size,1,1]))
	
	#b = bias_vector(num_in, num_out)
	out = tf.batch_matmul(layer_in, W)# + b
	if activation_fn != None:
		out = activation_fn(out)
		
	return out
	
	
def fc_layer(layer_in, num_in, num_out, activation_fn):
	batch_size = tf.shape(layer_in)[0]
	
	W = weight_matrix(num_in, num_out)
	b = bias_vector(num_in, num_out)
	out = tf.matmul(layer_in, W) + b
	
	if activation_fn != None:
		out = activation_fn(out)
		
	return out
		

# Doubles the number of features
def scale_grads(x):
	if grad_scaling_method == 'full':
		# Operations are element-wise
		return tf.sign(x)*tf.log(tf.maximum(tf.abs(x + tf.sign(x)),1e-2))
	return x
	

def inv_scale_grads(x):
	if grad_scaling_method == 'full':	
		# Operations are element-wise
		mask = tf.to_float(tf.greater(x,0.0))
		inv_mask = 1 - mask
		
		x_cond1 = tf.exp(x)-1
		x_cond2 = tf.exp(-x)*(tf.exp(x)-1)

		return x_cond1*mask + x_cond2*inv_mask		
	return x	
	
	
def np_inv_scale_grads(x):
	if grad_scaling_method == 'full':	
		# Operations are element-wise
		mask = np.greater(x,0.0)
		inv_mask = 1 - mask
		
		x_cond1 = np.exp(x)-1
		x_cond2 = np.exp(-x)*(np.exp(x)-1)

		return x_cond1*mask + x_cond2*inv_mask	
	return x
	

def tf_print(x):
	x = tf.Print(x,[x])
	return
	
	