from __future__ import division

import tensorflow as tf
import numpy as np

import rnn
import rnn_cell
from nn_utils import weight_matrix, bias_vector, fc_layer, fc_layer3, inv_scale_grads
from constants import rnn_size, num_rnn_layers, k, m, rnn_type, grad_scaling_method, \
		discount_rate, episode_length, loss_noise, seq_length
from snf import calc_snf_loss_tf, calc_grads_tf

class Optimizer(object):

	def __init__(self):
		# Input
		self.points = tf.placeholder(tf.float32, [seq_length,m,1], 'points') # Used to calculate loss only
		self.snf_losses = tf.placeholder(tf.float32, [seq_length], 'snf_losses')
		self.input_grads = tf.placeholder(tf.float32, [seq_length,None,1], 'input_grads')
		self.counters = tf.placeholder(tf.float32, [seq_length], name='counters')
		self.variances = tf.placeholder(tf.float32, [k,1], 'variances')
		self.weights = tf.placeholder(tf.float32, [k,1], 'weights')
		self.hyperplanes = tf.placeholder(tf.float32, [m,m,k], 'hyperplanes') # Points which define the hyperplanes
		self.step_size = tf.placeholder(tf.int32, [1], 'step_size') ### correct? # placeholder for RNN unrolling time step size.
		### Have seq_length as an argument for compatibility with compare
		
		# initial_rnn_state is given during evaluation but not during training
		self.initial_rnn_state = tf.placeholder_with_default(input=tf.zeros([m, num_rnn_layers*rnn_size]), shape=[None, num_rnn_layers*rnn_size])
		
		n_dims = tf.shape(self.input_grads)[2]
		
		points = tf.split(0, seq_length, self.points)
		snf_losses = tf.split(0, seq_length, self.snf_losses)
		input_grads = tf.split(0, seq_length, self.input_grads)
		counters = tf.split(0, seq_length, self.counters)

		# The scope allows these variables to be excluded from being reinitialized during the comparison phase
		with tf.variable_scope("a3c"):
			if rnn_type == 'rnn':
				cell = rnn_cell.BasicRNNCell(rnn_size,activation=tf.identity)
			elif rnn_type == 'gru':
				cell = rnn_cell.GRUCell(rnn_size)
			elif rnn_type == 'lstm':
				cell = rnn_cell.BasicLSTMCell(rnn_size)
				
			self.cell = rnn_cell.MultiRNNCell([cell] * num_rnn_layers)

			if rnn_type == 'lstm':
				raise NotImplementedError
		
			self.step_size = tf.tile(self.step_size, tf.pack([n_dims])) # m acts as the batch size
			
			rnn_state = self.initial_rnn_state
			
			grads = []
			for g in input_grads:
				g = tf.squeeze(g) # Remove the dimension of size 1
				g.set_shape([None,1]) # Necessary for running dynamic_rnn
				grads.append(g)
			
			# Unrolling step size is applied via self.step_size placeholder.
			# When forward propagating, step_size is 1.
			outputs, rnn_state = rnn.dynamic_rnn(self.cell,
									grads,
									initial_state = rnn_state,
									sequence_length = self.step_size,
									time_major = False)		
			
			self.total_loss = 0		
			outputs = tf.split(0, seq_length, outputs)
			
			for point,snf_loss,output,counter in zip(points,snf_losses,outputs,counters):
				output = tf.reshape(output,tf.pack([self.step_size[0],n_dims,rnn_size]))
			
				update = fc_layer3(output, num_in=rnn_size, num_out=1, activation_fn=None)
				update = tf.reshape(update, tf.pack([n_dims,1]))
				update = inv_scale_grads(update)
				
				new_point = tf.squeeze(point) + update		
				new_snf_loss = calc_snf_loss_tf(new_point, self.hyperplanes, self.variances, self.weights)
				
				# Add loss noise - reduce__mean is only to flatten
				new_snf_loss += tf.reduce_mean(tf.abs(new_snf_loss)*loss_noise*tf.random_uniform([1], minval=-1.0, maxval=1.0))
				
				#grads = calc_grads_tf(loss, new_point) ###
				
				#loss = self.snf_loss - self.new_snf_loss
				loss = tf.sign(tf.squeeze(snf_loss) - new_snf_loss)
				
				# Weight the loss by its position in the optimisation process
				tmp = tf.pow(discount_rate, episode_length - tf.squeeze(counter))
				w = (tmp*(1 - discount_rate))/tf.maximum(1 - tmp,1e-6) ### ordinarily causes a NaN error around iteration 3000
				self.total_loss += loss * w
			
			opt = tf.train.AdamOptimizer()
			self.train_step = opt.minimize(self.total_loss)
			

	# Update the parameters of another network (eg an MLP)
	def update_params(self, vars, h):
		total = 0
		ret = []

		for i,v in enumerate(vars):
			size = np.prod(list(v.get_shape()))
			size = tf.to_int32(size)
			var_grads = tf.slice(h,begin=[total,0],size=[size,-1])
			var_grads = tf.reshape(var_grads,v.get_shape())
			
			#if not grad_clip_value is None:
			#	var_grads = tf.clip_by_value(var_grads, -grad_clip_value, grad_clip_value)
			
			ret.append(v.assign_add(var_grads))
			size += total		
		return tf.group(*ret)
		
