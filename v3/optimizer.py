from __future__ import division

import tensorflow as tf
import numpy as np

import rnn
import rnn_cell
from nn_utils import weight_matrix, bias_vector, fc_layer, fc_layer3, inv_scale_grads
from constants import rnn_size, num_rnn_layers, k, m, rnn_type, grad_scaling_method, \
		discount_rate, episode_length, loss_noise
from snf import calc_snf_loss_tf, calc_grads_tf
from nn_utils import tf_print


class Optimizer(object):

	def __init__(self, seq_length):
		# Input
		self.points = tf.placeholder(tf.float32, [seq_length,m,1], 'points') # Used to calculate loss only
		self.snf_losses = tf.placeholder(tf.float32, [seq_length], 'snf_losses')
		self.input_grads = tf.placeholder(tf.float32, [seq_length,None,1], 'input_grads')
		self.counters = tf.placeholder(tf.float32, [seq_length], name='counters')
		self.variances = tf.placeholder(tf.float32, [k,1], 'variances')
		self.weights = tf.placeholder(tf.float32, [k,1], 'weights')
		self.hyperplanes = tf.placeholder(tf.float32, [m,m,k], 'hyperplanes') # Points which define the hyperplanes
		
		# initial_rnn_state is given during evaluation but not during training
		self.initial_rnn_state = tf.placeholder_with_default(input=tf.zeros([m, num_rnn_layers*rnn_size]), shape=[None, num_rnn_layers*rnn_size])
		
		n_dims = tf.shape(self.input_grads)[1]
		
		points = tf.split(0, seq_length, self.points)
		snf_losses = tf.split(0, seq_length, self.snf_losses)
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
			
			outputs, rnn_state = rnn.dynamic_rnn(self.cell,
									self.input_grads,
									initial_state = self.initial_rnn_state,
									sequence_length = seq_length*tf.ones(tf.pack([n_dims])),
									time_major = True)
			
			self.rnn_state_output = rnn_state
			
			self.total_loss = 0		
			outputs = tf.split(0, seq_length, outputs)
			
			snf_losses_output = []
			points_output = []
			grads_output = []
			
			for point,snf_loss,output,counter in zip(points,snf_losses,outputs,counters):
				output = tf.reshape(output,tf.pack([1,n_dims,rnn_size])) ### check
			
				update = fc_layer3(output, num_in=rnn_size, num_out=1, activation_fn=None)
				update = tf.reshape(update, tf.pack([n_dims,1]))
				update = inv_scale_grads(update)
				
				new_point = update + tf.squeeze(point, squeeze_dims=[0]) ### check squeeze_dims value
				
				points_output.append(new_point)	
				
				new_snf_loss = calc_snf_loss_tf(new_point, self.hyperplanes, self.variances, self.weights)
				
				# Add loss noise - reduce__mean is only to flatten
				new_snf_loss += tf.reduce_mean(tf.abs(new_snf_loss)*loss_noise*tf.random_uniform([1], minval=-1.0, maxval=1.0))
				snf_losses_output.append(new_snf_loss) ### without noise?
				
				g = calc_grads_tf(new_snf_loss, new_point) ### check
				grads_output.append(g)
				
				#loss = self.snf_loss - self.new_snf_loss
				loss = tf.sign(tf.squeeze(snf_loss) - new_snf_loss)
				
				# Weight the loss by its position in the optimisation process
				tmp = tf.pow(discount_rate, episode_length - tf.squeeze(counter))
				w = (tmp*(1 - discount_rate))/tf.maximum(1 - tmp,1e-6) ### ordinarily causes a NaN error around iteration 3000
				self.total_loss += loss * w
				
			# Cannot return lists as they are
			self.snf_losses_output = snf_losses_output[0] ### Indexing is admissible in all contexts?
			self.points_output = points_output[0]
			self.grads_output = grads_output[0]
			
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
		
