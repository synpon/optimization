from __future__ import division

import tensorflow as tf
import numpy as np

import rnn
import rnn_cell
from nn_utils import weight_matrix, bias_vector, fc_layer, fc_layer3, scale_grads, np_inv_scale_grads
from constants import rnn_size, num_rnn_layers, k, m, rnn_type, grad_scaling_method
from snf import calc_snf_loss_tf, calc_grads_tf

class Optimizer(object):

	def __init__(self):
		# Input
		self.point = tf.placeholder(tf.float32, [m,1], 'point') # Used to calculate loss only
		self.snf_loss = tf.placeholder(tf.float32, [1], 'snf_loss')
		self.variances = tf.placeholder(tf.float32, [k,1], 'variances')
		self.weights = tf.placeholder(tf.float32, [k,1], 'weights')
		self.hyperplanes = tf.placeholder(tf.float32, [m,m,k], 'hyperplanes') # Points which define the hyperplanes
		self.input_grads = tf.placeholder(tf.float32, [None,None,1], 'grads')
		
		grads = self.input_grads
		n_dims = tf.shape(grads)[1]	

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
			
			# placeholder for RNN unrolling time step size.
			self.step_size = tf.placeholder(tf.int32, [1], 'step_size')
			self.step_size = tf.tile(self.step_size, tf.pack([n_dims])) # m acts as the batch size
			
			self.initial_rnn_state = tf.placeholder(tf.float32, [None,self.cell.state_size], 'rnn_state')
			
			grads = tf.transpose(grads, perm=[1,0,2])

			# Unrolling LSTM up to LOCAL_T_MAX time steps.
			# When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
			# Unrolling step size is applied via self.step_size placeholder.
			# When forward propagating, step_size is 1.
			output, rnn_state = rnn.dynamic_rnn(self.cell,
									grads,
									initial_state = self.initial_rnn_state,
									sequence_length = self.step_size,
									time_major = False)#,
									#scope = scope)			
			
			self.output = tf.reshape(output,tf.pack([self.step_size[0],n_dims,rnn_size]))		
			self.rnn_state = rnn_state # [m, rnn_size*num_rnn_layers]
		
			update = fc_layer3(self.output, num_in=rnn_size, num_out=1, activation_fn=None)
			self.update = tf.reshape(update, tf.pack([n_dims,1]))
			self.new_point = self.point + self.update
			
			self.new_snf_loss = calc_snf_loss_tf(self.new_point, self.hyperplanes, self.variances, self.weights) ### Add noise?
			
			self.loss = tf.sign(self.snf_loss - self.new_snf_loss)
			
			self.grads = calc_grads_tf(self.loss, self.new_point)
			
			opt = tf.train.AdamOptimizer()
			self.train_step = opt.minimize(self.loss)
			
		self.trainable_vars = tf.trainable_variables()

		
	# Updates the RNN state
	def run_policy(self, sess, state):		
		feed_dict = {self.grads:state.grads, self.initial_rnn_state:self.rnn_state_out, self.step_size:np.ones([m])}
		[update, self.rnn_state_out] = sess.run([self.update, self.rnn_state], feed_dict=feed_dict)
		return update
	

	def sync_from(self, src_network, name=None):
		src_vars = src_network.trainable_vars
		dst_vars = self.trainable_vars

		sync_ops = []
		with tf.op_scope([], name, "A3CNet") as name:
			for(src_var, dst_var) in zip(src_vars, dst_vars):
				sync_op = tf.assign(dst_var, src_var)
				sync_ops.append(sync_op)

			return tf.group(*sync_ops, name=name)
			

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
		
