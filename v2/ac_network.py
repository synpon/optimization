from __future__ import division
import random
import tensorflow as tf
import numpy as np

import rnn
import rnn_cell
from nn_utils import weight_matrix, bias_vector, fc_layer, fc_layer3, scale_grads, np_inv_scale_grads
from constants import rnn_size, num_rnn_layers, m, rnn_type, grad_scaling_method


# Actor-Critic Network (policy and value network)
class A3CRNN(object):

	def __init__(self, num_trainable_vars):
		# Input
		self.grads = tf.placeholder(tf.float32, [None,None,1], 'grads')
		n_dims = tf.shape(self.grads)[1] # Dimensionality of the optimizee

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
			self.step_size = tf.placeholder(tf.int32, [None], 'step_size')
			#self.step_size = tf.tile(self.step_size, tf.pack([n_dims])) # m acts as the batch size
			
			self.initial_rnn_state = tf.placeholder(tf.float32, [None,self.cell.state_size], 'grad_rnn_state')
			
			grads = tf.transpose(self.grads, perm=[1,0,2]) ### correct?

			# Unrolling the RNN up to local_t_max time steps.
			# When episode terminates unrolling time steps becomes less than local_time_step.
			# Unrolling step size is applied via self.step_size placeholder.
			# When forward propagating, the step_size is 1.
			output, rnn_state = rnn.dynamic_rnn(self.cell, ### change to using the default rnn code
									grads,
									initial_state = self.initial_rnn_state,
									sequence_length = self.step_size, ### necessary?
									time_major = False)
			
			self.rnn_state = rnn_state # [m, cell.state_size]
		
			# policy - linear activation
			# fc_layer3 is needed for batch processing
			self.mean = fc_layer3(output, num_in=rnn_size, num_out=1, activation_fn=None)

			# The activation function ensures the variance is not negative
			self.variance = fc_layer3(output, num_in=rnn_size, num_out=1, activation_fn=tf.nn.relu) # softplus not used due to NaNs
			#tf.maximum(0.01,h) # protection against NaNs
			
			#===# Value network #===#		
			val_rnn_size = 4
			value_cell = rnn_cell.GRUCell(val_rnn_size) ### Use an LSTM?
			
			self.snf_loss = tf.placeholder(tf.float32, [None,1,1] , 'snf_loss') ### check dimensions
			self.initial_val_rnn_state = tf.placeholder(tf.float32, [None,value_cell.state_size], 'val_rnn_state')
			self.val_step_size = tf.placeholder(tf.int32, [None], 'step_size')
			
			snf_loss = tf.transpose(self.snf_loss, perm=[1,0,2])

			val_output, val_rnn_state = rnn.dynamic_rnn(value_cell,
												snf_loss,
												initial_state = self.initial_val_rnn_state,
												sequence_length = self.val_step_size, ### necessary?
												time_major = False)
											
			val_output = tf.reshape(val_output,[-1,val_rnn_size])
			self.val_rnn_state = val_rnn_state # [1, cell.state_size]

			v = fc_layer(val_output, num_in=val_rnn_size, num_out=1, activation_fn=None) ### Must be a scalar
			v = tf.squeeze(v)
			
		self.v = v
		
		if num_trainable_vars[0] == None:
			num_trainable_vars[0] = len(tf.trainable_variables())
		
		self.trainable_vars = tf.trainable_variables()[-num_trainable_vars[0]:]		
		self.reset_rnn_state()

		
	# Create placeholder variables in order to calculate the loss
	def prepare_loss(self, entropy_beta):
	
		# Taken action (input for policy)
		self.a = tf.placeholder(tf.float32, [None,m,1], 'a')
		a = tf.transpose(self.a, perm=[1,0,2])
	
		# Temporal difference (R-V) (input for policy)
		self.td = tf.placeholder(tf.float32, [None], 'td')
		
		# Entropy of the policy
		# Entropy encourages exploration, which it is positively correlated with. 
		# Therefore, higher entropy makes the loss function lower.
		entropy = -0.5*(tf.log(2*3.14*self.variance) + 1)

		# Policy loss (output)
		policy_loss = tf.nn.l2_loss(self.mean - a)*self.td - entropy*entropy_beta

		# R (input for value)
		self.R = tf.placeholder(tf.float32, [None], 'R')

		# Learning rate for critic is half of actor's, so multiply by 0.5
		value_loss = 0.5 * tf.nn.l2_loss(self.R - self.v)

		self.total_loss = policy_loss + value_loss

		
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
		
	# Updates the RNN state
	def run_policy_and_value(self, sess, state, snf, state_ops):	
		snf_loss = [snf.calc_loss(state.point, state_ops, sess)] ### reuse from the previous reward if possible
		snf_loss = np.reshape(snf_loss, [1,1,1])
		
		feed_dict = {self.grads:state.grads, 
						self.initial_rnn_state: self.rnn_state_out, 
						self.step_size: np.ones([m]),
						self.val_step_size: np.ones([1]),
						self.snf_loss: snf_loss,
						self.initial_val_rnn_state: self.val_rnn_state}
		[mean, variance, self.rnn_state_out, value] = sess.run([self.mean, self.variance, self.rnn_state, self.v], feed_dict=feed_dict)
		variance = np.maximum(variance,0.01)
		
		value = np.squeeze(value)
		return mean, variance, value

		
	# Updates the RNN state
	def run_policy(self, sess, state):
		feed_dict = {self.grads:state.grads, 
						self.initial_rnn_state:self.rnn_state_out, 
						self.step_size:np.ones([m])}
		[mean, variance, self.rnn_state_out] = sess.run([self.mean, self.variance, self.rnn_state], feed_dict=feed_dict)
		variance = np.maximum(variance,0.01)
		return mean, variance
	
	
	# Does not update the RNN state
	def run_value(self, sess, state, snf, state_ops):
		snf_loss = [snf.calc_loss(state.point, state_ops, sess)] ### reuse from the previous reward if possible	
		snf_loss = np.reshape(snf_loss, [1,1,1])
		
		feed_dict = {self.grads: state.grads, 
						self.initial_rnn_state: self.rnn_state_out,
						self.initial_val_rnn_state: self.val_rnn_state,
						self.step_size: np.ones([m]), 
						self.val_step_size: np.ones([1]),
						self.snf_loss: snf_loss}
						
		value = sess.run(self.v, feed_dict=feed_dict) # scalar
		return value 
		
		
	def reset_rnn_state(self): ### confusion with similar state ops in snf - put in separate classes?
		self.rnn_state_out = np.zeros([m,self.cell.state_size])
		self.val_rnn_state = np.zeros([1,4*num_rnn_layers])
		