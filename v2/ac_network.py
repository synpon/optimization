from __future__ import division
import random
import tensorflow as tf
import numpy as np

import rnn
import rnn_cell
from nn_utils import weight_matrix, bias_vector, fc_layer, fc_layer3, scale_grads, inv_scale_grads, scale_num
from constants import rnn_size, num_rnn_layers, m, rnn_type, grad_scaling_method, grad_scaling_factor, p


# Actor-Critic Network (policy and value network)
class A3CNet(object):

	# Create placeholder variables in order to calculate the loss
	def prepare_loss(self, entropy_beta):
	
		# Taken action (input for policy)
		self.a = tf.placeholder(tf.float32, [None,m,1], 'a')
	
		# Temporal difference (R-V) (input for policy)
		self.td = tf.placeholder(tf.float32, [None], 'td')
		
		# Entropy of the policy
		# Entropy encourages exploration, which it is positively correlated with. 
		# Therefore, higher entropy makes the loss function lower.
		entropy = -0.5*(tf.log(2*3.14*self.variance) + 1)

		# Policy loss (output)
		policy_loss = tf.nn.l2_loss(self.mean - self.a)*self.td - entropy*entropy_beta

		# R (input for value)
		self.r = tf.placeholder(tf.float32, [None], 'r')

		# Learning rate for critic is half of actor's, so multiply by 0.5
		value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

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
		

class A3CRNN(A3CNet):

	def __init__(self, num_trainable_vars):
		# Input
		self.grads = tf.placeholder(tf.float32, [None,None,1], 'grads')
		self.snf_loss = tf.placeholder(tf.float32, [None] , 'snf_loss')
		n_dims = tf.shape(self.grads)[1]
		
		grads = self.grads

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
			self.step_size = tf.placeholder(tf.int32, [1])
			self.step_size = tf.tile(self.step_size, tf.pack([n_dims])) # m acts as the batch size
			
			self.initial_rnn_state = tf.placeholder(tf.float32, [None,self.cell.state_size])
			
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
			self.rnn_state = rnn_state # [m, cell.state_size]
		
			# policy
			self.mean = fc_layer3(self.output, num_in=rnn_size, num_out=1, activation_fn=None)

			h = fc_layer3(self.output, num_in=rnn_size, num_out=1, activation_fn=tf.nn.relu) # softplus not used due to NaNs
			self.variance = tf.maximum(0.01,h) # protection against NaNs
			
			# value - linear output layer
			mean_output = tf.reduce_mean(self.output, reduction_indices=[1]) # [self.step_size[0], self.cell.state_size]
			
			### Needs more layers?
			snf_loss = tf.expand_dims(self.snf_loss, 1)
			mean_output_and_snf_loss = tf.concat(1, [mean_output, snf_loss])
			#mean_output_and_snf_loss = snf_loss ### works better?
			v_h = fc_layer(mean_output_and_snf_loss, num_in=rnn_size + 1, num_out=10, activation_fn=tf.nn.relu)
			v_h = fc_layer(v_h, num_in=10, num_out=10, activation_fn=tf.nn.relu)
			v = tf.contrib.layers.fully_connected(v_h, num_outputs=1, activation_fn=None)
			
		self.v = v
		
		if num_trainable_vars[0] == None:
			num_trainable_vars[0] = len(tf.trainable_variables())
		
		self.trainable_vars = tf.trainable_variables()[-num_trainable_vars[0]:]		
		self.reset_rnn_state()
		
		
	# Updates the RNN state
	def run_policy_and_value(self, sess, state, snf, state_ops):	
		snf_loss = [snf.calc_loss(state.point, state_ops, sess)] ### reuse from the previous reward if applicable
		
		feed_dict = {self.grads:state.grads, self.initial_rnn_state:self.rnn_state_out, self.step_size:np.ones([m]), self.snf_loss:snf_loss}
		[mean, variance, self.rnn_state_out, value] = sess.run([self.mean, self.variance, self.rnn_state, self.v], feed_dict=feed_dict)
		variance = np.maximum(variance,0.01)
		
		value = np.squeeze(value)
		return mean, variance, value

		
	# Updates the RNN state
	def run_policy(self, sess, state):		
		feed_dict = {self.grads:state.grads, self.initial_rnn_state:self.rnn_state_out, self.step_size:np.ones([m])}
		[mean, variance, self.rnn_state_out] = sess.run([self.mean, self.variance, self.rnn_state], feed_dict=feed_dict)
		variance = np.maximum(variance,0.01)
		return mean, variance
	
	
	# Does not update the RNN state
	def run_value(self, sess, state, snf, state_ops):
		snf_loss = [snf.calc_loss(state.point, state_ops, sess)] ### reuse from the previous reward if applicable	
		prev_rnn_state_out = self.rnn_state_out
		
		feed_dict = {self.grads:state.grads, self.initial_rnn_state:self.rnn_state_out, self.step_size:np.ones([m]), self.snf_loss:snf_loss}
		[value, self.rnn_state_out] = sess.run([self.v, self.rnn_state], feed_dict=feed_dict)		

	    # roll back RNN state
		self.rnn_state_out = prev_rnn_state_out ### necessary?
		
		value = np.squeeze(value)		
		return value # scalar
		
		
	def reset_rnn_state(self):
		self.rnn_state_out = np.zeros([m,self.cell.state_size])
		
		
# Feed-forward
class A3CFF(A3CNet):
	def __init__(self, num_trainable_vars):
	
		# Input
		# Third dimension is the number of features
		self.grads = tf.placeholder(tf.float32, [None,m,1], 'grads')
		self.update = tf.placeholder(tf.float32, [None,m,1], 'update') # Coordinate update
		self.rand = tf.placeholder_with_default(input=0.0, shape=[])

		# The scope allows these variables to be excluded from being reinitialized during the comparison phase
		with tf.variable_scope("a3c"):
			# fc_layer not used in order to extract W1 more easily
			self.W_m = weight_matrix(1,1) # Magnitude
			self.W_p = weight_matrix(1,1) # Probability of being positive
			
			# Greater than does not have a gradient - approximated with tanh
			self.W1 = self.W_m*(tf.nn.tanh(self.rand - self.W_p))
			
			self.b1 = bias_vector(1,1)
		
			self.W2 = weight_matrix(1,1)
			self.b2 = bias_vector(1,1)

			# policy
			self.mean = tf.mul(grads, self.W1) + self.b1
			self.variance = tf.maximum(0.01,tf.nn.relu(tf.mul(grads, self.W2) + self.b2)) # softplus causes NaNs in FF
			
			# value - linear output layer
			grads_and_update = tf.concat(2, [self.grads, self.update])

			# Twice as many inputs since grads and update are concatenated to make the input
			v_h = fc_layer(grads_and_update, num_in=2, num_out=10, activation_fn=tf.nn.relu)
			v = fc_layer(v_h, num_in=10, num_out=1, activation_fn=None)

		self.v = tf.reduce_mean(v) # Average over dimensions and convert to scalar
		
		if num_trainable_vars[0] == None:
			num_trainable_vars[0] = len(tf.trainable_variables())
		
		self.trainable_vars = tf.trainable_variables()[-num_trainable_vars[0]:]
		
	def run_policy(self, sess, grads):
		rand = random.uniform(-1,1)
		mean, variance = sess.run([self.mean,self.variance], feed_dict={self.grads:grads, self.rand:rand})
		variance = np.maximum(variance,0.01)
		return mean, variance

	def run_value(self, sess, grads, update):
		rand = random.uniform(-1,1)
		v_out = sess.run(self.v, feed_dict={self.grads:grads, self.update:update, self.rand:rand})
		return np.abs(v_out) # output is a scalar
		

		
		