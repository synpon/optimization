import tensorflow as tf
import numpy as np
import rnn_cell

from constants import rnn_size, num_rnn_layers, num_steps, m, rnn_type


# Actor-Critic Network (policy and value network)
class A3CNet(object):

	# Create placeholder variables in order to calculate the loss
	def prepare_loss(self, entropy_beta): ###
	
		# Taken action (input for policy)
		self.a = tf.placeholder(tf.float32, [1,m,1], 'a')
	
		# Temporal difference (R-V) (input for policy)
		self.td = tf.placeholder(tf.float32, [1], 'td')
		
		# Entropy of the policy
		entropy = -0.5*tf.log(2*3.14*self.variance) + 1 ### Treat dimensions of variance differently?

		# Policy loss (output)
		# Minus because this is for gradient ascent
		# Overlap between the distributions
		policy_loss = -(tf.nn.l2_loss(self.mean - self.a) * self.td + entropy*entropy_beta)

		# R (input for value)
		self.r = tf.placeholder(tf.float32, [1], 'r')

		# Learning rate for critic is half of actor's, so multiply by 0.5
		value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

		self.total_loss = policy_loss + value_loss
		

	def sync_from(self, src_network, name=None):
		src_vars = src_network.trainable_vars ### built-in function to do this?
		dst_vars = self.trainable_vars

		sync_ops = []
		with tf.op_scope([], name, "A3CNet") as name:
			for(src_var, dst_var) in zip(src_vars, dst_vars):
				sync_op = tf.assign(dst_var, src_var)
				sync_ops.append(sync_op)

			return tf.group(*sync_ops, name=name)

	def weight_matrix(self, n_in, n_out):
		with tf.variable_scope("weight"):
			return tf.Variable(tf.truncated_normal(stddev=0.1, shape=[1, n_in, n_out]))
	
	def bias_vector(self,n_out):
		with tf.variable_scope("bias"):
			return tf.Variable(tf.constant(0.1, shape=[n_out]))

	def _debug_save_sub(self, sess, prefix, var, name):
		var_val = var.eval(sess)
		var_val = np.reshape(var_val, (1, np.product(var_val.shape)))				 
		np.savetxt('./' + prefix + '_' + name + '.csv', var_val, delimiter=',')
	

class A3CRNN(A3CNet):
	# For an RNN, input is usually of shape [batch_size,num_steps]
	# Here they are both 1, as is the case for sampling in a generative RNN
	def __init__(self, num_trainable_vars):		

		# Input (not the cell state)
		batch_size = 1
		self.state = tf.placeholder(tf.float32, [batch_size,m,1])

		self.W1 = self.weight_matrix(rnn_size,1)
		self.W1 = tf.tile(self.W1,(batch_size,1,1))
		self.b1 = self.bias_vector(1)
		
		self.W2 = self.weight_matrix(rnn_size,1)
		self.W2 = tf.tile(self.W2,(batch_size,1,1))
		self.b2 = self.bias_vector(1)
		
		# Weights for value output layer
		self.W3 = self.weight_matrix(rnn_size,1)
		self.W3 = tf.tile(self.W3,(batch_size,1,1))
		self.b3 = self.bias_vector(1)
		
		#cell = rnn_cell.BasicLSTMCell(rnn_size)
		cell = rnn_cell.BasicRNNCell(rnn_size)

		self.rnn_state = tf.zeros([1,m,rnn_size])
		
		if rnn_type in ['rnn','gru']:
			self.rnn_state = self.state #[self.state,self.state]
		elif rnn_type in ['lstm']:
			raise NotImplementedError
		
		output, rnn_state_out = cell(self.state, self.rnn_state)
		self.rnn_state_out = rnn_state_out
	
		# policy
		self.mean = tf.batch_matmul(output, self.W1) + self.b1
		self.variance = tf.nn.softplus(tf.batch_matmul(output, self.W2) + self.b2)
		
		# value - linear output layer
		self.v = tf.batch_matmul(output, self.W3) + self.b3
		
		if num_trainable_vars[0] == None:
			num_trainable_vars[0] = len(tf.trainable_variables())
		
		self.trainable_vars = tf.trainable_variables()[-num_trainable_vars[0]:]
			
	def run_policy(self, sess, state, update_rnn_state):
		mean, variance, rnn_state = sess.run([self.mean,self.variance, self.rnn_state_out], feed_dict={self.state:state})
		if update_rnn_state:
			self.rnn_state = rnn_state
		return mean, variance
		
	def run_value(self, sess, state, update_rnn_state):
		[v_out, rnn_state] = sess.run([self.v, self.rnn_state_out], feed_dict = {self.state: state})
		if update_rnn_state:
			self.rnn_state = rnn_state	
		return v_out[0][0]
		
	def reset_state(self, batch_size, num_params):
		self.rnn_state = tf.zeros([batch_size,num_params,rnn_size])
		self.rnn_state = [self.state,self.state]
		
		
# Feed-forward
class A3CFF(A3CNet):
	def __init__(self, num_trainable_vars):
	
		# Input
		batch_size = 1
		self.state = tf.placeholder(tf.float32, [batch_size,m,1])

		with tf.variable_scope("A3CNet"):
			with tf.variable_scope("policy_mean"):
				self.W1 = self.weight_matrix(1,1)
				self.W1 = tf.tile(self.W1,(batch_size,1,1))
				self.b1 = self.bias_vector(1)
			
			with tf.variable_scope("policy_variance"):
				self.W2 = self.weight_matrix(1,1)
				self.W2 = tf.tile(self.W2,(batch_size,1,1))
				self.b2 = self.bias_vector(1)
			
			# weights for value output layer
			with tf.variable_scope("value"):
				self.W3 = self.weight_matrix(1,1)
				self.W3 = tf.tile(self.W3,(batch_size,1,1))
				self.b3 = self.bias_vector(1)

		# policy
		self.mean = tf.batch_matmul(self.state, self.W1) + self.b1
		self.variance = tf.nn.softplus(tf.batch_matmul(self.state, self.W2) + self.b2)
		
		# value - linear output layer
		self.v = tf.batch_matmul(self.state, self.W3) + self.b3
		
		if num_trainable_vars[0] == None:
			num_trainable_vars[0] = len(tf.trainable_variables())
		
		self.trainable_vars = tf.trainable_variables()[-num_trainable_vars[0]:]
		
	def run_policy(self, sess, state):
		mean, variance = sess.run([self.mean,self.variance], feed_dict={self.state:state})
		return mean, variance

	def run_value(self, sess, state):
		v_out = sess.run(self.v, feed_dict = {self.state : state})
		return v_out[0][0] # output is scalar
		
	def debug_save(self, sess, prefix): ### Change for LSTM or make general
		raise NotImplementedError

		
		