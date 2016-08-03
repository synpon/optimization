import tensorflow as tf
import numpy as np
from constants import rnn_size, num_rnn_layers, dropout_prob, num_steps, m


# Actor-Critic Network (policy and value network)
class AC3Net(object):

	# Create placeholder variables in order to calculate the loss
	def prepare_loss(self, entropy_beta):
	
		# taken action (input for policy)
		self.mean = tf.placeholder(tf.float32, [1,m,1]) ### Check shape
		self.variance = tf.placeholder(tf.float32, [1,m,1]) ### Check shape
	
		# temporal difference (R-V) (input for policy)
		self.td = tf.placeholder("float", [1])
		
		# Entropy of the policy
		entropy = -0.5*tf.reduce_sum(tf.log(2*3.14*variance)+1) ### Treat dimensions of variance differently?

		# Policy loss (output)
		# Minus because this is for gradient ascent
		# Overlap between the distributions
		policy_loss = -((self.pi - self.a) * self.td + entropy*entropy_beta)

		# R (input for value)
		self.r = tf.placeholder("float", [1])

		# Learning rate for critic is half of actor's, so multiply by 0.5
		value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

		self.total_loss = policy_loss + value_loss

	def sync_from(self, src_network, name=None):
		src_vars = src_network.trainable_vars ### built-in function to do this?
		dst_vars = self.trainable_vars

		sync_ops = []
		with tf.op_scope([], name, "AC3Net") as name:
			for(src_var, dst_var) in zip(src_vars, dst_vars):
				sync_op = tf.assign(dst_var, src_var)
				sync_ops.append(sync_op)

			return tf.group(*sync_ops, name=name)

	def weight_matrix(self,n_in,n_out):
		return tf.Variable(tf.truncated_normal(stddev=0.1, shape=[1, n_in, n_out]))
	
	def bias_vector(self,n_out):
		return tf.Variable(tf.constant(0.1, shape=[n_out]))

	def _debug_save_sub(self, sess, prefix, var, name):
		var_val = var.eval(sess)
		var_val = np.reshape(var_val, (1, np.product(var_val.shape)))				 
		np.savetxt('./' + prefix + '_' + name + '.csv', var_val, delimiter=',')
	

class AC3LSTM(AC3Net):
	# For an RNN, input is usually of shape [batch_size,num_steps]
	# Here they are both 1, as is the case for sampling in a generative RNN
	def __init__(self, num_actions, num_states, num_trainable_vars):		
		raise NotImplementedError
		
# Feed-forward
class AC3FF(AC3Net):
	def __init__(self, num_trainable_vars):
	
		# Input
		batch_size = 1
		self.state = (tf.float32, [batch_size,m,1])

		self.W1 = self.weight_matrix(1,1)
		self.W1 = tf.tile(self.W1,(batch_size,1,1))
		self.b1 = self.bias_vector(1)
		
		self.W2 = self.weight_matrix(1,1)
		self.W2 = tf.tile(self.W2,(batch_size,1,1))
		self.b2 = self.bias_vector(1)
		
		# weight for value output layer
		self.W3 = self.weight_matrix(1,1)
		self.W3 = tf.tile(self.W2,(batch_size,1,1))
		self.b3 = self.bias_vector(1)

		# policy - softmax over actions
		self.mean = tf.batch_matmul(self.state, self.W1) + self.b1
		self.variance = tf.softplus(tf.batch_matmul(self.state, self.W2) + self.b2)
		
		# value - linear output layer
		self.v = tf.matmul(self.state, self.W3) + self.b3
		
		if num_trainable_vars[0] == None:
			num_trainable_vars[0] = len(tf.trainable_variables())
		
		self.trainable_vars = tf.trainable_variables()[-num_trainable_vars[0]:]
		
	def run_policy(self, sess, state):
		pi_out = sess.run(self.pi, feed_dict = {self.state : [state]})
		return pi_out[0]

	def run_value(self, sess, state):
		v_out = sess.run(self.v, feed_dict = {self.state : [state]})
		return v_out[0][0] # output is scalar
		
	def debug_save(self, sess, prefix): ### Change for LSTM or make general
		raise NotImplementedError
