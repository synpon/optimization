import tensorflow as tf
import numpy as np
import rnn_cell

from constants import rnn_size, num_rnn_layers, num_steps, m, rnn_type, grad_scaling_method, grad_scaling_factor, p


# Actor-Critic Network (policy and value network)
class A3CNet(object):

	# Create placeholder variables in order to calculate the loss
	def prepare_loss(self, entropy_beta):
	
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
		### Print relative magnitudes, including the value loss
		# R (input for value)
		self.r = tf.placeholder(tf.float32, [1], 'r')

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

	def weight_matrix(self, n_in, n_out):
		with tf.variable_scope("weight"):
			d = 1.0/np.sqrt(n_in)
			return tf.Variable(tf.random_uniform(shape=[1, n_in, n_out], minval=-d, maxval=d))
	
	def bias_vector(self, n_in, n_out):
		with tf.variable_scope("bias"):
			d = 1.0/np.sqrt(n_in)
			return tf.Variable(tf.random_uniform(shape=[n_out], minval=-d, maxval=d))

	# Update the parameters of another network (eg an MLP)
	def update_params(self, vars, h):
		total = 0
		ret = []

		for i,v in enumerate(vars):
			size = np.prod(list(v.get_shape()))
			size = tf.to_int32(size)
			var_grads = tf.slice(h,begin=[0,total,0],size=[-1,size,-1])
			var_grads = tf.reshape(var_grads,v.get_shape())
			
			#if not grad_clip_value is None:
			#	var_grads = tf.clip_by_value(var_grads, -grad_clip_value, grad_clip_value)
			
			ret.append(v.assign_add(var_grads))
			size += total		
		return tf.group(*ret)
		

class A3CRNN(A3CNet):
	# For an RNN, input is usually of shape [batch_size,num_steps]
	# Here they are both 1, as is the case for sampling in a generative RNN
	def __init__(self, num_trainable_vars):		

		# Input
		batch_size = 1
		self.grads = tf.placeholder(tf.float32, [batch_size,m,1])
		
		grads = scale_grads(self.grads)

		self.W1 = self.weight_matrix(rnn_size,1)
		self.W1 = tf.tile(self.W1,(batch_size,1,1))
		self.b1 = self.bias_vector(1,1)
		
		self.W2 = self.weight_matrix(rnn_size,1)
		self.W2 = tf.tile(self.W2,(batch_size,1,1))
		self.b2 = self.bias_vector(1,1)
		
		# Weights for value output layer
		#self.W3 = self.weight_matrix(rnn_size,1)
		#self.W3 = tf.tile(self.W3,(batch_size,1,1))
		#self.b3 = self.bias_vector(1,1)
		
		if rnn_type == 'rnn':
			self.cell = rnn_cell.BasicRNNCell(rnn_size)
		elif rnn_type == 'gru':
			self.cell = rnn_cell.GRUCell(rnn_size)
		elif rnn_type == 'lstm':
			self.cell = rnn_cell.BasicLSTMCell(rnn_size)

		### Redo in the style of miyosuda's implementation?
		self.rnn_state = tf.zeros([1,m,rnn_size]) 

		if rnn_type == 'lstm':
			raise NotImplementedError
		
		output,rnn_state_out = self.cell(grads, self.rnn_state)
		self.rnn_state_out = rnn_state_out
	
		# policy
		self.mean = tf.batch_matmul(output, self.W1) + self.b1
		self.variance = tf.nn.softplus(tf.batch_matmul(output, self.W2) + self.b2)
		
		# value - linear output layer
		#self.v = tf.batch_matmul(output, self.W3) + self.b3
		
		if num_trainable_vars[0] == None:
			num_trainable_vars[0] = len(tf.trainable_variables())
		
		self.trainable_vars = tf.trainable_variables()[-num_trainable_vars[0]:]
			
			
	def run_policy(self, sess, state, update_rnn_state):
		state = np.reshape(state,[1,m,1])
		mean, variance, rnn_state = sess.run([self.mean,self.variance, self.rnn_state_out], feed_dict={self.grads:state})
		variance = np.maximum(variance,0.01)	
		if update_rnn_state:
			self.rnn_state = rnn_state
		return mean, variance
		
	#def run_value(self, sess, state, update_rnn_state):
	#	[v_out, rnn_state] = sess.run([self.v, self.rnn_state_out], feed_dict = {self.state: state})
	#	if update_rnn_state:
	#		self.rnn_state = rnn_state	
	#	return v_out[0][0]
		
	def reset_rnn_state(self, batch_size, num_params):
		self.rnn_state = tf.zeros([batch_size,num_params,rnn_size]) ### tensorflow (may need to be run) or numpy?
		
		if rnn_type == 'lstm':
			raise NotImplementedError
		
		
# Feed-forward
class A3CFF(A3CNet):
	def __init__(self, num_trainable_vars):
	
		# Input
		batch_size = 1
		self.grads = tf.placeholder(tf.float32, [batch_size,m,1], 'grads')
		self.update = tf.placeholder(tf.float32, [batch_size,m,1], 'update') # Coordinate update
		
		grads = scale_grads(self.grads) ### Add inverse scaling

		with tf.variable_scope("A3CNet"):
			with tf.variable_scope("policy_mean"):
				self.W1 = self.weight_matrix(1,1)
				self.W1 = tf.tile(self.W1,(batch_size,1,1))
				self.b1 = self.bias_vector(1,1)
			
			with tf.variable_scope("policy_variance"):
				self.W2 = self.weight_matrix(1,1)
				self.W2 = tf.tile(self.W2,(batch_size,1,1))
				self.b2 = self.bias_vector(1,1)
			
			# weights for value output layer
			with tf.variable_scope("value"):
				self.W3 = self.weight_matrix(2,1)
				self.W3 = tf.tile(self.W3,(batch_size,1,1))
				self.b3 = self.bias_vector(2,1)

		# policy
		self.mean = tf.batch_matmul(grads, self.W1) + self.b1
		self.variance = tf.nn.softplus(tf.batch_matmul(grads, self.W2) + self.b2)
		
		# value - linear output layer
		grads_and_update = tf.concat(2, [self.grads, self.update])
		v = tf.batch_matmul(grads_and_update, self.W3) + self.b3 # Scalar output so the activation function is linear
		self.v = tf.reduce_mean(v) # Average over dimensions and convert to scalar
		
		if num_trainable_vars[0] == None:
			num_trainable_vars[0] = len(tf.trainable_variables())
		
		self.trainable_vars = tf.trainable_variables()[-num_trainable_vars[0]:]
		
	def run_policy(self, sess, grads):
		grads = np.reshape(grads,[1,m,1])
		mean, variance = sess.run([self.mean,self.variance], feed_dict={self.grads:grads})
		variance = np.maximum(variance,0.01)
		return mean, variance

	def run_value(self, sess, grads, update):
		grads = np.reshape(grads,[1,m,1])
		update = np.reshape(update,[1,m,1])
		v_out = sess.run(self.v, feed_dict={self.grads:grads, self.update:update})
		return v_out # output is a scalar
		
		
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
		
		