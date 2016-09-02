from __future__ import division
import random
import tensorflow as tf
import numpy as np

#import rnn
#import rnn_cell
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
		# Minus because this is for gradient ascent
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
	# For an RNN, input is usually of shape [batch_size,num_steps]
	# Here they are both 1, as is the case for sampling in a generative RNN
	def __init__(self, num_trainable_vars):		
		### Add regularization of the parameters to control bias etc.?
		# Input
		self.grads = tf.placeholder(tf.float32, [None,m,1])
		self.update = tf.placeholder(tf.float32, [None,m,1], 'update') # Coordinate update
		#self.rand = tf.placeholder_with_default(input=tf.zeros([m,1],tf.float32), shape=[m,1])
		#self.rand = tf.placeholder(tf.float32, [None,m,1])
		
		grads = scale_grads(self.grads) ### Add inverse scaling

		# The scope allows these variables to be excluded from being reinitialized during the comparison phase
		with tf.variable_scope("a3c"):
			if rnn_type == 'rnn':
				self.cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
			elif rnn_type == 'gru':
				self.cell = tf.nn.rnn_cell.GRUCell(rnn_size)
			elif rnn_type == 'lstm':
				self.cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

			if rnn_type == 'lstm':
				raise NotImplementedError
			
			# placeholder for RNN unrolling time step size.
			self.step_size = tf.placeholder(tf.float32, [1])
			
			self.initial_rnn_state = tf.placeholder(tf.float32, [1,self.cell.state_size])

			# Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
			# When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
			# Unrolling step size is applied via self.step_size placeholder.
			# When forward propagating, step_size is 1.
			### Won't work unless dynamic_rnn is altered since it's 3D now
			### Could solve the problem by flattening the input so it's [batch_size x m, ]
			#grads = tf.reshape(grads,[-1])
			
			#scope = "net_" + str(thread_index)
			
			output, rnn_state = tf.nn.dynamic_rnn(self.cell,
									grads,
									initial_state = self.initial_rnn_state,
									sequence_length = self.step_size,
									time_major = False)#,
									#scope = scope)			
			
			self.output = tf.reshape(output,[m,rnn_size])
			self.rnn_state = rnn_state ###
		
			# policy
			self.mean = fc_layer(output, num_in=rnn_size, num_out=1, activation_fn=None)

			h = fc_layer(output, num_in=rnn_size, num_out=1, activation_fn=tf.nn.relu) # softplus not used due to NaNs
			self.variance = tf.maximum(0.01,h)
			
			# value - linear output layer
			grads_and_update = tf.concat(2, [self.grads, self.update])
			v_h = fc_layer(grads_and_update, num_in=2, num_out=10, activation_fn=tf.nn.relu)
			v = fc_layer(v_h, num_in=10, num_out=1, activation_fn=None)
		
		self.v = tf.reduce_mean(v) # Average over dimensions and convert to scalar
		
		if num_trainable_vars[0] == None:
			num_trainable_vars[0] = len(tf.trainable_variables())
		
		self.trainable_vars = tf.trainable_variables()[-num_trainable_vars[0]:]		
		self.reset_rnn_state()
			
			
	def run_policy(self, sess, grads):
		# Updates the RNN state
		feed_dict = {self.grads:grads, self.initial_rnn_state:self.rnn_state_out, self.step_size:[1]}
		[mean, variance, self.rnn_state_out] = sess.run([self.mean,self.variance, self.rnn_state], feed_dict=feed_dict)
		variance = np.maximum(variance,0.01)
		return mean, variance
	
	
	def run_value(self, sess, grads, update):
		# Does not update the RNN state
		prev_rnn_state_out = self.rnn_state_out
		feed_dict = {self.grads:grads, self.update:update, self.initial_rnn_state:self.rnn_state_out, self.step_size:[1]}
		[v_out, self.rnn_state_out] = sess.run([self.v, self.rnn_state], feed_dict=feed_dict)		

	    # roll back RNN state
		self.rnn_state_out = prev_rnn_state_out ### necessary?
		
		return np.abs(v_out) # output is a scalar
		
		
	def reset_rnn_state(self):
		self.rnn_state_out = np.zeros([1,self.cell.state_size])
		
		
# Feed-forward
class A3CFF(A3CNet):
	def __init__(self, num_trainable_vars):
	
		# Input
		# Third dimension is the number of features
		self.grads = tf.placeholder(tf.float32, [None,m,1], 'grads')
		self.update = tf.placeholder(tf.float32, [None,m,1], 'update') # Coordinate update
		self.rand = tf.placeholder_with_default(input=0.0, shape=[])
		
		grads = scale_grads(self.grads) ### Add inverse scaling

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
		
		