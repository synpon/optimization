from __future__ import division

import tensorflow as tf
from constants import rnn_size, m

class MLP:
	def __init__(self, opt_net):
		self.opt_net = opt_net
		self.batch_size = 64
		self.batches = 1000
		self.num_params = 7850

		# Define architecture
		self.x = tf.placeholder(tf.float32, [None, 784], 'x')
		self.y_ = tf.placeholder(tf.float32, [None, 10], 'y')
		
		# The scope is used to identify the right gradients to optimize
		with tf.variable_scope("mnist"):
			self.W = tf.Variable(tf.truncated_normal(stddev=0.1, shape=[784,10]))
			self.b = tf.Variable(tf.constant(0.1, shape=[10]))
			
		y = tf.nn.softmax(tf.matmul(self.x,self.W) + self.b)
		y = tf.clip_by_value(y, 1e-10, 1.0) # Prevent log(0) in the cross-entropy calculation
		
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1])) # Cross-entropy
		tf.scalar_summary('loss', self.loss)

		sgd_optimizer = tf.train.GradientDescentOptimizer(0.1)
		rmsprop_optimizer = tf.train.RMSPropOptimizer(0.001)
		adam_optimizer = tf.train.AdamOptimizer()
		
		grad_var_pairs = sgd_optimizer.compute_gradients(self.loss)
		grad_var_pairs = [i for i in grad_var_pairs if 'mnist/' in i[1].name]
		
		self.sgd_train_step = sgd_optimizer.apply_gradients(grad_var_pairs)
		self.rmsprop_train_step = rmsprop_optimizer.apply_gradients(grad_var_pairs)
		self.adam_train_step = adam_optimizer.apply_gradients(grad_var_pairs)
	
		#===# Opt net #===#
		grads,_ = zip(*grad_var_pairs)
		grads = [tf.reshape(i,(-1,1)) for i in grads]
		
		#grad_clip_value = None
		#if not grad_clip_value is None:
		#	grads = [tf.clip_by_value(g, -grad_clip_value, grad_clip_value) for g in grads]
			
		self.grads = tf.concat(0,grads)
		self.trainable_variables = [i for i in tf.trainable_variables() if 'mnist/' in i.name]		

		self.update = tf.placeholder(tf.float32,[self.num_params,1], 'update')
		self.opt_net_train_step = self.opt_net.update_params(self.trainable_variables, self.update)
		
		vars = [i for i in tf.all_variables() if not 'optimizer' in i.name]
		self.init = tf.initialize_variables(vars)
	