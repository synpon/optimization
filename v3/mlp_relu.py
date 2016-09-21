from __future__ import division

import tensorflow as tf
from constants import rnn_size

# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py

class MLP_RELU:
	def __init__(self, opt_net):
		self.opt_net = opt_net
		self.batch_size = 1 ### 32
		self.batches = 1000 ### Adjust
		self.num_params = 466442#tf.reduce_prod(tf.shape(grads)) ### calculate num_params automatically

		# Define architecture
		self.x = tf.placeholder(tf.float32, [None, 784])
		self.y_ = tf.placeholder(tf.float32, [None, 10])
		
		# The scope is used to identify the right gradients to optimize
		with tf.variable_scope("mnist"):
			h = tf.contrib.layers.fully_connected(inputs=self.x, num_outputs=256, activation_fn=tf.nn.relu)
			h = tf.contrib.layers.fully_connected(inputs=h, num_outputs=512, activation_fn=tf.nn.relu)
			h = tf.contrib.layers.fully_connected(inputs=h, num_outputs=256, activation_fn=tf.nn.relu)
			y = tf.contrib.layers.fully_connected(inputs=h, num_outputs=10, activation_fn=tf.nn.softmax)
			
		y = tf.clip_by_value(y, 1e-10, 1.0) # Prevent log(0) in the cross-entropy calculation
		
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1])) # Cross-entropy
		tf.scalar_summary('loss', self.loss)

		sgd_optimizer = tf.train.GradientDescentOptimizer(0.1)
		adam_optimizer = tf.train.AdamOptimizer()
		
		grad_var_pairs = sgd_optimizer.compute_gradients(self.loss)
		grad_var_pairs = [i for i in grad_var_pairs if 'mnist/' in i[1].name]
		
		self.sgd_train_step = sgd_optimizer.apply_gradients(grad_var_pairs)
		self.adam_train_step = adam_optimizer.apply_gradients(grad_var_pairs)
	
		#===# Opt net #===#
		grads,_ = zip(*grad_var_pairs)
		grads = [tf.reshape(i,(-1,1)) for i in grads]
		
		#if not grad_clip_value is None:
		#	grads = [tf.clip_by_value(g, -grad_clip_value, grad_clip_value) for g in grads]
			
		self.grads = tf.concat(0,grads)
		self.trainable_variables = [i for i in tf.trainable_variables() if 'mnist/' in i.name]		
		
		self.update = tf.placeholder(tf.float32,[1,self.num_params,1])
		update = tf.reshape(self.update,[self.num_params,1])
		self.opt_net_train_step = self.opt_net.update_params(self.trainable_variables, update)
		
		vars = [i for i in tf.all_variables() if not 'a3c' in i.name]
		self.init = tf.initialize_variables(vars)
		
		