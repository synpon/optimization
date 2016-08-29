from __future__ import division

import tensorflow as tf
from constants import use_rnn, rnn_size

# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py

class CNN:
	def __init__(self, opt_net):
		self.batch_size = 1 # 32
		self.batches = 1000 ### Adjust

		# Define architecture
		self.x = tf.placeholder(tf.float32, [None, 784])
		self.y_ = tf.placeholder(tf.float32, [None, 10])
		
		# The scope is used to identify the right gradients to optimize
		with tf.variable_scope("mnist"):
			x = tf.reshape(self.x, shape=[-1, 28, 28, 1])
			h = tf.contrib.layers.convolution2d(x, num_outputs=16, kernel_size=[8,8], stride=[1,1], padding='SAME', activation_fn=tf.nn.relu)
			h = tf.nn.max_pool(h, ksize=[1,4,4,1], strides=[1,2,2,1], padding='SAME')
			h = tf.contrib.layers.convolution2d(h, num_outputs=32, kernel_size=[4,4], stride=[1,1], padding='SAME', activation_fn=tf.nn.relu)
			h = tf.nn.max_pool(h, ksize=[1,4,4,1], strides=[1,2,2,1], padding='SAME')
			#h = tf.contrib.layers.convolution2d(h, num_outputs=128, kernel_size=[2,2], stride=[1,1], padding='SAME', activation_fn=tf.nn.relu)
			#h = tf.nn.max_pool(h, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
			h = tf.reshape(h, shape=[self.batch_size,1568])
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
			
		grads = tf.concat(0,grads)
		trainable_variables = [i for i in tf.trainable_variables() if 'mlp/' in i.name]		
		
		if use_rnn:
			grads = tf.reshape(grads,[None,1])
			opt_net.rnn_state = tf.zeros([None,rnn_size])
			output,_ = opt_net.cell(grads, opt_net.rnn_state)
			output = tf.reshape(output,[None,rnn_size])
			updates = tf.matmul(output, opt_net.W1) + opt_net.b1
		else:
			updates = tf.matmul(grads, opt_net.W1) + opt_net.b1
		
		# Apply updates to the parameters in the train net.
		self.opt_net_train_step = opt_net.update_params(trainable_variables, updates)
		
		vars = [i for i in tf.all_variables() if not 'a3c' in i.name]
		self.init = tf.initialize_variables(vars)
		
		