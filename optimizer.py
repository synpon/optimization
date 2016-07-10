from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

"""
Input: Gradients
Output: Change in parameters (exp transform to reverse scaling?)
Loss: Change in loss
"""

batch_size = 32
epochs = 100
#rnn_num_layers = 1
#rnn_size = 5
#rnn_seq_length = 5

grad_scaling_methods = ['scalar','full']
grad_scaling_method = grad_scaling_methods[1]
grad_scaling_factor = 0.1

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_losses = []
train_grads = []
train_params = []

##### Generate training data #####
with tf.variable_scope("mlp"):
	batch_size = 128
	batches = 1000

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	# Define architecture
	x = tf.placeholder(tf.float32, [None, 784])

	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x,W) + b)

	y_ = tf.placeholder(tf.float32, [None,10])
	loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # Cross-entropy

	optimizer = tf.train.GradientDescentOptimizer(0.5)
	
	grad_var_pairs = optimizer.compute_gradients(loss)
	grads,vars = zip(*grad_var_pairs)
	grads = [tf.reshape(i,(-1,1)) for i in grads]
	grads = tf.concat(0,grads)
	
	params = tf.trainable_variables()
	
	train_step = optimizer.apply_gradients(grad_var_pairs)

	init = tf.initialize_all_variables()

	sess = tf.Session()
	sess.run(init)

	for i in range(batches):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		# Train the batch, remembering the gradients and the loss
		tmp = sess.run([train_step,loss,grads] + params, feed_dict={x: batch_x, y_: batch_y})
		loss_,grads_ = tmp[1:3]
		params_ = tmp[3:]
		
		train_losses.append(loss_)
		train_grads.append(grads_)
		train_params.append(params_)
		
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	
print "\nTraining data generated"
print len(train_losses), len(train_params)

##### Train network #####
with tf.variable_scope("opt_net"):

	# Define architecture
	# Feed-forward
	x = tf.placeholder(tf.float32, [None,None,2])
	
	# Scale inputs
	scale_ = tf.placeholder(tf.float32)
	p_ = tf.placeholder(tf.float32)

	grad_threshold = tf.exp(-p_)

	if grad_scaling_method == 'scalar':
		y_ = x_*scale_
		
	elif grad_scaling_method == 'full':
		# Operations are element-wise
		mask = tf.greater(x_,grad_threshold)
		mask = tf.to_float(mask) # Convert from boolean
		inv_mask = 1 - mask
		
		x1_cond1 = tf.log(tf.abs(x_))/p_
		x2_cond1 = tf.sign(x_)
		x1_cond2 = -tf.ones(tf.shape(x_))
		x2_cond2 = tf.exp(p_)*x_
		
		x1 = x1_cond1*mask + x1_cond2*inv_mask
		x2 = x2_cond1*mask + x2_cond2*inv_mask	
	
	optimizer = tf.train.AdamOptimizer()




