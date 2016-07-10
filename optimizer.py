from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

"""
Input: Gradients
Output: Change in parameters (exp transform to reverse scaling?)
Loss: Change in loss
"""

summary_freq = 100

batch_size = 32
epochs = 100
#rnn = False # feed-forward otherwise
#rnn_num_layers = 1
#rnn_size = 5
#rnn_seq_length = 5

grad_scaling_methods = ['scalar','full']
grad_scaling_method = grad_scaling_methods[0]
grad_scaling_factor = 0.1

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_grads = []
train_losses = []
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
print len(train_losses)

##### Train network #####
with tf.variable_scope("opt_net"):

	batches = 1000

	# Define architecture
	# Feed-forward
	x_grads = tf.placeholder(tf.float32, [None,None,2])
	y_loss = tf.placeholder(tf.float32, [None])
	#y_params = tf.placeholder(tf.float32, [None,None,2])
	
	# Scale inputs
	scale = tf.placeholder(tf.float32)
	p = tf.placeholder(tf.float32)

	grad_threshold = tf.exp(-p)

	if grad_scaling_method == 'scalar':
		x = x_grads*scale
		
	elif grad_scaling_method == 'full':
		# Operations are element-wise
		mask = tf.greater(x_grads,grad_threshold)
		mask = tf.to_float(mask) # Convert from boolean
		inv_mask = 1 - mask
		
		x1_cond1 = tf.log(tf.abs(x_grads))/p
		x2_cond1 = tf.sign(x_grads)
		x1_cond2 = -tf.ones(tf.shape(x_grads))
		x2_cond2 = tf.exp(p)*x_grads
		
		x1 = x1_cond1*mask + x1_cond2*inv_mask
		x2 = x2_cond1*mask + x2_cond2*inv_mask
		
		x = tf.concat(2,[x1,x2])
		
	if grad_scaling_method == 'scalar':		
		W1 = tf.Variable(tf.zeros([1,4]))
	elif grad_scaling_factor == 'full':
		W1 = tf.Variable(tf.zeros([2,4]))
		
	b1 = tf.Variable(tf.zeros([4]))

	W1_1 = tf.reshape(W1,(1,-1,4)) # Convert from rank 2 to rank 3
	W1_1 = tf.tile(W1_1,(batch_size,1,1))

	h = tf.nn.relu(tf.batch_matmul(x,W1_1) + b1)
	
	W2 = tf.Variable(tf.zeros([4,1]))		
	b2 = tf.Variable(tf.zeros([1]))

	W2_1 = tf.reshape(W2,(1,-1,1)) # Convert from rank 2 to rank 3
	W2_1 = tf.tile(W2_1,(batch_size,1,1))
	
	h = tf.nn.relu(tf.batch_matmul(h,W2_1) + b2)
	
	### Make more efficient by using it in the style of AdamOptimizer etc.
	
	# Apply gradients to the parameters in the train net.
	
	# Calculate loss for the updated train net.
	
	# Change in loss as a result of the parameter update
	loss =  - y_loss
	
	optimizer = tf.train.AdamOptimizer()
	train_step = optimizer.minimize(loss)

	init = tf.initialize_all_variables()

	sess = tf.Session()
	sess.run(init)
	
	for i in range(epochs):
		print "Epoch ", i
		### Needs to be adapted for sequences
		perm = np.random.permutation(range(len(train_grads)))
		shuffled_train_grads = train_grads[perm]
		shuffled_train_losses = train_losses[perm]
		shuffled_train_params = train_params[perm]
		
		for start in range(0, len(train_grads), batch_size):
			end = start + batch_size
			train_grads_batch = shuffled_train_grads[start:end]
			train_params_batch = shuffled_train_params[start:end]
			train_losses_batch = shuffled_train_losses[start:end]
			
			_,loss_ = sess.run([train_step,loss], feed_dict={train_grads_batch,train_params_batch,train_losses_batch})
			if i % summary_freq == 0:
				print loss_


##### Compare optimizer performance using Tensorboard #####



