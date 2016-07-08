from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

"""
Input: Gradients
Output: Change in parameters (exp transform to reverse scaling?)
Loss: Change in loss
"""

batch_size = 32
epochs = 100
rnn_num_layers = 1 ### Implement
rnn_size = 5
rnn_seq_length = 5 ### Implement

grad_scaling_methods = ['scalar','full']
grad_scaling_method = grad_scaling_methods[1]
grad_scaling_factor = 0.1

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.variable_scope("mlp"):
	x = tf.placeholder(tf.float32, [None, 784])

	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))

	y = tf.nn.softmax(tf.matmul(x,W) + b)
	y_ = tf.placeholder(tf.float32, [None,10])
	loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # Cross-entropy

	
x = tf.placeholder(tf.float32, [None,None]) # [batch_size,num_params]

##### Scale inputs #####
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

#y = 

#loss = 

optimizer = tf.train.AdamOptimizer()
grad_var_pairs = optimizer.compute_gradients(loss)
train_step = optimizer.apply_gradients(grad_var_pairs)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(epochs):
	pass
	#sess.run()
	
	