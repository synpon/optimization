from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import control_flow_ops
import tensorflow as tf
import numpy as np
import time
import random

import rnn_cell
import rnn

"""
Input: Gradients
Output: Change in parameters (exp transform to reverse scaling?)
Loss: Change in loss

tensorboard --logdir=/tmp/logs ./ --host 0.0.0.0
http://ec2-52-48-79-131.eu-west-1.compute.amazonaws.com:6006/
"""

summary_freq = 10
summaries_dir = '/tmp/logs'
test_evaluation = True

use_rnn = False # feed-forward otherwise
max_seq_length = 5
rnn_size = 5
#rnn_num_layers = 1
grad_clip_value = None # Set to None to disable

grad_scaling_methods = ['scalar','full']
grad_scaling_method = grad_scaling_methods[0]
grad_scaling_factor = 0.1
p = 10.0

num_gaussians = 250 # Number of Gaussians
m = 10 # Number of dimensions
n = 1000 # Training set size, number of points
cov_range = [0,8]
cov_range[1] *= np.sqrt(m)
weight_gaussians = False
num_landscapes = 1 ### deprecated?

# Random noise is computed each time the point is processed while training the opt net
loss_noise = False
loss_noise_size = 0.2 # Determines the size of the standard deviation. The mean is zero.


def scale_grads(input):
	if grad_scaling_method == 'scalar':
		x = input*tf.constant(grad_scaling_factor)	
		
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
		
		x = tf.concat(2,[x1,x2])
		
	return x
	
	
def inv_scale_grads(input):
	if grad_scaling_method == 'scalar':
		x = input/tf.constant(grad_scaling_factor)	
		
	elif grad_scaling_method == 'full':	
		p_ = tf.constant(p)
	
		# Operations are element-wise
		a,b = tf.split(2,2,input)
		mask = tf.equal(tf.abs(b),1.0)
		mask = tf.to_float(mask) # Convert from boolean
		inv_mask = 1 - mask
		
		x_cond1 = tf.sign(b)*tf.exp(a*p_)
		x_cond2 = b/tf.exp(p_)
		
		x = x_cond1*mask + x_cond2*inv_mask
		
	return x
	
	
def gmm_loss(points, mean_vectors, inv_cov_matrices, gaussian_weights, num_points):
	points = tf.tile(points, multiples=[1,1,num_gaussians])
	mean_vectors = tf.tile(mean_vectors, multiples=[1,1,num_points]) 
	d = points - tf.transpose(mean_vectors,[2,1,0]) # n,m,num_gaussians

	losses = tf.batch_matmul(tf.transpose(d,[2,0,1]),inv_cov_matrices)
	# Follows the code in SciPy's multivariate_normal
	losses = tf.square(losses) # element-wise (num_gaussians,n,m)
	losses = tf.reduce_sum(losses,[2]) # Sum over the dimensions (num_gaussians,n)
	
	if weight_gaussians:
		gaussian_weights = tf.tile(gaussian_weights, multiples=[1,num_points])
		losses = tf.mul(losses,gaussian_weights)
	
	# The pdfs of the Gaussians are negative in order to create a minimization problem.
	losses = -tf.exp(-0.5*losses)
	losses = tf.reduce_mean(losses,[0]) # Average over the Gaussians
	return losses
	
	
class MLP:
	def __init__(self):
		self.batch_size = 1 # 32
		self.batches = 1000

		# Define architecture
		self.x = tf.placeholder(tf.float32, [None, 784])
		self.y_ = tf.placeholder(tf.float32, [None, 10])
		
		with tf.variable_scope("mlp"):
			self.W = tf.Variable(tf.truncated_normal(stddev=0.1, shape=[784,10]))
			self.b = tf.Variable(tf.constant(0.1, shape=[10]))
			
		y = tf.nn.softmax(tf.matmul(self.x,self.W) + self.b)
		y = tf.clip_by_value(y, 1e-10, 1.0) # Prevent log(0) in the cross-entropy calculation
		
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1])) # Cross-entropy
		tf.scalar_summary('loss', self.loss)

		### Make these global?
		sgd_optimizer = tf.train.GradientDescentOptimizer(0.5)
		adam_optimizer = tf.train.AdamOptimizer()
		
		grad_var_pairs = sgd_optimizer.compute_gradients(self.loss)
		grad_var_pairs = [i for i in grad_var_pairs if 'mlp/' in i[1].name]
		grads,vars = zip(*grad_var_pairs)
		grads = [tf.reshape(i,(-1,1)) for i in grads]
		
		if not grad_clip_value is None:
			grads = [tf.clip_by_value(g, -grad_clip_value, grad_clip_value) for g in grads]
			
		self.grads = tf.concat(0,grads)
		
		self.trainable_variables = [i for i in tf.trainable_variables() if 'mlp/' in i.name]
		
		self.sgd_train_step = sgd_optimizer.apply_gradients(grad_var_pairs)
		self.adam_train_step = adam_optimizer.apply_gradients(grad_var_pairs)

		self.init = tf.initialize_all_variables()	
	
	
	def init_opt_net_part(self):
		input = self.grads
		input = tf.reshape(input,[1,-1,1]) ### check

		h = opt_net.compute_updates(input,self.batch_size)
		
		# Apply updates to the parameters in the train net.
		self.opt_net_train_step = opt_net.update_params(self.trainable_variables, h)
		
		
	def fc_layer(input, size, act=tf.nn.relu):
		W = tf.Variable(tf.truncated_normal(stddev=0.1, shape=[size[0],size[1]]))
		b = tf.Variable(tf.constant(0.1,[size[1]]))
		h = act(tf.matmul(input,W) + b)
		return h

		
class OptNet(object):
	def __init__(self):
		self.epochs = 10
		self.batch_size = 32	

		# Define architecture
		self.x_grads = tf.placeholder(tf.float32, [None,None,1])
		self.y_losses = tf.placeholder(tf.float32, [None])
		
		self.input_points = tf.placeholder(tf.float32, [None,m,1])
		self.mean_vectors = tf.placeholder(tf.float32, [num_gaussians,m,1])
		self.inv_cov_matrices = tf.placeholder(tf.float32, [num_gaussians,m,m])
		self.gaussian_weights = tf.placeholder(tf.float32, [num_gaussians,1])
		self.true_batch_size = tf.placeholder(tf.int32)
		
		h = self.compute_updates(self.x_grads, self.true_batch_size)
		
		self.points = self.input_points + h
		new_losses = gmm_loss(self.points, self.mean_vectors, self.inv_cov_matrices, self.gaussian_weights, self.true_batch_size)
		
		# Change in loss as a result of the parameter update (ideally negative)
		self.loss = tf.reduce_mean(new_losses - self.y_losses) # Average over the batch
		
		self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

		self.init = tf.initialize_all_variables()

		
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
			
		return control_flow_ops.group(*ret)
		
		
	def compute_updates(self, input, batch_size):
		raise NotImplementedError("Abstract method")

		
class OptNetFF(OptNet):
	def __init__(self):
		self.feature_sizes = [1,4,1] ### [1,4,4,1] doesn't work
		assert self.feature_sizes[-1] == 1
		
		if grad_scaling_method == 'full':
			self.feature_sizes[0] *= 2
			self.feature_sizes[-1] *= 2
		### Numbers are too small to train effectively? Not enough variance?
		self.W = [tf.Variable(tf.truncated_normal(stddev=0.5, shape=[1,self.feature_sizes[i],self.feature_sizes[i+1]])) for i in range(len(self.feature_sizes) - 1)]
		self.b = [tf.Variable(tf.constant(0.1, shape=[self.feature_sizes[i+1]])) for i in range(len(self.feature_sizes) - 1)]
		#print [i.get_shape() for i in self.W]
		super(OptNetFF, self).__init__()

		
	# Used to update the MLP during evaluation
	def compute_updates(self, input, batch_size):
		x = scale_grads(input)
		self.W_1 = [tf.tile(i,(batch_size,1,1)) for i in self.W]
		### elu causes nan loss
		h = tf.nn.relu(tf.batch_matmul(x,self.W_1[0]) + self.b[0])
		#h = tf.nn.tanh(tf.batch_matmul(x,self.W_1[1]) + self.b[1])
		h = tf.batch_matmul(h,self.W_1[1]) + self.b[1] ### Linear output layer?
		h = inv_scale_grads(h)
		return h # Updates
		
		
class OptNetLSTM(OptNet):

	def __init__(self, batch_size, num_params):
		self.cell = rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0)
		self.reset_state(batch_size,num_params) # Initialize it in this case
		self.epochs = 4
		self.batch_size = 1

		# Define architecture
		self.x_grads = tf.placeholder(tf.float32, [batch_size,num_params,1])
		self.y_losses = tf.placeholder(tf.float32, [batch_size])
		
		self.input_points = tf.placeholder(tf.float32, [batch_size,m,1])
		self.mean_vectors = tf.placeholder(tf.float32, [num_gaussians,m,1])
		self.inv_cov_matrices = tf.placeholder(tf.float32, [num_gaussians,m,m])
		self.gaussian_weights = tf.placeholder(tf.float32, [num_gaussians,1])
		self.true_batch_size = tf.placeholder(tf.int32)
		
		x = [scale_grads(self.x_grads)]
		seq_length = [1]
		
		self.feature_sizes = [5,1]
		
		self.W = [tf.Variable(tf.truncated_normal(stddev=0.1, shape=[1,self.feature_sizes[i],self.feature_sizes[i+1]])) for i in range(len(self.feature_sizes) - 1)]
		self.b = [tf.Variable(tf.constant(0.1, shape=[self.feature_sizes[i+1]])) for i in range(len(self.feature_sizes) - 1)]

		# Compute one step and update the state
		outputs, self.state = rnn.rnn(self.cell, x, self.state, seq_length)
		outputs = outputs[0]
		self.W_1 = [tf.tile(i,(batch_size,1,1)) for i in self.W]
		h = tf.batch_matmul(outputs,self.W_1[0]) + self.b[0]
		
		self.points = self.input_points + h
		new_losses = gmm_loss(self.points, self.mean_vectors, self.inv_cov_matrices, self.gaussian_weights, self.true_batch_size)
		
		# Change in loss as a result of the parameter update (ideally negative)
		self.loss = tf.reduce_mean(new_losses - self.y_losses) # Average over the batch
		
		self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
		
		self.init = tf.initialize_all_variables()
		
	def reset_state(self, batch_size, num_params):
		self.state = tf.zeros([batch_size,num_params,rnn_size])
		self.state = [self.state,self.state]	
	
	# Used to update the MLP during evaluation
	### May require different versions for test and training
	def compute_updates(self, input, batch_size):
		x = [scale_grads(input)]
		seq_length = [1]
		
		# Compute one step and update the state
		outputs, self.state = rnn.rnn(self.cell, x, self.state, seq_length) ### [1,7850,5] - meant to happen?
		outputs = outputs[0]
		self.W_1 = [tf.tile(i,(batch_size,1,1)) for i in self.W]
		h = tf.batch_matmul(outputs,self.W_1[0]) + self.b[0]
		h = inv_scale_grads(h)
		return h


##### Generate training data #####
# Generate n points and their losses from one landscape
# Creating a sufficient training dataset would require this to be run multiple times
# Not probabilities so normalization is not necessary
# The gradient equals zero if and only if the loss equals zero

with tf.variable_scope("gmm"):
	gaussian_weights = tf.random_uniform(shape=(num_gaussians,1))
	mean_vectors = tf.random_uniform(shape=(num_gaussians,m,1), minval=0, maxval=1, dtype=tf.float32) # Each mean vector is a row vector

	# Covariance matrices must be positive-definite
	Q = tf.random_uniform(shape=(num_gaussians,m,m), minval=cov_range[0], maxval=cov_range[1])
	Q_T = tf.transpose(Q, perm=[0,2,1])

	D = [tf.abs(tf.random_uniform(shape=(m,), minval=cov_range[0], maxval=cov_range[1])) for i in range(num_gaussians)]
	D = [tf.diag(i) for i in D]
	D = tf.pack(D) # num_gaussians,m,m

	cov_matrices = tf.batch_matmul(tf.batch_matmul(Q_T,D),Q) # num_gaussians,m,m
	cov_matrices = tf.pow(cov_matrices,0.33)/m # re-scale
	inv_cov_matrices = tf.batch_matrix_inverse(cov_matrices) # num_gaussians,m,m

	points = tf.Variable(tf.random_uniform(shape=(n,m,1),dtype=tf.float32))
	losses = gmm_loss(points, mean_vectors, inv_cov_matrices, gaussian_weights, n)

	opt = tf.train.AdamOptimizer() # Used for initial generation of sequences and to compute gradients
	grads = opt.compute_gradients(losses)[0][0]

	
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

##### Train opt net #####
if use_rnn:
	opt_net = OptNetLSTM(1,m)
else:
	opt_net = OptNetFF()
	
sess.run(opt_net.init)

# One epoch is a pass through n points, not the total n*num_landscapes
print "Epoch\tLoss\t\tZeros\t\tTime(s)"
for epoch in range(opt_net.epochs):

	#print "Generating training data..."
	start = time.time()

	train_points = []
	train_losses = []
	train_grads = []
	train_mean_vectors = []
	train_inv_cov_matrices = []
	train_gaussian_weights = []

	for i in range(num_landscapes):
		all_train_data = sess.run([points, losses, grads, mean_vectors, inv_cov_matrices, gaussian_weights], feed_dict={})
		[train_points_i, train_losses_i, train_grads_i, train_mean_vectors_i, train_inv_cov_matrices_i, train_gaussian_weights_i] = all_train_data
		train_losses_i = np.transpose(train_losses_i)
		
		percentage_zeros = np.mean(np.equal(train_grads_i,np.zeros_like(train_grads_i)))
		
		train_points.append(train_points_i)
		train_losses.append(train_losses_i)
		train_grads.append(train_grads_i)
		train_mean_vectors.append(train_mean_vectors_i)
		train_inv_cov_matrices.append(train_inv_cov_matrices_i)
		train_gaussian_weights.append(train_gaussian_weights_i)

	start = time.time()
	perm = np.random.permutation(range(n))
	opt_net_losses = []
	
	for i in range(n):
		index = perm[i:i+opt_net.batch_size]
		
		# Each batch is from one landscape
		L = random.randint(0,num_landscapes-1) # Inclusive
		
		points_batch = train_points[L][index,:,:]
		losses_batch = train_losses[L][index]
		grads_batch = train_grads[L][index,:,:]
		true_batch_size = len(index)
		
		if loss_noise:
			std_dev = max(1e-30,loss_noise_size*np.mean(np.abs(grads_batch)))
			grads_batch += np.random.normal(0,std_dev,size=grads_batch.shape)
		
		if use_rnn:
			for j in range(max_seq_length):
				### Loss has to be calculated over the entire sequence?
				#opt_net.reset_state(1,m) ### Including this makes the time go from 8s to 180s
				_, loss_, points_batch = sess.run([opt_net.train_step, opt_net.loss, opt_net.points], 
									feed_dict={	opt_net.input_points: points_batch,
												opt_net.y_losses: losses_batch,
												opt_net.x_grads: grads_batch,
												opt_net.mean_vectors: train_mean_vectors[L],
												opt_net.inv_cov_matrices: train_inv_cov_matrices[L],
												opt_net.gaussian_weights: train_gaussian_weights[L],
												opt_net.true_batch_size: true_batch_size})
				
		else:
			_, loss_ = sess.run([opt_net.train_step, opt_net.loss], 
								feed_dict={	opt_net.input_points: points_batch,
											opt_net.y_losses: losses_batch,
											opt_net.x_grads: grads_batch,
											opt_net.mean_vectors: train_mean_vectors[L],
											opt_net.inv_cov_matrices: train_inv_cov_matrices[L],
											opt_net.gaussian_weights: train_gaussian_weights[L],
											opt_net.true_batch_size: true_batch_size})		
										
		#if i % summary_freq == 0:
		#	print loss_, i										
		opt_net_losses.append(loss_)
	
	print "%d\t%g\t%f\t%d" % (epoch, np.mean(opt_net_losses), percentage_zeros, time.time() - start)

mlp = MLP()
sess.run(mlp.init)

##### Compare optimizer performance using TensorBoard #####
# Optimizer parameters are now treated as fixed

if test_evaluation:
	print "\nRunning optimizer comparison..."

	if tf.gfile.Exists(summaries_dir):
		tf.gfile.DeleteRecursively(summaries_dir)
	tf.gfile.MakeDirs(summaries_dir)
	
	# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
	merged = tf.merge_all_summaries()

	sgd_writer = tf.train.SummaryWriter(summaries_dir + '/sgd')
	adam_writer = tf.train.SummaryWriter(summaries_dir + '/adam')
	opt_net_writer = tf.train.SummaryWriter(summaries_dir + '/opt_net')

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	
	# SGD
	sess.run(mlp.init) # Reset parameters of net to be trained
	for i in range(mlp.batches):
		batch_x, batch_y = mnist.train.next_batch(mlp.batch_size)
		summary,_ = sess.run([merged, mlp.sgd_train_step], feed_dict={mlp.x: batch_x, mlp.y_: batch_y})
		sgd_writer.add_summary(summary,i)
	accuracy = sess.run(mlp.accuracy, feed_dict={mlp.x: mnist.test.images, mlp.y_: mnist.test.labels})
	print "SGD accuracy: %f" % accuracy
	sgd_writer.close()
	
	# Adam
	sess.run(mlp.init) # Reset parameters of net to be trained
	for i in range(mlp.batches):
		batch_x, batch_y = mnist.train.next_batch(mlp.batch_size)
		summary,_ = sess.run([merged, mlp.adam_train_step], feed_dict={mlp.x: batch_x, mlp.y_: batch_y})
		adam_writer.add_summary(summary,i)
	accuracy = sess.run(mlp.accuracy, feed_dict={mlp.x: mnist.test.images, mlp.y_: mnist.test.labels})
	print "Adam accuracy: %f" % accuracy
	adam_writer.close()
	
	# Opt net
	if use_rnn:
		opt_net.reset_state(mlp.batch_size, 7850) ### Should not be hardcoded
		sess.run([opt_net.init],feed_dict={})
	mlp.init_opt_net_part()
	
	for i in range(10):
		sess.run(mlp.init) # Reset parameters of net to be trained
		for j in range(mlp.batches):
			batch_x, batch_y = mnist.train.next_batch(mlp.batch_size)
			summary,_ = sess.run([merged, mlp.opt_net_train_step], feed_dict={mlp.x: batch_x, mlp.y_: batch_y})
			opt_net_writer.add_summary(summary,j)
		accuracy = sess.run(mlp.accuracy, feed_dict={mlp.x: mnist.test.images, mlp.y_: mnist.test.labels})
		print "Opt net accuracy: %f" % accuracy
	opt_net_writer.close()
