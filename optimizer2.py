from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import control_flow_ops
import tensorflow as tf
import numpy as np

"""
Input: Gradients
Output: Change in parameters (exp transform to reverse scaling?)
Loss: Change in loss

tensorboard --logdir=/tmp/logs ./ --host 0.0.0.0
http://ec2-52-48-79-131.eu-west-1.compute.amazonaws.com:6006/
"""

summary_freq = 100
summaries_dir = '/tmp/logs'
test_evaluation = True
#rnn = False # feed-forward otherwise
#rnn_num_layers = 1
#rnn_size = 5
seq_length = 1
grad_clip_value = None # Set to None to disable

grad_scaling_methods = ['scalar','full']
grad_scaling_method = grad_scaling_methods[1]
grad_scaling_factor = 0.1
p = 10.0

num_gaussians = 50 # Number of Gaussians
m = 5 # Number of dimensions
n = 5000 # Training set size, number of points
cov_range = [0,1]
weight_gaussians = False


def scale_grads(input):
	if grad_scaling_method == 'scalar':
		input = input*tf.constant(grad_scaling_factor)	
		
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
		
		input = tf.concat(2,[x1,x2])
		
	return input
	
### Should all be logged? How would this affect the shape?
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
		self.batch_size = 1 # 128
		self.batches = 1000

		# Define architecture
		self.x = tf.placeholder(tf.float32, [None, 784])
		self.y_ = tf.placeholder(tf.float32, [None, 10])
		
		self.W = tf.Variable(tf.truncated_normal(stddev=0.1, shape=[784,10]))
		self.b = tf.Variable(tf.constant(0.1, shape=[10]))
		y = tf.nn.softmax(tf.matmul(self.x,self.W) + self.b)
		
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1])) # Cross-entropy
		tf.scalar_summary('loss', self.loss)

		sgd_optimizer = tf.train.GradientDescentOptimizer(0.5)
		adam_optimizer = tf.train.AdamOptimizer()
		
		grad_var_pairs = sgd_optimizer.compute_gradients(self.loss)
		grad_var_pairs = [i for i in grad_var_pairs if i[0] is not None] ### Fix properly
		grads,vars = zip(*grad_var_pairs)
		grads = [tf.reshape(i,(-1,1)) for i in grads]
		
		if not grad_clip_value is None:
			grads = [tf.clip_by_value(g, -grad_clip_value, grad_clip_value) for g in grads]
			
		self.grads = tf.concat(0,grads)
		
		### 7 trainable variables - should be 2
		### 	points and the 4 opt net variables
		#print tf.trainable_variables()
		#print [i.get_shape() for i in tf.trainable_variables()]
		#print "---"
		self.trainable_variables = tf.trainable_variables()[4:] ### fix properly
		
		self.sgd_train_step = sgd_optimizer.apply_gradients(grad_var_pairs)
		self.adam_train_step = adam_optimizer.apply_gradients(grad_var_pairs)

		self.init = tf.initialize_all_variables()	
	
		input = self.grads
		input = tf.reshape(input,[1,-1,1]) ### check
		input = scale_grads(input)

		### Share code with opt net - activations functions could become different
		# Apply gradient update from the opt-net		
		W1_1 = tf.reshape(opt_net.W1,(-1,opt_net.feature_sizes[0],opt_net.feature_sizes[1])) # Convert from rank 2 to rank 3
		W1_1 = tf.tile(W1_1,(1,1,1))
		h = tf.nn.relu(tf.batch_matmul(input,W1_1) + opt_net.b1)

		W2_1 = tf.reshape(opt_net.W2,(-1,opt_net.feature_sizes[1],opt_net.feature_sizes[2])) # Convert from rank 2 to rank 3
		W2_1 = tf.tile(W2_1,(1,1,1))
		h = tf.nn.relu(tf.batch_matmul(h,W2_1) + opt_net.b2) # Gradients
		
		# Apply gradients to the parameters in the train net.
		total = 0
		ret = []

		for i,v in enumerate(self.trainable_variables):
			size = np.prod(list(v.get_shape()))
			size = tf.to_int32(size)
			var_grads = tf.slice(h,begin=[0,total,0],size=[-1,size,-1])
			var_grads = tf.reshape(var_grads,v.get_shape())
			ret.append(v.assign_add(var_grads))
			size += total
		self.var_grads = input
		self.opt_net_train_step = control_flow_ops.group(*ret)		
		
	def fc_layer(input, size, act=tf.nn.relu):
		W = tf.Variable(tf.truncated_normal(stddev=0.1, shape=[size[0],size[1]]))
		b = tf.Variable(tf.constant(0.1,[size[1]]))
		h = act(tf.matmul(input,W) + b)
		return h

		
class OptNet:
	### Put train and test versions in separate functions
	def __init__(self):
		self.epochs = 4
		self.batch_size = 32
		self.feature_sizes = [1,4,1]
		assert self.feature_sizes[-1] == 1
		
		if grad_scaling_method == 'full':
			self.feature_sizes[0] = 2

		# Define architecture
		# Feed-forward
		self.x_grads = tf.placeholder(tf.float32, [None,None,1])
		self.y_losses = tf.placeholder(tf.float32, [None])
		
		self.input_points = tf.placeholder(tf.float32, [None,m,1])
		self.mean_vectors = tf.placeholder(tf.float32, [num_gaussians,m,1])
		self.inv_cov_matrices = tf.placeholder(tf.float32, [num_gaussians,m,m])
		self.gaussian_weights = tf.placeholder(tf.float32, [num_gaussians,1])
		self.true_batch_size = tf.placeholder(tf.int32)
		
		x = scale_grads(self.x_grads)

		self.W1 = tf.Variable(tf.truncated_normal(stddev=0.1, shape=[self.feature_sizes[0],self.feature_sizes[1]]))
		W1_1 = tf.reshape(self.W1,(-1,self.feature_sizes[0],self.feature_sizes[1])) # Convert from rank 2 to rank 3
		self.W1_1 = tf.tile(W1_1,(self.true_batch_size,1,1))
		self.b1 = tf.Variable(tf.constant(0.1, shape=[self.feature_sizes[1]]))

		h = tf.nn.relu(tf.batch_matmul(x,self.W1_1) + self.b1)

		self.W2 = tf.Variable(tf.truncated_normal(stddev=0.1, shape=[4,1]))
		W2_1 = tf.reshape(self.W2,(-1,self.feature_sizes[1],self.feature_sizes[2])) # Convert from rank 2 to rank 3
		self.W2_1 = tf.tile(W2_1,(self.true_batch_size,1,1))
		self.b2 = tf.Variable(tf.constant(0.1, shape=[self.feature_sizes[2]]))

		h = tf.nn.relu(tf.batch_matmul(h,self.W2_1) + self.b2) # Gradients
		
		# Apply gradients to the parameters in the train net.
		#update_params(net,h) ###

		points = self.input_points + h
		new_losses = gmm_loss(points, self.mean_vectors, self.inv_cov_matrices, self.gaussian_weights, self.true_batch_size)
		
		# Change in loss as a result of the parameter update (ideally negative)
		self.loss = tf.reduce_mean(new_losses - self.y_losses) # Average over the batch
		
		self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

		self.init = tf.initialize_all_variables()
		
	def update_params(net,h):
		### Use in the style of the inbuilt optimizers
		total = 0
		
		for i,v in enumerate(net.trainable_variables):
			size = np.prod(list(v.get_shape()))
			size = tf.to_int32(size)
			var_grads = tf.slice(h,begin=[0,total,0],size=[-1,size,-1])
			var_grads = tf.reshape(var_grads,v.get_shape())
			
			if not grad_clip_value is None:
				var_grads = tf.clip_by_value(var_grads, -grad_clip_value, grad_clip_value)

			v.assign_add(var_grads)
			size += total

##### Generate training data #####
# Generate n points and their losses from one landscape
# Creating a sufficient training dataset would require this to be run multiple times
# Not probabilities so normalization is not necessary
### To do: Generate points from multiple landscapes

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

### rescale cov_matrices to roughly follow the desired random uniform distribution?

points = tf.Variable(tf.random_uniform(shape=(n,m,1),dtype=tf.float32))
losses = gmm_loss(points, mean_vectors, inv_cov_matrices, gaussian_weights, n)

opt = tf.train.GradientDescentOptimizer(0.5) # Only used to compute gradients, not optimize
grads = opt.compute_gradients(losses)[0][0]

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "Generating training data..."
all_train_data = sess.run([points, losses, grads, mean_vectors, inv_cov_matrices, gaussian_weights], feed_dict={})
[train_points, train_losses, train_grads, train_mean_vectors, train_inv_cov_matrices, train_gaussian_weights] = all_train_data
train_losses = np.transpose(train_losses)

##### Train opt net #####
opt_net = OptNet()
sess.run(opt_net.init)

for epoch in range(opt_net.epochs):
	print "Epoch ", epoch
	
	perm = np.random.permutation(range(n))
	opt_net_losses = []
	
	for i in range(n):
		index = perm[i:i+opt_net.batch_size]
		points_batch = train_points[index,:,:]		
		losses_batch = train_losses[index]
		grads_batch = train_grads[index,:,:]
		true_batch_size = len(index)
		
		_,loss_ = sess.run([opt_net.train_step,opt_net.loss], 
							feed_dict={	opt_net.input_points: points_batch,
										opt_net.y_losses: losses_batch,
										opt_net.x_grads: grads_batch,
										opt_net.mean_vectors: train_mean_vectors,
										opt_net.inv_cov_matrices: train_inv_cov_matrices,
										opt_net.gaussian_weights: train_gaussian_weights,
										opt_net.true_batch_size: true_batch_size})

		#if i % summary_freq == 0:
		#print loss_, i										
		opt_net_losses.append(loss_)
	
	print np.mean(opt_net_losses)

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
	sess.run(mlp.init)
	for i in range(mlp.batches):
		batch_x, batch_y = mnist.train.next_batch(mlp.batch_size)
		summary,_ = sess.run([merged, mlp.sgd_train_step], feed_dict={mlp.x: batch_x, mlp.y_: batch_y})
		sgd_writer.add_summary(summary,i)
		
	sgd_writer.close()
	
	# Adam
	sess.run(mlp.init)
	for i in range(mlp.batches):
		batch_x, batch_y = mnist.train.next_batch(mlp.batch_size)
		summary,_ = sess.run([merged, mlp.adam_train_step], feed_dict={mlp.x: batch_x, mlp.y_: batch_y})
		adam_writer.add_summary(summary,i)
		
	adam_writer.close()
	
	# Opt net
	sess.run(mlp.init) # Reset parameters of net to be trained
	for i in range(mlp.batches):
		batch_x, batch_y = mnist.train.next_batch(mlp.batch_size)
		summary,_,l = sess.run([merged, mlp.opt_net_train_step, mlp.var_grads], feed_dict={mlp.x: batch_x, mlp.y_: batch_y})
		#print l
		opt_net_writer.add_summary(summary,i)
	
	opt_net_writer.close()
