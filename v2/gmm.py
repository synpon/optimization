import tensorflow as tf

from constants import m, n, num_gaussians, cov_range, weight_gaussians

class GMM(object):
	def __init__(self):
		self.gaussian_weights = tf.random_uniform(shape=(num_gaussians,1))
		self.mean_vectors = tf.random_uniform(shape=(num_gaussians,m,1), minval=0, maxval=1, dtype=tf.float32) # Each mean vector is a row vector

		# Covariance matrices must be positive-definite
		Q = tf.random_uniform(shape=(num_gaussians,m,m), minval=cov_range[0], maxval=cov_range[1])
		Q_T = tf.transpose(Q, perm=[0,2,1])

		D = [tf.abs(tf.random_uniform(shape=(m,), minval=cov_range[0], maxval=cov_range[1])) for i in range(num_gaussians)]
		D = [tf.diag(i) for i in D]
		D = tf.pack(D) # num_gaussians,m,m

		cov_matrices = tf.batch_matmul(tf.batch_matmul(Q_T,D),Q) # num_gaussians,m,m
		cov_matrices = tf.pow(cov_matrices,0.33)/m # re-scale
		self.inv_cov_matrices = tf.batch_matrix_inverse(cov_matrices) # num_gaussians,m,m

		point_mean_vectors = tf.tile(self.mean_vectors,[tf.to_int32(tf.ceil(n/num_gaussians)),1,1])
		point_mean_vectors = tf.random_crop(point_mean_vectors,[n,m,1])
		points = tf.Variable(point_mean_vectors + 0.2*tf.random_uniform(shape=(n,m,1),dtype=tf.float32))
		
		self.losses = self.gmm_loss(points, self.mean_vectors, self.inv_cov_matrices, self.gaussian_weights, n)

		opt = tf.train.AdamOptimizer() # Used for initial generation of sequences and to compute gradients
		self.grads = opt.compute_gradients(self.losses)[0][0]

		
	def gmm_loss(self, points, num_points): ### Is num_points necessary?
		points = tf.tile(points, multiples=[1,1,num_gaussians])
		mean_vectors = tf.tile(self.mean_vectors, multiples=[1,1,num_points]) 
		d = points - tf.transpose(mean_vectors,[2,1,0]) # n,m,num_gaussians

		losses = tf.batch_matmul(tf.transpose(d,[2,0,1]), self.inv_cov_matrices)
		# Follows the code in SciPy's multivariate_normal
		losses = tf.square(losses) # element-wise (num_gaussians,n,m)
		losses = tf.reduce_sum(losses,[2]) # Sum over the dimensions (num_gaussians,n)
		
		if weight_gaussians:
			self.gaussian_weights = tf.tile(self.gaussian_weights, multiples=[1,num_points])
			losses = tf.mul(losses,self.gaussian_weights)
		
		# The pdfs of the Gaussians are negative in order to create a minimization problem.
		losses = -tf.exp(-0.5*losses)
		losses = tf.reduce_mean(losses,[0]) # Average over the Gaussians
		return losses
		
		
	def gen_points(self, num_points):
		point_mean_vectors = tf.tile(self.mean_vectors,[tf.to_int32(tf.ceil(num_points/num_gaussians)),1,1])
		point_mean_vectors = tf.random_crop(point_mean_vectors,[num_points,m,1])
		points = tf.Variable(point_mean_vectors + 0.2*tf.random_uniform(shape=(num_points,m,1),dtype=tf.float32))
		return points
		
	def act(self, state, action):
		# There are no terminal states
		state += action
		loss = self.gmm_loss(state,1)
		reward = -loss
		return reward, state