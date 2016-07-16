import tensorflow as tf
import numpy as np

num_gaussians = 10 # Number of Gaussians
m = 100 # Number of dimensions
n = 1000 # Training set size, number of points

# Not probabilities so normalization is not necessary
gaussian_weights = tf.Variable(tf.random_uniform(shape=(num_gaussians)))

mean_vectors = tf.Variable(tf.random_uniform(shape=(num_gaussians,m,1),dtype=tf.float32)) # Each mean vector is a row vector

# Covariance matrices must be positive-definite
Q = tf.Variable(tf.random_uniform(shape=(num_gaussians,m,m)))
Q_T = tf.transpose(Q, perm=[0,2,1])
D = tf.diag(tf.abs(tf.random_uniform(m,1))) # Diagonal matrix formed from a vector
cov_matrices = Q_T*D*Q # num_gaussians,m,m

points = tf.Variable(tf.random_uniform(shape=(n,m,1),dtype=tf.float32))

# Compute the distance for every point from every mean vector
tmp_points = tf.tile(points, multiples=[1,1,num_gaussians])
tmp_mean_vectors = tf.tile(mean_vectors, multiples=[1,1,n]) 
d = tmp_points - tf.transpose(tmp_mean_vectors,[2,1,0]) # (n,m,num_gaussians)

inv_cov_matrices = tf.batch_matrix_inverse(cov_matrices)

# The pdfs of the Gaussians are negative in order to create a minimization problem.
losses = tf.batch_matmul(tf.transpose(d,[2,0,1]),inv_cov_matrices)
losses = tf.batch_matmul(losses,tf.transpose(d,[2,1,0]))
losses = -tf.exp(-0.5*losses)
losses = tf.reduce_sum(losses,[0,1]) # Sum over the Gaussians and dimensions

sess = tf.Session()