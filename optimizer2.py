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
grad_clip_value = 0.01 # Set to None to disable

grad_scaling_methods = ['scalar','full']
grad_scaling_method = grad_scaling_methods[0]
grad_scaling_factor = 0.1


# Generate n points and their losses from one landscape
# Creating a sufficient training dataset would require this to be run multiple times

# Numbers are very small, from around 10^-30 to 10^-5

num_gaussians = 100 # Number of Gaussians
m = 100 # Number of dimensions
n = 10000 # Training set size, number of points

# Not probabilities so normalization is not necessary
gaussian_weights = tf.Variable(tf.random_uniform(shape=(num_gaussians,)))

mean_vectors = tf.Variable(tf.random_uniform(shape=(num_gaussians,m,1),dtype=tf.float32)) # Each mean vector is a row vector

# Covariance matrices must be positive-definite
Q = tf.Variable(tf.random_uniform(shape=(num_gaussians,m,m)))
Q_T = tf.transpose(Q, perm=[0,2,1])

D = [tf.abs(tf.random_uniform(shape=(m,))) for i in range(num_gaussians)]
D = [tf.diag(i) for i in D]
D = tf.pack(D) # num_gaussians,m,m

cov_matrices = tf.batch_matmul(tf.batch_matmul(Q_T,D),Q) # num_gaussians,m,m

points = tf.Variable(tf.random_uniform(shape=(n,m,1),dtype=tf.float32))

# Compute the distance for every point from every mean vector
tmp_points = tf.tile(points, multiples=[1,1,num_gaussians])
tmp_mean_vectors = tf.tile(mean_vectors, multiples=[1,1,n]) 
d = tmp_points - tf.transpose(tmp_mean_vectors,[2,1,0]) # n,m,num_gaussians

inv_cov_matrices = tf.batch_matrix_inverse(cov_matrices) # num_gaussians,m,m

losses = tf.batch_matmul(tf.transpose(d,[2,0,1]),inv_cov_matrices)
# Follows the code in SciPy's multivariate_normal
losses = tf.square(losses) # element-wise (num_gaussians,n,m)
losses = tf.reduce_sum(losses,[2]) # Sum over the dimensions
# The pdfs of the Gaussians are negative in order to create a minimization problem.
losses = -tf.exp(-0.5*losses)
losses = tf.reduce_mean(losses,[0]) # Average over the Gaussians

opt = tf.train.GradientDescentOptimizer(0.5) # Only used to compute gradients, not optimize
# Computes gradients for all variables. Only the gradients for 'points' are needed
grads = opt.compute_gradients(losses)[3][0]

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)
print sess.run(losses)

# Optimize using Adam
