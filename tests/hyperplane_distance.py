from __future__ import division

import numpy as np
import tensorflow as tf

m = 2 # dimensionality

def hyperplane_point_dist(point,hyperplane_points):
	X = np.matrix(hyperplane_points)
	k = np.ones((m,1))
	a = np.dot(np.linalg.inv(X), k) # vector
	D = np.dot(point,a) - 1 # scalar
	D /= np.linalg.norm(a)
	return np.abs(D)
	
	
point_ = np.reshape(np.array([0,3]),[1,m])
hyperplane_points_ = np.array([[4,2],[-1,6]])
	
print hyperplane_point_dist(np.array([0,3]),[np.array([4,2]),np.array([-1,6])])

sess = tf.Session()

point = tf.placeholder(tf.float32, [1,m])
X = tf.placeholder(tf.float32, [m,m])
k = tf.ones((m,1))
a = tf.matmul(tf.matrix_inverse(X), k) # vector
D = tf.matmul(point,a) - 1 # scalar
norm = tf.sqrt(tf.reduce_sum(tf.square(a)))
D /= norm
D = tf.abs(D)

print sess.run(D,feed_dict={point:point_, X:hyperplane_points_})[0][0]
