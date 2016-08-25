from __future__ import division

import tensorflow as tf
import numpy as np

from constants import m, grad_noise
from snf import SNF, StateOps


class SGD(object):
	def __init__(self):
		self.lr = 0.1
		
		
class Adam(object):
	def __init__(self):
		self.lr = 0.001
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.epsilon = 1e-8
		
sgd = SGD()
adam = Adam()

sess = tf.Session()

# Percentage of zero losses 
def proportion_zeros(snf):
	n = 1000
	points = snf.gen_points(n)
	losses = snf.calc_loss(points)
	z = np.zeros_like(losses)
	z = np.equal(z,losses).astype(int)
	print "Zeros: ", np.mean(z)
	
	
def optimize(point, snf, optimizer):
	print "\nLoss \t\t Grad sizes"	
	M = np.zeros_like(point)
	V = np.zeros_like(point)
	
	state_ops = StateOps()
	start_loss = None
	
	# Run SGD or Adam for 1000 steps
	for i in range(1,1000):
		losses = []
		grad_sizes = []
		
		feed_dict = {state_ops.point: point, 
					state_ops.means: snf.means, 
					state_ops.variances: snf.variances,
					state_ops.weights: snf.weights}
		grads = sess.run([state_ops.grads],feed_dict=feed_dict)
		
		grads = grads[0]
		
		if grad_noise > 0:
			grads += np.abs(grads)*grad_noise*np.random.random((m,1))
		
		grad_sizes.append(np.mean(abs(grads)))
		
		if optimizer == 'sgd':
			point += -sgd.lr*grads
			
		elif optimizer == 'adam':
			t = i
			lr_t = adam.lr
			lr_t *= np.sqrt(1 - np.power(adam.beta2,t))
			lr_t /= (1 - np.power(adam.beta1,t))

			# Copied from the Chainer implementation
			M += (1 - adam.beta1) * (grads - M)
			V += (1 - adam.beta2) * (grads * grads - V)
			point -= lr_t * M / (np.sqrt(V) + adam.epsilon)			
		
		loss = snf.calc_loss(point)
		losses.append(loss)
		
		if i == 1:
			start_loss = loss
		
		if i % 100 == 0:
			print "{:4g} \t {:4g}".format(np.mean(losses), np.mean(grad_sizes))
			losses = []
			grad_sizes = []
			
	loss_change = loss - start_loss
	print "Total change in loss: ", loss_change
	print "Relative change magnitude: ", np.abs(loss_change)/np.abs(start_loss)
			
			
def main():
	snf = SNF()
	proportion_zeros(snf)
	
	point = np.random.rand(m,1)

	print "\nOptimizing with SGD"
	optimize(point,snf,'sgd') ### changes point?
	
	point = np.random.rand(m,1)
	print "\nOptimizing with Adam"
	optimize(point,snf,'adam')


if __name__ == "__main__":
	main()
