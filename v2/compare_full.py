from __future__ import division

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from mlp import MLP
from mlp_relu import MLP_RELU
#from cnn import CNN
#from lm import LM
from optimizer import Optimizer
from constants import summaries_dir, save_path, seq_length

"""
rm nohup.out; nohup python -u compare_full.py &
tensorboard --logdir=/tmp/logs ./ --host 0.0.0.0
http://ec2-52-48-79-131.eu-west-1.compute.amazonaws.com:6006/
"""

runs = 1 ###

sess = tf.Session()
opt_net = Optimizer()
	
# Load model
saver = tf.train.Saver(tf.trainable_variables())
saver.restore(sess, save_path)

net = MLP(opt_net)
sess.run(net.init)

print "\nRunning optimizer comparison..."
results = np.zeros([net.batches,5])

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def test_inbuilt_optimizer(opt_step, index):
	for i in range(runs):
		sess.run(net.init) # Reset parameters of net to be trained
		for j in range(net.batches):
			batch_x, batch_y = mnist.train.next_batch(net.batch_size)
			train_loss,_ = sess.run([net.loss, opt_step], feed_dict={net.x: batch_x, net.y_: batch_y})
			results[j,index] += train_loss
			results[j,0] = j
		#accuracy = sess.run(net.accuracy, feed_dict={net.x: mnist.test.images, net.y_: mnist.test.labels})	
	return
	
test_inbuilt_optimizer(net.sgd_train_step,1)
print "SGD complete"
test_inbuilt_optimizer(net.rmsprop_train_step,2)
print "RMSProp complete"
test_inbuilt_optimizer(net.adam_train_step,3)
print "Adam complete"

for i in range(runs):
	sess.run(net.init) # Reset parameters of the net to be trained
	rnn_state = np.zeros([net.num_params, net.opt_net.cell.state_size])
	print i
	for j in range(net.batches):
		batch_x, batch_y = mnist.train.next_batch(net.batch_size)
		
		# Compute gradients
		train_loss, grads = sess.run([net.loss, net.grads], feed_dict={net.x:batch_x, net.y_:batch_y})
		results[j,4] += train_loss
		
		# Compute update
		feed_dict = {net.opt_net.input_grads: np.reshape(grads,[1,-1,1]), 
					net.opt_net.initial_rnn_state: rnn_state}
		[update, rnn_state] = sess.run([net.opt_net.update, net.opt_net.rnn_state_out_compare], feed_dict=feed_dict)
		
		# Update MLP parameters
		sess.run([net.opt_net_train_step], feed_dict={net.update:update})
		
	#accuracy = sess.run(net.accuracy, feed_dict={net.x: mnist.test.images, net.y_: mnist.test.labels})
	#print "Opt net accuracy: %f" % accuracy
print "Opt net complete"
	
results /= runs

np.savetxt("results.csv", results, delimiter=",")