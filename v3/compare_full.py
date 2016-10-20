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

runs = 5

sess = tf.Session()
	
opt_net = Optimizer()
	
# Load model
saver = tf.train.Saver(tf.trainable_variables())
saver.restore(sess, save_path)

net = MLP(opt_net)
sess.run(net.init)

print "\nRunning optimizer comparison..."

if tf.gfile.Exists(summaries_dir):
	tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.merge_all_summaries()

sgd_writer = tf.train.SummaryWriter(summaries_dir + '/sgd')
rmsprop_writer = tf.train.SummaryWriter(summaries_dir + '/rmsprop')
adam_writer = tf.train.SummaryWriter(summaries_dir + '/adam')
opt_net_writer = tf.train.SummaryWriter(summaries_dir + '/opt_net')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def test_inbuilt_optimizer(writer,optimizer):
	for i in range(runs):
		sess.run(net.init) # Reset parameters of net to be trained
		for j in range(net.batches):
			batch_x, batch_y = mnist.train.next_batch(net.batch_size)
			summary,_ = sess.run([merged, optimizer], feed_dict={net.x: batch_x, net.y_: batch_y})
			writer.add_summary(summary,j)
		accuracy = sess.run(net.accuracy, feed_dict={net.x: mnist.test.images, net.y_: mnist.test.labels})
	return
	
test_inbuilt_optimizer(sgd_writer, net.sgd_train_step)
print "SGD complete"
test_inbuilt_optimizer(rmsprop_writer, net.rmsprop_train_step)
print "RMSProp complete"
test_inbuilt_optimizer(adam_writer, net.adam_train_step)
print "Adam complete"

for i in range(runs):
	sess.run(net.init) # Reset parameters of the net to be trained

	rnn_state = np.zeros([net.num_params, net.opt_net.cell.state_size])

	for j in range(net.batches):
		batch_x, batch_y = mnist.train.next_batch(net.batch_size)
		
		# Compute gradients
		summary, grads = sess.run([merged, net.grads], feed_dict={net.x:batch_x, net.y_:batch_y})
		
		# Compute update
		feed_dict = {net.opt_net.input_grads: np.reshape(grads,[1,-1,1]), 
					net.opt_net.initial_rnn_state: rnn_state}
		[update, rnn_state] = sess.run([net.opt_net.update, net.opt_net.rnn_state_out_compare], feed_dict=feed_dict)
		
		# Update MLP parameters
		_ = sess.run([net.opt_net_train_step], feed_dict={net.update:update})	
		opt_net_writer.add_summary(summary,j)
		
	accuracy = sess.run(net.accuracy, feed_dict={net.x: mnist.test.images, net.y_: mnist.test.labels})
	print "Opt net accuracy: %f" % accuracy
	
sgd_writer.close()
rmsprop_writer.close()
adam_writer.close()
opt_net_writer.close()

