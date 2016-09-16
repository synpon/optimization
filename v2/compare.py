from __future__ import division

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from mlp import MLP
from mlp_relu import MLP_RELU
from cnn import CNN
from ac_network import A3CRNN, A3CFF
from constants import use_rnn, summaries_dir, save_path, m, rnn_size

sess = tf.Session()

if use_rnn:
	opt_net = A3CRNN([None])
else:
	opt_net = A3CFF([None])
	
# Load model
saver = tf.train.Saver(tf.trainable_variables())
saver.restore(sess, save_path)

if not use_rnn:
	print "Loaded: W: %f\tb: %f" % (sess.run(opt_net.W1)[0], sess.run(opt_net.b1)[0])

net = MLP_RELU(opt_net)
sess.run(net.init)

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
for i in range(1):
	sess.run(net.init) # Reset parameters of net to be trained
	for j in range(net.batches):
		batch_x, batch_y = mnist.train.next_batch(net.batch_size)
		summary,_ = sess.run([merged, net.sgd_train_step], feed_dict={net.x: batch_x, net.y_: batch_y})
		sgd_writer.add_summary(summary,j)
	accuracy = sess.run(net.accuracy, feed_dict={net.x: mnist.test.images, net.y_: mnist.test.labels})
	print "SGD accuracy: %f" % accuracy
sgd_writer.close()

# Adam
sess.run(net.init) # Reset parameters of net to be trained
for i in range(net.batches):
	batch_x, batch_y = mnist.train.next_batch(net.batch_size)
	summary,_ = sess.run([merged, net.adam_train_step], feed_dict={net.x: batch_x, net.y_: batch_y})
	adam_writer.add_summary(summary,i)
accuracy = sess.run(net.accuracy, feed_dict={net.x: mnist.test.images, net.y_: mnist.test.labels})
print "Adam accuracy: %f" % accuracy
adam_writer.close()

for i in range(10):
	sess.run(net.init) # Reset parameters of net to be trained

	rnn_state_out = np.zeros([net.num_params, net.opt_net.cell.state_size])

	for j in range(net.batches):
		batch_x, batch_y = mnist.train.next_batch(net.batch_size)
		
		# Compute gradients
		grads = sess.run([net.grads], feed_dict={net.x:batch_x, net.y_:batch_y})
		
		# Compute update
		feed_dict = {net.opt_net.grads:grads, net.opt_net.initial_rnn_state:rnn_state_out, net.opt_net.step_size:np.ones([net.num_params])}
		[mean, rnn_state_out] = sess.run([net.opt_net.mean, net.opt_net.rnn_state], feed_dict=feed_dict)
		
		# Update MLP parameters
		_ = sess.run([net.opt_net_train_step], feed_dict={net.update:mean})
		
		opt_net_writer.add_summary(summary,j)
	accuracy = sess.run(net.accuracy, feed_dict={net.x: mnist.test.images, net.y_: mnist.test.labels})
	print "Opt net accuracy: %f" % accuracy
opt_net_writer.close()

