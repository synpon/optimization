"""
python tensorboard_comparison.py
tensorboard --logdir=/tmp/logs ./ --host 0.0.0.0
http://ec2-52-48-79-131.eu-west-1.compute.amazonaws.com:6006/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

fake_data = False
max_steps = 100
learning_rate = 0.5
dropout = 0.9
data_dir = '/tmp/data'
summaries_dir = '/tmp/logs'

def weight_variable(shape):
	initial = tf.zeros(shape=shape)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.zeros(shape=shape)
	return tf.Variable(initial)

def nn_layer(input_tensor, input_dim, output_dim, act):
	weights = weight_variable([input_dim, output_dim])
	biases = bias_variable([output_dim])
	activations = act(tf.matmul(input_tensor, weights) + biases)
	return activations


def train():
	mnist = input_data.read_data_sets(data_dir, one_hot=True, fake_data=fake_data)

	sess = tf.InteractiveSession()

	x = tf.placeholder(tf.float32, [None, 784], name='x-input')
	y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

	keep_prob = tf.placeholder(tf.float32)

	y = nn_layer(x, 784, 10, tf.nn.softmax)

	loss = -tf.reduce_mean(y_ * tf.log(y))
	tf.scalar_summary('loss', loss)

	sgd_train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	adam_train_step = tf.train.AdamOptimizer().minimize(loss)

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
	merged = tf.merge_all_summaries()
	train_writer = tf.train.SummaryWriter(summaries_dir + '/train', sess.graph)
	tf.initialize_all_variables().run()

	def feed_dict(train):
		if train or fake_data:
			xs, ys = mnist.train.next_batch(100, fake_data=fake_data)
			k = dropout
		else:
			xs, ys = mnist.test.images, mnist.test.labels
			k = 1.0
		return {x: xs, y_: ys, keep_prob: k}

	for i in range(max_steps):
		acc = sess.run([accuracy], feed_dict=feed_dict(False))
		
		if i % 10 == 0:	
			print('Accuracy at step %s: %s' % (i, acc))
		
		summary, _ = sess.run([merged, sgd_train_step], feed_dict=feed_dict(True))
		train_writer.add_summary(summary, i)
		
	train_writer.close()


def main(_):
	if tf.gfile.Exists(summaries_dir):
		tf.gfile.DeleteRecursively(summaries_dir)
	tf.gfile.MakeDirs(summaries_dir)
	train()


if __name__ == '__main__':
	tf.app.run()