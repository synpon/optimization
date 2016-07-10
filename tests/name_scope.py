from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

# Creates a training set
# Gradients at each time step

class ShallowNet:
	def __init__(self):
		self.x = tf.placeholder(tf.float32, [None, 784])
		self.y_ = tf.placeholder(tf.float32, [None,10])

		W = tf.Variable(tf.zeros([784, 10]))
		b = tf.Variable(tf.zeros([10]))
		y = tf.nn.softmax(tf.matmul(self.x,W) + b)

		self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1])) # Cross-entropy

		optimizer = tf.train.GradientDescentOptimizer(0.5)
		
		grad_var_pairs = optimizer.compute_gradients(self.loss)
		self.train_step = optimizer.apply_gradients(grad_var_pairs)
		
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		self.init = tf.initialize_all_variables()
	
	
class MLP:
	def __init__(self):
		self.x = tf.placeholder(tf.float32, [None, 784])
		self.y_ = tf.placeholder(tf.float32, [None,10])

		W1 = tf.Variable(tf.truncated_normal(shape=[784, 500],stddev=0.1))
		b1 = tf.Variable(tf.constant(0.1,shape=[500]))
		
		h = tf.nn.relu(tf.matmul(self.x,W1) + b1)
		h = tf.nn.dropout(h, 0.9)
		
		W2 = tf.Variable(tf.truncated_normal(shape=[500, 10],stddev=0.1))
		b2 = tf.Variable(tf.constant(0.1,shape=[10]))
		
		y = tf.nn.softmax(tf.matmul(h,W2) + b2)
		y = tf.clip_by_value(y, 0.001, 1) # Fix nan losses for relu

		self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1])) # Cross-entropy

		optimizer = tf.train.AdamOptimizer()
		
		grad_var_pairs = optimizer.compute_gradients(self.loss)
		self.train_step = optimizer.apply_gradients(grad_var_pairs)
		
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		self.init = tf.initialize_all_variables()
		
		
		
def main():
	batch_size = 128
	batches = 1000

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	shallow_net = ShallowNet()
	mlp = MLP()

	sess = tf.Session()
	sess.run(shallow_net.init)
	sess.run(mlp.init)

	for i in range(batches):
		batch_x, batch_y = mnist.train.next_batch(batch_size)

		_,loss_ = sess.run([shallow_net.train_step,shallow_net.loss], feed_dict={shallow_net.x: batch_x, shallow_net.y_: batch_y})
		print "shallow: ", loss_
			
		_,loss_ = sess.run([mlp.train_step,mlp.loss], feed_dict={mlp.x: batch_x, mlp.y_: batch_y})
		print "mlp: ", loss_

	#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
	main()