from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

# Creates a training set
# Gradients at each time step

def main():
	batch_size = 128
	epochs = 1000

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	x = tf.placeholder(tf.float32, [None, 784])

	W = tf.Variable(tf.zeros([784, 10]), name="W")
	b = tf.Variable(tf.zeros([10]), name="b")
	y = tf.nn.softmax(tf.matmul(x,W) + b)

	y_ = tf.placeholder(tf.float32, [None,10])
	loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # Cross-entropy

	optimizer = tf.train.GradientDescentOptimizer(0.5)
	
	grad_var_pairs = optimizer.compute_gradients(loss)
	grads,vars = zip(*grad_var_pairs)
	grads = [tf.reshape(i,(-1,1)) for i in grads]
	grads = tf.concat(0,grads)
	
	params = tf.trainable_variables()

	train_step = optimizer.apply_gradients(grad_var_pairs)

	init = tf.initialize_all_variables()

	sess = tf.Session()
	sess.run(init)

	for i in range(epochs):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		# Train the batch, remembering the gradients and the loss
		tmp = sess.run([train_step,loss,grads] + params, feed_dict={x: batch_x, y_: batch_y})
		loss_,grads_ = tmp[1:3]
		params_ = tmp[3:]
		
		# Set variables
		#W = params[0]
		#b = params[1]
		
		for i,var in enumerate(tf.trainable_variables()):
			var = params[i]
		
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
	main()