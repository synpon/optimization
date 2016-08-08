import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from mlp import MLP
from ac_network import A3CRNN, A3CFF
from constants import use_rnn, summaries_dir, save_path

sess = tf.Session()

opt_net = A3CFF([None])
		
# Load model
saver = tf.train.Saver(tf.trainable_variables())
saver.restore(sess, save_path)

mlp = MLP(opt_net,sess)
sess.run(mlp.init)

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
sess.run(mlp.init) # Reset parameters of net to be trained
for i in range(mlp.batches):
	batch_x, batch_y = mnist.train.next_batch(mlp.batch_size)
	summary,_ = sess.run([merged, mlp.sgd_train_step], feed_dict={mlp.x: batch_x, mlp.y_: batch_y})
	sgd_writer.add_summary(summary,i)
accuracy = sess.run(mlp.accuracy, feed_dict={mlp.x: mnist.test.images, mlp.y_: mnist.test.labels})
print "SGD accuracy: %f" % accuracy
sgd_writer.close()

# Adam
sess.run(mlp.init) # Reset parameters of net to be trained
for i in range(mlp.batches):
	batch_x, batch_y = mnist.train.next_batch(mlp.batch_size)
	summary,_ = sess.run([merged, mlp.adam_train_step], feed_dict={mlp.x: batch_x, mlp.y_: batch_y})
	adam_writer.add_summary(summary,i)
accuracy = sess.run(mlp.accuracy, feed_dict={mlp.x: mnist.test.images, mlp.y_: mnist.test.labels})
print "Adam accuracy: %f" % accuracy
adam_writer.close()

# Opt net
#if use_rnn:
	#opt_net.reset_state(mlp.batch_size, 7850) ### Should not be hardcoded
	#sess.run([opt_net.init],feed_dict={})
	
#mlp.init_opt_net_part()

for i in range(10):
	sess.run(mlp.init) # Reset parameters of net to be trained
	for j in range(mlp.batches):
		batch_x, batch_y = mnist.train.next_batch(mlp.batch_size)
		summary,_ = sess.run([merged, mlp.opt_net_train_step], feed_dict={mlp.x: batch_x, mlp.y_: batch_y})
		opt_net_writer.add_summary(summary,j)
	accuracy = sess.run(mlp.accuracy, feed_dict={mlp.x: mnist.test.images, mlp.y_: mnist.test.labels})
	print "Opt net accuracy: %f" % accuracy
opt_net_writer.close()

