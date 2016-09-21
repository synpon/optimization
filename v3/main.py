from __future__ import division
import argparse

import tensorflow as tf
import numpy as np

import rnn
import rnn_cell
from constants import num_iterations, max_seq_length, save_path, summary_freq
from snf import SNF, State, StateOps
from optimizer import Optimizer


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--save', '-s', dest='save_model', action='store_true')
	parser.set_defaults(save=False)
	args = parser.parse_args()

	if args.save_model:
		print "Model will be saved"
	else:
		print "Model will not be saved"
		
	sess = tf.Session()
		
	state_ops = StateOps()
	opt_net = Optimizer()
	
	init = tf.initialize_all_variables()
	sess.run(init)
	
	losses = []

	for i in range(num_iterations):
		snf = SNF()
		state = State(snf, state_ops, sess) # Starting point
		
		for j in range(max_seq_length):
			update = opt_net.run_policy(sess, state)
			### Run backprop - outside this loop?
			loss, state = snf.act(state, update, state_ops, sess)
			losses.append(loss)
			
		if i % summary_freq == 0:
			print "%d\t%.4f" % (i, np.mean(losses))
			losses = []
			
	# Save model
	if args.save_model:
		saver = tf.train.Saver(tf.trainable_variables())
		saver.save(sess, save_path)
		print "Model saved"

if __name__ == "__main__":
	main()