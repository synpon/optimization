from __future__ import division
import argparse
import random

import tensorflow as tf
import numpy as np
import scipy.stats as ss

import rnn
import rnn_cell
from constants import num_iterations, seq_length, save_path, summary_freq, \
    num_samples_for_gamma, replay_mem_start_size, replay_memory_max_size, \
	num_SNFs
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
	
	snfs = []
	# Generate the set of SNFs
	print "Generating SNFs..."
	for i in range(num_SNFs):
		snf = SNF()
		losses = []
		
		for j in range(num_samples_for_gamma):
			state = State(snf, state_ops, sess)
			loss = snf.calc_loss(state.point, state_ops, sess)
			# The Gamma distribution is only defined for positive numbers
			losses.append(-loss)
	
		# Fit Gamma distribution
		shape, loc, scale = ss.gamma.fit(losses)
	   
		snf.gamma_dist_params = [shape, loc, scale]
		snfs.append(snf)
	
	print "Initializing replay memory..."
	replay_memory = []
	# Add some initial states to the replay memory
	for i in range(replay_mem_start_size):
	    snf = random.choice(snfs)
	    state = State(snf, state_ops, sess)
	    state.snf = snf
	    replay_memory.append(state)
	
	init = tf.initialize_all_variables()
	sess.run(init)
	
	losses = []

	for i in range(num_iterations):
		snf = SNF()
		state = State(snf, state_ops, sess) # Starting point
		
		for j in range(seq_length):
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