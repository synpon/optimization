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
	num_SNFs, num_rnn_layers, rnn_size, m
from snf import SNF, State, StateOps
from optimizer import Optimizer

#rm nohup.out; nohup python -u main.py -s &

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
		# Initializer computes a random point and the SNF loss
	    state = State(snf, state_ops, sess)
	    replay_memory.append(state)
	
	init = tf.initialize_all_variables()
	sess.run(init)
	
	losses = []

	# Training loop
	for i in range(num_iterations):
	    # Retrieve a starting point from the replay memory
		state = random.choice(replay_memory)
		snf = state.snf
		shape, loc, scale = snf.gamma_dist_params
		
		rnn_state = np.zeros([m,rnn_size*num_rnn_layers])
		
		for j in range(seq_length):
			### Handle the RNN state
			feed_dict = {opt_net.point:state.point, 
							opt_net.snf_loss:[state.loss],
							opt_net.variances:snf.variances, 
							opt_net.weights:snf.weights, 
							opt_net.hyperplanes:snf.hyperplanes, 
							opt_net.grads:state.grads,
							opt_net.step_size:np.ones([m]),
							opt_net.initial_rnn_state:rnn_state}
							
			loss,new_point,_ = sess.run([opt_net.loss, opt_net.new_point, opt_net.train_step], feed_dict=feed_dict)
			losses.append(loss)
			
			new_state = state
			new_state.point = new_point
			###new_state.grads?
			replay_memory.append(new_state)
			
		### Run backprop
			
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
	
