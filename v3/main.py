from __future__ import division
import argparse
import random

import tensorflow as tf
import numpy as np

from constants import num_iterations, seq_length, save_path, summary_freq, \
    episode_length, replay_mem_start_size, replay_memory_max_size, \
	num_SNFs, num_rnn_layers, rnn_size, m
from snf import SNF, State, StateOps
from optimizer import Optimizer

"""
rm nohup.out; nohup python -u main.py -s &

pyflakes main.py compare.py optimizer.py constants.py snf.py nn_utils.py
pychecker main.py
"""

### grad scaling has not been inverted
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
		
		for j in range(seq_length):
			feed_dict = {opt_net.point: state.point, 
							opt_net.snf_loss: [state.loss],
							opt_net.variances: snf.variances, 
							opt_net.weights: snf.weights, 
							opt_net.hyperplanes: snf.hyperplanes, 
							opt_net.input_grads: state.grads,
							opt_net.step_size: np.ones([m]),
							opt_net.initial_rnn_state: state.rnn_state,
							opt_net.state_index: state.counter}
							
			loss,new_point,rnn_state,grads,_ = sess.run([opt_net.loss, 
														opt_net.new_point, 
														opt_net.rnn_state, 
														opt_net.grads, 
														opt_net.train_step], 
														feed_dict=feed_dict)
			losses.append(loss)
			
			state.point = new_point
			state.rnn_state = rnn_state
			state.grads = grads
			
			state.counter += 1
			if state.counter >= episode_length:
				snf = random.choice(snfs)
				state = State(snf, state_ops, sess)
			
			replay_memory.append(state)
			
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
	
