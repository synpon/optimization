from __future__ import division
import argparse
import random

import tensorflow as tf
import numpy as np

from constants import num_iterations, seq_length, save_path, summary_freq, \
    episode_length, replay_mem_start_size, replay_memory_max_size, \
	num_SNFs, num_rnn_layers, rnn_size, m, net_sync_freq
from snf import SNF, State, StateOps
from optimizer import Optimizer

"""
rm nohup.out; nohup python -u main.py -s &

pyflakes main.py compare.py optimizer.py constants.py snf.py nn_utils.py
pychecker main.py
"""
### Check the signs are correct
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
	
	with tf.variable_scope("opt1"):
		opt_net = Optimizer()
		
	with tf.variable_scope("opt2"):
		opt_net2 = Optimizer() # Only used for generating new states
	
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
		
		if state.counter >= episode_length:
			snf = random.choice(snfs)
			state = State(snf, state_ops, sess)
		
		#===# Generate states #===#
		states_seq = []
		
		for j in range(seq_length):		
			feed_dict = {opt_net2.point: state.point, 
							opt_net2.snf_loss: [state.loss],
							opt_net2.variances: snf.variances, 
							opt_net2.weights: snf.weights, 
							opt_net2.hyperplanes: snf.hyperplanes, 
							opt_net2.input_grads: state.grads,
							opt_net2.step_size: np.ones([m]),
							opt_net2.initial_rnn_state: state.rnn_state,
							opt_net2.state_index: state.counter}
							
			snf_loss, new_point, rnn_state, grads = sess.run([opt_net2.new_snf_loss,
														opt_net2.new_point, 
														opt_net2.rnn_state, 
														opt_net2.grads], 
														feed_dict=feed_dict)
			
			state.loss = snf_loss
			state.point = new_point
			state.rnn_state = rnn_state
			state.grads = grads
			state.counter += 1
			
			replay_memory.append(state)
			states_seq.append(state)

		#===# Train the optimizer #===#
		points = [state.point for state in states_seq]
		snf_losses = [state.loss for state in states_seq]
		grads = [state.point for state in states_seq]
		counters = [state.point for state in states_seq]
		
		feed_dict = {opt_net.points: points, 
						opt_net.snf_losses: snf_losses,
						opt_net.input_grads: grads,
						opt_net.counters: counters,
						opt_net.variances: snf.variances,
						opt_net.weights: snf.weights, 
						opt_net.hyperplanes: snf.hyperplanes, 
						opt_net.step_size: np.ones([m])}
						
		# The RNN state is initialised from a zero-matrix
		loss,_ = sess.run([opt_net.loss, opt_net.train_step], feed_dict=feed_dict)
		losses.append(loss)		
		
		# Synchronize optimizers
		if i & net_sync_freq == 0 and i > 0:
			opt1_vars = [j for j in tf.trainable_variables() if 'opt1' in j.name]
			opt2_vars = [j for j in tf.trainable_variables() if 'opt2' in j.name]
			for v1,v2 in zip(opt1_vars,opt2_vars):
				# The net which generates states copies the variables of the net which is trained.
				v2.assign(v1)
			
		if i % summary_freq == 0 and i > 0:
			print "%d\t%.5f" % (i, np.mean(losses))
			losses = []
			
	# Save model
	if args.save_model:
		vars_to_save = [j for j in tf.trainable_variables() if 'opt1' in j.name]
		saver = tf.train.Saver(vars_to_save)
		saver.save(sess, save_path)
		print "Model saved"

if __name__ == "__main__":
	main()
	
