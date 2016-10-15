from __future__ import division
import argparse
import random

import tensorflow as tf
import numpy as np

from constants import num_iterations, seq_length, save_path, summary_freq, \
    episode_length, replay_mem_start_size, replay_memory_max_size, \
	num_SNFs, num_rnn_layers, rnn_size, m, net_sync_freq, batch_size, save_freq
from snf import SNF, State, StateOps
from optimizer import Optimizer

"""
rm nohup.out; nohup python -u main.py -s &

pyflakes main.py compare.py optimizer.py constants.py snf.py nn_utils.py mlp.py
pylint --rcfile=pylint.cfg main.py compare.py optimizer.py constants.py snf.py nn_utils.py mlp.py
"""

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
		opt_net = Optimizer("opt1")
	
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
	
	best_loss = np.float('inf')

	# Training loop
	for i in range(num_iterations):
		batch_losses = []
		batch_snf_losses = []
		batch_grads = []
		batch_counters = []
		
		for j in range(batch_size):
			# Retrieve a random starting point from the replay memory
			state = random.choice(replay_memory)
			snf = state.snf
			
			if state.counter >= episode_length:
				snf = random.choice(snfs)
				state = State(snf, state_ops, sess)
				
			### The RNN state is initialised from a zero-matrix?
			
			feed_dict = {opt_net.point: state.point,
							opt_net.variances: snf.variances, 
							opt_net.weights: snf.weights, 
							opt_net.hyperplanes: snf.hyperplanes,
							opt_net.initial_rnn_state: state.rnn_state}
			
			res = sess.run([opt_net.new_point,
							opt_net.rnn_state_out,
							opt_net.snf_loss_change,
							opt_net.total_loss]
							+ [g for g,v in opt_net.gvs], 
							feed_dict=feed_dict)
														
			new_point, rnn_state_out, snf_loss_change, loss = res[0:4]
			grads_out = res[4:]
			
			# Prepare a new state to add to the replay memory
			state = State(snf, state_ops, sess)
			state.point = new_point
			state.rnn_state = rnn_state_out
			
			# Prevent these attributes from being used until their values are overridden
			state.loss = None
			state.grads = None
			
			state.counter += seq_length
				
			# Only the last state is added. Adding more may result in a loss 
			# of diversity in the replay memory
			replay_memory.append(state)
			
			if len(replay_memory) > replay_memory_max_size:
				replay_memory = replay_memory[-replay_memory_max_size:]
			
			batch_counters.append(state.counter)
			batch_losses.append(loss)
			batch_snf_losses.append(snf_loss_change)
			batch_grads.append(grads_out)
		
		loss = np.mean(batch_losses)
		sign_loss = np.mean(np.sign(batch_snf_losses))
		avg_counter = np.mean(batch_counters)
		avg_loss_change_sign = np.mean(batch_snf_losses)

		total_grads = batch_grads[0]
		
		for j in range(1,batch_size):
			for k in range(len(batch_grads[j])):
				total_grads[k] += batch_grads[j][k]
		
		total_grads = [j/batch_size for j in total_grads]

		#===# Train the optimizer #===#	
		# By the derivative sum rule, the average of the derivatives (calculated here)
		# is identical to the derivative of the average (the usual method).
		feed_dict = {}
		for j in range(len(opt_net.grads_input)):
			feed_dict[opt_net.grads_input[j][0]] = total_grads[j]
		
		_ = sess.run([opt_net.train_step], feed_dict=feed_dict)
			
		if i % summary_freq == 0 and i > 0:
			print "{:>3}{:>10.3}{:>10.3}{:>10.3}".format(i, loss, avg_loss_change_sign, avg_counter)
			
		# Save model
		if loss < best_loss:
			best_loss = loss
			if args.save_model:
				vars_to_save = [j for j in tf.trainable_variables() if 'opt1' in j.name]
				saver = tf.train.Saver(vars_to_save)
				saver.save(sess, save_path)
				print "{:>3}{:>10.3}{:>10.3}{:>10.3} (S)".format(i, loss, avg_loss_change_sign, avg_counter)

if __name__ == "__main__":
	main()
	
