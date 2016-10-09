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
		opt_net = Optimizer(seq_length,"opt1")
		
	with tf.variable_scope("opt2"):
		opt_net2 = Optimizer(1,"opt2") # Only used for generating new states
	
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

	# Training loop
	for i in range(num_iterations):
		batch_losses = []
		batch_grads = []
		batch_counters = []
		
		for j in range(batch_size):
			# Retrieve a starting point from the replay memory
			state = random.choice(replay_memory)
			snf = state.snf
			
			if state.counter >= episode_length:
				snf = random.choice(snfs)
				state = State(snf, state_ops, sess)
			
			#===# Generate states #===#
			states_seq = []
			
			# For loop is necessary since all the points are taken as input, not computed internally
			
			for k in range(seq_length):	
				feed_dict = {opt_net2.points: [state.point], 
								opt_net2.snf_losses: [state.loss],
								opt_net2.variances: snf.variances, 
								opt_net2.weights: snf.weights, 
								opt_net2.hyperplanes: snf.hyperplanes, 
								opt_net2.input_grads: state.grads,
								opt_net2.initial_rnn_state: state.rnn_state}
				
				# rnn_state is omitted - no need to fix this
				snf_loss, new_point, grads = sess.run([opt_net2.snf_losses_output,
															opt_net2.points_output,
															opt_net2.grads_output], 
															feed_dict=feed_dict)
				
				state.loss = snf_loss
				state.point = new_point
				state.grads = grads
				state.counter += 1
				
				states_seq.append(state)
				
			# Only the last state is added. Adding more may result in a loss 
			# of diversity in the replay memory
			replay_memory.append(states_seq[-1])
			
			if len(replay_memory) > replay_memory_max_size:
				replay_memory = replay_memory[-replay_memory_max_size:]
				
			#===# Calculate loss for the optimizer #===#
			# The RNN state is initialised from a zero-matrix
			points = [state.point for state in states_seq]
			snf_losses = [state.loss for state in states_seq]
			grads = [np.reshape(state.grads,[m,1]) for state in states_seq]
			counters = [state.counter for state in states_seq]
			
			feed_dict = {opt_net.points: points, 
							opt_net.snf_losses: snf_losses,
							opt_net.input_grads: grads,
							opt_net.counters: counters,
							opt_net.variances: snf.variances,
							opt_net.weights: snf.weights, 
							opt_net.hyperplanes: snf.hyperplanes}
									
			res = sess.run([opt_net.total_loss] + [g for g,v in opt_net.gvs], feed_dict=feed_dict)
			loss = res[0]
			grads_out = res[1:]
			
			batch_counters.append(np.mean(counters))
			batch_losses.append(loss)
			batch_grads.append(grads_out)
		
		loss = np.mean(batch_losses)
		sign_loss = np.mean(np.sign(batch_losses))
		avg_counter = np.mean(batch_counters)

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
		
		# Synchronize optimizers
		if i & net_sync_freq == 0 and i > 0:
			opt1_vars = [j for j in tf.trainable_variables() if 'opt1' in j.name]
			opt2_vars = [j for j in tf.trainable_variables() if 'opt2' in j.name]
			for v1,v2 in zip(opt1_vars,opt2_vars):
				# The net which generates states copies the variables of the net which is trained.
				v2.assign(v1)
			
		if i % summary_freq == 0 and i > 0:
			print "{:>3}{:>10.3}{:>10.3}{:>10.3}".format(i, loss, sign_loss, avg_counter)
			
		# Save model
		if i % save_freq == 0 and i > 0:
			if args.save_model:
				vars_to_save = [j for j in tf.trainable_variables() if 'opt1' in j.name]
				saver = tf.train.Saver(vars_to_save)
				saver.save(sess, save_path)
				print "Model saved"

if __name__ == "__main__":
	main()
	
