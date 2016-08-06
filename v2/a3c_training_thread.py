import tensorflow as tf
import numpy as np
import random

from accum_trainer import AccumTrainer
from ac_network import AC3LSTM, AC3FF
from gmm import GMM

from constants import discount_rate, local_t_max, entropy_beta, use_lstm, m

class A3CTrainingthread(object):
	def __init__(self,
			 sess,
			 thread_index,
			 global_network,
			 initial_learning_rate,
			 learning_rate_input,
			 grad_applier,
			 max_global_time_step,
			 num_trainable_vars):

		self.thread_index = thread_index
		self.learning_rate_input = learning_rate_input
		self.max_global_time_step = max_global_time_step	
		
		if use_lstm:
			initializer = tf.random_uniform_initializer(-0.1, 0.1)		
			with tf.variable_scope("model"+str(thread_index), reuse=None, initializer=initializer):
				self.local_network = AC3LSTM(num_trainable_vars)
		else:
			self.local_network = AC3FF(num_trainable_vars)
			
		self.local_network.prepare_loss(entropy_beta)

		self.trainer = AccumTrainer()
		self.trainer.prepare_minimize(self.local_network.total_loss, self.local_network.trainable_vars)

		self.accum_gradients = self.trainer.accumulate_gradients() ###
		self.reset_gradients = self.trainer.reset_gradients()
	
		self.apply_gradients = grad_applier.apply_gradients(
			global_network.trainable_vars,
			self.trainer.get_accum_grad_list() )

		self.sync = self.local_network.sync_from(global_network)
		self.local_t = 0
		self.initial_learning_rate = initial_learning_rate
		self.episode_reward = 0


	def _anneal_learning_rate(self, global_time_step):
		learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
		if learning_rate < 0:
			learning_rate = 0
		return learning_rate


	# Run for one episode
	def thread(self, sess, global_t):
		states = []
		actions = []
		rewards = []
		values = []
		
		if use_lstm:
			self.local_network.reset_state(1,m)
			
		# reset accumulated gradients
		sess.run(self.reset_gradients)
		# copy weights from shared to local
		sess.run(self.sync)
		start_local_t = self.local_t
	
		gmm = GMM() # Generate a landscape
		state = gmm.gen_point() # The state is a point in the landscape
		
		discounted_reward = 0
		
		for i in range(local_t_max):
			if use_lstm:
				mean,variance = self.local_network.run_policy(sess, state, update_rnn_state=True)
			else:
				mean,variance = self.local_network.run_policy(sess, state)

			action = gmm.choose_action(mean,variance) # Calculate update
			states.append(state)
			actions.append(action)
			
			if use_lstm:
				# Do not update the state again
				value_ = self.local_network.run_value(sess, state, update_rnn_state=False)
			else:
				value_ = self.local_network.run_value(sess, state)
				
			values.append(value_)

			# State is the point, action is the update
			reward, next_state = gmm.act(state,action)
			self.episode_reward += reward

			rewards.append(reward)

			self.local_t += 1
			state = next_state
			
		discounted_reward = (discount_rate**i)*self.episode_reward ### What triggered terminal previously?
		self.episode_reward = 0

		R = 0.0 ### Necessary?

		if use_lstm:
			# Do not update the state again
			R = self.local_network.run_value(sess, state, update_rnn_state=False) 
		else:
			R = self.local_network.run_value(sess, state) 

		# Order from the final time point to the first
		actions.reverse()
		states.reverse()
		rewards.reverse()
		values.reverse()

		# compute and accumulate gradients
		for (a, r, state, V) in zip(actions, rewards, states, values):
			R = r + discount_rate * R
			td = R - V # temporal difference

			sess.run(self.accum_gradients,
								feed_dict = {
									self.local_network.state: state,
									self.local_network.a: a,
									self.local_network.td: td,
									self.local_network.r: R})
			
		cur_learning_rate = self._anneal_learning_rate(global_t)

		sess.run(self.apply_gradients, feed_dict = {self.learning_rate_input: cur_learning_rate})

		# local step
		diff_local_t = self.local_t - start_local_t
		return diff_local_t, discounted_reward
		