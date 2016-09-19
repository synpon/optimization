from __future__ import division

import tensorflow as tf
import numpy as np
import random

from accum_trainer import AccumTrainer
from ac_network import A3CRNN, A3CFF
from snf import SNF, State, StateOps
from constants import local_t_max, entropy_beta, use_rnn, m, discount_rate, \
						termination_prob, max_time_steps, lr_high, lr_low


class A3CTrainingthread(object):
	def __init__(self,
			 thread_index,
			 global_network,
			 learning_rate_input,
			 grad_applier,
			 num_trainable_vars,
			 snf,
			 sess):
			 
		# All ops to be executed in a thread must be defined here since tf.Graph is not thread-safe.
		
		self.thread_index = thread_index
		self.learning_rate_input = learning_rate_input
		self.snf = snf
		self.state_ops = StateOps()
		self.episode_rewards = []
		
		if use_rnn:
			#initializer = tf.random_uniform_initializer(-0.1, 0.1)		
			with tf.variable_scope("model"+str(thread_index), reuse=None):#, initializer=initializer):
				self.local_network = A3CRNN(num_trainable_vars)
		else:
			self.local_network = A3CFF(num_trainable_vars)
			
		self.local_network.prepare_loss(entropy_beta)

		self.trainer = AccumTrainer()
		self.trainer.prepare_minimize(self.local_network.total_loss, self.local_network.trainable_vars)

		self.accum_gradients = self.trainer.accumulate_gradients()
		self.reset_gradients = self.trainer.reset_gradients()
		self.apply_gradients = grad_applier.apply_gradients(
			global_network.trainable_vars,
			self.trainer.get_accum_grad_list() )

		self.sync = self.local_network.sync_from(global_network)
		self.local_t = 0
		
		if not use_rnn:
			self.W = global_network.W1
			
		self.snf = SNF() # Generate a new landscape
		self.state = State(self.snf, self.state_ops, sess) # Generate a new starting point on the landscape


	def _anneal_learning_rate(self, global_time_step):
		t = global_time_step / max_time_steps # Proportion of total time elapsed
		lr = lr_high*t + (1-t)*lr_low
		return lr


	# Not one episode
	def thread(self, sess, global_t):
		states = []
		actions = []
		rewards = []
		values = []
		snf_losses = []
		
		terminal_end = False
			
		# reset accumulated gradients
		sess.run(self.reset_gradients)
		
		# copy weights from shared to local
		sess.run(self.sync)
		
		start_local_t = self.local_t
		
		snf_losses.append(self.snf.calc_loss(self.state.point, self.state_ops, sess))
		
		if use_rnn:
			start_rnn_state = self.local_network.rnn_state_out

		episode_losses = []
		prev_snf_loss = None
		
		for i in range(local_t_max):
			mean,variance,value = self.local_network.run_policy_and_value(sess, self.state, self.snf, self.state_ops)
			
			action = self.snf.choose_action(mean,variance) # Calculate update
			
			states.append(self.state)
			actions.append(action)
			values.append(value)

			# State is the point, action is the update
			snf_loss, next_state = self.snf.act(self.state, action, self.state_ops, sess)
			if prev_snf_loss is None:
				reward = 0
			else:
				reward = np.sign(prev_snf_loss - snf_loss)
			prev_snf_loss = snf_loss
			
			### Consider making the reward the change in loss?
			self.episode_rewards.append(reward)
			rewards.append(reward)
			snf_losses.append(-reward)

			self.local_t += 1
			self.state = next_state

			terminal = random.random() < termination_prob
				
			if terminal:
				terminal_end = True
				self.snf = SNF()
				self.state = State(self.snf,self.state_ops,sess)
				
				#if len(self.episode_rewards) >= 2:
				#	print "Episode reward: %4f\t%4f\t%d" % (self.episode_rewards[-1] - self.episode_rewards[0], np.mean(self.episode_rewards), global_t)
				#else:
				#	print "Episode reward cannot be computed"
					
				self.episode_rewards = []
				if use_rnn:
					self.local_network.reset_rnn_state()
				break

		snf_losses = snf_losses[:-1] # Remove last entry
				
		R = 0.0
		if not terminal_end:
			R = self.local_network.run_value(sess, self.state, self.snf, self.state_ops)

		# Order from the final time point to the first
		actions.reverse()
		states.reverse()
		rewards.reverse()
		values.reverse()
		snf_losses.reverse()
		
		batch_a = []
		batch_grads = []
		batch_td = []
		batch_R = []
		batch_snf_loss = []
		
		assert len(actions) == len(rewards) == len(states) == len(values) == len(snf_losses)
		
		# compute and accumulate gradients
		for (a, r, s, V, snf_loss) in zip(actions, rewards, states, values, snf_losses):
			R = r + discount_rate * R
			td = R - V # temporal difference
			
			batch_a.append(a)
			batch_grads.append(s.grads)
			batch_td.append(td)
			batch_R.append(R)
			batch_snf_loss.append(snf_loss)
			
		if use_rnn:
			batch_a.reverse()
			batch_grads.reverse()
			batch_td.reverse()
			batch_R.reverse()
			batch_snf_loss.reverse()
			
			step_size = len(batch_a)
			batch_grads = np.concatenate(batch_grads, axis=0)
			batch_a = np.concatenate(batch_a, axis=0)
				
			_, loss = sess.run([self.accum_gradients, self.local_network.total_loss], 
								feed_dict = {
									self.local_network.grads: batch_grads,
									self.local_network.a: batch_a,
									self.local_network.td: batch_td,
									self.local_network.r: batch_R,
									self.local_network.snf_loss: batch_snf_loss,
									self.local_network.initial_rnn_state: start_rnn_state,
									self.local_network.step_size: step_size*np.ones([m])})
		else:
			batch_grads = np.concatenate(batch_grads, axis=0)
			batch_a = np.concatenate(batch_a, axis=0)
			
			_, loss = sess.run([self.accum_gradients, self.local_network.total_loss], 
								feed_dict = {
									self.local_network.grads: batch_grads,
									self.local_network.a: batch_a,
									self.local_network.td: batch_td,
									self.local_network.r: batch_R})

		episode_losses.append(loss)
			 
		cur_learning_rate = self._anneal_learning_rate(global_t)

		sess.run(self.apply_gradients, feed_dict = {self.learning_rate_input: cur_learning_rate})

		diff_local_t = self.local_t - start_local_t # Amount to increment global_t by
		
		if len(self.episode_rewards) >= 2:
			episode_reward = self.episode_rewards[-1] - self.episode_rewards[0] # Change in the SNF loss - should be negative
		else:
			episode_reward = 0
			
		return diff_local_t, episode_reward, np.mean(episode_losses)
		
