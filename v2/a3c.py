#rm nohup.out && python -u a3c.py &

import tensorflow as tf
import numpy as np

import random
import math
import os
import threading
import signal

from ac_network import AC3LSTM, AC3FF
from a3c_training_thread import A3CTrainingthread
from rmsprop_applier import RMSPropApplier

from constants import num_threads, initial_alpha_low, \
	initial_alpha_high, initial_alpha_log_rate, max_time_steps, \
	checkpoint_dir, log_file, rmsp_epsilon, rmsp_alpha, grad_norm_clip, \
	use_lstm, summary_freq


def log_uniform(low, high, rate):
	log_low = math.log(low)
	log_high = math.log(high)
	v = log_low * (1-rate) + log_high * rate
	return math.exp(v)

	
# Stops memory leaks from unterminated threads
# along with signal.pause() and signal.signal(...)
def signal_handler(signal, frame): ### How is frame used?
	global stop_requested
	print('Requesting stop')
	stop_requested = True
	
	
def train_function(parallel_index):
	global global_t
	global graph
	train_thread_class = train_thread_classes[parallel_index]
	discounted_rewards = []
	count = 0 # Unlike global_t, count is thread-specific

	while True:
		count += 1
		if stop_requested:
			break
		
		if global_t > max_time_steps:
			print "thread %d reached max steps" % parallel_index
			break
		
		diff_global_t, reward = train_thread_class.thread(sess, global_t)
		discounted_rewards.append(reward)
		global_t += diff_global_t
		
		if count % summary_freq == 0:
			print "Reward: ", float(np.mean(discounted_rewards))
			discounted_rewards = []

# Globals
stop_requested = False	
global_t = 0
num_trainable_vars = [None]
graph = tf.Graph()			
			
with graph.as_default(), tf.Session() as sess:

	if use_lstm:
		global_network = AC3LSTM(num_trainable_vars)
	else:
		global_network = AC3FF(num_trainable_vars)
		
	initial_learning_rate = log_uniform(initial_alpha_low, initial_alpha_high, initial_alpha_log_rate)
	learning_rate_input = tf.placeholder("float")

	grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
									decay = rmsp_alpha,
									momentum = 0.0,
									epsilon = rmsp_epsilon,
									clip_norm = grad_norm_clip)
				
	train_threads = []
	train_thread_classes = []

	for i in range(num_threads):
		train_thread_class = A3CTrainingthread(sess, i, global_network, initial_learning_rate,
											learning_rate_input, grad_applier, max_time_steps, num_trainable_vars)
											
		train_thread_classes.append(train_thread_class)
		train_threads.append(threading.Thread(target=train_function, args=(i,)))
		
	init = tf.initialize_all_variables()
	sess.run(init)

	signal.signal(signal.SIGINT, signal_handler)
		
	for t in train_threads:
		t.start()
		
	signal.pause()
		
	for t in train_threads:
		t.join()


