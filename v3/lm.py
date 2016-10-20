# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
python lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import ptb_reader as reader


class LM(object):

	def __init__(self, config, opt_net):

		self.opt_net = opt_net
		self.batch_size = 1 ### 32
		self.batches = 1000
		###self.num_params = 
	
		self.batch_size = batch_size = config.batch_size ### Must be 1 - alter it in SmallConfig
		self.num_steps = num_steps = config.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size

		self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
		self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

		# Slightly better results can be obtained with forget gate biases
		# initialized to 1 but the hyperparameters of the model would need to be
		# different than reported in the paper.
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
		if config.keep_prob < 1:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
					lstm_cell, output_keep_prob=config.keep_prob)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

		self._initial_state = cell.zero_state(batch_size, tf.float32)

		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [vocab_size, size])
			inputs = tf.nn.embedding_lookup(embedding, self._input_data)

		if config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		inputs = [tf.squeeze(input_, [1])
							  for input_ in tf.split(1, num_steps, inputs)]
		outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

		output = tf.reshape(tf.concat(1, outputs), [-1, size])
		softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
		softmax_b = tf.get_variable("softmax_b", [vocab_size])
		logits = tf.matmul(output, softmax_w) + softmax_b
		loss = tf.nn.seq2seq.sequence_loss_by_example(
				[logits],
				[tf.reshape(self._targets, [-1])],
				[tf.ones([batch_size * num_steps])])
		self._cost = cost = tf.reduce_sum(loss) / batch_size
		self._final_state = state

		self._lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
		
		sgd_optimizer = tf.train.GradientDescentOptimizer(0.1)
		rmsprop_optimizer = tf.train.RMSPropOptimizer(0.001)
		adam_optimizer = tf.train.AdamOptimizer()
		
		#self._train_op = optimizer.apply_gradients(zip(grads, tvars))
		
		grad_var_pairs = sgd_optimizer.compute_gradients(self.loss)
		grad_var_pairs = [i for i in grad_var_pairs if 'mnist/' in i[1].name]
		
		self.sgd_train_step = sgd_optimizer.apply_gradients(grad_var_pairs)
		self.rmsprop_train_step = rmsprop_optimizer.apply_gradients(grad_var_pairs)
		self.adam_train_step = adam_optimizer.apply_gradients(grad_var_pairs)
		
		#===# Opt net #===#
		grads,_ = zip(*grad_var_pairs)
		grads = [tf.reshape(i,(-1,1)) for i in grads]
		
		#grad_clip_value = None
		#if not grad_clip_value is None:
		#	grads = [tf.clip_by_value(g, -grad_clip_value, grad_clip_value) for g in grads]
			
		self.grads = tf.concat(0,grads)
		self.trainable_variables = [i for i in tf.trainable_variables() if 'mnist/' in i.name]		

		self.update = tf.placeholder(tf.float32,[self.num_params,1], 'update')
		self.opt_net_train_step = self.opt_net.update_params(self.trainable_variables, self.update)
		
		vars = [i for i in tf.all_variables() if not 'optimizer' in i.name]
		self.init = tf.initialize_variables(vars)
		

	def assign_lr(self, session, lr_value):
		session.run(tf.assign(self.lr, lr_value))

	@property
	def input_data(self):
		return self._input_data

	@property
	def targets(self):
		return self._targets

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op


class SmallConfig(object):
	"""Small config."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 13
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000


class MediumConfig(object):
	"""Medium config."""
	init_scale = 0.05
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 35
	hidden_size = 650
	max_epoch = 6
	max_max_epoch = 39
	keep_prob = 0.5
	lr_decay = 0.8
	batch_size = 20
	vocab_size = 10000


class LargeConfig(object):
	"""Large config."""
	init_scale = 0.04
	learning_rate = 1.0
	max_grad_norm = 10
	num_layers = 2
	num_steps = 35
	hidden_size = 1500
	max_epoch = 14
	max_max_epoch = 55
	keep_prob = 0.35
	lr_decay = 1 / 1.15
	batch_size = 20
	vocab_size = 10000


class TestConfig(object):
	"""Tiny config, for testing."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 1
	num_layers = 1
	num_steps = 2
	hidden_size = 2
	max_epoch = 1
	max_max_epoch = 1
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000


def run_epoch(session, m, data, eval_op, verbose=False):
	"""Runs the model on the given data."""
	epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = m.initial_state.eval()
	for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
																										m.num_steps)):
		cost, state, _ = session.run([m.cost, m.final_state, eval_op],
																 {m.input_data: x,
																	m.targets: y,
																	m.initial_state: state})
		costs += cost
		iters += m.num_steps

		if verbose and step % (epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" %
						(step * 1.0 / epoch_size, np.exp(costs / iters),
						 iters * m.batch_size / (time.time() - start_time)))

	return np.exp(costs / iters)


def main(_):
	data_path = 'simple-examples/data/'
	raw_data = reader.ptb_raw_data(data_path)
	train_data, _, _, _ = raw_data

	config = SmallConfig()

	with tf.Graph().as_default(), tf.Session() as session:
		tf.initialize_all_variables().run()

		for i in range(config.max_max_epoch):
			lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
			m.assign_lr(session, config.learning_rate * lr_decay)

			train_perplexity = run_epoch(session, m, train_data, m.train_op,
																	 verbose=True)
			print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

if __name__ == "__main__":
	tf.app.run()