from __future__ import division

import tensorflow as tf
import numpy as np

import rnn
import rnn_cell
from nn_utils import weight_matrix, bias_vector, fc_layer, fc_layer3, inv_scale_grads
from constants import rnn_size, num_rnn_layers, k, m, rnn_type, grad_scaling_method, \
		episode_length, loss_noise, osc_control, seq_length
import snf
from nn_utils import tf_print


class Optimizer(object):

	def __init__(self):
		# Input
		self.point = tf.placeholder(tf.float32, [m,1], 'points') # Used in training only
		self.variances = tf.placeholder(tf.float32, [k,1], 'variances')
		self.weights = tf.placeholder(tf.float32, [k,1], 'weights')
		self.hyperplanes = tf.placeholder(tf.float32, [m,m,k], 'hyperplanes') # Points which define the hyperplanes
			
		if rnn_type == 'lstm':
			self.initial_rnn_state = tf.placeholder_with_default(input=tf.zeros([m, 2*num_rnn_layers*rnn_size]), shape=[None, 2*num_rnn_layers*rnn_size])
		else:
			# initial_rnn_state is passed during evaluation but not during training
			# each dimension has an independent hidden state, required in order to simulate Adam, RMSProp etc.
			self.initial_rnn_state = tf.placeholder_with_default(input=tf.zeros([m, num_rnn_layers*rnn_size]), shape=[None, num_rnn_layers*rnn_size])

		# The scope allows these variables to be excluded from being reinitialized during the comparison phase
		with tf.variable_scope("optimizer"):
			if rnn_type == 'rnn':
				cell = rnn_cell.BasicRNNCell(rnn_size)
			elif rnn_type == 'gru':
				cell = rnn_cell.GRUCell(rnn_size)
			elif rnn_type == 'lstm':
				cell = rnn_cell.LSTMCell(rnn_size)
				
			self.cell = rnn_cell.MultiRNNCell([cell] * num_rnn_layers)
			
			updates = []
			snf_losses = []
			
			# Arguments passed to the condition and body functions
			time = tf.constant(0)
			point = self.point
			
			snf_loss = snf.calc_snf_loss_tf(point, self.hyperplanes, self.variances, self.weights)
			snf_losses.append(snf_loss)
			snf_grads = snf.calc_grads_tf(snf_loss,point)
			snf_grads = tf.squeeze(snf_grads, [0])
			
			snf_loss_ta = tf.TensorArray(dtype=tf.float32, size=seq_length)
			update_ta = tf.TensorArray(dtype=tf.float32, size=seq_length)
			rnn_state = tf.zeros([m,rnn_size])
			
			loop_vars = [time, point, snf_grads, rnn_state, snf_loss_ta, update_ta, self.hyperplanes, self.variances, self.weights]
			
			def condition(time, point, snf_grads, rnn_state, snf_loss_ta, update_ta, hyperplanes, variances, weights):
				return tf.less(time,seq_length)
				
			def body(time, point, snf_grads, rnn_state, snf_loss_ta, update_ta, hyperplanes, variances, weights):
				
				h, rnn_state_out = self.cell(snf_grads, rnn_state)

				# Final layer of the optimizer
				# Cannot use fc_layer due to a 'must be from the same frame' error
				d = np.sqrt(1.0)/np.sqrt(rnn_size+1) ### should be sqrt(2, 3 or 6?)
				initializer = tf.random_uniform_initializer(-d, d)
				W = tf.get_variable("W", [rnn_size,1], initializer=initializer)
				
				# No bias, linear activation function
				update = tf.matmul(h,W)
				update = tf.reshape(update, [m,1])
				update = inv_scale_grads(update)
				
				new_point = point + update
				
				snf_loss = snf.calc_snf_loss_tf(new_point, hyperplanes, variances, weights)
				snf_losses.append(snf_loss)
				
				snf_loss_ta = snf_loss_ta.write(time, snf_loss)
				update_ta = update_ta.write(time, update)
				
				snf_grads_out = snf.calc_grads_tf(snf_loss,point)
				snf_grads_out = tf.reshape(snf_grads_out,[m,1])
				
				time += 1
				return [time, new_point, snf_grads_out, rnn_state_out, snf_loss_ta, update_ta, hyperplanes, variances, weights]		
			
			# Do the computation
			with tf.variable_scope("o1"):
				res = tf.while_loop(condition, body, loop_vars)
			
			self.new_point = res[1]
			self.rnn_state_out = res[3]		
			losses = res[4].pack()
			updates = res[5].pack()
			
			# Total change in the SNF loss
			# Improvement: 2 - 3 = -1 (small loss)
			snf_loss_change = losses[seq_length - 1] - losses[0]
			snf_loss_change = tf.maximum(snf_loss_change,2*snf_loss_change) # Asymmetric loss
			self.loss_change_sign = tf.sign(snf_loss_change)
			
			# Oscillation cost
			overall_update = tf.zeros([m,1])
			norm_sum = 0.0
				
			for i in range(seq_length):
				overall_update += updates[i,:,:]
				norm_sum += tf_norm(updates[i,:,:])
				
			osc_cost = norm_sum/tf_norm(overall_update)	# > 1
			
			self.total_loss = snf_loss_change*tf.pow(osc_cost,tf.sign(snf_loss_change))
			
			#===# Model training #===#
			#opt = tf.train.RMSPropOptimizer(0.01,momentum=0.5)
			opt = tf.train.AdamOptimizer()
			vars = tf.trainable_variables()
			
			gvs = opt.compute_gradients(self.total_loss, vars)
			
			self.gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for (grad, var) in gvs]

			self.grads_input = [(tf.placeholder(tf.float32, shape=v.get_shape()), v) for (g,v) in gvs]
			self.train_step = opt.apply_gradients(self.grads_input)
			
			#===# Comparison code #===#
			self.input_grads = tf.placeholder(tf.float32, [1,None,1], 'input_grads') ### Remove first dimension?
			input_grads = tf.squeeze(self.input_grads, [0])
			
			with tf.variable_scope("o1", reuse=True) as scope:
				h, self.rnn_state_out_compare = self.cell(input_grads, self.initial_rnn_state)
			
				W = tf.get_variable("W")
				update = tf.matmul(h,W)
				
				update = tf.reshape(update, [-1,1])
				self.update = inv_scale_grads(update)
			
			
	# Update the parameters of another network (eg an MLP)
	def update_params(self, vars, update):
		total = 0
		ret = []

		for i,v in enumerate(vars):
			size = np.prod(list(v.get_shape()))
			size = tf.to_int32(size)
			var_grads = tf.slice(update,begin=[total,0],size=[size,-1])
			var_grads = tf.reshape(var_grads,v.get_shape())
			ret.append(v.assign_add(var_grads))
			size += total		
		return tf.group(*ret)
		
		
def tf_norm(v):
	return tf.sqrt(tf.reduce_sum(tf.square(v)))
		
