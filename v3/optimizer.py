from __future__ import division

import tensorflow as tf
import numpy as np

import rnn
import rnn_cell
from nn_utils import weight_matrix, bias_vector, fc_layer, fc_layer3, inv_scale_grads
from constants import rnn_size, num_rnn_layers, k, m, rnn_type, grad_scaling_method, \
		episode_length, loss_noise, osc_control
import snf
from nn_utils import tf_print


class Optimizer(object):

	def __init__(self, seq_length,scope_name):
		# Input
		self.point = tf.placeholder(tf.float32, [m,1], 'points') # Used in training only
		self.variances = tf.placeholder(tf.float32, [k,1], 'variances')
		self.weights = tf.placeholder(tf.float32, [k,1], 'weights')
		self.hyperplanes = tf.placeholder(tf.float32, [m,m,k], 'hyperplanes') # Points which define the hyperplanes
			
		if rnn_type == 'lstm':
			self.initial_rnn_state = tf.placeholder_with_default(input=tf.zeros([m, 2*num_rnn_layers*rnn_size]), shape=[None, 2*num_rnn_layers*rnn_size])
		else:
			# initial_rnn_state is given during evaluation but not during training
			# each dimension has an independent hidden state, required in order to simulate Adam, RMSProp etc.
			self.initial_rnn_state = tf.placeholder_with_default(input=tf.zeros([m, num_rnn_layers*rnn_size]), shape=[None, num_rnn_layers*rnn_size])

		# The scope allows these variables to be excluded from being reinitialized during the comparison phase
		with tf.variable_scope("a3c"): ### rename scope
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
			time = tf.constant(0.0) ### check incrementing works
			point = self.point
			
			snf_loss = snf.calc_snf_loss_tf(point, self.hyperplanes, self.variances, self.weights)
			snf_losses.append(snf_loss)
			snf_grads = snf.calc_grads_tf(snf_loss,point)
			snf_grads = tf.squeeze(snf_grads, [0])
			
			loss = tf.constant(0.0)
			
			rnn_state = tf.zeros([m,rnn_size])
			loop_vars = [time, point, snf_grads, rnn_state, loss]
			
			def condition(time, point, snf_grads, rnn_state, loss):
				return tf.less(time,seq_length)
				
			# Function defined inside __init__ in order to avoid passing hyperplanes, variances etc. as arguments
			def body(time, point, snf_grads, rnn_state, loss):
				time += 1

				h, rnn_state_out = self.cell(snf_grads, rnn_state)
	
				# Final layer of the optimizer
				update = fc_layer(h, num_in=rnn_size, num_out=1, activation_fn=None, bias=False)
				update = tf.reshape(update, [m,1])
				#update.set_shape()
				
				new_point = point + update
				
				snf_loss = snf.calc_snf_loss_tf(new_point, self.hyperplanes, self.variances, self.weights)
				snf_losses.append(snf_loss)
				loss += snf_loss
				
				snf_grads_out = snf.calc_grads_tf(snf_loss,point)
				snf_grads_out = tf.reshape(snf_grads_out,[m,1])
				
				return [time, new_point, snf_grads_out, rnn_state_out, loss]
			
			# Total change in the SNF loss
			#self.total_loss = 0 ###			
			
			# Do the computation
			res = tf.while_loop(condition, body, loop_vars)
			self.rnn_state_out = res[3]
			self.total_loss = res[4]
			
			# Oscillation cost
			overall_update = tf.zeros([m])
			norm_sum = 0
				
			for i in range(len(updates)):
				overall_update += updates[i]
				norm_sum += tf_norm(updates[i])
				
			osc_cost = tf_norm(overall_update)/norm_sum
				
			#self.total_loss += osc_cost
				
			#===# SNF outputs #===#
			# Used when filling the replay memory during training
			# Indexing is admissible here as these 3 variables only need to be returned when seq_length = 1
			#self.snf_losses_output = snf_losses_output[0]
			#self.points_output = points_output[0]
			#self.grads_output = grads_output[0]
			
			#===# Model training #===#
			opt = tf.train.RMSPropOptimizer(0.01,momentum=0.5)
			vars = [i for i in tf.trainable_variables() if scope_name in i.name] ### could be unreliable in the future
			
			gvs = opt.compute_gradients(self.total_loss, vars) ###
			print gvs
			self.gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for (grad, var) in gvs]

			self.grads_input = [(tf.placeholder(tf.float32, shape=v.get_shape()), v) for (g,v) in gvs]
			self.train_step = opt.apply_gradients(self.grads_input)

			
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
		
