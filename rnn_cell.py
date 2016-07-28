from __future__ import absolute_import
from __future__ import division

import collections
import six
import tensorflow as tf


def _is_sequence(seq):
	return (isinstance(seq, collections.Sequence)
					and not isinstance(seq, six.string_types))


def _sequence_like(instance, args):
	try:
		assert isinstance(instance, tuple)
		assert isinstance(instance._fields, collections.Sequence)
		assert all(isinstance(f, six.string_types) for f in instance._fields)
		# This is a namedtuple
		return type(instance)(*args)
	except (AssertionError, AttributeError):
		# Not a namedtuple
		return type(instance)(args)


def _packed_state_with_indices(structure, flat, index):
	packed = []
	for s in structure:
		if _is_sequence(s):
			new_index, child = _packed_state_with_indices(s, flat, index)
			packed.append(_sequence_like(s, child))
			index = new_index
		else:
			packed.append(flat[index])
			index += 1
	return (index, packed)


def _yield_unpacked_state(state):
	for s in state:
		if _is_sequence(s):
			for si in _yield_unpacked_state(s):
				yield si
		else:
			yield s


def _unpacked_state(state):
	if not _is_sequence(state):
		raise TypeError("state must be a sequence")
	return list(_yield_unpacked_state(state))


def _packed_state(structure, state):
	if not _is_sequence(structure):
		raise TypeError("structure must be a sequence")
	if not _is_sequence(state):
		raise TypeError("state must be a sequence")

	flat_structure = _unpacked_state(structure)
	if len(flat_structure) != len(state):
		raise ValueError(
				"Internal error: Could not pack state.	Structure had %d elements, but "
				"state had %d elements.	Structure: %s, state: %s."
				% (len(flat_structure), len(state), structure, state))

	(_, packed) = _packed_state_with_indices(structure, state, 0)
	return _sequence_like(structure, packed)


class RNNCell(object):

	def __call__(self, inputs, state, scope=None):
		raise NotImplementedError("Abstract method")

	@property
	def state_size(self):
		raise NotImplementedError("Abstract method")

	@property
	def output_size(self):
		raise NotImplementedError("Abstract method")

	def zero_state(self, batch_size, dtype):
		raise NotImplementedError


class BasicRNNCell(RNNCell):
	def __init__(self, num_units, activation=tf.nn.tanh):
		self._num_units = num_units
		self._activation = activation

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):	# "BasicRNNCell"
			output = self._activation(_linear([inputs, state], self._num_units, True))
		return output, output


class GRUCell(RNNCell):
	def __init__(self, num_units, activation=tf.nn.tanh):
		self._num_units = num_units
		self._activation = activation

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):	# "GRUCell"
			with tf.variable_scope("Gates"): 	# Reset gate and update gate.
				# We start with bias of 1.0 to not reset and not update.
				r, u = tf.split(2, 2, _linear([inputs, state],2 * self._num_units, True, 1.0))
				r, u = tf.nn.sigmoid(r), tf.nn.sigmoid(u)
			with tf.variable_scope("Candidate"):
				c = self._activation(_linear([inputs, r * state], self._num_units, True))
			new_h = u * state + (1 - u) * c
		return new_h, new_h


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
	__slots__ = ()


class BasicLSTMCell(RNNCell):
	def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
		self._num_units = num_units
		self._forget_bias = forget_bias
		self._activation = activation

	@property
	def state_size(self):
		return (LSTMStateTuple(self._num_units, self._num_units))

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):	# "BasicLSTMCell"
			# Parameters of gates are concatenated into one multiply for efficiency.
			c, h = state
			
			concat = _linear([inputs, h], 4 * self._num_units, True)

			# i = input_gate, j = new_input, f = forget_gate, o = output_gate
			i, j, f, o = tf.split(2, 4, concat)

			new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
							 self._activation(j))
			new_h = self._activation(new_c) * tf.nn.sigmoid(o)

			new_state = LSTMStateTuple(new_c, new_h)
			return new_h, new_state


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
	if args is None or (_is_sequence(args) and not args):
		raise ValueError("`args` must be specified")
	if not _is_sequence(args):
		args = [args]

	# Calculate the total size of arguments on dimension 2.
	total_arg_size = 0
	shapes = [a.get_shape().as_list() for a in args]
	for shape in shapes:
		if len(shape) != 3:
			raise ValueError("Linear is expecting 3D arguments: %s" % str(shapes))
		else:
			total_arg_size += shape[2]

	# Now the computation.
	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [1,total_arg_size, output_size])
		# Batch size is 1
		res = tf.batch_matmul(tf.concat(2, args), matrix)
		
		if not bias:
			return res
		bias_term = tf.get_variable(
				"Bias", [output_size],
				initializer=tf.constant_initializer(bias_start))
	return res + bias_term
	
	