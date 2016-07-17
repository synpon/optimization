from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import six

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.platform import tf_logging as logging


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
		state_size = self.state_size
		if _is_sequence(state_size):
			state_size_flat = _unpacked_state(state_size)
			zeros_flat = [
					array_ops.zeros(array_ops.pack([batch_size, s]), dtype=dtype)
					for s in state_size_flat]
			for s, z in zip(state_size_flat, zeros_flat):
				z.set_shape([None, s])
			zeros = _packed_state(structure=state_size, state=zeros_flat)
		else:
			zeros = array_ops.zeros(
					array_ops.pack([batch_size, state_size]), dtype=dtype)
			zeros.set_shape([None, state_size])

		return zeros


class BasicRNNCell(RNNCell):
	def __init__(self, num_units, activation=tanh):
		self._num_units = num_units
		self._activation = activation

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope=None):
		with vs.variable_scope(scope or type(self).__name__):	# "BasicRNNCell"
			output = self._activation(_linear([inputs, state], self._num_units, True))
		return output, output


class GRUCell(RNNCell):
	def __init__(self, num_units, activation=tanh):
		self._num_units = num_units
		self._activation = activation

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope=None):
		with vs.variable_scope(scope or type(self).__name__):	# "GRUCell"
			with vs.variable_scope("Gates"):	# Reset gate and update gate.
				# We start with bias of 1.0 to not reset and not update.
				r, u = array_ops.split(1, 2, _linear([inputs, state],2 * self._num_units, True, 1.0))
				r, u = sigmoid(r), sigmoid(u)
			with vs.variable_scope("Candidate"):
				c = self._activation(_linear([inputs, r * state], self._num_units, True))
			new_h = u * state + (1 - u) * c
		return new_h, new_h


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
	__slots__ = ()


class BasicLSTMCell(RNNCell):
	def __init__(self, num_units, forget_bias=1.0, state_is_tuple=False, activation=tanh):
		if not state_is_tuple:
			logging.warn(
					"%s: Using a concatenated state is slower and will soon be "
					"deprecated.	Use state_is_tuple=True." % self)

		self._num_units = num_units
		self._forget_bias = forget_bias
		self._state_is_tuple = state_is_tuple
		self._activation = activation

	@property
	def state_size(self):
		return (LSTMStateTuple(self._num_units, self._num_units)
						if self._state_is_tuple else 2 * self._num_units)

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope=None):
		with vs.variable_scope(scope or type(self).__name__):	# "BasicLSTMCell"
			# Parameters of gates are concatenated into one multiply for efficiency.
			if self._state_is_tuple:
				c, h = state
			else:
				c, h = array_ops.split(1, 2, state)
			concat = _linear([inputs, h], 4 * self._num_units, True)

			# i = input_gate, j = new_input, f = forget_gate, o = output_gate
			i, j, f, o = array_ops.split(1, 4, concat)

			new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
							 self._activation(j))
			new_h = self._activation(new_c) * sigmoid(o)

			if self._state_is_tuple:
				new_state = LSTMStateTuple(new_c, new_h)
			else:
				new_state = array_ops.concat(1, [new_c, new_h])
			return new_h, new_state


class MultiRNNCell(RNNCell):
	def __init__(self, cells, state_is_tuple=False):
		if not cells:
			raise ValueError("Must specify at least one cell for MultiRNNCell.")
		self._cells = cells
		self._state_is_tuple = state_is_tuple
		if not state_is_tuple:
			if any(_is_sequence(c.state_size) for c in self._cells):
				raise ValueError("Some cells return tuples of states, but the flag "
												 "state_is_tuple is not set.	State sizes are: %s"
												 % str([c.state_size for c in self._cells]))

	@property
	def state_size(self):
		if self._state_is_tuple:
			return tuple(cell.state_size for cell in self._cells)
		else:
			return sum([cell.state_size for cell in self._cells])

	@property
	def output_size(self):
		return self._cells[-1].output_size

	def __call__(self, inputs, state, scope=None):
		with vs.variable_scope(scope or type(self).__name__):	# "MultiRNNCell"
			cur_state_pos = 0
			cur_inp = inputs
			new_states = []
			for i, cell in enumerate(self._cells):
				with vs.variable_scope("Cell%d" % i):
					if self._state_is_tuple:
						if not _is_sequence(state):
							raise ValueError(
									"Expected state to be a tuple of length %d, but received: %s"
									% (len(self.state_size), state))
						cur_state = state[i]
					else:
						cur_state = array_ops.slice(
								state, [0, cur_state_pos], [-1, cell.state_size])
						cur_state_pos += cell.state_size
					cur_inp, new_state = cell(cur_inp, cur_state)
					new_states.append(new_state)
		new_states = (tuple(new_states) if self._state_is_tuple
									else array_ops.concat(1, new_states))
		return cur_inp, new_states


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
	if args is None or (_is_sequence(args) and not args):
		raise ValueError("`args` must be specified")
	if not _is_sequence(args):
		args = [args]

	# Calculate the total size of arguments on dimension 1.
	total_arg_size = 0
	shapes = [a.get_shape().as_list() for a in args]
	for shape in shapes:
		if len(shape) != 2:
			raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
		if not shape[1]:
			raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
		else:
			total_arg_size += shape[1]

	# Now the computation.
	with vs.variable_scope(scope or "Linear"):
		matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
		if len(args) == 1:
			res = math_ops.matmul(args[0], matrix)
		else:
			res = math_ops.matmul(array_ops.concat(1, args), matrix)
		if not bias:
			return res
		bias_term = vs.get_variable(
				"Bias", [output_size],
				initializer=init_ops.constant_initializer(bias_start))
	return res + bias_term