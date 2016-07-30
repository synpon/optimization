from __future__ import absolute_import
from __future__ import division

from tensorflow.python.framework import tensor_shape
import rnn_cell
import tensorflow as tf


_is_sequence = rnn_cell._is_sequence
_unpacked_state = rnn_cell._unpacked_state
_packed_state = rnn_cell._packed_state


def rnn(cell, inputs, initial_state,
				sequence_length, scope=None):

	if not isinstance(cell, rnn_cell.RNNCell):
		raise TypeError("cell must be an instance of RNNCell")
	if not isinstance(inputs, list):
		raise TypeError("inputs must be a list")
	if not inputs:
		raise ValueError("inputs must not be empty")

	outputs = []
	# Create a new scope in which the caching device is either
	# determined by the parent scope, or is set to place the cached
	# Variable using the same placement as for the rest of the RNN.
	with tf.variable_scope(scope or "RNN") as varscope:
		if varscope.caching_device is None:
			varscope.set_caching_device(lambda op: op.device)

		(fixed_batch_size, params_size, input_size) = inputs[0].get_shape().with_rank(3)

		batch_size = fixed_batch_size.value

		state = initial_state
		sequence_length = tf.to_int32(sequence_length)

		zero_output = tf.zeros(
				tf.pack([batch_size, params_size, cell.output_size]), inputs[0].dtype)
		zero_output.set_shape(
				tensor_shape.TensorShape([fixed_batch_size.value, params_size, cell.output_size]))
				
		min_sequence_length = tf.reduce_min(sequence_length)
		max_sequence_length = tf.reduce_max(sequence_length)

		for time, input_ in enumerate(inputs):
			if time > 0:
				varscope.reuse_variables()

			call_cell = lambda: cell(input_, state)

			(output, state) = _rnn_step(
					time=time,
					sequence_length=sequence_length,
					min_sequence_length=min_sequence_length,
					max_sequence_length=max_sequence_length,
					zero_output=zero_output,
					state=state,
					call_cell=call_cell,
					state_size=cell.state_size)

			outputs.append(output)

		return (outputs, state)


def _rnn_step(time, sequence_length, min_sequence_length, max_sequence_length,
		zero_output, state, call_cell, state_size, skip_conditionals=False):

	state_is_tuple = _is_sequence(state)
	# Convert state to a list for ease of use
	state = list(_unpacked_state(state)) if state_is_tuple else [state]
	state_shape = [s.get_shape() for s in state]

	def _copy_some_through(new_output, new_state):
		# Use broadcasting select to determine which values should get
		# the previous state & zero output, and which values should get
		# a calculated state & output.
		copy_cond = (time >= sequence_length)

		return ([tf.select(copy_cond, zero_output, new_output)]
						+ [tf.select(copy_cond, old_s, new_s)
							 for (old_s, new_s) in zip(state, new_state)])
							 

	def _maybe_copy_some_through():
		# Run RNN step. Pass through either no or some past state.
		new_output, new_state = call_cell()
		new_state = (
				list(_unpacked_state(new_state)) if state_is_tuple else [new_state])

		if len(state) != len(new_state):
			raise ValueError(
					"Input and output state tuple lengths do not match: %d vs. %d"
					% (len(state), len(new_state)))

		# if t < min_seq_len: calculate and return everything
		# else copy some of it through
		return tf.cond(time < min_sequence_length, lambda: [new_output] + new_state,
						lambda: _copy_some_through(new_output, new_state))
	
	if skip_conditionals:
		# Instead of using conditionals, perform the selective copy at all time
		# steps. This is faster when max_seq_len is equal to the number of unrolls
		# (which is typical for dynamic_rnn).
		new_output, new_state = call_cell()
		new_state = (
				list(_unpacked_state(new_state)) if state_is_tuple else [new_state])

		if len(state) != len(new_state):
			raise ValueError(
					"Input and output state tuple lengths do not match: %d vs. %d"
					% (len(state), len(new_state)))

		final_output_and_state = _copy_some_through(new_output, new_state)
	else:
		empty_update = lambda: [zero_output] + list(state)

		# if t >= max_seq_len: copy all state through, output zeros
		# otherwise calculation is required: copy some or all of it through
		final_output_and_state = tf.cond(time >= max_sequence_length, 
														empty_update,
														_maybe_copy_some_through)

	(final_output, final_state) = (
			final_output_and_state[0], final_output_and_state[1:])

	final_output.set_shape(zero_output.get_shape())
	for final_state_i, state_shape_i in zip(final_state, state_shape):
		final_state_i.set_shape(state_shape_i)

	if state_is_tuple:
		return (
				final_output,
				_packed_state(structure=state_size, state=final_state))
	else:
		return (final_output, final_state[0])


def _reverse_seq(input_seq, lengths):
	if lengths is None:
		return list(reversed(input_seq))

	input_shape = tensor_shape.matrix(None, None)
	for input_ in input_seq:
		input_shape.merge_with(input_.get_shape())
		input_.set_shape(input_shape)

	# Join into (time, batch_size, depth)
	s_joined = tf.pack(input_seq)

	if lengths is not None:
		lengths = tf.to_int64(lengths)

	# Reverse along dimension 0
	s_reversed = tf.reverse_sequence(s_joined, lengths, 0, 1)
	# Split again into list
	result = tf.unpack(s_reversed)
	for r in result:
		r.set_shape(input_shape)
	return result
