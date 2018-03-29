from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.contrib.distributions import RelaxedBernoulli

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))

class LSTMStateTuple(_LSTMStateTuple):
    """ Stores c=hidden state and h=output """
    __slots__ = ()

    def dtype(self):
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
        return c.dtype

class _LayerRNNCell(RNNCell):
    def __call__(self, inputs, state, scope=None, *args, **kwargs):
        return base_layer.Layer.__call__(self, inputs, state, scope=scope, 
                *args, **kwargs)

class BasicLSTMCell(_LayerRNNCell):

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True):
        super(BasicLSTMCell, self).__init__()

        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = tanh

    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units))
        
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
    	if inputs_shape[1].value is None:
      		raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    	input_depth = inputs_shape[1].value
    	h_depth = self._num_units
    	self._kernel = self.add_variable(
        	_WEIGHTS_VARIABLE_NAME,
        	shape=[input_depth + h_depth, 4 * self._num_units])
    	self._bias = self.add_variable(
        	_BIAS_VARIABLE_NAME,
        	shape=[4 * self._num_units],
        	initializer=init_ops.zeros_initializer(dtype=self.dtype))

    	self.built = True

    def call(self, inputs, state):
        # Gumbel Softmax
        rb = RelaxedBernoulli(probs=[0.5])

        one = constant_op.constant(1, dtype=dtypes.int32)
        c, h = state
        gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)
        print(f)
        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)

        new_c = c * sigmoid(f + forget_bias_tensor) + sigmoid(i) * self._activation(j)
        new_h = self._activation(new_c) * sigmoid(o)

        new_state = LSTMStateTuple(new_c, new_h)

        return new_h, new_state


