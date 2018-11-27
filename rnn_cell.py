# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import ops
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

from . import rnn_cell_impl

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def _get_concat_variable(name, shape, dtype, num_shards):
  """Get a sharded variable concatenated into one tensor."""
  sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
  if len(sharded_variable) == 1:
    return sharded_variable[0]

  concat_name = name + "/concat"
  concat_full_name = vs.get_variable_scope().name + "/" + concat_name + ":0"
  for value in ops.get_collection(ops.GraphKeys.CONCATENATED_VARIABLES):
    if value.name == concat_full_name:
      return value

  concat_variable = array_ops.concat(sharded_variable, 0, name=concat_name)
  ops.add_to_collection(ops.GraphKeys.CONCATENATED_VARIABLES, concat_variable)
  return concat_variable


def _get_sharded_variable(name, shape, dtype, num_shards):
  """Get a list of sharded variables with the given dtype."""
  if num_shards > shape[0]:
    raise ValueError("Too many shards: shape=%s, num_shards=%d" % (shape,
                                                                   num_shards))
  unit_shard_size = int(math.floor(shape[0] / num_shards))
  remaining_rows = shape[0] - unit_shard_size * num_shards

  shards = []
  for i in range(num_shards):
    current_size = unit_shard_size
    if i < remaining_rows:
      current_size += 1
    shards.append(
        vs.get_variable(
            name + "_%d" % i, [current_size] + shape[1:], dtype=dtype))
  return shards


def _norm(g, b, inp, scope):
  shape = inp.get_shape()[-1:]
  gamma_init = init_ops.constant_initializer(g)
  beta_init = init_ops.constant_initializer(b)
  with vs.variable_scope(scope):
    # Initialize beta and gamma for use by layer_norm.
    vs.get_variable("gamma", shape=shape, initializer=gamma_init)
    vs.get_variable("beta", shape=shape, initializer=beta_init)
  normalized = layers.layer_norm(inp, reuse=True, scope=scope)
  return normalized


# pylint: disable=protected-access
_Linear = core_rnn_cell._Linear  # pylint: disable=invalid-name


_REGISTERED_OPS = None


class IndyGRUCell(rnn_cell_impl.LayerRNNCell):
  r"""Independently Gated Recurrent Unit cell.

  Based on IndRNNs (https://arxiv.org/abs/1803.04831) and similar to GRUCell,
  yet with the \(U_r\), \(U_z\), and \(U\) matrices in equations 5, 6, and
  8 of http://arxiv.org/abs/1406.1078 respectively replaced by diagonal
  matrices, i.e. a Hadamard product with a single vector:

    $$r_j = \sigma\left([\mathbf W_r\mathbf x]_j +
      [\mathbf u_r\circ \mathbf h_{(t-1)}]_j\right)$$
    $$z_j = \sigma\left([\mathbf W_z\mathbf x]_j +
      [\mathbf u_z\circ \mathbf h_{(t-1)}]_j\right)$$
    $$\tilde{h}^{(t)}_j = \phi\left([\mathbf W \mathbf x]_j +
      [\mathbf u \circ \mathbf r \circ \mathbf h_{(t-1)}]_j\right)$$

  where \(\circ\) denotes the Hadamard operator. This means that each IndyGRU
  node sees only its own state, as opposed to seeing all states in the same
  layer.

  TODO(gonnet): Write a paper describing this and add a reference here.

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight
      matrices applied to the input.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               dtype=None):
    super(IndyGRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError(
          "Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

    input_depth = inputs_shape[1].value
    # pylint: disable=protected-access
    self._gate_kernel_w = self.add_variable(
        "gates/%s_w" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
        shape=[input_depth, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_kernel_u = self.add_variable(
        "gates/%s_u" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
        shape=[1, 2 * self._num_units],
        initializer=init_ops.random_uniform_initializer(
            minval=-1, maxval=1, dtype=self.dtype))
    self._gate_bias = self.add_variable(
        "gates/%s" % rnn_cell_impl._BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(self._bias_initializer
                     if self._bias_initializer is not None else
                     init_ops.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel_w = self.add_variable(
        "candidate/%s" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
        shape=[input_depth, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_kernel_u = self.add_variable(
        "candidate/%s_u" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
        shape=[1, self._num_units],
        initializer=init_ops.random_uniform_initializer(
            minval=-1, maxval=1, dtype=self.dtype))
    self._candidate_bias = self.add_variable(
        "candidate/%s" % rnn_cell_impl._BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(self._bias_initializer
                     if self._bias_initializer is not None else
                     init_ops.zeros_initializer(dtype=self.dtype)))
    # pylint: enable=protected-access

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""

    gate_inputs = math_ops.matmul(inputs, self._gate_kernel_w) + (
        gen_array_ops.tile(state, [1, 2]) * self._gate_kernel_u)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = math_ops.matmul(inputs, self._candidate_kernel_w) + (
        r_state * self._candidate_kernel_u)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    new_h = u * state + (1 - u) * c
    return new_h, new_h
