import copy
import pdb

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
import numpy as np

import os

import tensorflow.contrib as tfc
import tensorflow_probability as tfp
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import random_ops


def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = tf.max(x, axis=axis, keepdims=True)
    return tf.log(tf.reduce_sum(tf.exp(x - x_max),
                  axis=axis, keepdims=True))+x_max


def safe_div(numerator, denominator, name):
    return array_ops.where(
        math_ops.greater(denominator, 0),
        math_ops.truediv(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)


def pinv(a, rcond=None, validate_args=False, name=None):
  with tf.name_scope(name, 'pinv', [a, rcond]):
    a = tf.convert_to_tensor(a, name='a')

    if not a.dtype.is_floating:
      raise TypeError('Input `a` must have `float`-like `dtype` '
                      '(saw {}).'.format(a.dtype.name))
    if a.shape.ndims is not None:
      if a.shape.ndims < 2:
        raise ValueError('Input `a` must have at least 2 dimensions '
                         '(saw: {}).'.format(a.shape.ndims))
    elif validate_args:
      assert_rank_at_least_2 = tf.assert_rank_at_least(
          a, rank=2,
          message='Input `a` must have at least 2 dimensions.')
      with tf.control_dependencies([assert_rank_at_least_2]):
        a = tf.identity(a)

    dtype = a.dtype.as_numpy_dtype

    if rcond is None:
      def get_dim_size(dim):
        if a.shape.ndims is not None and a.shape[dim].value is not None:
          return a.shape[dim].value
        return tf.shape(a)[dim]
      num_rows = get_dim_size(-2)
      num_cols = get_dim_size(-1)
      if isinstance(num_rows, int) and isinstance(num_cols, int):
        max_rows_cols = float(max(num_rows, num_cols))
      else:
        max_rows_cols = tf.cast(tf.maximum(num_rows, num_cols), dtype)
      rcond = 10. * max_rows_cols * np.finfo(dtype).eps

    rcond = tf.convert_to_tensor(rcond, dtype=dtype, name='rcond')

    # Calculate pseudo inverse via SVD.
    # Note: if a is symmetric then u == v. (We might observe additional
    # performance by explicitly setting `v = u` in such cases.)
    [
        singular_values,         # Sigma
        left_singular_vectors,   # U
        right_singular_vectors,  # V
    ] = tf.linalg.svd(a, full_matrices=False, compute_uv=True)

    # Saturate small singular values to inf. This has the effect of make
    # `1. / s = 0.` while not resulting in `NaN` gradients.
    cutoff = rcond * tf.reduce_max(singular_values, axis=-1)
    singular_values = tf.where(
        singular_values > cutoff[..., tf.newaxis],
        singular_values,
        tf.fill(tf.shape(singular_values), np.array(np.inf, dtype)))

    # Although `a == tf.matmul(u, s * v, transpose_b=True)` we swap
    # `u` and `v` here so that `tf.matmul(pinv(A), A) = tf.eye()`, i.e.,
    # a matrix inverse has "transposed" semantics.
    a_pinv = tf.matmul(
        right_singular_vectors / singular_values[..., tf.newaxis, :],
        left_singular_vectors,
        adjoint_b=True)

    if a.shape.ndims is not None:
      a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))

    return a_pinv


def prepare_batch(minibatch_index, X, Y, S, seq_len, batch_size, new_batch):
    if new_batch is True:
        xf0 = X[int(minibatch_index * batch_size): int((minibatch_index + 1) * batch_size)]
        yf0 = Y[int(minibatch_index * batch_size): int((minibatch_index + 1) * batch_size)]
        sf0 = S[int(minibatch_index * batch_size): int((minibatch_index + 1) * batch_size)]

        ll = copy.copy(xf0.shape[0])

        XL = list()
        SL = list()
        YL = list()
        PS = list()
        PM = list()
        PT = list()
        PMeta = list()
        IT = list()
        IM = list()
        IS = list()
        length_list = np.zeros(shape=[batch_size, 1])
        for i in range(ll):
            yf = yf0[i, :, :2]
            m = ~(yf == 0).all(1)
            yf = yf[m]
            length_list[i, 0] = yf.shape[0]

        max_length = np.max(length_list)

        for i in range(ll):
            yf = yf0[i, :, :]
            sf = sf0[i, :, :]
            xf = xf0[i, :, :]

            m = ~(yf[:, :2] == 0).all(1)
            yf = yf[m]
            sf = sf[m]
            xf = xf[m]

            time_temp = copy.copy(xf[:, 0])
            # max_tt = np.max(time_temp)
            min_tt = np.min(time_temp)

            # xf[:, 0] = copy.copy((time_temp - min_tt) / (max_tt - min_tt))
            xf[:, 0] = copy.copy(time_temp - min_tt)

            delta = int(max_length - yf.shape[0])

            z1 = np.zeros(shape=[delta, xf.shape[1]])
            z2 = np.zeros(shape=[delta, sf.shape[1]])
            z3 = np.zeros(shape=[delta, yf.shape[1]])

            z1n = np.concatenate([xf, z1], axis=0)
            z2n = np.concatenate([sf, z2], axis=0)
            z3n = np.concatenate([yf, z3], axis=0)

            yf = np.expand_dims(z3n, axis=0)
            sf = np.expand_dims(z2n, axis=0)
            xf = np.expand_dims(z1n, axis=0)

            imeas = 3
            meas00 = copy.copy(xf[:, :imeas, 1:])
            state00 = copy.copy(yf[:, :imeas, :])
            time00 = copy.copy(xf[:, :imeas, 0, np.newaxis])

            prev_state = copy.copy(state00[:, -1, np.newaxis, :])
            prev_meas = copy.copy(meas00[:, -1, np.newaxis, :])
            prev_time = copy.copy(time00[:, -1, np.newaxis, 0, np.newaxis])
            prev_meta = copy.copy(sf[:, -1, np.newaxis, 0, np.newaxis])

            x = copy.copy(xf[:, 1 + imeas:, :])
            y = copy.copy(yf[:, 1 + imeas:, :])
            s = copy.copy(sf[:, 1 + imeas:, :])

            XL.append(x)
            YL.append(y)
            SL.append(s)
            PS.append(prev_state)
            PM.append(prev_meas)
            PT.append(prev_time)
            PMeta.append(prev_meta)
            IM.append(meas00)
            IS.append(state00)
            IT.append(time00)

            del x, y, s, xf, yf, sf, meas00, time00

        xout = np.concatenate(XL, axis=0)
        yout = np.concatenate(YL, axis=0)
        sout = np.concatenate(SL, axis=0)
        psout = np.concatenate(PS, axis=0)
        pmout = np.concatenate(PM, axis=0)
        ptout = np.concatenate(PT, axis=0)
        pmetaout = np.concatenate(PMeta, axis=0)
        im0out = np.concatenate(IM, axis=0)
        is0out = np.concatenate(IS, axis=0)
        it0out = np.concatenate(IT, axis=0)

        if seq_len > yout.shape[1]:
            delta = int(seq_len - yout.shape[1])

            # z1 = np.zeros(shape=[delta, xout.shape[2]])
            # z2 = np.zeros(shape=[delta, sout.shape[2]])
            # z3 = np.zeros(shape=[delta, yout.shape[2]])

            init_z_y = np.zeros([yout.shape[0], delta, yout.shape[2]])
            init_z_x = np.zeros([xout.shape[0], delta, xout.shape[2]])
            init_z_s = np.zeros([sout.shape[0], delta, sout.shape[2]])

            xout = np.concatenate([xout, init_z_x], axis=1)
            yout = np.concatenate([yout, init_z_y], axis=1)
            sout = np.concatenate([sout, init_z_s], axis=1)

        tw = np.ceil(xout.shape[1] / seq_len)
        twd = int(tw * seq_len - xout.shape[1])

        init_z_y = np.zeros([yout.shape[0], twd, yout.shape[2]])
        init_z_x = np.zeros([xout.shape[0], twd, xout.shape[2]])
        init_z_s = np.zeros([sout.shape[0], twd, sout.shape[2]])

        xout = np.concatenate([xout, init_z_x], axis=1)
        yout = np.concatenate([yout, init_z_y], axis=1)
        sout = np.concatenate([sout, init_z_s], axis=1)

    else:
        xout = copy.copy(X)
        yout = copy.copy(Y)
        sout = copy.copy(S)
        psout = []
        pmout = []
        ptout = []
        pmetaout = []
        im0out = []
        is0out = []
        it0out = []

    return xout, yout, sout, psout, pmout, ptout, pmetaout, im0out, is0out, it0out, max_length


def prepare_batch_testing(xf):

    XL = list()
    PM = list()
    PT = list()
    IT = list()
    IM = list()

    time_temp = copy.copy(xf[:, 0])
    # max_tt = np.max(time_temp)
    min_tt = np.min(time_temp)

    # xf[:, 0] = copy.copy((time_temp - min_tt) / (max_tt - min_tt))
    xf[:, 0] = copy.copy(time_temp - min_tt)

    imeas = 3
    meas00 = copy.copy(xf[:, :imeas, 1:])
    time00 = copy.copy(xf[:, :imeas, 0, np.newaxis])

    prev_meas = copy.copy(meas00[:, -1, np.newaxis, :])
    prev_time = copy.copy(time00[:, -1, np.newaxis, 0, np.newaxis])

    x = copy.copy(xf[:, 1 + imeas:, :])

    XL.append(x)
    PM.append(prev_meas)
    PT.append(prev_time)
    IM.append(meas00)
    IT.append(time00)

    xout = np.concatenate(XL, axis=0)
    pmout = np.concatenate(PM, axis=0)
    ptout = np.concatenate(PT, axis=0)
    im0out = np.concatenate(IM, axis=0)
    it0out = np.concatenate(IT, axis=0)

    init_z_x = np.zeros([xout.shape[0], twd, xout.shape[2]])

    xout = np.concatenate([xout, init_z_x], axis=1)

    return xout, pmout, ptout, im0out, it0out


def get_feed_time_asynch(x, y, meta, prev_x, prev_y, prev_time, prev_meta, max_seq, step, num_state, window_mode=True):
    xt = copy.copy(prev_x)
    yt = copy.copy(prev_y)
    mt = copy.copy(prev_meta)
    # sl = copy.copy(prev_sl)
    time = copy.copy(prev_time)

    if window_mode:

        r1 = step * max_seq
        r2 = r1 + max_seq
        slc_length = max_seq

        if step > 0:
            xtemp = x[:, r1:r2, :]
            # xtemp = xtemp[:, np.newaxis, :]

            xt = np.concatenate([xt, xtemp[:, :, 1:]], axis=1)
            xt = copy.copy(xt[:, -slc_length:, :])

            mtemp = meta[:, r1:r2, :1]
            # mtemp = mtemp[:, np.newaxis, :]
            mt = np.concatenate([mt, mtemp], axis=1)

            mt = mt[:, -slc_length:, :]

            time_temp = copy.copy(xtemp[:, :, 0])
            time_temp = time_temp[:, :, np.newaxis]
            time = np.concatenate([time, time_temp], axis=1)
            time = time[:, -slc_length:, :]

            ytemp = y[:, r1:r2, :num_state]
            # ytemp = ytemp[:, np.newaxis, :]

            yt = np.concatenate([yt, ytemp], axis=1)
            yt0 = copy.copy(yt[:, -slc_length:, :])
        else:
            xt = x[:, r1:r2, 1:]
            mt = mt[:, r1:r2, :]
            time = x[:, r1:r2, 0, np.newaxis]
            yt0 = y[:, r1:r2, :]
    else:
        slc_length = 1
        xtemp = x[:, step, :]
        xtemp = xtemp[:, np.newaxis, :]

        xt = np.concatenate([xt, xtemp[:, :, 1:]], axis=1)
        xt = copy.copy(xt[:, -slc_length:, :])

        mtemp = meta[:, step, :1]
        mtemp = mtemp[:, np.newaxis, :]
        mt = np.concatenate([mt, mtemp], axis=1)

        mt = mt[:, -slc_length:, :]

        time_temp = copy.copy(xtemp[:, :, 0])
        time_temp = time_temp[:, :, np.newaxis]
        time = np.concatenate([time, time_temp], axis=1)
        time = time[:, -slc_length:, :]

        ytemp = y[:, step, :num_state]
        ytemp = ytemp[:, np.newaxis, :]

        yt = np.concatenate([yt, ytemp], axis=1)
        yt0 = copy.copy(yt[:, -slc_length:, :])

    # xt = current measurements
    # yt0 = current truth state
    # It = Identity matrix
    # time = current time
    # pst = previous truth state
    # mt = current meta data
    # seqlen = sequence length indicator
    # seqweight = sequence weight indicator
    # prev_measurement = previous measurement
    # y_pred = next truth truth (for prediction)
    return xt, yt0, time, mt


def set_sequence_weights(batch_size, seqlen, current_data, r1, r2):
    seqlen = np.ones(shape=[batch_size, ])
    int_time = np.zeros(shape=[batch_size, seqlen])
    seqweight = np.zeros(shape=[batch_size, seqlen])

    for i in range(batch_size):
        current_data = current_data[i, :, :3]
        m = ~(current_yt == 0).all(1)
        yf = current_yt[m]
        seq = yf.shape[0]
        seqlen[i] = seq
        int_time[i, :] = range(r1, r2)
        seqweight[i, :] = m.astype(int)

    return seqlen, int_time, seqweight


def get_zero_state(layers, units, batch_size, n, std=0.3):
    state_list = list()
    if n == 2:
        for i in range(layers):
            s = list()
            s.append(np.zeros(shape=(batch_size, units), dtype=np.float32) + np.random.normal(loc=0.0, scale=std))
            # s.append(np.zeros(shape=(batch_size, units), dtype=np.float64) + np.random.normal(loc=0.0, scale=0.05))
            state_list.append(tuple(s))
    elif n == 3:
        state_list = np.zeros(shape=(layers, batch_size, units), dtype=np.float32) + np.random.normal(loc=0.0, scale=std)
        # s.append(np.zeros(shape=(batch_size, units), dtype=np.float64) + np.random.normal(loc=0.0, scale=0.05))
    elif n == 4:
        state_list = np.zeros(shape=(batch_size, units), dtype=np.float32) + np.random.normal(loc=0.0, scale=std)
        # s.append(np.zeros(shape=(batch_size, units), dtype=np.float64) + np.random.normal(loc=0.0, scale=0.05))
    elif n == 5:
        for i in range(layers):
            s = list()
            s.append(np.zeros(shape=(batch_size, units), dtype=np.float32) + np.random.normal(loc=0.0, scale=std))
            s.append(np.zeros(shape=(batch_size, units), dtype=np.float32) + np.random.normal(loc=0.0, scale=std))
            state_list.append(tuple(s))
        state_list.append(np.zeros(shape=(batch_size, int(units / 2)), dtype=np.float64))
        state_list.append(np.zeros(shape=(), dtype=np.int32))
        state_list.append(np.zeros(shape=(batch_size, 5), dtype=np.float64))
        state_list.append(tuple())
    return state_list


def initialize_run_variables():
    out_plot_filter = list()
    out_plot_smooth = list()
    out_plot_refined = list()
    time_vals = list()
    meas_plot = list()
    truth_plot = list()
    q_plots = list()
    q_plott = list()
    q_plotr = list()
    qt_plot = list()
    rt_plot = list()
    at_plot = list()

    return out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None,
              dtype=tf.float32):

    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable(scope + 'beta', initializer=tf.zeros(params_shape, dtype=dtype), dtype=dtype)
        gamma = tf.get_variable(scope + 'gamma', initializer=tf.ones(params_shape, dtype=dtype), dtype=dtype)
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    """Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=None, reuse=reuse, name='Q')  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None, reuse=reuse, name='K')  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None, reuse=reuse, name='V')  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        # outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            outputs = tf.where(tf.is_nan(masks), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs, name='alpha')  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        dropout_rate = 1. - dropout_rate
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = normalize(outputs, reuse=reuse, dtype=dtype)  # (N, T_q, C)

    return outputs


def weighted_mape_tf(y_true, y_pred, weight, den=None, name=''):
    with tf.variable_scope('weighted_mape'):
        mult = 1
        num = tf.sqrt(tf.square(tf.subtract(y_true * mult, y_pred * mult))) * math_ops.to_double(weight)

        # den = tf.reduce_sum(tf.sqrt(tf.sqrt(tf.square(y_true))) * mult, name=name+'mean')
        if den is not None:
            den = tf.clip_by_value(den, clip_value_min=0.0001, clip_value_max=1e9)
            wmape = num
        else:
            wmape = num

        wmape = tf.reduce_sum(wmape, name=name+'_sum')
        return wmape


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

