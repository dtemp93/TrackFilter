import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
from tensorflow.python.ops import rnn_cell_impl

_state_size_with_prefix = rnn_cell_impl._zero_state_tensors
import os


def attention(batch_size, max_seq, n_hidden, out, series_k, scope=None):

    with tf.variable_scope('encoder') as scope:
        try:
            mean = 0.0
            stddev = 1.0 / (n_hidden * max_seq)
            We = tf.get_variable(name='We', dtype=tf.float64, shape=[max_seq, out.shape[1].value],
                                 initializer=tf.truncated_normal_initializer(mean, stddev, dtype=tf.float64))
            Ve = tf.get_variable(name='Ve', dtype=tf.float64, shape=[1, max_seq],
                                 initializer=tf.truncated_normal_initializer(mean, stddev, dtype=tf.float64))
        except ValueError:
            scope.reuse_variables()
            We = tf.get_variable('We', dtype=tf.float64)
            Ve = tf.get_variable('Ve', dtype=tf.float64)
    W_e = tf.tile(tf.expand_dims(We, 0), [batch_size, 1, 1])  # b*T*2d
    brcast = tf.nn.tanh(tf.matmul(W_e, out) + series_k)  # b,T,T + b,T,1 = b, T, T
    V_e = tf.tile(tf.expand_dims(Ve, 0), [batch_size, 1, 1])  # b,1,T

    return tf.matmul(V_e, brcast)  # b,1,T


def conv1d(inputs,
           out_channels,
           filter_width=2,
           stride=1,
           padding='VALID',
           data_format='NHWC',
           gain=np.sqrt(2),
           activation=tf.nn.relu,
           bias=False,
           name='',
           dtype=tf.float32):

    in_channels = inputs.get_shape().as_list()[-1]

    stddev = gain / np.sqrt(filter_width ** 2 * in_channels)
    w_init = tf.random_normal_initializer(stddev=stddev, dtype=dtype)

    w = tf.get_variable(name=name+'_w',
                        shape=(filter_width, in_channels, out_channels),
                        initializer=w_init, dtype=dtype)

    outputs = tf.nn.conv1d(inputs,
                           w,
                           stride=stride,
                           padding=padding,
                           data_format=data_format,
                           name=name+'_conv1d')

    if bias:
        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable(name=name+'_b',
                            shape=(out_channels,),
                            initializer=b_init, dtype=dtype)

        outputs = outputs + tf.expand_dims(tf.expand_dims(b, 0), 0)

    if activation:
        outputs = activation(outputs)

    return outputs


def zero_state(cell, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      cell: RNNCell.
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = cell.state_size
    if nest.is_sequence(state_size):
        state_size_flat = nest.flatten(state_size)
        zeros_flat = [
            tf.zeros(
              tf.stack(_state_size_with_prefix(s, batch_size, dtype=dtype)))
            for s in state_size_flat]
        for s, z in zip(state_size_flat, zeros_flat):
            z.set_shape(_state_size_with_prefix(s, batch_size, dtype=dtype))
        zeros = nest.pack_sequence_as(structure=state_size,
                                      flat_sequence=zeros_flat)
    else:
        zeros_size = _state_size_with_prefix(state_size, batch_size, dtype=dtype)
        zeros = tf.zeros(tf.stack(zeros_size), dtype=dtype)
        zeros.set_shape(_state_size_with_prefix(state_size, batch_size, dtype=dtype))

    return zeros


def get_initial_cell_state(cell, initializer, batch_size, dtype):
    """Return state tensor(s), initialized with initializer.
    Args:
      cell: RNNCell.
      batch_size: int, float, or unit Tensor representing the batch size.
      initializer: function with two arguments, shape and dtype, that
          determines how the state is initialized.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` initialized
      according to the initializer.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = cell.state_size
    if nest.is_sequence(state_size):
        state_size_flat = nest.flatten(state_size)
        init_state_flat = [
            initializer([s], batch_size, dtype, i)
                for i, s in enumerate(state_size_flat)]
        init_state = nest.pack_sequence_as(structure=state_size,
                                           flat_sequence=init_state_flat)
    else:
        init_state_size = _state_size_with_prefix(state_size, batch_size, dtype=dtype)
        init_state = initializer(init_state_size, batch_size, dtype, None)

    return init_state


def make_variable_state_initializer(**kwargs):
    def variable_state_initializer(shape, batch_size, dtype, index):
        args = kwargs.copy()

        if args.get('name'):
            args['name'] = args['name'] + '_' + str(index)
        else:
            args['name'] = 'init_state_' + str(index)

        args['shape'] = shape
        args['dtype'] = dtype

        var = tf.get_variable(**args)
        var = tf.expand_dims(var, 0)
        var = tf.tile(var, tf.stack([batch_size] + [1] * len(shape)))
        # var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
        return var

    return variable_state_initializer


def make_gaussian_state_initializer(initializer, deterministic_tensor=None, stddev=0.3):
    def gaussian_state_initializer(shape, batch_size, dtype, index):
        init_state = initializer(shape, batch_size, dtype, index)
        if deterministic_tensor is not None:
            return tf.cond(deterministic_tensor,
                lambda: init_state,
                lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev, dtype=tf.float64))
        else:
            return init_state + tf.random_normal(tf.shape(init_state), stddev=stddev, dtype=tf.float64)
    return gaussian_state_initializer


def get_chol_vars(self):
    with tf.variable_scope("muW", reuse=tf.AUTO_REUSE):
        self.muW = tf.get_variable(name='muW', shape=[self.Q, self.ydim], initializer=tf.random_normal_initializer(stddev=0.1), dtype=tf.float32)  # [Q x D]
    with tf.variable_scope("logSigmaW", reuse=tf.AUTO_REUSE):
        self.logSigmaW = tf.get_variable(name='logSigmaW', shape=[self.Q, self.ydim], initializer=tf.constant_initializer(-2.0), dtype=tf.float32)  # [Q x D]
    with tf.variable_scope("muZ", reuse=tf.AUTO_REUSE):
        self.muZ = tf.constant(np.zeros((self.Q, self.ydim)), name='muZ', dtype=tf.float32)  # [Q x D]
    with tf.variable_scope("logSigmaZ", reuse=tf.AUTO_REUSE):
        self.logSigmaZ = tf.constant(self.logSigmaZval * np.ones((self.Q, self.ydim)), name='logSigmaZ', dtype=tf.float32)  # [Q x D]

    return self.muW, self.logSigmaW, self.muZ, self.logSigmaZ


def filter_layer(self, x, filter_width=-1, y_width=-1, residual=True, name='', act=tf.nn.selu, cov=False):
    xav = tfc.layers.xavier_initializer()
    if name in self.filters:
        var_reuse = True
    else:
        var_reuse = False
        self.filters[name] = name

    with tf.variable_scope(name+'filter', reuse=var_reuse):
        filter_in_size = int(x.get_shape()[-1])
        if filter_width == -1:
            filter_width = filter_in_size

        w1 = tf.get_variable(name+'_filter_w1', [filter_width, filter_width], tf.float32, initializer=xav)
        b1 = tf.get_variable(name+'_filter_b1', [1, filter_width], tf.float32, initializer=xav)
        z1 = tf.add(b1, tf.matmul(x, w1))
        if act is not None:
            o1 = act(z1, name='o1')
        else:
            o1 = z1

        w2 = tf.get_variable(name + '_filter_w2', [filter_width, filter_width], tf.float32, initializer=xav)
        b2 = tf.get_variable(name + '_filter_b2', [1, filter_width], tf.float32, initializer=xav)
        z2 = tf.add(b2, tf.matmul(o1, w2))
        if act is not None:
            o2 = act(z2, name='o2')
        else:
            o2 = z2

        if residual is False:
            o2_final = o2
            if cov is True:
                o2_final = tf.exp(o2)

        else:
            wr1 = tf.get_variable(name+'_filter_wr1', [filter_width, filter_width], tf.float32, initializer=xav)
            br1 = tf.get_variable(name+'_filter_br1', [1, filter_width], tf.float32, initializer=xav)
            zr1 = tf.add(br1, tf.matmul(o2, wr1))
            if act is not None:
                or1 = act(zr1, name=name+'_filter_or1')
            else:
                or1 = zr1

            wr2 = tf.get_variable(name + '_filter_wr2', [filter_width, filter_width], tf.float32, initializer=xav)
            br2 = tf.get_variable(name + '_filter_br2', [1, filter_width], tf.float32, initializer=xav)
            zr2 = tf.add(br2, tf.matmul(or1, wr2))
            if act is not None:
                or2 = act(zr2, name=name + '_filter_or2')
            else:
                or2 = zr2

            o2_final = tf.add(o2, or2)

        if y_width != -1:
            wp1 = tf.get_variable(name + '_project_w1', [filter_width, y_width], tf.float32, initializer=xav)
            bp1 = tf.get_variable(name + '_project_b1', [1, y_width], tf.float32, initializer=xav)
            zr0 = tf.add(bp1, tf.matmul(o2_final, wp1))
            if cov is True:
                zr0 = tf.exp(zr0)
            return zr0
        else:
            return o2_final


def trk_rnn(self, x, st_onehots, prev_full_state, width=-1, name=''):
    if name in self.rnns:
        var_reuse = True
    else:
        var_reuse=False
        self.rnns[name] = name

    with tf.variable_scope(name+'rnn', reuse=var_reuse):
        trk_lstm = tfc.rnn.BasicLSTMCell(width, state_is_tuple=True)
        st_ohrc = tf.reshape(st_onehots, tf.reshape([self.batch_size, self.num_sensors*self.num_trks, 1]))
        st_inverse = tf.reshape(tf.subtract(tf.ones_like(st_ohrc), st_ohrc), [self.batch_size, self.num_sensors*self.num_trks, 1])
        rnn_prev_state = [0, 1]
        rnn_prev_state[0] = tf.reduce_sum(tf.multiply(prev_full_state[0], st_ohrc), axis=1)
        rnn_prev_state[1] = tf.reduce_sum(tf.multiply(prev_full_state[1], st_ohrc), axis=1)

        y, rnn_new_state = trk_lstm(x, rnn_prev_state)

        # new_full_state[0] = tf.add(tf.multiply(tf.reshape(rnn_new_state[0], [self.batch_size, 1, self.num_trks]), st_ohrc),
        #                            tf.multiply(prev_full_state[0], st_inverse))
        # new_full_state[1] = tf.add(tf.multiply(tf.reshape(rnn_new_state[1], [self.batch_size, 1, self.num_trks]), st_ohrc),
        #                            tf.multiply(prev_full_state[1], st_inverse))


def alpha_rnn(self, inputs, cell, state=None, u=None, buffer=None, reuse=None, init_buffer=False, name='alpha'):
    # Increase the number of hidden units if we also learn u (learn_u=True)
    num_units = self.dim_a

    # Overwrite input buffer
    if init_buffer:
        buffer = tf.zeros((tf.shape(inputs[0])[0], self.dim_a, self.dim_y), dtype=tf.float32)

    # If K == 1, return inputs
    if self.K == 1:
        return tf.ones([self.batch_size, self.K]), state, u, buffer

    with tf.variable_scope(name, reuse=reuse):
        output, state = cell(inputs, state)

        alpha = tfc.layers.fully_connected(output[:, :num_units], self.K, activation_fn=tf.nn.softmax, scope='alpha_var', reuse=tf.AUTO_REUSE)
        u = tfc.layers.fully_connected(output[:, num_units:], self.dim_u, activation_fn=None, scope='u_var', reuse=tf.AUTO_REUSE)
    # u = tf.reshape(u, [self.batch_size, self.max_seq, self.dim_u])
    return alpha, state, u, buffer


def weighted_mape_tf(y_true, y_pred, weight, tot=1., name=''):
    with tf.variable_scope('weighted_mape'):
        mult = 1
        # tot = tf.reduce_sum(tf.abs(y_true))
        # tot = tf.clip_by_value(tot, clip_value_min=(0.001/6378137), clip_value_max=tot)
        num = tf.reduce_sum(tf.square(tf.subtract(y_true * mult, y_pred * mult)) * math_ops.to_double(weight), name=name+'sum')
        # num = tf.clip_by_value(num, clip_value_min=0., clip_value_max=1000)
        den = tf.reduce_mean(tf.square(y_true * mult), name=name+'mean')
        den = tf.clip_by_value(den, clip_value_min=1., clip_value_max=1e9, name=name+'clip')
        # wmape = tf.realdiv(tf.reduce_sum(tf.abs(tf.subtract(y_true, y_pred)) * math_ops.to_float(weight)), tot)*100

        # wmape = tf.realdiv(num, den) * 100

        wmape = num / tot

        return wmape


def msec(y_true, y_pred, weight, tot=1):
    mult = 1
    tot = tf.reduce_sum(tf.abs(y_true))
    tot = tf.clip_by_value(tot, clip_value_min=(0.001/6378137), clip_value_max=tot)
    # loss = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(y_true * mult, y_pred * mult))) * math_ops.to_double(weight))
    # num = tf.clip_by_value(num, clip_value_min=0., clip_value_max=1000)
    # den = tf.reduce_sum(tf.square(y_true * mult))
    # den = tf.clip_by_value(den, clip_value_min=1., clip_value_max=tf.reduce_mean(tot))
    loss = tf.realdiv(tf.reduce_mean(tf.abs(tf.subtract(y_true, y_pred)) * math_ops.to_double(weight)), tot)*100

    return loss


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def mse(y_true, y_pred, weight, denom):

    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(y_true, y_pred), axis=1, keepdims=True)))

    return loss


def normed_mse(y_true, y_pred, weight, denom):
    # mean_true, var_true = tf.nn.moments(y_true, axes=[1])
    # mean_pred, var_pred = tf.nn.moments(y_pred, axes=[1])
    # mean_true = tf.reduce_mean(y_true, axis=1, keepdims=True)
    # std_true = reduce_std(y_true, axis=1, keepdims=True)
    # mean_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    # std_pred = reduce_std(y_pred, axis=1, keepdims=True)
    #
    # std_true = tf.clip_by_value(std_true, clip_value_min=(0.01 / 6378137), clip_value_max=std_true)
    # std_pred = tf.clip_by_value(std_pred, clip_value_min=(0.01 / 6378137), clip_value_max=std_pred)
    #
    # norm_true = (y_true - mean_true) / std_true
    # norm_pred = (y_pred - mean_pred) / std_pred
    nmse_a = 0.
    nmse_b = 0.
    if y_pred.get_shape().ndims == 2:  # [batch_size, n_feature]
        nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(y_true, y_pred), axis=1, keepdims=True))
        nmse_b = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=1, keepdims=True))
        nmse_b = tf.clip_by_value(nmse_b, clip_value_min=1e-10, clip_value_max=1e9)
    elif y_pred.get_shape().ndims == 3:  # [batch_size, w, h]
        nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(y_true, y_pred), axis=[1, 2], keepdims=True))
        nmse_b = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=[1, 2], keepdims=True))
        # denom = tf.ones_like(nmse_a) * denom
        nmse_b = tf.clip_by_value(nmse_b, clip_value_min=(0.0001 / 6378137), clip_value_max=1e9)
    elif y_pred.get_shape().ndims == 4:  # [batch_size, w, h, c]
        nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(y_true, y_pred), axis=[1, 2, 3], keepdims=True))
        nmse_b = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=[1, 2, 3], keepdims=True))
        # denom = tf.ones_like(nmse_a) * denom
        nmse_b = tf.clip_by_value(nmse_b, clip_value_min=(0.0001 / 6378137), clip_value_max=1e9)
    #
    loss = tf.reduce_mean((nmse_a / nmse_b) * weight)

    # loss = tf.cast(tf.losses.huber_loss(norm_true, norm_pred, weights=weight, delta=tf.cast(std_true/6, tf.float32)), tf.float64)
    # loss = tf.cast(tf.losses.mean_squared_error(norm_true, norm_pred, weights=weight), tf.float64)

    return loss


# def pinv(A, reltol=1e-7):
#     # Compute the SVD of the input matrix A
#     s, u, v = tf.svd(A)
#
#     # Invert s, clear entries lower than reltol*s[0].
#     atol = tf.reduce_max(s, keepdims=True) * reltol
#     # s = tf.boolean_mask(s, s > atol)
#     s = tf.where(s > atol, s, tf.ones_like(s)*atol)
#     s_inv = tf.matrix_diag(1. / s)
#
#     # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
#     return tf.matmul(v, tf.matmul(s_inv, u, transpose_b=True))


def softplus_and_shift(x, shift=1e-5, mult=None, name=None):
  with tf.name_scope(name, 'softplus_and_shift', [x, shift]):
    x = tf.convert_to_tensor(x, name='x')
    y = tf.nn.softplus(x, 'softplus')
    # y = tf.sqrt(tf.square(x))
    if mult is not None:
      y *= mult
    if shift is not None:
      y += shift
    return y


def tril_with_diag_softplus_and_shift(x, diag_shift=1e-5, diag_mult=None, name=None):
  with tf.name_scope(name, 'tril_with_diag_softplus_and_shift',
                     [x, diag_shift]):
    x = tf.convert_to_tensor(x, name='x'+name)
    x = tfd.fill_triangular(x, name)
    diag = softplus_and_shift(tf.matrix_diag_part(x), diag_shift, diag_mult, name)
    x2 = tf.tanh(x) * 0.99
    x2 = x2 * tf.cast(tf.sqrt(1e-12), tf.float64)
    x3 = tf.matrix_set_diag(x2, diag, name)
    return x3


def multivariate_normal_tril(
    x,
    dims,
    layer_fn=tf.layers.dense,
    loc_fn=lambda x: x,
    scale_fn=tril_with_diag_softplus_and_shift,
    diag_shift=1e-5,
    name=None,
    reuse=False):

  with tf.name_scope(name, 'multivariate_normal_tril', [x, dims]):
    x = tf.convert_to_tensor(x, name='x')
    x = layer_fn(x, dims + dims * (dims + 1) // 2, name=name, reuse=reuse)
    return tfd.MultivariateNormalTriL(
        loc=loc_fn(x[..., :dims]),
        scale_tril=scale_fn(x[..., dims:], diag_shift=diag_shift))


def denormalize_statenp(meas_norm0, state_norm0, meanv, stdv):

    zero_rows = (state_norm0 == 0).all(2)
    state = (state_norm0 * stdv) + meanv
    meas = (meas_norm0 * stdv[:3]) + meanv[:3]

    for i in range(state_norm0.shape[0]):
        zz = zero_rows[i, :, np.newaxis]
        state[i, :, :] = np.where(zz, np.zeros_like(state[i, :, :]), state[i, :, :])
        meas[i, :, :] = np.where(zz, np.zeros_like(meas[i, :, :]), meas[i, :, :])

    return meas, state


def normalize_statenp(meas0, state0):
    # rb = 0
    # x0 = meas0
    # y0 = state0
    # nz = np.nonzero(x0[rb, :, 0])[0][0]
    #
    # time = x0[rb, nz:, 0]
    # xm = x0[rb, nz:, 1] * 1
    # ym = x0[rb, nz:, 2] * 1
    # zm = x0[rb, nz:, 3] * 1
    #
    # xt = y0[rb, nz:, 0] * 1
    # yt = y0[rb, nz:, 1] * 1
    # zt = y0[rb, nz:, 2] * 1
    #
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.subplot(311)
    # plt.scatter(time, xm)
    # plt.plot(time, xt, 'r')
    #
    # plt.subplot(312)
    # plt.scatter(time, ym)
    # plt.plot(time, yt, 'r')
    #
    # plt.subplot(313)
    # plt.scatter(time, zm)
    # plt.plot(time, zt, 'r')
    # plt.pause(0.01)
    # plt.show()

    zero_rows = (state0 == 0).all(2)
    state = state0[~zero_rows, :]
    # meas = meas0[~zero_rows, :]

    meanv = np.mean(state, 0)
    stdv = np.std(state, 0)

    # meanvm = np.mean(meas[:, 1:], 0)
    # stdvm = np.std(meas[:, 1:], 0)

    state_norm = (state0 - meanv) / stdv
    meas_norm = (meas0[:, :, 1:] - meanv[:3]) / stdv[:3]

    for i in range(state0.shape[0]):
        zz = zero_rows[i, :, np.newaxis]
        state_norm[i, :, :] = np.where(zz, np.zeros_like(state_norm[i, :, :]), state_norm[i, :, :])
        meas_norm[i, :, :] = np.where(zz, np.zeros_like(meas_norm[i, :, :]), meas_norm[i, :, :])

    meas_norm = np.concatenate([meas0[:, :, 0, np.newaxis], meas_norm], axis=2)

    return meas0, meas_norm, state0, state_norm, meanv, stdv
#
#
# def denormalize_state(state):
#     pos = tf.concat([state[:, 0, tf.newaxis], state[:, 4, tf.newaxis], state[:, 8, tf.newaxis]], axis=1)
#     vel = tf.concat([state[:, 1, tf.newaxis], state[:, 5, tf.newaxis], state[:, 9, tf.newaxis]], axis=1)
#     acc = tf.concat([state[:, 2, tf.newaxis], state[:, 6, tf.newaxis], state[:, 10, tf.newaxis]], axis=1)
#     jer = tf.concat([state[:, 3, tf.newaxis], state[:, 7, tf.newaxis], state[:, 11, tf.newaxis]], axis=1)
#
#     posn = pos * (1. * 6378137)
#     veln = vel * (1e-3 * 6378137)
#     accn = acc * (4e-5 * 6378137)
#     jern = jer * (4e-5 * 6378137)
#
#     state = tf.concat([posn, veln, accn, jern], axis=1)
#
#     return state
#
#
# def denormalize_state2(state):
#     pos = tf.concat([state[:, :, 0, tf.newaxis], state[:, :, 4, tf.newaxis], state[:, :, 8, tf.newaxis]], axis=2)
#     vel = tf.concat([state[:, :, 1, tf.newaxis], state[:, :, 5, tf.newaxis], state[:, :, 9, tf.newaxis]], axis=2)
#     acc = tf.concat([state[:, :, 2, tf.newaxis], state[:, :, 6, tf.newaxis], state[:, :, 10, tf.newaxis]], axis=2)
#     jer = tf.concat([state[:, :, 3, tf.newaxis], state[:, :, 7, tf.newaxis], state[:, :, 11, tf.newaxis]], axis=2)
#
#     posn = pos * (1. * 6378137)
#     veln = vel * (1e-3 * 6378137)
#     accn = acc * (4e-5 * 6378137)
#     jern = jer * (4e-5 * 6378137)
#
#     state = tf.concat([posn, veln, accn, jern], axis=2)
#
#     return state

# def normalize_state(state):
#
#     pos = tf.concat([state[:, 0, tf.newaxis], state[:, 4, tf.newaxis], state[:, 8, tf.newaxis]], axis=1)
#     vel = tf.concat([state[:, 1, tf.newaxis], state[:, 5, tf.newaxis], state[:, 9, tf.newaxis]], axis=1)
#     acc = tf.concat([state[:, 2, tf.newaxis], state[:, 6, tf.newaxis], state[:, 10, tf.newaxis]], axis=1)
#     jer = tf.concat([state[:, 3, tf.newaxis], state[:, 7, tf.newaxis], state[:, 11, tf.newaxis]], axis=1)
#
#     posn = pos / (1. * 6378137)
#     veln = vel / (1e-3 * 6378137)
#     accn = acc / (4e-5 * 6378137)
#     jern = jer / (4e-5 * 6378137)
#
#     state = tf.concat([posn, veln, accn, jern], axis=1)
#
#     return state


def get_QP(dt, om, zm, I_3z, I_4z, zb, dimension=3, sjix=50e-6, sjiy=50e-6, sjiz=50e-6, aji=0.1):

    dt = dt[:, tf.newaxis, :]

    dt7 = dt ** 7
    dt6 = dt ** 6
    dt5 = dt ** 5
    dt4 = dt ** 4
    dt3 = dt ** 3
    dt2 = dt ** 2

    aj = aji[:, :, tf.newaxis]

    aj7 = tf.pow(aj, 7)
    aj6 = tf.pow(aj, 6)
    aj5 = tf.pow(aj, 5)
    aj4 = tf.pow(aj, 4)
    aj3 = tf.pow(aj, 3)
    aj2 = tf.pow(aj, 2)

    q11 = dt7 / 252
    q22 = dt5 / 20
    q33 = dt3 / 3
    q44 = dt

    q12 = dt6 / 72
    q13 = dt5 / 30
    q14 = dt4 / 24
    q23 = dt4 / 8
    q24 = dt3 / 6
    q34 = dt2 / 2

    emadt = tf.exp(-aj * dt)

    q11j = ((1 / (2 * aj7)) * (((aj5 * dt5) / 10) - ((aj4 * dt4) / 2) + ((4 * aj3 * dt3) / 3)
                               + (2 * aj * dt) - (2 * aj2 * dt2) - 3 + (4 * emadt) + (2 * aj2 * dt2 * emadt) - tf.exp(-2 * aj * dt)))

    q22j = ((1 / (2 * aj5)) * (1 - tf.exp(-2 * aj * dt) + ((2 * aj3 * dt3) / 2) + (2 * aj * dt) - (2 * aj2 * dt2) - (4 * aj * dt * emadt)))
    q33j = ((1 / (2 * aj3)) * (4 * emadt + (2 * aj * dt) - (tf.exp(-2 * aj * dt)) - 3))
    q44j = ((1 / (2 * aj)) * (1 - tf.exp(-2 * aj * dt)))

    q12j = (1 / (2 * aj6)) * (1 - (2 * aj * dt) + (2 * aj2 * dt2) - (aj3 * dt3) + ((aj4 * dt4) / 4)
                              + tf.exp(-2 * aj * dt) + (2 * aj * dt * emadt) - (2 * emadt) - (aj2 * dt2 * emadt))
    q13j = (1 / (2 * aj5)) * (((aj3 * dt3) / 3) + (2 * aj * dt) - (aj2 * dt2) - 3
                              + (4 * emadt) + (aj2 * dt2 * emadt) - tf.exp(-2 * aj * dt))

    # q14j = ((1 / (2 * aj4)) * (1 - (2 * tf.exp(-2 * aj * dt)) - (aj2 * dt2 * emadt) + tf.exp(-2 * aj * dt)))
    q14j = ((1 / (2 * aj4)) * (1 + tf.exp(-2 * aj * dt) - (2 * emadt) - (aj2 * dt2 * emadt)))

    q23j = ((1 / (2 * aj4)) * (1 - (2 * aj * dt) + (aj2 * dt2) + (2 * aj * dt * emadt) + tf.exp(-2 * aj * dt) - 2 * emadt))
    q24j = ((1 / (2 * aj3)) * (1 - 2 * aj * dt * emadt - tf.exp(-2 * aj * dt)))
    q34j = ((1 / (2 * aj2)) * (1 - 2 * emadt + tf.exp(-2 * aj * dt)))

    pj = ((2 - (2 * aj * dt) + (aj2 * dt2) - 2 * emadt) / (2 * aj3))
    qj = ((emadt - 1 + (aj * dt)) / aj2)
    rj = ((1 - emadt) / aj)
    sj = emadt

    sj1 = 2 * tf.cast(sjix[:, :, tf.newaxis], dtype=tf.float64) * aj
    sj2 = 2 * tf.cast(sjiy[:, :, tf.newaxis], dtype=tf.float64) * aj
    sj3 = 2 * tf.cast(sjiz[:, :, tf.newaxis], dtype=tf.float64) * aj

    if dimension == 4:

        zeta1j = tf.concat(
            [tf.concat([q11j, q12j, q13j, q14j], axis=2), tf.concat([q12j, q22j, q23j, q24j], axis=2), tf.concat([q13j, q23j, q33j, q34j], axis=2), tf.concat([q14j, q24j, q34j, q44j], axis=2)],
            axis=1) * sj1

        zeta2j = tf.concat(
            [tf.concat([q11j, q12j, q13j, q14j], axis=2), tf.concat([q12j, q22j, q23j, q24j], axis=2), tf.concat([q13j, q23j, q33j, q34j], axis=2), tf.concat([q14j, q24j, q34j, q44j], axis=2)],
            axis=1) * sj2

        zeta3j = tf.concat(
            [tf.concat([q11j, q12j, q13j, q14j], axis=2), tf.concat([q12j, q22j, q23j, q24j], axis=2), tf.concat([q13j, q23j, q33j, q34j], axis=2), tf.concat([q14j, q24j, q34j, q44j], axis=2)],
            axis=1) * sj3

        Q = tf.concat([tf.concat([zeta1j, I_4z, I_4z], axis=2), tf.concat([I_4z, zeta2j, I_4z], axis=2), tf.concat([I_4z, I_4z, zeta3j], axis=2)], axis=1)

        phi = tf.concat([tf.concat([om, dt, q34, pj], axis=2), tf.concat([zm, om, dt, qj], axis=2), tf.concat([zm, zm, om, rj], axis=2), tf.concat([zm, zm, zm, sj], axis=2)], axis=1)

        A = tf.concat([tf.concat([phi, I_4z, I_4z], axis=2), tf.concat([I_4z, phi, I_4z], axis=2), tf.concat([I_4z, I_4z, phi], axis=2)], axis=1)

        # zeta1 = tf.concat([tf.concat([q11, q12, q13, q14], axis=2), tf.concat([q12, q22, q23, q24], axis=2), tf.concat([q13, q23, q33, q34], axis=2), tf.concat([q14, q24, q34, q44], axis=2)], axis=1) * sj1
        # zeta2 = tf.concat([tf.concat([q11, q12, q13, q14], axis=2), tf.concat([q12, q22, q23, q24], axis=2), tf.concat([q13, q23, q33, q34], axis=2), tf.concat([q14, q24, q34, q44], axis=2)], axis=1) * sj2
        # zeta3 = tf.concat([tf.concat([q11, q12, q13, q14], axis=2), tf.concat([q12, q22, q23, q24], axis=2), tf.concat([q13, q23, q33, q34], axis=2), tf.concat([q14, q24, q34, q44], axis=2)], axis=1) * sj3
        # Q = tf.concat([tf.concat([zeta1, I_4z, I_4z], axis=2), tf.concat([I_4z, zeta2, I_4z], axis=2), tf.concat([I_4z, I_4z, zeta3], axis=2)], axis=1)
        #
        # phi2 = tf.concat([tf.concat([om, dt, q34, q24], axis=2), tf.concat([zm, om, dt, q34], axis=2), tf.concat([zm, zm, om, dt], axis=2), tf.concat([zm, zm, zm, om], axis=2)], axis=1)
        # A = tf.concat([tf.concat([phi2, I_4z, I_4z], axis=2), tf.concat([I_4z, phi2, I_4z], axis=2), tf.concat([I_4z, I_4z, phi2], axis=2)], axis=1)

        # tb = tf.concat([tf.concat([q34, pj], axis=2), tf.concat([q44, qj], axis=2), tf.concat([om, rj], axis=2), tf.concat([zm, sj], axis=2)], axis=1)
        # tb = tf.concat([pj, qj, rj, sj], axis=1)
        tb = tf.concat([q34, q44, om, zm], axis=1)

        # B = tf.concat([tf.concat([tb, zb, zb], axis=2), tf.concat([zb, tb, zb], axis=2), tf.concat([zb, zb, tb], axis=2)], axis=1)
        B = tf.concat([tf.concat([tb, zb, zb], axis=2), tf.concat([zb, tb, zb], axis=2), tf.concat([zb, zb, tb], axis=2)], axis=1)

    elif dimension == 3:
        zeta1 = tf.concat([tf.concat([q22, q23, q24], axis=2), tf.concat([q23, q33, q34], axis=2), tf.concat([q24, q34, q44], axis=2)], axis=1)
        zeta2 = tf.concat([tf.concat([q22, q23, q24], axis=2), tf.concat([q23, q33, q34], axis=2), tf.concat([q24, q34, q44], axis=2)], axis=1)
        zeta3 = tf.concat([tf.concat([q22, q23, q24], axis=2), tf.concat([q23, q33, q34], axis=2), tf.concat([q24, q34, q44], axis=2)], axis=1)
        Q = tf.scalar_mul(2, tf.concat([tf.concat([zeta1, I_3z, I_3z], axis=2), tf.concat([I_3z, zeta2, I_3z], axis=2), tf.concat([I_3z, I_3z, zeta3], axis=2)], axis=1))

        phi = tf.concat([tf.concat([om, dt, q34], axis=2), tf.concat([zm, om, dt], axis=2), tf.concat([zm, zm, dt], axis=2)], axis=1)
        A = tf.concat([tf.concat([phi, I_3z, I_3z], axis=2), tf.concat([I_3z, phi, I_3z], axis=2), tf.concat([I_3z, I_3z, phi], axis=2)], axis=1)
        B = A

    return Q, A, B, A


def R2(truth, prediction, weight):

    mu = tf.expand_dims(tf.reduce_mean(prediction, reduction_indices=[0]), axis=0)  # predicted mean
    rse = tf.reduce_mean(tf.pow(tf.subtract(truth, mu), 2) * weight)  # residual from mean prediction
    mset = tf.reduce_mean(tf.pow(tf.subtract(prediction, truth), 2) * weight)
    mset = tf.clip_by_value(mset, clip_value_min=1e-12, clip_value_max=mset)
    invR2 = tf.div(rse, mset)

    loss = invR2 + mset

    # total_error = tf.reduce_sum(tf.square(tf.sub(truth, tf.reduce_mean(prediction))))
    # unexplained_error = tf.reduce_sum(tf.square(tf.sub(truth, prediction)))
    # R_squared = tf.sub(1, tf.div(unexplained_error, total_error))

    return loss


def safe_norm(x, epsilon=1e-12, axis=None, keepdims=True):
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) + epsilon)


def log_sum_exp(x, axis=None):
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    val = tf.log(tf.reduce_sum(tf.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return val


def nl(loss, update_condition, name, epsilon=1e-10):
    loss_value = tf.Variable(1.0, name=name, trainable=False, dtype=tf.float32)
    loss_value_updated = tf.cond(update_condition,
                                 lambda: loss_value.assign(loss),
                                 lambda: loss_value)
    nld = loss / (loss_value_updated + epsilon)

    return nld


def ut_meas_rae(X, Wm, Wc, R, meas_mat, batch_size):
    # Y = X[:, :3, :]
    # east = R * np.sin(A) * np.cos(E)
    # north = R * np.cos(E) * np.cos(A)
    # up = R * np.sin(E)
    #
    # cosPhi = np.cos(lat)
    # sinPhi = np.sin(lat)
    # cosLambda = np.cos(lon)
    # sinLambda = np.sin(lon)
    #
    # tv = cosPhi * up - sinPhi * north
    # wv = sinPhi * up + cosPhi * north
    # uv = cosLambda * tv - sinLambda * east
    # vv = sinLambda * tv + cosLambda * east
    #
    # ecef_base = np.concatenate([uv, vv, wv], axis=2)
    # # ENU = np.concatenate([east, north, up], axis=2)
    #
    # a = 6378137
    # e_sq = 0.00669437999014132
    # chi = a / np.sqrt(1 - e_sq * (np.sin(lat) ** 2))
    #
    # x = (chi + alt) * np.cos(lat) * np.cos(lon)
    # y = (chi + alt) * np.cos(lat) * np.sin(lon)
    # z = (chi + alt - e_sq * chi) * np.sin(lat)
    #
    # ecef_ref = np.expand_dims(np.array([x, y, z]), axis=0)

    Y = tf.matmul(X, meas_mat, transpose_a=True, transpose_b=True)
    Y = tf.transpose(Y, [0, 2, 1])
    y = tf.zeros([batch_size, Y.shape[1], 1], dtype=tf.float64)
    for q in range(Y.shape[2]):
        y = y + tf.expand_dims(tf.expand_dims(Wm[:, q], 1) * Y[:, :, q], axis=2)
    # y = tf.expand_dims(Wm, axis=2) * Y
    # y = tf.reduce_mean(Y, axis=2, keepdims=True)
    Y1 = Y - tf.tile(y, [1, 1, Y.shape[2]])
    P = (tf.matmul(tf.matmul(Y1, tf.matrix_diag(Wc)), tf.transpose(Y1, [0, 2, 1])) / (1 ** 2)) + R
    # P = tf.matmul(Y1, tf.transpose(Y1, [0, 2, 1])) + R

    return y, Y, P, Y1


def ut_meas_cw(X, weights, R, meas_mat, batch_size):
    # Y = X[:, :3, :]
    Y = tf.matmul(X, meas_mat, transpose_a=True, transpose_b=True)
    Y = tf.transpose(Y, [0, 2, 1])
    y = tf.zeros([batch_size, Y.shape[1], 1], dtype=tf.float64)
    weights = tf.cast(weights, tf.float64)
    y = tf.matmul(Y, weights[:, :, tf.newaxis]) / Y.shape[2].value
    # for q in range(Y.shape[2]):
    #     y = y + tf.expand_dims(tf.expand_dims(Wm[:, q], 1) * Y[:, :, q], axis=2)
    # y = tf.expand_dims(Wm, axis=2) * Y
    # y = tf.reduce_mean(Y, axis=2, keepdims=True)
    Y1 = Y - tf.tile(y, [1, 1, Y.shape[2]])
    P = tf.matmul(tf.matmul(Y1, tf.matrix_diag(weights)), tf.transpose(Y1, [0, 2, 1])) + R
    # P = tf.matmul(Y1, tf.transpose(Y1, [0, 2, 1])) + R

    return y, Y, P, Y1


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
  """Cosine decay schedule with warm up period.
  Cosine annealing learning rate as described in:
    Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
    ICLR 2017. https://arxiv.org/abs/1608.03983
  In this schedule, the learning rate grows linearly from warmup_learning_rate
  to learning_rate_base for warmup_steps, then transitions to a cosine decay
  schedule.
  Args:
    global_step: int64 (scalar) tensor representing global step.
    learning_rate_base: base learning rate.
    total_steps: total number of training steps.
    warmup_learning_rate: initial learning rate for warm up.
    warmup_steps: number of warmup steps.
    hold_base_rate_steps: Optional number of steps to hold base learning rate
      before decaying.
  Returns:
    a (scalar) float tensor representing learning rate.
  Raises:
    ValueError: if warmup_learning_rate is larger than learning_rate_base,
      or if warmup_steps is larger than total_steps.
  """
  if total_steps < warmup_steps:
    raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
  learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
      np.pi *
      (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
      ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
  if hold_base_rate_steps > 0:
    learning_rate = tf.where(global_step > warmup_steps + hold_base_rate_steps,
                             learning_rate, learning_rate_base)
  if warmup_steps > 0:
    if learning_rate_base < warmup_learning_rate:
      raise ValueError('learning_rate_base must be larger or equal to '
                       'warmup_learning_rate.')
    slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
    warmup_rate = slope * tf.cast(global_step,
                                  tf.float32) + warmup_learning_rate
    learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                             learning_rate)
  return tf.where(global_step > total_steps, 0.0, learning_rate,
                  name='learning_rate')


from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import random_ops

_Linear = core_rnn_cell._Linear

def _random_exp_initializer(minval, maxval, seed=None, dtype=dtypes.float64):
  """Returns an exponential distribution initializer.

  Args:
    minval: float or a scalar float Tensor. With value > 0. Lower bound of the
        range of random values to generate.
    maxval: float or a scalar float Tensor. With value > minval. Upper bound of
        the range of random values to generate.
    seed: An integer. Used to create random seeds.
    dtype: The data type.

  Returns:
    An initializer that generates tensors with an exponential distribution.
  """

  def _initializer(shape, dtype=dtype, partition_info=None):
    del partition_info  # Unused.
    return math_ops.exp(
        random_ops.random_uniform(
            shape, math_ops.log(minval), math_ops.log(maxval), dtype,
            seed=seed))

  return _initializer

class PhasedLSTMCell(rnn_cell_impl.RNNCell):
  """Phased LSTM recurrent network cell.

  https://arxiv.org/pdf/1610.09513v1.pdf
  """

  def __init__(self,
               num_units,
               use_peepholes=False,
               leak=0.001,
               ratio_on=0.1,
               trainable_ratio_on=True,
               period_init_min=1.0,
               period_init_max=1000.0,
               reuse=None):
    """Initialize the Phased LSTM cell.

    Args:
      num_units: int, The number of units in the Phased LSTM cell.
      use_peepholes: bool, set True to enable peephole connections.
      leak: float or scalar float Tensor with value in [0, 1]. Leak applied
          during training.
      ratio_on: float or scalar float Tensor with value in [0, 1]. Ratio of the
          period during which the gates are open.
      trainable_ratio_on: bool, weather ratio_on is trainable.
      period_init_min: float or scalar float Tensor. With value > 0.
          Minimum value of the initialized period.
          The period values are initialized by drawing from the distribution:
          e^U(log(period_init_min), log(period_init_max))
          Where U(.,.) is the uniform distribution.
      period_init_max: float or scalar float Tensor.
          With value > period_init_min. Maximum value of the initialized period.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope. If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(PhasedLSTMCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._leak = tf.cast(leak, tf.float64)
    self._ratio_on = ratio_on
    self._trainable_ratio_on = trainable_ratio_on
    self._period_init_min = tf.cast(period_init_min, tf.float64)
    self._period_init_max = tf.cast(period_init_max, tf.float64)
    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _mod(self, x, y):
    """Modulo function that propagates x gradients."""
    return array_ops.stop_gradient(math_ops.mod(x, y) - x) + x

  def _get_cycle_ratio(self, time, phase, period):
    """Compute the cycle ratio in the dtype of the time."""
    phase_casted = math_ops.cast(phase, dtype=time.dtype)
    period_casted = math_ops.cast(period, dtype=time.dtype)
    shifted_time = time - phase_casted
    cycle_ratio = self._mod(shifted_time, period_casted) / period_casted
    return math_ops.cast(cycle_ratio, dtype=tf.float64)

  def call(self, inputs, state, scope=None):
    """Phased LSTM Cell.

    Args:
      inputs: A tuple of 2 Tensor.
         The first Tensor has shape [batch, 1], and type float32 or float64.
         It stores the time.
         The second Tensor has shape [batch, features_size], and type float32.
         It stores the features.
      state: rnn_cell_impl.LSTMStateTuple, state from previous timestep.

    Returns:
      A tuple containing:
      - A Tensor of float32, and shape [batch_size, num_units], representing the
        output of the cell.
      - A rnn_cell_impl.LSTMStateTuple, containing 2 Tensors of float32, shape
        [batch_size, num_units], representing the new state and the output.
    """
    (c_prev, h_prev) = state
    (time, x) = inputs

    in_mask_gates = [x, h_prev]
    if self._use_peepholes:
      in_mask_gates.append(c_prev)

    with vs.variable_scope("mask_gates"):
      if self._linear1 is None:
        self._linear1 = _Linear(in_mask_gates, 2 * self._num_units, True)

      mask_gates = math_ops.sigmoid(self._linear1(in_mask_gates))
      [input_gate, forget_gate] = array_ops.split(
          axis=1, num_or_size_splits=2, value=mask_gates)

    with vs.variable_scope("new_input"):
      if self._linear2 is None:
        self._linear2 = _Linear([x, h_prev], self._num_units, True)
      new_input = math_ops.tanh(self._linear2([x, h_prev]))

    new_c = (c_prev * forget_gate + input_gate * new_input)

    in_out_gate = [x, h_prev]
    if self._use_peepholes:
      in_out_gate.append(new_c)

    with vs.variable_scope("output_gate"):
      if self._linear3 is None:
        self._linear3 = _Linear(in_out_gate, self._num_units, True)
      output_gate = math_ops.sigmoid(self._linear3(in_out_gate))

    new_h = math_ops.tanh(new_c) * output_gate

    period = vs.get_variable(
        "period", [self._num_units],
        initializer=_random_exp_initializer(self._period_init_min,
                                            self._period_init_max, dtype=tf.float64),
        dtype=tf.float64)

    phase = vs.get_variable(
        "phase", [self._num_units],
        initializer=init_ops.random_uniform_initializer(0.,
                                                        period.initial_value, dtype=tf.float64),
        dtype=tf.float64)

    ratio_on = vs.get_variable(
        "ratio_on", [self._num_units],
        initializer=init_ops.constant_initializer(self._ratio_on, dtype=tf.float64),
        trainable=self._trainable_ratio_on,
        dtype=tf.float64)

    cycle_ratio = self._get_cycle_ratio(time, phase, period)

    k_up = 2 * cycle_ratio / ratio_on
    k_down = 2 - k_up
    k_closed = self._leak * cycle_ratio

    k = array_ops.where(cycle_ratio < ratio_on, k_down, k_closed)
    k = array_ops.where(cycle_ratio < 0.5 * ratio_on, k_up, k)

    new_c = k * new_c + (1 - k) * c_prev
    new_h = k * new_h + (1 - k) * h_prev

    new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)

    return new_h, new_state


def discriminator_loss(real, fake):

    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    loss = real_loss + fake_loss

    return loss


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
