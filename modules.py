# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import tensorflow as tf


def batch_norm(inputs, name_scope, is_training, epsilon=1e-3, decay=0.99, dtype=tf.float32):
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        size = inputs.get_shape().as_list()[1]

        scale = tf.get_variable(
            'scale', [size], initializer=tf.constant_initializer(0.1, dtype=dtype), dtype=dtype)
        offset = tf.get_variable('offset', [size], dtype=dtype)

        population_mean = tf.get_variable(
            'population_mean', [size],
            initializer=tf.zeros_initializer(dtype=dtype), trainable=False, dtype=dtype)
        population_var = tf.get_variable(
            'population_var', [size],
            initializer=tf.ones_initializer(dtype=dtype), trainable=False, dtype=dtype)
        batch_mean, batch_var = tf.nn.moments(inputs, [0])

        # The following part is based on the implementation of :
        # https://github.com/cooijmanstim/recurrent-batch-normalization
        train_mean_op = tf.assign(
            population_mean,
            population_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(
            population_var, population_var * decay + batch_var * (1 - decay))

        if is_training is True:
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(
                    inputs, batch_mean, batch_var, offset, scale, epsilon)
        else:
            return tf.nn.batch_normalization(
                inputs, population_mean, population_var, offset, scale,
                epsilon)


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


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]
     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]
     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        # outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        # if scale:
        #     outputs = outputs * (num_units ** 0.5)

    return lookup_table


def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.
    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        # outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        # if scale:
        #     outputs = outputs * num_units ** 0.5

        return lookup_table


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        dtype=tf.float32):
    '''Applies multihead attention.

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
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=None, reuse=reuse)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None, reuse=reuse)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None, reuse=reuse)  # (N, T_k, C)

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
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

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
        outputs = normalize(outputs, reuse=reuse, dtype=dtype)  # (N, T_q, C)

    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None,
                dtype=tf.float32):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.elu, "use_bias": True, "reuse": reuse}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True, "reuse": reuse}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs, reuse=reuse, dtype=dtype)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)

# Imports
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

def temporal_padding(x, padding=(1, 1)):
    """Pads the middle dimension of a 3D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.
    # Returns
        A padded 3D tensor.
    """
    assert len(padding) == 2
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.pad(x, pattern)

def attentionBlock(x, counters, dropout):
    """self attention block
    # Arguments
        x: Tensor of shape [N, L, Cin]
        counters: to keep track of names
        dropout: add dropout after attention
    # Returns
        A tensor of shape [N, L, Cin]
    """

    k_size = x.get_shape()[-1].value
    v_size = x.get_shape()[-1].value

    name = get_name('attention_block', counters)
    with tf.variable_scope(name):
        # [N, L, k_size]
        key = tf.layers.dense(x, units=k_size, activation=None, use_bias=False,
                              kernel_initializer=tf.random_normal_initializer(0, 0.01))
        key = tf.nn.dropout(key, 1.0 - dropout)
        # [N, L, k_size]
        query = tf.layers.dense(x, units=k_size, activation=None, use_bias=False,
                                kernel_initializer=tf.random_normal_initializer(0, 0.01))
        query = tf.nn.dropout(query, 1.0 - dropout)
        value = tf.layers.dense(x, units=v_size, activation=None, use_bias=False,
                                kernel_initializer=tf.random_normal_initializer(0, 0.01))
        value = tf.nn.dropout(value, 1.0 - dropout)

        logits = tf.matmul(query, key, transpose_b=True)
        logits = logits / np.sqrt(k_size)
        weights = tf.nn.softmax(logits, name="attention_weights")
        output = tf.matmul(weights, value)

    return output


@add_arg_scope
def weightNormConvolution1d(x, num_filters, dilation_rate, filter_size=3, stride=[1],
                            pad='VALID', init_scale=1., init=False, gated=False,
                            counters={}, reuse=False, dtype=tf.float32):
    """a dilated convolution with weight normalization (Salimans & Kingma 2016)
       Note that init part is NEVER used in our code
       It relates to the data-dependent init in original paper
    # Arguments
        x: A tensor of shape [N, L, Cin]
        num_filters: number of convolution filters
        dilation_rate: dilation rate / holes
        filter_size: window / kernel width of each filter
        stride: stride in convolution
        gated: use gated linear units (Dauphin 2016) as activation
    # Returns
        A tensor of shape [N, L, num_filters]
    """
    name = get_name('weight_norm_conv1d', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # currently this part is never used
        if init:
            print("initializing weight norm")
            # data based initialization of parameters
            V = tf.get_variable('V', [filter_size, int(x.get_shape()[-1]), num_filters],
                                dtype, tf.random_normal_initializer(0, 0.01),
                                trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1])

            # pad x
            left_pad = dilation_rate * (filter_size - 1)
            x = temporal_padding(x, (left_pad, 0))
            x_init = tf.nn.convolution(x, V_norm, pad, stride, [dilation_rate])
            #x_init = tf.nn.conv2d(x, V_norm, [1]+stride+[1], pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1])
            scale_init = init_scale/tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=dtype, initializer=scale_init,
                                trainable=True)
            b = tf.get_variable('b', dtype=dtype, initializer=-m_init*scale_init,
                                trainable=True)
            x_init = tf.reshape(scale_init, [1, 1, num_filters]) \
                                * (x_init - tf.reshape(m_init, [1, 1, num_filters]))
            # apply nonlinearity
            x_init = tf.nn.elu(x_init)
            return x_init

        else:
            # Gating mechanism (Dauphin 2016 LM with Gated Conv. Nets)
            if gated:
                num_filters = num_filters * 2

            # size of V is L, Cin, Cout
            V = tf.get_variable('V', [filter_size, int(x.get_shape()[-1]), num_filters],
                                dtype, tf.random_normal_initializer(0, 0.01),
                                trainable=True)
            g = tf.get_variable('g', shape=[num_filters], dtype=dtype,
                                initializer=tf.constant_initializer(1.), trainable=True)
            b = tf.get_variable('b', shape=[num_filters], dtype=dtype,
                                initializer=None, trainable=True)

            # size of input x is N, L, Cin

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1])

            # pad x for causal convolution
            left_pad = dilation_rate * (filter_size - 1)
            x = temporal_padding(x, (left_pad, 0))

            # calculate convolutional layer output
            x = tf.nn.bias_add(tf.nn.convolution(x, W, pad, stride, [dilation_rate]), b)

            # GLU
            if gated:
                split0, split1 = tf.split(x, num_or_size_splits=2, axis=2)
                split1 = tf.sigmoid(split1)
                x = tf.multiply(split0, split1)
            # ReLU
            else:
                # apply nonlinearity
                x = tf.nn.elu(x)

            print(x.get_shape())

            return x


def TemporalBlock(input_layer, out_channels, filter_size, stride, dilation_rate, counters,
                  dropout, init=False, atten=False, use_highway=False, gated=False, dtype=tf.float32):
    """temporal block in TCN (Bai 2018)
    # Arguments
        input_layer: A tensor of shape [N, L, Cin]
        out_channels: output dimension
        filter_size: receptive field of a conv. filter
        stride: same as what's need in conv. function
        dilation_rate: holes inbetween
        counters: to keep track of layer names
        dropout: prob. to drop weights
        atten: (not in TCN) add self attention block after Conv.
        use_highway: (not in TCN) use highway as residual connection
        gated: (not in TCN) use gated linear unit as activation
        init: (NEVER used) data-dependent initialization
    # Returns
        A tensor of shape [N, L, out_channels]
    """
    keep_prob = 1.0 - dropout

    in_channels = input_layer.get_shape()[-1]
    name = get_name('temporal_block', counters)
    with tf.variable_scope(name):

        # num_filters is the hidden units in TCN
        # which is the number of out channels
        conv1 = weightNormConvolution1d(input_layer, out_channels, dilation_rate,
                                        filter_size, [stride], counters=counters,
                                        init=init, gated=gated, dtype=dtype)
        # set noise shape for spatial dropout
        # refer to https://colab.research.google.com/drive/1la33lW7FQV1RicpfzyLq9H0SH1VSD4LE#scrollTo=TcFQu3F0y-fy
        # shape should be [N, 1, C]
        noise_shape = (tf.shape(conv1)[0], tf.constant(1), tf.shape(conv1)[2])
        out1 = tf.nn.dropout(conv1, keep_prob, noise_shape)
        if atten:
            out1 = attentionBlock(out1, counters, dropout)

        conv2 = weightNormConvolution1d(out1, out_channels, dilation_rate, filter_size,
            [stride], counters=counters, init=init, gated=gated, dtype=dtype)
        out2 = tf.nn.dropout(conv2, keep_prob, noise_shape)
        if atten:
            out2 = attentionBlock(out2, counters, dropout)

        # highway connetions or residual connection
        residual = None
        if use_highway:
            W_h = tf.get_variable('W_h', [1, int(input_layer.get_shape()[-1]), out_channels],
                                  dtype, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_h = tf.get_variable('b_h', shape=[out_channels], dtype=dtype,
                                  initializer=None, trainable=True)
            H = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)

            W_t = tf.get_variable('W_t', [1, int(input_layer.get_shape()[-1]), out_channels],
                                  dtype, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_t = tf.get_variable('b_t', shape=[out_channels], dtype=dtype,
                                  initializer=None, trainable=True)
            T = tf.nn.bias_add(tf.nn.convolution(input_layer, W_t, 'SAME'), b_t)
            T = tf.nn.sigmoid(T)
            residual = H*T + input_layer * (1.0 - T)
        elif in_channels != out_channels:
            W_h = tf.get_variable('W_h', [1, int(input_layer.get_shape()[-1]), out_channels],
                                  dtype, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_h = tf.get_variable('b_h', shape=[out_channels], dtype=dtype,
                                  initializer=None, trainable=True)
            residual = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)
        else:
            print("no residual convolution")

        res = input_layer if residual is None else residual

        return tf.nn.elu(out2 + res)


def TemporalConvNet(input_layer, num_channels, sequence_length, kernel_size=2,
                    dropout=tf.constant(0.0, dtype=tf.float64), init=False,
                    atten=False, use_highway=False, use_gated=False, dtype=tf.float32):
    """A stacked dilated CNN architecture described in Bai 2018
    # Arguments
        input_layer: Tensor of shape [N, L, Cin]
        num_channels: # of filters for each CNN layer
        kernel_size: kernel for every CNN layer
        dropout: channel dropout after CNN
        atten: (not in TCN) add self attention block after Conv.
        use_highway: (not in TCN) use highway as residual connection
        gated: (not in TCN) use gated linear unit as activation
        init: (NEVER used) data-dependent initialization
    # Returns
        A tensor of shape [N, L, num_channels[-1]]
    """
    num_levels = len(num_channels)
    counters = {}
    for i in range(num_levels):
        print(i)
        dilation_size = 2 ** i
        out_channels = num_channels[i]
        input_layer = TemporalBlock(input_layer, out_channels, kernel_size, stride=1, dilation_rate=dilation_size,
                                    counters=counters, dropout=dropout, init=init, atten=atten, gated=use_gated, dtype=dtype)

    return input_layer



def attention_time(inputs, attention_size, time_major=False, return_alphas=False, dtype=tf.float32):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1, dtype=dtype), name='w_omega')
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1, dtype=dtype), name='b_omega')
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1, dtype=dtype), name='u_omega')

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas