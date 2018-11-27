import os
import numpy as np
import tensorflow as tf
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

tfd = tfp.distributions
_state_size_with_prefix = rnn_cell_impl._zero_state_tensors


def weighted_mape_tf(y_true, y_pred, weight, divide=False, name=''):
    with tf.variable_scope('weighted_mape'):
        mult = 1
        num = tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(y_true * mult, y_pred * mult))) * math_ops.to_double(weight), name=name+'sum')

        if divide is True:
            den = tf.reduce_sum(tf.sqrt(tf.sqrt(tf.square(y_true))) * mult, name=name+'mean')
            den = tf.clip_by_value(den, clip_value_min=0.001, clip_value_max=1e9)
            wmape = num / den
        else:
            wmape = num

        return wmape


def get_QP(dt, om, zm, I_3z, I_4z, zb, dimension=3, sjix=50e-6, sjiy=50e-6, sjiz=50e-6, aji=0.1):

    dt = dt[:, tf.newaxis, :]

    dt7 = dt ** 7
    dt6 = dt ** 6
    dt5 = dt ** 5
    dt4 = dt ** 4
    dt3 = dt ** 3
    dt2 = dt ** 2

    aj = aji[:, :, tf.newaxis]

    # aj7 = tf.pow(aj, 7)
    # aj6 = tf.pow(aj, 6)
    # aj5 = tf.pow(aj, 5)
    # aj4 = tf.pow(aj, 4)
    # aj3 = tf.pow(aj, 3)
    # aj2 = tf.pow(aj, 2)

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

    # emadt = tf.exp(-aj * dt)

    # q11j = ((1 / (2 * aj7)) * (((aj5 * dt5) / 10) - ((aj4 * dt4) / 2) + ((4 * aj3 * dt3) / 3)
    #                            + (2 * aj * dt) - (2 * aj2 * dt2) - 3 + (4 * emadt) + (2 * aj2 * dt2 * emadt) - tf.exp(-2 * aj * dt)))
    #
    # q22j = ((1 / (2 * aj5)) * (1 - tf.exp(-2 * aj * dt) + ((2 * aj3 * dt3) / 2) + (2 * aj * dt) - (2 * aj2 * dt2) - (4 * aj * dt * emadt)))
    # q33j = ((1 / (2 * aj3)) * (4 * emadt + (2 * aj * dt) - (tf.exp(-2 * aj * dt)) - 3))
    # q44j = ((1 / (2 * aj)) * (1 - tf.exp(-2 * aj * dt)))
    #
    # q12j = (1 / (2 * aj6)) * (1 - (2 * aj * dt) + (2 * aj2 * dt2) - (aj3 * dt3) + ((aj4 * dt4) / 4)
    #                           + tf.exp(-2 * aj * dt) + (2 * aj * dt * emadt) - (2 * emadt) - (aj2 * dt2 * emadt))
    # q13j = (1 / (2 * aj5)) * (((aj3 * dt3) / 3) + (2 * aj * dt) - (aj2 * dt2) - 3
    #                           + (4 * emadt) + (aj2 * dt2 * emadt) - tf.exp(-2 * aj * dt))
    #
    # # q14j = ((1 / (2 * aj4)) * (1 - (2 * tf.exp(-2 * aj * dt)) - (aj2 * dt2 * emadt) + tf.exp(-2 * aj * dt)))
    # q14j = ((1 / (2 * aj4)) * (1 + tf.exp(-2 * aj * dt) - (2 * emadt) - (aj2 * dt2 * emadt)))
    #
    # q23j = ((1 / (2 * aj4)) * (1 - (2 * aj * dt) + (aj2 * dt2) + (2 * aj * dt * emadt) + tf.exp(-2 * aj * dt) - 2 * emadt))
    # q24j = ((1 / (2 * aj3)) * (1 - 2 * aj * dt * emadt - tf.exp(-2 * aj * dt)))
    # q34j = ((1 / (2 * aj2)) * (1 - 2 * emadt + tf.exp(-2 * aj * dt)))
    #
    # pj = ((2 - (2 * aj * dt) + (aj2 * dt2) - 2 * emadt) / (2 * aj3))
    # qj = ((emadt - 1 + (aj * dt)) / aj2)
    # rj = ((1 - emadt) / aj)
    # sj = emadt

    sj1 = 2 * tf.cast(sjix[:, :, tf.newaxis], dtype=tf.float64) * aj
    sj2 = 2 * tf.cast(sjiy[:, :, tf.newaxis], dtype=tf.float64) * aj
    sj3 = 2 * tf.cast(sjiz[:, :, tf.newaxis], dtype=tf.float64) * aj

    if dimension == 4:

        # zeta1j = tf.concat(
        #     [tf.concat([q11j, q12j, q13j, q14j], axis=2), tf.concat([q12j, q22j, q23j, q24j], axis=2), tf.concat([q13j, q23j, q33j, q34j], axis=2), tf.concat([q14j, q24j, q34j, q44j], axis=2)],
        #     axis=1) * sj1
        #
        # zeta2j = tf.concat(
        #     [tf.concat([q11j, q12j, q13j, q14j], axis=2), tf.concat([q12j, q22j, q23j, q24j], axis=2), tf.concat([q13j, q23j, q33j, q34j], axis=2), tf.concat([q14j, q24j, q34j, q44j], axis=2)],
        #     axis=1) * sj2
        #
        # zeta3j = tf.concat(
        #     [tf.concat([q11j, q12j, q13j, q14j], axis=2), tf.concat([q12j, q22j, q23j, q24j], axis=2), tf.concat([q13j, q23j, q33j, q34j], axis=2), tf.concat([q14j, q24j, q34j, q44j], axis=2)],
        #     axis=1) * sj3
        #
        # Q = tf.concat([tf.concat([zeta1j, I_4z, I_4z], axis=2), tf.concat([I_4z, zeta2j, I_4z], axis=2), tf.concat([I_4z, I_4z, zeta3j], axis=2)], axis=1)
        #
        # phi = tf.concat([tf.concat([om, dt, q34, pj], axis=2), tf.concat([zm, om, dt, qj], axis=2), tf.concat([zm, zm, om, rj], axis=2), tf.concat([zm, zm, zm, sj], axis=2)], axis=1)
        #
        # A = tf.concat([tf.concat([phi, I_4z, I_4z], axis=2), tf.concat([I_4z, phi, I_4z], axis=2), tf.concat([I_4z, I_4z, phi], axis=2)], axis=1)

        zeta1 = tf.concat([tf.concat([q11, q12, q13, q14], axis=2),
                           tf.concat([q12, q22, q23, q24], axis=2),
                           tf.concat([q13, q23, q33, q34], axis=2),
                           tf.concat([q14, q24, q34, q44], axis=2)], axis=1) * sj1
        zeta2 = tf.concat([tf.concat([q11, q12, q13, q14], axis=2),
                           tf.concat([q12, q22, q23, q24], axis=2),
                           tf.concat([q13, q23, q33, q34], axis=2),
                           tf.concat([q14, q24, q34, q44], axis=2)], axis=1) * sj2
        zeta3 = tf.concat([tf.concat([q11, q12, q13, q14], axis=2),
                           tf.concat([q12, q22, q23, q24], axis=2),
                           tf.concat([q13, q23, q33, q34], axis=2),
                           tf.concat([q14, q24, q34, q44], axis=2)], axis=1) * sj3

        Q = tf.concat([tf.concat([zeta1, I_4z, I_4z], axis=2),
                       tf.concat([I_4z, zeta2, I_4z], axis=2),
                       tf.concat([I_4z, I_4z, zeta3], axis=2)], axis=1)

        phi2 = tf.concat([tf.concat([om, dt, q34, q24], axis=2),
                          tf.concat([zm, om, dt, q34], axis=2),
                          tf.concat([zm, zm, om, dt], axis=2),
                          tf.concat([zm, zm, zm, om], axis=2)], axis=1)

        A = tf.concat([tf.concat([phi2, I_4z, I_4z], axis=2),
                       tf.concat([I_4z, phi2, I_4z], axis=2),
                       tf.concat([I_4z, I_4z, phi2], axis=2)], axis=1)

        tb = tf.concat([tf.concat([q34, q24], axis=2),
                        tf.concat([q44, q34], axis=2),
                        tf.concat([om, dt], axis=2),
                        tf.concat([zm, om], axis=2)], axis=1)
        # tb = tf.concat([pj, qj, rj, sj], axis=1)
        # tb = tf.concat([q24, q34, dt, om], axis=1)

        # B = tf.concat([tf.concat([tb, zb, zb], axis=2), tf.concat([zb, tb, zb], axis=2), tf.concat([zb, zb, tb], axis=2)], axis=1)
        B = tf.concat([tf.concat([tb, zb, zb], axis=2),
                       tf.concat([zb, tb, zb], axis=2),
                       tf.concat([zb, zb, tb], axis=2)], axis=1)

    elif dimension == 3:
        zeta1 = tf.concat([tf.concat([q22, q23, q24], axis=2), tf.concat([q23, q33, q34], axis=2), tf.concat([q24, q34, q44], axis=2)], axis=1)
        zeta2 = tf.concat([tf.concat([q22, q23, q24], axis=2), tf.concat([q23, q33, q34], axis=2), tf.concat([q24, q34, q44], axis=2)], axis=1)
        zeta3 = tf.concat([tf.concat([q22, q23, q24], axis=2), tf.concat([q23, q33, q34], axis=2), tf.concat([q24, q34, q44], axis=2)], axis=1)
        Q = tf.scalar_mul(2, tf.concat([tf.concat([zeta1, I_3z, I_3z], axis=2), tf.concat([I_3z, zeta2, I_3z], axis=2), tf.concat([I_3z, I_3z, zeta3], axis=2)], axis=1))

        phi = tf.concat([tf.concat([om, dt, q34], axis=2), tf.concat([zm, om, dt], axis=2), tf.concat([zm, zm, dt], axis=2)], axis=1)
        A = tf.concat([tf.concat([phi, I_3z, I_3z], axis=2), tf.concat([I_3z, phi, I_3z], axis=2), tf.concat([I_3z, I_3z, phi], axis=2)], axis=1)
        B = A

    return Q, A, B, A


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
