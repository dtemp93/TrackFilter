from propagation_utils import *
from helper3 import *
from helper2 import *
from load_all_data_4 import DataServerLive
from modules import *
import math
from plotting import *
from attention import *

import tensorflow as tf
import tensorflow.contrib as tfc
from tensorflow.contrib.layers import fully_connected as FCL
import numpy as np
import tensorflow_probability as tfp
from tensorflow.python.ops import init_ops

tfd = tfp.distributions
tfb = tfp.bijectors

# # sensor_locations
# [0, 0.0449, 8]
# [-0.18, 0.0449, 8]
# [0.45, 0.0449, 8]

setattr(tfc.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)


def create_dist(loc, scale, scope):
    # qmatrix = FCL(tf.cast(scale, tf.float32), self.num_state, activation_fn=tf.nn.tanh, scope=scope)
    # qmatrix = tf.reshape(qmatrix, [self.batch_size, self.num_state, self.num_state])
    # qloc = FCL(tf.cast(loc, tf.float64), self.num_state, activation_fn=None, scope='mu')
    # qchol = tfd.matrix_diag_transform(tf.cast(qmatrix, tf.float32), tf.nn.softplus)
    # qchol = tf.linalg.LinearOperatorLowerTriangular(qchol).to_dense()
    x = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
    return x


class Filter(object):
    def __init__(self, sess, trainable_state=False, state_type='GRU', mode='training',
                 data_dir='', filter_name='', plot_dir='', save_dir='', log_dir='',
                 F_hidden=12, R_hidden=12, num_state=12, num_meas=3, max_seq=2,
                 max_epoch=10000, RE=6378137, GM=398600441890000, batch_size=10,
                 window_mode=False, pad_front=False, constant=False):

        self.sess = sess
        self.mode = mode
        self.max_seq = max_seq
        self.num_mixtures = 10
        self.max_sj = 100
        self.min_sj = 1e-3
        self.max_at = 1
        self.min_at = 0.1
        self.train_init_state = trainable_state
        self.F_hidden = F_hidden
        self.R_hidden = R_hidden
        self.num_state = num_state
        self.num_meas = num_meas
        self.plot_dir = plot_dir
        self.checkpoint_dir = save_dir
        self.log_dir = log_dir
        self.GM = GM
        self.max_epoch = max_epoch
        self.RE = RE
        self.state_type = state_type
        self.window_mode = window_mode
        self.filter_name = filter_name
        self.pad_front = pad_front
        self.constant = constant

        self.batch_size_np = batch_size
        # tf.set_random_seed(1)

        # print('Using Advanced Datagen 2 ')
        # self.meas_dir = 'D:/TrackFilterData/Delivery_13/5k25Hz_oop_broad_data/NoiseRAE/'
        # self.state_dir = 'D:/TrackFilterData/Delivery_13/5k25Hz_oop_broad_data/Translate/'

        print('Using Advanced Datagen 2 ')
        # self.meas_dir = 'D:/TrackFilterData/Delivery_11/5k25Hz_oop_data/NoiseRAE/'
        self.meas_dir = data_dir + '/NoiseRAE/'
        self.state_dir = data_dir + '/Translate/'
        # self.state_dir_rae = data_dir + ''
        # self.state_dir = 'D:/TrackFilterData/Delivery_11/5k25Hz_oop_data/Translate/'

        # self.log_dir = self.log_dir + '/' + filter_name
        # summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        self.global_step = tf.Variable(initial_value=0, name="global_step", trainable=False, collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES], dtype=tf.int32)
        self.batch_step = tf.Variable(0.0, trainable=False)
        self.drop_rate = tf.Variable(0.5, trainable=False, dtype=tf.float64)
        self.learning_rate_inp = tf.Variable(0.0, trainable=False, dtype=tf.float64)
        self.deterministic = tf.constant(False)

        # Meta Variables
        plen = int(self.max_seq)
        pi_val = tf.constant(math.pi, dtype=tf.float64)

        self.filters = {}
        self.hiways = {}
        self.projects = {}
        self.vdtype = tf.float64
        self.vdp_np = np.float64
        # fading = tf.Variable(0.9, trainable=False) * tf.pow(0.99, (global_step / (1500*40/self.max_seq))) + 0.05

        self.seqlen = tf.placeholder(tf.int32, [None])
        self.int_time = tf.placeholder(tf.float64, [None, self.max_seq])
        self.batch_size = tf.shape(self.seqlen)[0]

        alpha = 1e-3 * tf.ones([self.batch_size, 1], dtype=self.vdtype)
        beta = 2. * tf.ones([self.batch_size, 1], dtype=self.vdtype)
        k = 0. * tf.ones([self.batch_size, 1], dtype=self.vdtype)

        L = tf.cast(self.num_state, dtype=self.vdtype)
        lam = alpha * (L + k) - L
        c1 = L + lam
        tmat = tf.ones([1, 2 * self.num_state], dtype=self.vdtype)
        self.Wm = tf.concat([(lam / c1), (0.5 / c1) * tmat], axis=1)
        Wc1 = tf.expand_dims(copy.copy(self.Wm[:, 0]), axis=1) + (tf.ones_like(alpha, dtype=self.vdtype) - (alpha) + beta)
        self.Wc = tf.concat([Wc1, copy.copy(self.Wm[:, 1:])], axis=1)
        self.c = tf.sqrt(c1)

    def hiway_layer(self, x, name='', dtype=tf.float32):
        if name in self.hiways:
            var_reuse = True
        else:
            var_reuse = False
            self.hiways[name] = name
        with tf.variable_scope(name + 'hiway', reuse=tf.AUTO_REUSE):

            factor = 2.0
            n = 1.
            # trunc_stddev = math.sqrt(1.3 * factor / n)
            trunc_stddev = 0.01
            hiway_width = int(x.get_shape()[-1])
            wxy = tf.get_variable(name + '_hiway_wxy', None, dtype, tf.random_normal([int(x.get_shape()[-1]), hiway_width], stddev=trunc_stddev, dtype=dtype))
            bxy = tf.get_variable(name + '_hiway_bxy', None, dtype, tf.random_normal([hiway_width], stddev=0.2 / hiway_width, dtype=dtype))
            zy = tf.add(bxy, tf.matmul(x, wxy))
            # z1_bn = self.batch_norm_wrapper(z1, live, is_training=mode,layer=name+'_filter_l1')
            hy = tf.nn.elu(zy, name='hiway_y')

            wxc = tf.get_variable(name + '_hiway_wxc', None, dtype, tf.random_normal([int(x.get_shape()[-1]), hiway_width], stddev=trunc_stddev, dtype=dtype))
            bxc = tf.get_variable(name + '_hiway_bxc', None, dtype, tf.constant(-2.0, shape=[hiway_width], dtype=dtype))
            zc = tf.add(bxc, tf.matmul(x, wxc))
            # z1_bn = self.batch_norm_wrapper(z1, live, is_training=mode,layer=name+'_filter_l1')
            hiway_t = tf.nn.sigmoid(zc, name='hiway_t')
            hiway_c = tf.subtract(tf.constant(1.0, dtype=dtype), hiway_t)

            y = tf.add(tf.multiply(hy, hiway_t), tf.multiply(x, hiway_c))
            return y, hiway_t

    def filterh(self, x, filter_width=-1, y_width=-1, name='', dtype=tf.float32):
        if name in self.filters:
            var_reuse = True
        else:
            var_reuse = False
            self.filters[name] = name
        with tf.variable_scope(name + 'filter', reuse=tf.AUTO_REUSE):
            filter_in_size = int(x.get_shape()[-1])
            if filter_width == -1:  # caller didn't supply, so use x width
                filter_width = int(x.get_shape()[-1])
            if filter_width != int(x.get_shape()[-1]):
                xf, wp = self.project(x, y_width=filter_width, name=name + 'f_input_project', dtype=dtype)
            else:
                xf = x

            o1, o1_t = self.hiway_layer(xf, name=name + '1', dtype=dtype)
            o2, o2_t = self.hiway_layer(o1, name=name + '2', dtype=dtype)
            o3, o3_t = self.hiway_layer(o2, name=name + '3', dtype=dtype)
            o4, o4_t = self.hiway_layer(o3, name=name + '4', dtype=dtype)
            if y_width != -1 and y_width != filter_width:
                return self.project(o4, y_width=y_width, name=name + '_filter_project_to_y', dtype=dtype)[0], o4_t
            else:
                return o4, o4_t

    def hiway_layer2(self, x, name='', dtype=tf.float32):
        if name in self.hiways:
            var_reuse = True
        else:
            var_reuse = False
            self.hiways[name] = name
        with tf.variable_scope(name + 'hiway', reuse=tf.AUTO_REUSE):

            factor = 2.0
            n = 1.
            # trunc_stddev = math.sqrt(1.3 * factor / n)
            trunc_stddev = 0.01
            hiway_width = int(x.get_shape()[-1])
            wxy = tf.get_variable(name + '_hiway_wxy', None, dtype, tf.random_normal([int(x.get_shape()[-1]), hiway_width], stddev=trunc_stddev, dtype=dtype))
            bxy = tf.get_variable(name + '_hiway_bxy', None, dtype, tf.random_normal([hiway_width], stddev=0.2 / hiway_width, dtype=dtype))
            # zy = tf.add(bxy, tf.matmul(x, wxy))
            zy = tf.add(bxy, tf.tensordot(x, wxy, axes=1))
            # z1_bn = self.batch_norm_wrapper(z1, live, is_training=mode,layer=name+'_filter_l1')
            hy = tf.nn.elu(zy, name='hiway_y')

            wxc = tf.get_variable(name + '_hiway_wxc', None, dtype, tf.random_normal([int(x.get_shape()[-1]), hiway_width], stddev=trunc_stddev, dtype=dtype))
            bxc = tf.get_variable(name + '_hiway_bxc', None, dtype, tf.constant(-2.0, shape=[hiway_width], dtype=dtype))
            # zc = tf.add(bxc, tf.matmul(x, wxc))
            zc = tf.add(bxc, tf.tensordot(x, wxc, axes=1))
            # z1_bn = self.batch_norm_wrapper(z1, live, is_training=mode,layer=name+'_filter_l1')
            hiway_t = tf.nn.sigmoid(zc, name='hiway_t')
            hiway_c = tf.subtract(tf.constant(1.0, dtype=dtype), hiway_t)

            y = tf.add(tf.multiply(hy, hiway_t), tf.multiply(x, hiway_c))
            return y, hiway_t

    def filterh2(self, x, filter_width=-1, y_width=-1, name='', dtype=tf.float32):
        if name in self.filters:
            var_reuse = True
        else:
            var_reuse = False
            self.filters[name] = name
        with tf.variable_scope(name + 'filter', reuse=tf.AUTO_REUSE):
            filter_in_size = int(x.get_shape()[-1])
            if filter_width == -1:  # caller didn't supply, so use x width
                filter_width = int(x.get_shape()[-1])
            if filter_width != int(x.get_shape()[-1]):
                xf, wp = self.project2(x, y_width=filter_width, name=name + 'f_input_project', dtype=dtype)
            else:
                xf = x

            o1, o1_t = self.hiway_layer2(xf, name=name + '1', dtype=dtype)
            o2, o2_t = self.hiway_layer2(o1, name=name + '2', dtype=dtype)
            o3, o3_t = self.hiway_layer2(o2, name=name + '3', dtype=dtype)
            o4, o4_t = self.hiway_layer2(o3, name=name + '4', dtype=dtype)
            if y_width != -1 and y_width != filter_width:
                return self.project2(o4, y_width=y_width, name=name + '_filter_project_to_y', dtype=dtype)[0], o4_t
            else:
                return o4, o4_t

    def filter_layer(self, x, filter_width=-1, y_width=-1, weight_sd=0.01, residual=True, name=''):
        if name in self.filters:
            var_reuse = True
        else:
            var_reuse = False
            self.filters[name] = name
        with tf.variable_scope(name + 'fl', reuse=var_reuse):
            filter_in_size = int(x.get_shape()[-1])
            if filter_width == -1:  # caller didn't supply, so use x width
                filter_width = int(x.get_shape()[-1])

            w1 = tf.get_variable(name + '_fl_w1', None, tf.float64,
                                 tf.random_normal([filter_in_size, filter_width], stddev=weight_sd, dtype=tf.float64))
            b1 = tf.get_variable(name + '_fl_b1', None, tf.float64,
                                 tf.random_normal([filter_width], stddev=weight_sd, dtype=tf.float64))
            z1 = tf.add(b1, tf.matmul(x, w1))
            o1 = tf.nn.elu(z1, name=name + '_fl_o1')
            # o1 = tf.nn.dropout(o1, keep_prob)

            w2 = tf.get_variable(name + '_fl_w2', None, tf.float64,
                                 tf.random_normal([filter_width, filter_width], stddev=weight_sd, dtype=tf.float64))
            b2 = tf.get_variable(name + '_fl_b2', None, tf.float64,
                                 tf.random_normal([filter_width], stddev=weight_sd, dtype=tf.float64))
            z2 = tf.add(b2, tf.matmul(o1, w2))
            o2 = tf.nn.elu(z2, name=name + '_fl_o2')
            # o2d = tf.nn.dropout(o2, keep_prob)

            if residual == False:
                o2_final = o2
            else:
                layer_name = 'residual_1'
                wr1 = tf.get_variable(name + '_fl_wr1', None, tf.float64,
                                      tf.random_normal([filter_width, filter_width], stddev=weight_sd, dtype=tf.float64))
                br1 = tf.get_variable(name + '_fl_br1', None, tf.float64,
                                      tf.random_normal([filter_width], stddev=weight_sd, dtype=tf.float64))
                zr1 = tf.add(br1, tf.matmul(o2, wr1))
                or1 = tf.nn.elu(zr1, name=name + '_fl_or1')
                # or1 = tf.nn.dropout(o1, keep_prob)

                layer_name = 'residual_2';
                wr2 = tf.get_variable(name + '_fl_wr2', None, tf.float64,
                                      tf.random_normal([filter_width, filter_width], stddev=weight_sd, dtype=tf.float64))
                br2 = tf.get_variable(name + '_fl_br2', None, tf.float64,
                                      tf.random_normal([filter_width], stddev=weight_sd, dtype=tf.float64))
                zr2 = tf.add(br2, tf.matmul(or1, wr2))
                # zr2_bn=self.batch_norm_wrapper(zr2, live, is_training=mode,layer=name+'_filter_bnr2')
                or2 = tf.nn.elu(zr2, name=name + '_or2')

                o2_final = tf.add(o2, or2)
            if y_width != -1 and y_width != filter_width:
                return self.project(o2_final, y_width=y_width, name=name + '_fl_prj_to_y')
            else:
                return o2_final

    def project(self, x, y_width, name='', dtype=tf.float64):
        if name in self.projects:
            var_reuse = True
        else:
            var_reuse = False
            self.projects[name] = name
        with tf.variable_scope(name + 'project', reuse=var_reuse):
            factor = 2.0
            n = 1.
            # trunc_stddev = math.sqrt(1.3 * factor / n)
            trunc_stddev = 0.01

            filter_in_size = int(x.get_shape()[-1])
            w1 = tf.get_variable(name + '_w1', None, dtype, tf.random_normal([filter_in_size, y_width], stddev=trunc_stddev, dtype=dtype))
            b1 = tf.get_variable(name + '_b1', None, dtype, tf.random_normal([y_width], stddev=trunc_stddev, dtype=dtype))
            y = tf.add(tf.matmul(x, w1), b1)
            return y

    def project2(self, x, y_width, name='', dtype=tf.float32):
        if name in self.projects:
            var_reuse = True
        else:
            var_reuse = False
            self.projects[name] = name
        with tf.variable_scope(name + 'project', reuse=var_reuse):
            factor = 2.0
            n = 1.
            # trunc_stddev = math.sqrt(1.3 * factor / n)
            trunc_stddev = 0.01

            filter_in_size = int(x.get_shape()[-1])
            w1 = tf.get_variable(name + '_w1', None, dtype, tf.random_normal([filter_in_size, y_width], stddev=trunc_stddev, dtype=dtype))
            b1 = tf.get_variable(name + '_b1', None, dtype, tf.random_normal([y_width], stddev=trunc_stddev, dtype=dtype))
            # y = tf.add(tf.matmul(x, w1), b1)
            y = tf.add(tf.tensordot(x, w1, axes=1), b1)
            return y, w1

    def alpha(self, current_time, int_time, dt, pstate, meas_rae, cov_est, Q_est, R_est, LLA, sensor_onehot, state1=None, state2=None, state3=None):

        lat = LLA[:, 0, tf.newaxis]
        lon = LLA[:, 1, tf.newaxis]
        alt = LLA[:, 2, tf.newaxis]

        R = meas_rae[:, 0, tf.newaxis]
        A = meas_rae[:, 1, tf.newaxis]
        E = meas_rae[:, 2, tf.newaxis]

        east = (R * tf.sin(A) * tf.cos(E))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        north = (R * tf.cos(E) * tf.cos(A))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        up = (R * tf.sin(E))  # * ((tf.exp(tf.negative(tf.pow(se, 2) / 2))))

        cosPhi = tf.cos(lat)
        sinPhi = tf.sin(lat)
        cosLambda = tf.cos(lon)
        sinLambda = tf.sin(lon)

        tv = cosPhi * up - sinPhi * north
        wv = sinPhi * up + cosPhi * north
        uv = cosLambda * tv - sinLambda * east
        vv = sinLambda * tv + cosLambda * east

        meas_uvw1 = tf.concat([uv, vv, wv], axis=1)

        # pre_residual = meas_uvw1[:, :, tf.newaxis] - tf.matmul(self.meas_mat, pstate[:, :, tf.newaxis])

        # acc_jer = tf.concat([pstate[:, 2, tf.newaxis], pstate[:, 3, tf.newaxis],
        #                      pstate[:, 6, tf.newaxis], pstate[:, 7, tf.newaxis],
        #                      pstate[:, 10, tf.newaxis], pstate[:, 11, tf.newaxis]], axis=1)

        # acc = tf.concat([pstate[:, 2, tf.newaxis], pstate[:, 6, tf.newaxis], pstate[:, 10, tf.newaxis]], axis=1)
        # jer = tf.concat([pstate[:, 3, tf.newaxis], pstate[:, 7, tf.newaxis], pstate[:, 11, tf.newaxis]], axis=1)

        # acc = normalize(acc, epsilon=1e-15, scope='ln1', reuse=tf.AUTO_REUSE, dtype=self.vdtype)
        # jer = normalize(jer, epsilon=1e-15, scope='ln2', reuse=tf.AUTO_REUSE, dtype=self.vdtype)
        cov_diag = tf.matrix_diag_part(cov_est)
        # Q_diag = tf.matrix_diag_part(Q_est)
        # R_diag = tf.matrix_diag_part(R_est)

        # pstate_n = normalize(pstate, epsilon=1e-15, scope='ln_state', reuse=tf.AUTO_REUSE, dtype=self.vdtype)
        # cov_diag_n = normalize(cov_diag, epsilon=1e-15, scope='ln_cov', reuse=tf.AUTO_REUSE, dtype=self.vdtype)
        # Q_diag_n = normalize(Q_diag, epsilon=1e-15, scope='ln_Q', reuse=tf.AUTO_REUSE, dtype=self.vdtype)
        # R_diag_n = normalize(R_diag, epsilon=1e-15, scope='ln_R', reuse=tf.AUTO_REUSE, dtype=self.vdtype)
        # pre_res_n = normalize(pre_residual[:, :, 0], epsilon=1e-15, scope='ln_res', reuse=tf.AUTO_REUSE, dtype=self.vdtype)

        layer_input = tf.concat([cov_diag], axis=1)
        rnn_inp = tf.concat([dt, layer_input, sensor_onehot], axis=1)
        # rnn_inp = FCL(layer_input, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp', reuse=tf.AUTO_REUSE)

        # rnn_inpr = self.filter_layer(layer_input, name='rnn_inp')
        # rnn_inp = tf.concat([layer_input, rnn_inpr], axis=1)
        # rnn_inp = FCL(layer_input, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp2', reuse=tf.AUTO_REUSE)
        rnn_inp = tfc.layers.dropout(rnn_inp, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_inputs')
        # rnn_inpa, _ = self.filterh(layer_input, y_width=self.F_hidden, name='rnn_inp/A', dtype=self.vdtype)
        # rnn_inpa = FCL(layer_input, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp/a', reuse=tf.AUTO_REUSE)

        with tf.variable_scope('Source_Track_Forward/state', reuse=tf.AUTO_REUSE):
            (out1, state1) = self.source_fwf((int_time, rnn_inp), state=state1)

        # out1 = normalize(out1, epsilon=1e-15, scope='ln2', reuse=tf.AUTO_REUSE, dtype=self.vdtype)

        with tf.variable_scope('Source_Track_Forward2/state', reuse=tf.AUTO_REUSE):
            (out2, state2) = self.source_fwf2((int_time, rnn_inp), state=state2)

        rm0 = FCL(tf.concat([out1], axis=1), 7, activation_fn=None, scope='r1/state', reuse=tf.AUTO_REUSE)
        rm1 = FCL(rm0, 7, activation_fn=None, scope='r2/state', reuse=tf.AUTO_REUSE)
        d_mult = (tf.nn.sigmoid(rm1[:, -1:]) * 50) + tf.ones_like(rm1[:, -1:])*1
        rd = tril_with_diag_softplus_and_shift(rm1[:, :6], diag_shift=1e-10, diag_mult=d_mult, name='tril/rd')
        # rd = FCL(rm1, 3, activation_fn=tf.nn.softplus, scope='rd', reuse=tf.AUTO_REUSE)
        # rd = tf.nn.sigmoid(rm1, 'rd') * tf.sqrt(pre_residual[:, :, 0])

        # sr = FCL(rm1[:, 0, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, scope='sr', reuse=tf.AUTO_REUSE) * 100 + tf.ones_like(rm1[:, 0, tf.newaxis]) * 1
        # sa = FCL(rm1[:, 1, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, scope='sa', reuse=tf.AUTO_REUSE) * 1e-1 + tf.ones_like(rm1[:, 0, tf.newaxis]) * 1e-3
        # se = FCL(rm1[:, 2, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, scope='sz', reuse=tf.AUTO_REUSE) * 1e-1 + tf.ones_like(rm1[:, 0, tf.newaxis]) * 1e-3

        # eastn = (sr * tf.sin(sa) * tf.cos(se))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        # northn = (sr * tf.cos(se) * tf.cos(sa))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        # upn = (sr * tf.sin(se))  # * ((tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        #
        # enu_noise = tf.concat([eastn, northn, upn], axis=1)
        #
        # tvn = cosPhi * upn - sinPhi * northn
        # wvn = sinPhi * upn + cosPhi * northn
        # uvn = cosLambda * tvn - sinLambda * eastn
        # vvn = sinLambda * tvn + cosLambda * eastn
        #
        # meas_uvw_noise = tf.concat([uvn, vvn, wvn], axis=1)
        #
        # meas_uvw = meas_uvw1 + meas_uvw_noise
        #
        # uvw_diff = meas_uvw1 - meas_uvw
        # uvw_cov = tf.matmul(uvw_diff[:, :, tf.newaxis], uvw_diff[:, :, tf.newaxis], transpose_b=True)
        #
        # uvw_diag = tf.matrix_diag_part(uvw_cov)
        # uvw_diag = tf.where(uvw_diag < 1, tf.ones_like(uvw_diag), uvw_diag)

        rdist = tfd.MultivariateNormalTriL(loc=None, scale_tril=rd)
        # rdist = tfd.MultivariateNormalDiag(loc=None, scale_diag=rd)
        Rt = rdist.covariance()

        meas_uvw = meas_uvw1  # + rdist.mean()

        # alpha0 = FCL(out[:, hp:], out[:, hp:].shape[1].value, activation_fn=None, scope='alpha0', reuse=tf.AUTO_REUSE)
        # alpha1 = FCL(alpha0, alpha0.shape[1].value, activation_fn=None, scope='alpha1', reuse=tf.AUTO_REUSE)
        # alpha = FCL(alpha1, self.num_mixtures, activation_fn=tf.nn.softmax, scope='alpha', reuse=tf.AUTO_REUSE)

        # u0 = FCL(out[:, hp:hp+6], 3, activation_fn=None, scope='u/1', reuse=tf.AUTO_REUSE)
        # ul = FCL(u0, 3, activation_fn=None, scope='u/2', reuse=tf.AUTO_REUSE)
        # u = FCL(ul, 3, activation_fn=None, scope='u/3', reuse=tf.AUTO_REUSE)

        u = tf.zeros([self.batch_size, 3], self.vdtype)

        qm0 = FCL(tf.concat([out2], axis=1), 3, activation_fn=None, scope='q1/state', reuse=tf.AUTO_REUSE)
        qm1 = FCL(qm0, 3, activation_fn=None, scope='q2/state', reuse=tf.AUTO_REUSE)
        qm2 = FCL(qm1, 3, activation_fn=tf.nn.softplus, scope='q3/state', reuse=tf.AUTO_REUSE)
        # qm2 = tf.nn.sigmoid(qm1, 'q3')

        # sjx = qm2[:, -3, tf.newaxis] * 100 + self.om[:, :, 0] * 0.1
        sjx = self.om[:, :, 0] * 100

        # sjy = qm2[:, -2, tf.newaxis] * 100 + self.om[:, :, 0] * 0.1
        sjy = self.om[:, :, 0] * 100

        # sjz = qm2[:, -1, tf.newaxis] * 100 + self.om[:, :, 0] * 0.1
        sjz = self.om[:, :, 0] * 100

        # Q_list = list()
        # A_list = list()
        # B_list = list()
        #
        # sj_inc = (self.max_sj - self.min_sj) / self.num_mixtures
        # # at_inc = (self.max_at - self.min_at) / self.num_mixtures
        #
        # for ppp in range(self.num_mixtures):
        #     sj = self.min_sj + sj_inc * ppp
        #     # at = self.min_at + at_inc * ppp
        #     Qtemp, Atemp, Btemp, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
        #                                     dimension=int(self.num_state / 3),
        #                                     sjix=self.om[:, :, 0] * sj ** 2,
        #                                     sjiy=self.om[:, :, 0] * sj ** 2,
        #                                     sjiz=self.om[:, :, 0] * sj ** 2,
        #                                     aji=self.om[:, :, 0] * 1.0)
        #     Q_list.append(Qtemp)
        #     A_list.append(Atemp)
        #     B_list.append(Btemp)
        #
        # At = tf.stack(A_list, axis=1)[0, :, :, :]
        # Qt = tf.stack(Q_list, axis=1)[0, :, :, :]
        # Bt = tf.stack(B_list, axis=1)[0, :, :, :]
        #
        # At = tf.matmul(alpha, tf.reshape(At, [-1, self.num_state * self.num_state]))
        # At = tf.reshape(At, [-1, self.num_state, self.num_state])
        #
        # Qt = tf.matmul(alpha, tf.reshape(Qt, [-1, self.num_state * self.num_state]))
        # Qt = tf.reshape(Qt, [-1, self.num_state, self.num_state])
        #
        # Bt = tf.matmul(alpha, tf.reshape(Bt, [-1, self.num_state * 3]))
        # Bt = tf.reshape(Bt, [-1, self.num_state, 3])

        Qt, At, Bt, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                               dimension=int(self.num_state / 3),
                               sjix=self.om[:, :, 0] * sjx ** 2,
                               sjiy=self.om[:, :, 0] * sjy ** 2,
                               sjiz=self.om[:, :, 0] * sjz ** 2,
                               aji=self.om[:, :, 0] * 1.0)

        # qc = tf.cholesky(Qt)
        # ac = tf.cholesky(At)
        # rc = tf.cholesky(Rt)

        # qdist = tfd.MultivariateNormalDiag(loc=None, scale_diag=tf.sqrt(tf.matrix_diag_part(Qt)))
        qdist = tfd.MultivariateNormalTriL(loc=None, scale_tril=tf.cholesky(Qt))
        Qt = qdist.covariance()

        return meas_uvw, Qt, At, Rt, Bt, u, state1, state2, state3

    def beta(self, current_time, int_time, pos_res, dt, mu_pred, Sigma_pred, sensor_onehot, cur_weight, state3=None):

        weight = cur_weight[:, :, tf.newaxis]

        layer_input = tf.concat([dt, pos_res, tf.matrix_diag_part(Sigma_pred)], axis=1)

        rnn_inpr = self.filter_layer(layer_input, name='rnn_inp_beta')
        rnn_inp = tf.concat([layer_input, rnn_inpr], axis=1)
        rnn_inp = FCL(rnn_inp, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp2_beta', reuse=tf.AUTO_REUSE)
        rnn_inp = tfc.layers.dropout(rnn_inp, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_inputs_beta')

        with tf.variable_scope('Source_Track_Forward3/state', reuse=tf.AUTO_REUSE):
            (out3, state3) = self.source_fwf3((int_time, rnn_inp), state=state3)

        cov_jer = tf.concat([tf.concat([Sigma_pred[:, 3, 3, tf.newaxis, tf.newaxis], Sigma_pred[:, 3, 7, tf.newaxis, tf.newaxis], Sigma_pred[:, 3, 11, tf.newaxis, tf.newaxis]], axis=2),
                             tf.concat([Sigma_pred[:, 7, 3, tf.newaxis, tf.newaxis], Sigma_pred[:, 7, 7, tf.newaxis, tf.newaxis], Sigma_pred[:, 7, 11, tf.newaxis, tf.newaxis]], axis=2),
                             tf.concat([Sigma_pred[:, 11, 3, tf.newaxis, tf.newaxis], Sigma_pred[:, 11, 7, tf.newaxis, tf.newaxis], Sigma_pred[:, 11, 11, tf.newaxis, tf.newaxis]], axis=2)],
                             axis=1)

        # smooth_jer = tf.concat([mu_pred[:, 3, tf.newaxis], mu_pred[:, 7, tf.newaxis], mu_pred[:, 11, tf.newaxis]], axis=1)

        n_samples = 10
        jerk_dist = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=cov_jer)
        jerk_samples = jerk_dist.sample(n_samples)
        jerk_samples = tf.transpose(jerk_samples, [1, 0, 2])

        ts = tf.tile(mu_pred[:, tf.newaxis, :], [1, n_samples, 1])

        all_states = tf.concat([ts[:, :, 0, tf.newaxis], ts[:, :, 1, tf.newaxis], ts[:, :, 2, tf.newaxis], jerk_samples[:, :, 0, tf.newaxis],
                                ts[:, :, 4, tf.newaxis], ts[:, :, 5, tf.newaxis], ts[:, :, 6, tf.newaxis], jerk_samples[:, :, 1, tf.newaxis],
                                ts[:, :, 8, tf.newaxis], ts[:, :, 9, tf.newaxis], ts[:, :, 10, tf.newaxis], jerk_samples[:, :, 2, tf.newaxis]], axis=2)

        # rnn_out = self.filter_layer(tf.concat([out3, rnn_inp], axis=1), name='rnn_out_beta')
        beta0 = FCL(out3, n_samples, activation_fn=None, scope='rnn_out2_beta0', reuse=tf.AUTO_REUSE)
        beta1 = FCL(beta0, n_samples, activation_fn=None, scope='rnn_out2_beta1', reuse=tf.AUTO_REUSE)
        beta = FCL(beta1, n_samples, activation_fn=tf.nn.softmax, scope='rnn_out2_beta', reuse=tf.AUTO_REUSE)
        beta = beta[:, tf.newaxis, :]

        mu_out = tf.squeeze(tf.matmul(beta, all_states), 1)

        # gain = tf.reshape(rnn_out2, [self.batch_size, 3])
        # gain = tf.where(tf.equal(cur_weight * gain, tf.zeros_like(gain)), tf.zeros_like(gain), gain)
        # pos_pred0 = tf.concat([mu_pred[:, 0, tf.newaxis], mu_pred[:, 4, tf.newaxis], mu_pred[:, 8, tf.newaxis]], axis=1)
        # pos_pred = pos_pred0 + gain * pos_res
        # mu_pred = mu_pred + tf.squeeze(tf.matmul(gain, pos_res[:, :, tf.newaxis]), -1)

        # mu_pred = tf.concat([pos_pred[:, 0, tf.newaxis], mu_pred[:, 1, tf.newaxis], mu_pred[:, 2, tf.newaxis], mu_pred[:, 3, tf.newaxis],
        #                      pos_pred[:, 1, tf.newaxis], mu_pred[:, 5, tf.newaxis], mu_pred[:, 6, tf.newaxis], mu_pred[:, 7, tf.newaxis],
        #                      pos_pred[:, 2, tf.newaxis], mu_pred[:, 9, tf.newaxis], mu_pred[:, 10, tf.newaxis], mu_pred[:, 11, tf.newaxis]], axis=1)

        return mu_out, state3

    def forward_step_fn(self, params, inputs):
        """
        Forward step over a batch, to be used in tf.scan
        :param params:
        :param inputs: (batch_size, variable dimensions)
        :return:
        """

        current_time = inputs[:, 0, tf.newaxis]
        prev_time = inputs[:, 1, tf.newaxis]
        int_time = inputs[:, 2, tf.newaxis]
        meas_rae = inputs[:, 3:6]
        cur_weight = inputs[:, 6, tf.newaxis]
        LLA = inputs[:, 7:10]
        sensor_onehot = inputs[:, 10:]

        weight = cur_weight[:, :, tf.newaxis]
        # inv_weight = tf.ones_like(weight) - weight

        _, _, mu_t0, Sigma_t0, _, state1, state2, state3, Qt0, Rt0, _, _, _, _, _ = params

        dt = current_time - prev_time
        dt = tf.where(dt <= 1 / 100, tf.ones_like(dt) * 1 / 25, dt)

        meas_uvw, Qt, At, Rt, Bt, u, state1, state2, state3 = self.alpha(current_time, int_time, dt, mu_t0, meas_rae,
                                                                         Sigma_t0, Qt0, Rt0, LLA, sensor_onehot,
                                                                         state1=state1, state2=state2, state3=state3)  # (bs, k)

        # Am = tf.expand_dims(self.c, axis=2) * tf.cholesky(tf.cast(Sigma_pred, self.vdtype))
        # Y = tf.tile(tf.expand_dims(mu_pred, axis=2), [1, 1, self.num_state])
        # X = tf.concat([tf.expand_dims(mu_pred, axis=2), Y + Am, Y - Am], axis=2)
        # X = tf.transpose(X, [0, 2, 1])

        mu_pred = tf.squeeze(tf.matmul(At, tf.expand_dims(mu_t0, 2)), -1) + tf.squeeze(tf.matmul(Bt, u[:, :, tf.newaxis]), -1)
        # mu_pred = mu_pred + tf.squeeze(tf.matmul(Bt, u[:, :, tf.newaxis]), -1)
        Sigma_pred = tf.matmul(tf.matmul(At, Sigma_t0), At, transpose_b=True) + Qt
        # Sigma_pred1 = tf.matmul(tf.matmul(At, Sigma_t), At, transpose_b=True)

        pos_res1 = meas_uvw[:, :, tf.newaxis] - tf.matmul(self.meas_mat, mu_pred[:, :, tf.newaxis])
        # pos_res1 = tf.where(cur_weight == 0, tf.zeros_like(pos_res1), pos_res1)
        # x1, X1, P1, X2 = ut_state_batch_no_prop(X, self.Wm, self.Wc, Qt, self.num_state, self.batch_size, At, dt)
        # z1, Z1, P2, Z2 = ut_meas(X1, self.Wm, self.Wc, Rt, self.meas_mat, self.batch_size)

        # P12 = tf.matmul(tf.matmul(X2, tf.matrix_diag(self.Wc)), Z2, transpose_b=True)
        # Rt = tf.eye(3, 3, batch_shape=self.batch_size) * 100
        # sc = tf.matrix_inverse(Sigma_pred)
        # rc = tf.matrix_inverse(Rt)

        sp1 = tf.matmul(tf.matmul(self.meas_mat, Sigma_pred), self.meas_mat, transpose_b=True)
        S = sp1 + Rt

        # tf.Print(Sigma_pred, [Sigma_pred], "covariance")
        # tf.Print(Rt, [Rt], "meas_covariance")

        S_inv = tf.matrix_inverse(S)
        gain = tf.matmul(tf.matmul(Sigma_pred, self.meas_mat, transpose_b=True), S_inv)
        gain = tf.where(tf.equal(weight * gain, tf.zeros_like(gain)), tf.zeros_like(gain), gain)
        # gain = tf.matmul(P12, tf.matrix_inverse(P2)) * cur_weight[:, tf.newaxis, :]

        mu_t = mu_pred[:, :, tf.newaxis] + tf.matmul(gain, pos_res1)
        mu_t = mu_t[:, :, 0]

        # mu_t, state3 = self.beta(current_time, int_time, pos_res1[:, :, 0], dt, mu_t, Sigma_pred0, sensor_onehot, cur_weight, state3=state3)

        I_KC = self.I_12 - tf.matmul(gain, self.meas_mat)  # (bs, dim_z, dim_z)
        Sigma_t = tf.matmul(tf.matmul(I_KC, Sigma_pred), I_KC, transpose_b=True) + tf.matmul(tf.matmul(gain, Rt), gain, transpose_b=True)
        Sigma_t = (Sigma_t + tf.transpose(Sigma_t, [0, 2, 1])) / 2

        # Am = tf.expand_dims(self.c, axis=2) * tf.cholesky(tf.cast(Sigma_t, self.vdtype))
        # Y = tf.tile(tf.expand_dims(mu_t, axis=2), [1, 1, self.num_state])
        # X = tf.concat([tf.expand_dims(mu_t, axis=2), Y + Am, Y - Am], axis=2)
        # X = tf.transpose(X, [0, 2, 1])

        # mu_pred, _, Sigma_pred, _ = ut_state_batch(X, self.Wm, self.Wc, Qt, self.num_state, self.batch_size, At, dt, self.sensor_ecef)
        # mu_pred = mu_pred[:, :, 0]

        # ballistic = propagatef2(mu_t, dt)

        # mu_pred = tf.squeeze(tf.matmul(At, tf.expand_dims(mu_t, 2)), -1) + tf.squeeze(tf.matmul(Bt, u[:, :, tf.newaxis]), -1)
        # # mu_pred = mu_pred + tf.squeeze(tf.matmul(Bt, u[:, :, tf.newaxis]), -1)
        # Sigma_pred = tf.matmul(tf.matmul(At, Sigma_t), At, transpose_b=True) + Qt
        # # Sigma_pred1 = tf.matmul(tf.matmul(At, Sigma_t), At, transpose_b=True)

        return mu_pred, Sigma_pred, mu_t, Sigma_t, meas_uvw, state1, state2, state3, Qt, Rt, At, Bt, S_inv, weight, u

    def backward_step_fn(self, params, inputs):
        """
        Backwards step over a batch, to be used in tf.scan
        :param params:
        :param inputs: (batch_size, variable dimensions)
        :return:
        """
        mu_back, Sigma_back = params
        mu_pred_tp1, Sigma_pred_tp1, mu_filt_t, Sigma_filt_t, A, weight = inputs

        # J_t = tf.matmul(tf.reshape(tf.transpose(tf.matrix_inverse(Sigma_pred_tp1), [0, 2, 1]), [-1, self.dim_z]),
        #                 self.A)
        # J_t = tf.transpose(tf.reshape(J_t, [-1, self.dim_z, self.dim_z]), [0, 2, 1])
        J_t = tf.matmul(tf.transpose(A, [0, 2, 1]), tf.matrix_inverse(Sigma_pred_tp1))
        J_t = tf.matmul(Sigma_filt_t, J_t)

        # J_t = tf.where(tf.equal(weight * J_t, tf.zeros_like(J_t)), tf.zeros_like(J_t), J_t)
        # J_t = tf.zeros_like(J_t)

        mu_back = mu_filt_t + tf.matmul(J_t, mu_back - mu_pred_tp1)
        Sigma_back = Sigma_filt_t + tf.matmul(J_t, tf.matmul(Sigma_back - Sigma_pred_tp1, J_t, adjoint_b=True))

        return mu_back, Sigma_back

    def compute_forwards(self):
        """Compute the forward step in the Kalman filter.
           The forward pass is intialized with p(z_1)=N(self.mu, self.Sigma).
           We then return the mean and covariances of the predictive distribution p(z_t|z_tm1,u_t), t=2,..T+1
           and the filtering distribution p(z_t|x_1:t,u_1:t), t=1,..T
           We follow the notation of Murphy's book, section 18.3.1
        """

        self.mu = self.state_input
        self.Sigma = tf.reshape(self.P_inp, [self.batch_size, self.num_state, self.num_state])

        all_time = tf.concat([self.prev_time[:, tf.newaxis], tf.stack(self.current_timei, axis=1)], axis=1)
        meas_rae = tf.concat([self.prev_measurement[:, tf.newaxis], tf.stack(self.measurement, axis=1)], axis=1)
        meas_time = all_time[:, 1:, :]
        prev_time = all_time[:, :-1, :]

        dt0 = meas_time[:, 0, :] - prev_time[:, 0, :]

        int_time = self.int_time

        sensor_lla = tf.expand_dims(self.sensor_lla, axis=1)
        sensor_lla = tf.tile(sensor_lla, [1, meas_rae.shape[1], 1])

        sensor_onehot = tf.expand_dims(self.sensor_onehots, axis=1)
        sensor_onehot = tf.tile(sensor_onehot, [1, meas_rae.shape[1], 1])

        inputs = tf.concat([meas_time, prev_time, int_time[:, :, tf.newaxis], meas_rae[:, 1:, :], self.seqweightin[:, :, tf.newaxis], sensor_lla[:, 1:, :], sensor_onehot[:, 1:, :]], axis=2)

        cov_input = self.prev_covariance_estimate
        init_Q = self.Q_inp
        init_R = self.R_inp

        meas_uvw, Qt, At, Rt, Bt, u, state1, state2, state3 = self.alpha(prev_time[:, 0, :], int_time[:, 0, tf.newaxis], dt0, self.prev_state_estimate, meas_rae[:, 0, :],
                                                                         cov_input, init_Q, init_R, sensor_lla[:, 0, :], sensor_onehot[:, 0, :],
                                                                         state1=self.state_fw_in_state, state2=self.state_fw_in_state2, state3=self.state_fw_in_state3)

        state1 = self.state_fw_in_state
        state2 = self.state_fw_in_state2

        # pos_res = meas_uvw - tf.squeeze(tf.matmul(self.meas_mat, self.prev_state_estimate[:, :, tf.newaxis]), -1)

        # mu_pred, state3 = self.beta(prev_time[:, 0, :], int_time[:, 0, tf.newaxis], pos_res, dt0, self.mu, self.Sigma, sensor_onehot[:, 0, :], self.seqweightin[:, 0, tf.newaxis], state3=state3)

        # init_Q = tf.ones([self.batch_size, 12, 12], self.vdtype)
        init_Si = tf.ones([self.batch_size, 3, 3], self.vdtype)
        init_A = tf.ones([self.batch_size, 12, 12], self.vdtype)
        init_B = tf.ones([self.batch_size, 12, 3], self.vdtype)
        # meas_uvw = tf.zeros([self.batch_size, 3], self.vdtype)
        init_weight = tf.ones([self.batch_size, 1, 1], self.vdtype)

        state3 = self.state_fw_in_state3

        forward_states = tf.scan(self.forward_step_fn, tf.transpose(inputs, [1, 0, 2]),
                                 initializer=(self.mu, self.Sigma, self.mu, self.Sigma, meas_uvw,
                                              state1, state2, state3,
                                              init_Q, init_R, init_A, init_B, init_Si,
                                              init_weight, u),
                                 parallel_iterations=1, name='forward')
        return forward_states

    def compute_backwards(self, forward_states):
        mu_pred, Sigma_pred, mu_filt, Sigma_filt, meas_uvw, state1, state2, state3, Q, R, A, B, S_inv, weights, u = forward_states

        mu_pred = tf.expand_dims(mu_pred, 3)
        mu_filt = tf.expand_dims(mu_filt, 3)
        # The tf.scan below that does the smoothing is initialized with the filtering distribution at time T.
        # following the derivarion in Murphy's book, we then need to discard the last time step of the predictive
        # (that will then have t=2,..T) and filtering distribution (t=1:T-1)
        states_scan = [mu_pred[:-1, :, :, :],
                       Sigma_pred[:-1, :, :, :],
                       mu_filt[:-1, :, :, :],
                       Sigma_filt[:-1, :, :, :],
                       A[:-1],
                       weights[:-1]]

        # Reverse time dimension
        dims = [0]
        for i, state in enumerate(states_scan):
            states_scan[i] = tf.reverse(state, dims)

        # Compute backwards states
        backward_states = tf.scan(self.backward_step_fn, states_scan,
                                  initializer=(mu_filt[-1, :, :, :], Sigma_filt[-1, :, :, :]), parallel_iterations=1,
                                  name='backward')

        # Reverse time dimension
        backward_states = list(backward_states)
        dims = [0]
        for i, state in enumerate(backward_states):
            backward_states[i] = tf.reverse(state, dims)

        # Add the final state from the filtering distribution
        backward_states[0] = tf.concat([backward_states[0], mu_filt[-1:, :, :, :]], axis=0)
        backward_states[1] = tf.concat([backward_states[1], Sigma_filt[-1:, :, :, :]], axis=0)

        # Remove extra dimension in the mean
        backward_states[0] = backward_states[0][:, :, :, 0]

        return backward_states, Q, R, A, B, S_inv, u, meas_uvw, state1, state2, state3

    def filter(self):
        mu_pred, Sigma_pred, mu_filt, Sigma_filt, meas_uvw, state1, state2, state3, Q, R, A, B, S_inv, weights, u = forward_states = \
            self.compute_forwards()

        state1c = tf.transpose(state1[0], [1, 0, 2])
        state1h = tf.transpose(state1[1], [1, 0, 2])
        state1_out = tf.contrib.rnn.LSTMStateTuple(state1c[:, -1, :], state1h[:, -1, :])

        state2c = tf.transpose(state2[0], [1, 0, 2])
        state2h = tf.transpose(state2[1], [1, 0, 2])
        state2_out = tf.contrib.rnn.LSTMStateTuple(state2c[:, -1, :], state2h[:, -1, :])

        state3c = tf.transpose(state3[0], [1, 0, 2])
        state3h = tf.transpose(state3[1], [1, 0, 2])
        state3_out = tf.contrib.rnn.LSTMStateTuple(state3c[:, -1, :], state3h[:, -1, :])

        # if is_training:
        forward_states_filter = [mu_filt, Sigma_filt]
        # else:
        forward_states_pred = [mu_pred, Sigma_pred]

        # Swap batch dimension and time dimension
        forward_states_filter[0] = tf.transpose(forward_states_filter[0], [1, 0, 2])
        forward_states_filter[1] = tf.transpose(forward_states_filter[1], [1, 0, 2, 3])

        forward_states_pred[0] = tf.transpose(forward_states_pred[0], [1, 0, 2])
        forward_states_pred[1] = tf.transpose(forward_states_pred[1], [1, 0, 2, 3])

        return tuple(forward_states_filter), tf.transpose(A, [1, 0, 2, 3]), tf.transpose(Q, [1, 0, 2, 3]), \
               tf.transpose(R, [1, 0, 2, 3]), tf.transpose(B, [1, 0, 2, 3]), tf.transpose(S_inv, [1, 0, 2, 3]),\
               tf.transpose(u, [1, 0, 2]), tf.transpose(meas_uvw, [1, 0, 2]), tuple(forward_states_pred), state1_out, state2_out, state3_out

    def smooth(self):
        backward_states, Q, R, A, B, S_inv, u, meas_uvw, state1, state2, state3 = self.compute_backwards(self.compute_forwards())

        state1c = tf.transpose(state1[0], [1, 0, 2])
        state1h = tf.transpose(state1[1], [1, 0, 2])
        state1_out = tf.contrib.rnn.LSTMStateTuple(state1c[:, -1, :], state1h[:, -1, :])

        state2c = tf.transpose(state2[0], [1, 0, 2])
        state2h = tf.transpose(state2[1], [1, 0, 2])
        state2_out = tf.contrib.rnn.LSTMStateTuple(state2c[:, -1, :], state2h[:, -1, :])

        state3c = tf.transpose(state3[0], [1, 0, 2])
        state3h = tf.transpose(state3[1], [1, 0, 2])
        state3_out = tf.contrib.rnn.LSTMStateTuple(state3c[:, -1, :], state3h[:, -1, :])

        # Swap batch dimension and time dimension
        backward_states[0] = tf.transpose(backward_states[0], [1, 0, 2])
        backward_states[1] = tf.transpose(backward_states[1], [1, 0, 2, 3])
        return tuple(backward_states), tf.transpose(A, [1, 0, 2, 3]), tf.transpose(Q, [1, 0, 2, 3]), \
               tf.transpose(R, [1, 0, 2, 3]), tf.transpose(B, [1, 0, 2, 3]), tf.transpose(S_inv, [1, 0, 2, 3]), \
               tf.transpose(u, [1, 0, 2]), tf.transpose(meas_uvw, [1, 0, 2]), state1_out, state2_out, state3_out

    def _sast(self, a, s):
        _, dim_1, dim_2 = s.get_shape().as_list()
        sast = tf.matmul(tf.reshape(s, [-1, dim_2]), a, transpose_b=True)
        sast = tf.transpose(tf.reshape(sast, [-1, dim_1, dim_2]), [0, 2, 1])
        sast = tf.matmul(s, sast)
        return sast

    def filter_measurement(self, mu_pred0, Sigma_pred0, current_time, prev_time, int_time, meas_rae, LLA, state1, state2, state3):

        dt = current_time - prev_time
        dt = tf.where(dt <= 1 / 100, tf.ones_like(dt) * 1 / 25, dt)

        meas_uvw, Qt, At, Rt, Bt, u, state1, state2, state3 = self.alpha(current_time, int_time, dt, mu_pred0, meas_rae,
                                                                         tf.reshape(Sigma_pred0, [self.batch_size, self.num_state ** 2]), LLA,
                                                                         state1=state1, state2=state2, state3=state3)  # (bs, k)

        pos_res1 = meas_uvw[:, :, tf.newaxis] - tf.matmul(self.meas_mat, mu_pred0[:, :, tf.newaxis])
        sp1 = tf.matmul(tf.matmul(self.meas_mat, Sigma_pred0), self.meas_mat, transpose_b=True)
        S = sp1 + Rt
        S_inv = tf.matrix_inverse(S)
        gain = tf.matmul(tf.matmul(Sigma_pred0, self.meas_mat, transpose_b=True), S_inv)

        mu_t = mu_pred0[:, :, tf.newaxis] + tf.matmul(gain, pos_res1)
        mu_t = mu_t[:, :, 0]

        I_KC = self.I_12 - tf.matmul(gain, self.meas_mat)  # (bs, dim_z, dim_z)

        Sigma_t = tf.matmul(tf.matmul(I_KC, Sigma_pred0), I_KC, transpose_b=True) + tf.matmul(tf.matmul(gain, Rt), gain, transpose_b=True)
        Sigma_t = (Sigma_t + tf.transpose(Sigma_t, [0, 2, 1])) / 2

        mu_pred = tf.squeeze(tf.matmul(At, tf.expand_dims(mu_t, 2)), -1) + tf.squeeze(tf.matmul(Bt, u[:, :, tf.newaxis]), -1)
        Sigma_pred = tf.matmul(tf.matmul(At, Sigma_t), At, transpose_b=True) + Qt

        return mu_pred, Sigma_pred, state1, state2, state3, Qt, At, Rt

    def get_elbo(self, backward_states):

        mu_smooth = backward_states[0]
        Sigma_smooth = backward_states[1]

        ssdiag = tf.matrix_diag_part(Sigma_smooth)
        ssdiag = tf.where(tf.less_equal(ssdiag, tf.zeros_like(ssdiag)), tf.ones_like(ssdiag) * 1, ssdiag)
        Sigma_smooth = tf.matrix_set_diag(Sigma_smooth, ssdiag)

        all_truth = tf.stack(self.truth_state, axis=1)
        mvn_smooth = tfd.MultivariateNormalTriL(mu_smooth, tf.cholesky(Sigma_smooth))
        # mvn_smooth = tfd.MultivariateNormalDiag(mu_smooth, tf.sqrt(tf.matrix_diag_part(Sigma_smooth)))
        self.mvn_inv = tf.matrix_inverse(mvn_smooth.covariance())

        # mvn_smooth_error = tfd.MultivariateNormalTriL(None, tf.cholesky(Sigma_smooth))
        z_smooth = mvn_smooth.sample()
        # z_smooth = mu_smooth
        # self.final_state2 = z_smooth
        self.state_error = all_truth - mu_smooth
        self.state_error = tf.where(self.state_error < 1e-3, tf.ones_like(self.state_error) * 1e-3, self.state_error)
        self.state_error = self.state_error[:, :, :, tf.newaxis]

        M1 = tf.matmul(self.state_error, self.mvn_inv, transpose_a=True)
        M2 = tf.sqrt(tf.square(tf.matmul(M1, self.state_error)))
        self.MDP = tf.squeeze(tf.sqrt(M2/12), -1)

        truth_pos = tf.concat([all_truth[:, :, 0, tf.newaxis], all_truth[:, :, 4, tf.newaxis], all_truth[:, :, 8, tf.newaxis]], axis=2)
        truth_vel = tf.concat([all_truth[:, :, 1, tf.newaxis], all_truth[:, :, 5, tf.newaxis], all_truth[:, :, 9, tf.newaxis]], axis=2)
        truth_acc = tf.concat([all_truth[:, :, 2, tf.newaxis], all_truth[:, :, 6, tf.newaxis], all_truth[:, :, 10, tf.newaxis]], axis=2)
        truth_jer = tf.concat([all_truth[:, :, 3, tf.newaxis], all_truth[:, :, 7, tf.newaxis], all_truth[:, :, 11, tf.newaxis]], axis=2)

        smooth_pos = tf.concat([z_smooth[:, :, 0, tf.newaxis], z_smooth[:, :, 4, tf.newaxis], z_smooth[:, :, 8, tf.newaxis]], axis=2)
        smooth_vel = tf.concat([z_smooth[:, :, 1, tf.newaxis], z_smooth[:, :, 5, tf.newaxis], z_smooth[:, :, 9, tf.newaxis]], axis=2)
        smooth_acc = tf.concat([z_smooth[:, :, 2, tf.newaxis], z_smooth[:, :, 6, tf.newaxis], z_smooth[:, :, 10, tf.newaxis]], axis=2)
        smooth_jer = tf.concat([z_smooth[:, :, 3, tf.newaxis], z_smooth[:, :, 7, tf.newaxis], z_smooth[:, :, 11, tf.newaxis]], axis=2)

        pos_error = truth_pos - smooth_pos
        vel_error = truth_vel - smooth_vel
        acc_error = truth_acc - smooth_acc
        jer_error = truth_jer - smooth_jer

        Az_tm1 = tf.matmul(self.ao_list[:, :-1], tf.expand_dims(all_truth[:, :-1], 3))
        Bz_tm1 = tf.matmul(self.bo_list[:, :-1], tf.expand_dims(self.uo_list[:, :-1], 3))
        mu_transition = Az_tm1[:, :, :, 0] + Bz_tm1[:, :, :, 0]
        z_t_transition = all_truth[:, 1:, :]

        trans_centered = z_t_transition - mu_transition
        self.to_list = trans_centered
        self.trans1 = mu_transition
        self.trans2 = z_t_transition

        Qdiag = tf.matrix_diag_part(self.qo_list[:, :-1])
        # mvn_transition = tfd.MultivariateNormalTriL(tf.zeros(self.num_state, dtype=self.vdtype), tf.cholesky(self.qo_list[:, :-1]))
        # mvn_transition = tfd.MultivariateNormalDiag(None, tf.sqrt(Qdiag))
        # log_prob_transition = mvn_transition.log_prob(trans_centered) * self.seqweightin[:, :-1]

        # trans_centered_pv = tf.concat([trans_centered[:, :, 0, tf.newaxis], trans_centered[:, :, 1, tf.newaxis], trans_centered[:, :, 4, tf.newaxis],
        #                                trans_centered[:, :, 5, tf.newaxis], trans_centered[:, :, 8, tf.newaxis], trans_centered[:, :, 9, tf.newaxis]], axis=2)
        #
        # Qdiag_pv = tf.concat([Qdiag[:, :, 0, tf.newaxis], Qdiag[:, :, 1, tf.newaxis], Qdiag[:, :, 4, tf.newaxis],
        #                       Qdiag[:, :, 5, tf.newaxis], Qdiag[:, :, 8, tf.newaxis], Qdiag[:, :, 9, tf.newaxis]], axis=2)
        #
        # trans_centered_aj = tf.concat([trans_centered[:, :, 2, tf.newaxis], trans_centered[:, :, 3, tf.newaxis], trans_centered[:, :, 6, tf.newaxis],
        #                                trans_centered[:, :, 7, tf.newaxis], trans_centered[:, :, 9, tf.newaxis], trans_centered[:, :, 11, tf.newaxis]], axis=2)
        #
        # Qdiag_aj = tf.concat([Qdiag[:, :, 2, tf.newaxis], Qdiag[:, :, 3, tf.newaxis], Qdiag[:, :, 6, tf.newaxis],
        #                       Qdiag[:, :, 7, tf.newaxis], Qdiag[:, :, 9, tf.newaxis], Qdiag[:, :, 11, tf.newaxis]], axis=2)

        trans_centered_j = tf.concat([trans_centered[:, :, 3, tf.newaxis], trans_centered[:, :, 7, tf.newaxis], trans_centered[:, :, 11, tf.newaxis]], axis=2)

        Qdiag_j = tf.concat([Qdiag[:, :, 3, tf.newaxis], Qdiag[:, :, 7, tf.newaxis], Qdiag[:, :, 11, tf.newaxis]], axis=2)

        # trans_centered_a = tf.concat([trans_centered[:, :, 2, tf.newaxis], trans_centered[:, :, 6, tf.newaxis], trans_centered[:, :, 10, tf.newaxis]], axis=2)

        # Qdiag_a = tf.concat([Qdiag[:, :, 2, tf.newaxis], Qdiag[:, :, 6, tf.newaxis], Qdiag[:, :, 10, tf.newaxis]], axis=2)

        mvn_transition = tfd.MultivariateNormalDiag(None, tf.sqrt(Qdiag_j))
        # mvn_transition = tfd.MultivariateNormalTriL(None, tf.cholesky(Qdiag_aj))
        log_prob_transition = mvn_transition.log_prob(trans_centered_j) * self.seqweightin[:, :-1]

        self.y_t_resh = tf.concat([all_truth[:, :, 0, tf.newaxis], all_truth[:, :, 4, tf.newaxis], all_truth[:, :, 8, tf.newaxis]], axis=2)
        # self.Cz_t = tf.concat([z_smooth[:, :, 0, tf.newaxis], z_smooth[:, :, 4, tf.newaxis], z_smooth[:, :, 8, tf.newaxis]], axis=2)
        # self.y_t_resh = tf.matmul(z_smooth, self.meas_mat, transpose_b=True)
        self.Cz_t = self.new_meas
        emiss_centered = (self.Cz_t - self.y_t_resh) + tf.ones_like(self.Cz_t)*1e-12
        # emiss_centered = tf.where(emiss_centered < 1., tf.sqrt(emiss_centered), emiss_centered)
        mvn_emission = tfd.MultivariateNormalTriL(None, tf.cholesky(self.ro_list))
        # mvn_emission = tfd.MultivariateNormalDiag(None, tf.sqrt(tf.matrix_diag_part(self.ro_list)))

        # ## Distribution of the initial state p(z_1|z_0)
        # z_0 = z_smooth[:, 0, :]
        # mvn_0 = tfd.MultivariateNormalTriL(self.mu, tf.cholesky(self.Sigma))

        # Compute terms of the lower bound
        # We compute the log-likelihood *per frame*
        num_el = tf.reduce_sum(self.seqweightin)  # / tf.cast(self.batch_size, self.vdtype)
        # num_el2 = tf.reduce_sum(self.seqweightin) * tf.cast(self.batch_size, self.vdtype)

        state_error_pos = truth_pos - smooth_pos
        meas_error_pos = tf.sqrt(tf.square(truth_pos - self.new_meas))
        meas_error_pos = meas_error_pos + tf.ones_like(meas_error_pos) * 1e-12

        # meas_dist = tfd.MultivariateNormalTriL(meas_error_pos, tf.cholesky(meas_error_pos_cov))
        meas_dist = tfd.MultivariateNormalDiag(None, tf.sqrt(meas_error_pos))
        self.meas_error = tf.truediv(tf.reduce_sum(tf.negative(meas_dist.log_prob(state_error_pos)) * self.seqweightin), num_el*3.)

        # self.aux_meas_loss = tf.where(state_error_pos > meas_error_pos, tf.ones_like(state_error_pos)*10000, tf.zeros_like(state_error_pos))
        # self.aux_meas_loss = tf.truediv(tf.reduce_sum(self.aux_meas_loss * self.seqweightin[:, :, tf.newaxis]), num_el)
        acceptable_acc_error = tf.ones_like(acc_error) * 5

        acc_dist = tfd.MultivariateNormalDiag(None, tf.sqrt(acceptable_acc_error))
        self.acc_error = tf.truediv(tf.reduce_sum(tf.negative(acc_dist.log_prob(acc_error)) * self.seqweightin), num_el)

        cov_pos = tf.concat([tf.concat([Sigma_smooth[:, :, 0, 0, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 0, 4, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 0, 8, tf.newaxis, tf.newaxis]], axis=3),
                             tf.concat([Sigma_smooth[:, :, 4, 0, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 4, 4, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 4, 8, tf.newaxis, tf.newaxis]], axis=3),
                             tf.concat([Sigma_smooth[:, :, 8, 0, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 8, 4, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 8, 8, tf.newaxis, tf.newaxis]], axis=3)],
                            axis=2)

        cov_vel = tf.concat([tf.concat([Sigma_smooth[:, :, 1, 1, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 1, 5, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 1, 9, tf.newaxis, tf.newaxis]], axis=3),
                             tf.concat([Sigma_smooth[:, :, 5, 1, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 5, 5, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 5, 9, tf.newaxis, tf.newaxis]], axis=3),
                             tf.concat([Sigma_smooth[:, :, 9, 1, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 9, 5, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 9, 9, tf.newaxis, tf.newaxis]], axis=3)],
                            axis=2)

        pos_error = pos_error[:, :, :, tf.newaxis]
        meas_error = (truth_pos - smooth_pos)
        meas_error = tf.where(meas_error<1e-6, tf.ones_like(meas_error)*1e-6, meas_error)
        meas_error = meas_error[:, :, :, tf.newaxis]
        # M1P = tf.matmul(pos_error, tf.matrix_inverse(cov_pos), transpose_a=True)
        # M1P = tf.matmul(meas_error, self.si_list, transpose_a=True)
        # M2P = tf.matmul(M1P, meas_error)
        # self.MDP = tf.squeeze(M2P, -1)
        self.MDPi = (tf.ones_like(self.MDP, self.vdtype) / self.MDP) * self.seqweightin[:, :, tf.newaxis]
        # self.maha_loss = tf.truediv(tf.reduce_sum((self.MDP+self.MDPi)*self.seqweightin[:, :, tf.newaxis]), num_el)
        self.maha_loss = tf.truediv(tf.reduce_sum((self.MDP + self.MDPi)*self.seqweightin[:, :, tf.newaxis]), num_el)
        # self.maha_loss = tf.truediv(tf.reduce_sum(self.MDP*self.seqweightin[:, :, tf.newaxis]), num_el)

        train_cov00 = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=Sigma_smooth)
        train_cov_pos = tfd.MultivariateNormalFullCovariance(loc=smooth_pos, covariance_matrix=cov_pos)
        train_cov_vel = tfd.MultivariateNormalFullCovariance(loc=smooth_vel, covariance_matrix=cov_vel)
        self.trace_loss = tf.truediv(tf.reduce_sum(tf.sqrt(tf.pow(tf.matrix_diag_part(Sigma_smooth), 2))), num_el)

        self.error_loss_pos = tf.truediv(tf.reduce_sum(tf.negative(train_cov_pos.log_prob(truth_pos)) * self.seqweightin), num_el * 3.)
        self.error_loss_vel = tf.truediv(tf.reduce_sum(tf.negative(train_cov_vel.log_prob(truth_vel)) * self.seqweightin), num_el * 3.)
        self.error_loss_full = tf.truediv(tf.reduce_sum(tf.negative(train_cov00.log_prob(self.state_error[:, :, :, 0])) * self.seqweightin), (num_el * 12.))
        self.entropy = tf.truediv(tf.reduce_sum(mvn_smooth.log_prob(z_smooth) * self.seqweightin), num_el*12.)
        self.rl = tf.truediv(tf.reduce_sum(tf.negative(mvn_emission.log_prob(emiss_centered)) * self.seqweightin), num_el*3.)
        # self.maha_out = tf.truediv(tf.reduce_sum(MD * self.seqweightin[:, :, tf.newaxis]), num_el)

        # self.MD0 = tf.truediv(tf.reduce_sum(tf.abs((tf.ones_like(MD) - MD)) * self.seqweightin[:, :, tf.newaxis]), num_el)
        # self.MDPL = tf.truediv(tf.reduce_sum(tf.abs((tf.ones_like(self.MDP) - self.MDP)) * self.seqweightin[:, :, tf.newaxis]), num_el)

        self.z_smooth = mu_smooth
        self.num_el = num_el
        self.error_loss_Q = tf.truediv(tf.reduce_sum(tf.negative(log_prob_transition)), num_el*3.)
        # self.error_loss_Q = tf.reduce_sum(0.0)

    def build_model(self, is_training):

        with tf.variable_scope('Input_Placeholders'):
            self.DROPOUT = tf.placeholder(self.vdtype)
            self.update_condition = tf.placeholder(tf.bool, name='update_condition')
            self.meanv = tf.placeholder(self.vdtype, shape=(1, self.num_state), name='meanv')
            # self.stdv = tf.placeholder(self.vdtype, shape=(1, self.num_state), name='stdv')

            self.grad_clip = tf.placeholder(self.vdtype, name='grad_clip')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.measurement = [tf.placeholder(self.vdtype, shape=(None, self.num_meas), name="meas_uvw_{}".format(t)) for t in range(self.max_seq)]

            self.sensor_ecef = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name='sen_ecef')
            self.sensor_lla = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name='sen_lla')
            self.sensor_onehots = tf.placeholder(self.vdtype, shape=(None, 3), name='sen_onehot')

            self.prev_measurement = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name="px")
            self.prev_covariance_estimate = tf.placeholder(self.vdtype, shape=(None, self.num_state, self.num_state), name="pcov")
            self.prev_time = tf.placeholder(self.vdtype, shape=(None, 1), name="ptime")
            self.prev_state_truth = tf.placeholder(self.vdtype, shape=(None, self.num_state), name="ptruth")
            self.prev_state_estimate = tf.placeholder(self.vdtype, shape=(None, self.num_state), name="prev_state_estimate")

            self.current_timei = [tf.placeholder(self.vdtype, shape=(None, 1), name="current_time_{}".format(t)) for t in range(self.max_seq)]
            self.P_inp = tf.placeholder(self.vdtype, shape=(None, self.num_state, self.num_state), name="p_inp")
            self.Q_inp = tf.placeholder(self.vdtype, shape=(None, self.num_state, self.num_state), name="q_inp")
            self.R_inp = tf.placeholder(self.vdtype, shape=(None, self.num_meas, self.num_meas), name="r_inp")
            self.state_input = tf.placeholder(self.vdtype, shape=(None, self.num_state), name="state_input")
            self.truth_state = [tf.placeholder(self.vdtype, shape=(None, self.num_state), name="y_truth_{}".format(t)) for t in range(self.max_seq)]
            self.seqweightin = tf.placeholder(self.vdtype, [None, self.max_seq])

            self.init_c_fwf = tf.placeholder(name='init_c_fwf/state', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.init_h_fwf = tf.placeholder(name='init_h_fwf/state', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_fw_in_state = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwf, self.init_h_fwf)

            self.init_c_fwf2 = tf.placeholder(name='init_c_fwf2/state', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.init_h_fwf2 = tf.placeholder(name='init_h_fwf2/state', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_fw_in_state2 = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwf2, self.init_h_fwf2)

            self.init_c_fwf3 = tf.placeholder(name='init_c_fwf3/state', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.init_h_fwf3 = tf.placeholder(name='init_h_fwf3/state', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_fw_in_state3 = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwf3, self.init_h_fwf3)

            if self.state_type == 'GRU':
                cell_type = tfc.rnn.IndyGRUCell
            elif self.state_type == 'LSTM':
                cell_type = tfc.rnn.IndyLSTMCell
            elif self.state_type == 'PLSTM':
                cell_type = PhasedLSTMCell
            else:
                cell_type = tfc.rnn.GRUCell

            use_dropout = False
            with tf.variable_scope('Source_Track_Forward/state'):
                if use_dropout:
                    self.source_fwf = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
                                                             input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
                else:
                    if is_training:
                        self.source_fwf = cell_type(self.F_hidden, ratio_on=0.5, period_init_min=1, period_init_max=2750)
                    else:
                        self.source_fwf = cell_type(self.F_hidden, ratio_on=0.5, period_init_min=1, period_init_max=2750, leak=0.0)

            with tf.variable_scope('Source_Track_Forward2/state'):
                if use_dropout:
                    self.source_fwf2 = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
                                                              input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
                else:
                    if is_training:
                        self.source_fwf2 = cell_type(self.F_hidden, ratio_on=0.5, period_init_min=1, period_init_max=2750)
                    else:
                        self.source_fwf2 = cell_type(self.F_hidden, ratio_on=0.5, period_init_min=1, period_init_max=2750, leak=0.0)

            with tf.variable_scope('Source_Track_Forward3/state'):
                if use_dropout:
                    self.source_fwf3 = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
                                                              input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
                else:
                    if is_training:
                        self.source_fwf3 = cell_type(self.F_hidden, ratio_on=0.5, period_init_min=1, period_init_max=2750)
                    else:
                        self.source_fwf3 = cell_type(self.F_hidden, ratio_on=0.5, period_init_min=1, period_init_max=2750, leak=0.0)

            self.I_3 = tf.scalar_mul(1.0, tf.eye(3, batch_shape=[self.batch_size], dtype=self.vdtype))
            self.I_12 = tf.scalar_mul(1.0, tf.eye(12, batch_shape=[self.batch_size], dtype=self.vdtype))
            self.I_3z = tf.scalar_mul(0.0, tf.eye(3, batch_shape=[self.batch_size], dtype=self.vdtype))
            self.I_4z = tf.scalar_mul(0.0, tf.eye(4, batch_shape=[self.batch_size], dtype=self.vdtype))
            self.I_6z = tf.scalar_mul(0.0, tf.eye(6, batch_shape=[self.batch_size], dtype=self.vdtype))
            self.om = tf.ones([self.batch_size, 1, 1], dtype=self.vdtype)
            self.zb = tf.zeros([self.batch_size, 4, 1], dtype=self.vdtype)
            self.zm = tf.zeros([self.batch_size, 1, 1], dtype=self.vdtype)
            omp = np.ones([1, 1], self.vdp_np)
            zmp = np.zeros([1, 1], self.vdp_np)

            m1 = np.concatenate([omp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp], axis=1).astype(self.vdp_np)
            m2 = np.concatenate([zmp, zmp, zmp, zmp, omp, zmp, zmp, zmp, zmp, zmp, zmp, zmp], axis=1).astype(self.vdp_np)
            m3 = np.concatenate([zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, omp, zmp, zmp, zmp], axis=1).astype(self.vdp_np)
            self.meas_mat = tf.tile(tf.expand_dims(tf.concat([m1, m2, m3], axis=0), axis=0), [self.batch_size, 1, 1])

        smooth, A, Q, R, B, S_inv, u, meas_uvw, state1_out, state2_out, state3_out = self.smooth()  # for plotting smoothed posterior
        _, _, _, _, _, _, u, _, prediction, _, _, _ = self.filter()  # for plotting smoothed posterior

        filter, _, _, _, _, _, _, _, _, _, _, _ = self.filter()  # for plotting forward pass

        self.ao_list = A
        self.qo_list = Q
        self.ro_list = R
        self.uo_list = u
        self.bo_list = B
        self.si_list = S_inv

        self.new_meas = meas_uvw
        self.final_state_filter = filter[0]
        self.final_state_pred = prediction[0]
        self.final_cov_filter = filter[1]
        self.final_cov_pred = prediction[1]
        self.state_fwf = state1_out
        self.state_fwf2 = state2_out
        self.state_fwf3 = state3_out

        _y = tf.stack(self.truth_state, axis=1)

        # filter = smooth

        self.get_elbo(filter)

        total_weight = tf.cast(self.seqweightin, self.vdtype)
        tot = tf.cast(self.max_seq, self.vdtype)
        loss_func = weighted_mape_tf

        pos1m_err = loss_func(_y[:, :, 0], self.new_meas[:, :, 0], total_weight, tot)
        pos2m_err = loss_func(_y[:, :, 4], self.new_meas[:, :, 1], total_weight, tot)
        pos3m_err = loss_func(_y[:, :, 8], self.new_meas[:, :, 2], total_weight, tot)

        pos1e_err = loss_func(_y[:, :, 0], self.z_smooth[:, :, 0], total_weight, tot)
        pos2e_err = loss_func(_y[:, :, 4], self.z_smooth[:, :, 4], total_weight, tot)
        pos3e_err = loss_func(_y[:, :, 8], self.z_smooth[:, :, 8], total_weight, tot)

        # state_loss_pos100 = tf.where(pos1e_err > pos1m_err, 10000*tf.ones_like(pos1e_err), pos1e_err)
        # state_loss_pos200 = tf.where(pos2e_err > pos2m_err, 10000*tf.ones_like(pos1e_err), pos2e_err)
        # state_loss_pos300 = tf.where(pos3e_err > pos3m_err, 10000*tf.ones_like(pos1e_err), pos3e_err)

        state_loss_pos100 = loss_func(_y[:, :, 0], self.z_smooth[:, :, 0], total_weight, tot)
        state_loss_pos200 = loss_func(_y[:, :, 4], self.z_smooth[:, :, 4], total_weight, tot)
        state_loss_pos300 = loss_func(_y[:, :, 8], self.z_smooth[:, :, 8], total_weight, tot)
        state_loss_vel100 = loss_func(_y[:, :, 1], self.z_smooth[:, :, 1], total_weight, tot)  # / 1000
        state_loss_vel200 = loss_func(_y[:, :, 5], self.z_smooth[:, :, 5], total_weight, tot)  # / 1000
        state_loss_vel300 = loss_func(_y[:, :, 9], self.z_smooth[:, :, 9], total_weight, tot)  # / 1000
        state_loss_acc100 = loss_func(_y[:, :, 2], self.z_smooth[:, :, 2], total_weight, tot)  # / 10000
        state_loss_acc200 = loss_func(_y[:, :, 6], self.z_smooth[:, :, 6], total_weight, tot)  # / 10000
        state_loss_acc300 = loss_func(_y[:, :, 10], self.z_smooth[:, :, 10], total_weight, tot)  # / 10000
        state_loss_j100 = loss_func(_y[:, :, 3], self.z_smooth[:, :, 3], total_weight, tot)  # / 100000
        state_loss_j200 = loss_func(_y[:, :, 7], self.z_smooth[:, :, 7], total_weight, tot)  # / 100000
        state_loss_j300 = loss_func(_y[:, :, 11], self.z_smooth[:, :, 11], total_weight, tot)  # / 100000

        self.SLPf1 = state_loss_pos100 + state_loss_pos200 + state_loss_pos300
        self.SLVf1 = state_loss_vel100 + state_loss_vel200 + state_loss_vel300
        self.SLAf1 = state_loss_acc100 + state_loss_acc200 + state_loss_acc300
        self.SLJf1 = state_loss_j100 + state_loss_j200 + state_loss_j300

        self.SLPf1 = tf.truediv(self.SLPf1, self.num_el * 3.)
        self.SLVf1 = tf.truediv(self.SLVf1, self.num_el * 3.)
        self.SLAf1 = tf.truediv(self.SLAf1, self.num_el * 3.)
        self.SLJf1 = tf.truediv(self.SLJf1, self.num_el * 3.)

        self.rmse_pos = self.SLPf1
        self.rmse_vel = self.SLVf1
        self.rmse_acc = self.SLAf1
        self.rmse_jer = self.SLJf1
        self.maha_out = tf.reduce_mean(self.MDP)

        self.learning_rate = tf.train.exponential_decay(self.learning_rate_inp, global_step=self.global_step, decay_steps=5, decay_rate=0.95, staircase=True)

        with tf.variable_scope("TrainOps"):
            print('cov_update gradients...')

            # tfc.opt.MomentumWOptimizer(learning_rate=self.learning_rate, momentum=0.9, weight_decay=1e-10, name='r3')
            # tfc.opt.AdamWOptimizer(weight_decay=1e-7, learning_rate=self.learning_rate, name='opt')

            opt = tfc.opt.MultitaskOptimizerWrapper(tfc.opt.AdamWOptimizer(weight_decay=1e-10, learning_rate=self.learning_rate, name='opt'))

            # lower_bound = self.rl*1 + self.entropy*1 + self.error_loss_full*1 + self.meas_error*1 + self.error_loss_Q*0 + self.acc_error*0 + self.error_loss_pos*0 + self.maha_loss*0
            # lower_bound = self.rl*1 + self.error_loss_pos*1 + self.rmse_pos*0 + self.maha_loss*0 + self.error_loss_Q + self.meas_error
            lower_bound = self.rl + self.entropy + self.error_loss_Q*0 + self.maha_loss * 1 + self.rmse_pos + self.error_loss_full*0 + self.meas_error

            gradvars = opt.compute_gradients(lower_bound)

            gradvars1, _ = tfc.opt.clip_gradients_by_global_norm(gradvars, 1.)
            self.train_op = opt.apply_gradients(gradvars, global_step=self.global_step)

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Total traininable network parameters:: ' + str(total_parameters))

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        # tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in gradvars1])

        tf.summary.scalar("Total_Loss", lower_bound)
        tf.summary.scalar("Measurement", self.meas_error)
        tf.summary.scalar("Acceleration", self.acc_error)
        tf.summary.scalar("MeasurementCovariance", self.rl)
        tf.summary.scalar("TransitionCovariance", self.error_loss_Q)
        tf.summary.scalar("Entropy", self.entropy)
        tf.summary.scalar("PositionCovariance", self.error_loss_pos)
        tf.summary.scalar("VelocityCovariance", self.error_loss_vel)
        tf.summary.scalar("TotalCovariance", self.error_loss_full)
        tf.summary.scalar("MahalanobisLoss", self.maha_loss)
        tf.summary.scalar("MahalanobisDistance", tf.reduce_mean(self.MDP))
        tf.summary.scalar("MahalanobisInverse", tf.reduce_mean(self.MDPi))
        tf.summary.scalar("RMSE_pos", self.rmse_pos)
        tf.summary.scalar("RMSE_vel", self.rmse_vel)
        tf.summary.scalar("RMSE_acc", self.rmse_acc)
        tf.summary.scalar("RMSE_jer", self.rmse_jer)

    def train(self, data_rate, max_exp_seq):

        # rho0 = 1.22  # kg / m**3
        # k0 = 0.14141e-3
        # area = 0.25  # / self.RE  # meter squared
        # cd = 0.03  # unitless
        # gmn = self.GM / (self.RE ** 3)

        # OOP
        aeg_loc = [None] * 3
        tpy_loc = [None] * 3
        pat_loc = [None] * 3
        aeg_loc[0], aeg_loc[1], aeg_loc[2] = [0.17, -0.085], [0.00001, -0.089], [-0.089, -0.084]
        pat_loc[0], pat_loc[1], pat_loc[2] = [-0.0823, 0.0359], [0.0269, 0.0359], [0.0898, -0.1556]
        tpy_loc[0], tpy_loc[1], tpy_loc[2] = [0.15, 0.0449], [0.00001, 0.09], [-0.09, 0.06]

        # # ADVANCED
        # aeg_loc = [None] * 3
        # tpy_loc = [None] * 3
        # pat_loc = [None] * 3
        # aeg_loc[0], aeg_loc[1], aeg_loc[2] = [0.17, -0.085], [0.00001, -0.089], [-0.089, -0.084]
        # pat_loc[0], pat_loc[1], pat_loc[2] = [1e-5, -0.0449], [0.09, 0.0449], [0.0898, -0.1556]
        # tpy_loc[0], tpy_loc[1], tpy_loc[2] = [-0.45, 0.0449], [0.00001, 0.0449], [0.18, 0.0449]

        shuffle_data = False
        self.data_rate = data_rate
        self.max_exp_seq = max_exp_seq
        # initialize all variables
        tf.global_variables_initializer().run()

        summary = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        try:
            save_files = os.listdir(self.checkpoint_dir + '/')
            save_files = natsorted(save_files, reverse=True)
            recent = str.split(save_files[1], '_')
            start_epoch = recent[2]
            step = str.split(recent[3], '.')[0]
            print("Resuming run from epoch " + str(start_epoch) + ' and step ' + str(step))
            step = int(step)
        except:
            print("Beginning New Run ")
            start_epoch = 0
            step = 0
            print('Loading filter...')
        try:
            self.saver = tf.train.import_meta_graph(self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step) + '.meta')
            self.saver.restore(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step))
            # self.saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step) + '.meta'))
            print("filter restored.")
        except:
            self.saver = tf.train.Saver(save_relative_paths=True)
            start_epoch = 0
            step = 0
            print("Could not restore filter")

        e = int(start_epoch)

        ds = DataServerLive(self.meas_dir, self.state_dir)

        # plot_count = 0
        # train_writer = tf.summary.FileWriter('./log/0/train/', self.sess.graph)

        # lr = 1e-2 / 0.9
        for epoch in range(int(start_epoch), self.max_epoch):

            n_train_batches = int(ds.num_train / self.batch_size_np)

            for minibatch_index in range(n_train_batches):

                testing = False
                # if minibatch_index % 100 == 0 and minibatch_index != 0:
                #     testing = True
                #     # print('Testing filter for epoch ' + str(epoch))
                # else:
                #     testing = False
                    # print('Training filter for epoch ' + str(epoch))

                # Data is unnormalized at this point
                x_train, y_train, batch_number, total_batches, ecef_ref, lla_data = ds.load(batch_size=self.batch_size_np, constant=self.constant, test=testing,
                                                                                            max_seq_len=self.max_exp_seq, HZ=self.data_rate)

                # if shuffle_data:
                # shuf = np.arange(x_train.shape[0])
                # np.random.shuffle(shuf)
                # x_train = x_train[shuf]
                # y_train = y_train[shuf]

                lla_datar = copy.copy(lla_data)

                # GET SENSOR ONE HOTS
                sensor_onehots = np.zeros([self.batch_size_np, 3])
                for i in range(lla_datar.shape[0]):
                    cur_lla = list(lla_datar[i, :2])
                    if cur_lla in aeg_loc:
                        vec = [1, 0, 0]
                    elif cur_lla in pat_loc:
                        vec = [0, 1, 0]
                    elif cur_lla in tpy_loc:
                        vec = [0, 0, 1]
                    else:
                        pdb.set_trace()
                        print("unknown sensor location")
                        pass
                    sensor_onehots[i, :] = vec

                lla_datar[:, 0] = lla_datar[:, 0] * np.pi / 180
                lla_datar[:, 1] = lla_datar[:, 1] * np.pi / 180
                # print("Batch Number: {0:2d} out of {1:2d}".format(batch_number, total_batches))

                x_train = np.concatenate([x_train[:, :, 0, np.newaxis], x_train[:, :, 4:7]], axis=2)  # rae measurements

                y_uvw = y_train[:, :, :3] - np.ones_like(y_train[:, :, :3]) * ecef_ref[:, np.newaxis, :]
                zero_rows = (y_train[:, :, :3] == 0).all(2)
                for i in range(y_train.shape[0]):
                    zz = zero_rows[i, :, np.newaxis]
                    y_uvw[i, :, :] = np.where(zz, np.zeros_like(y_uvw[i, :, :]), y_uvw[i, :, :])

                y_train = np.concatenate([y_uvw, y_train[:, :, 3:]], axis=2)

                permute_dims = False
                if permute_dims:
                    x_train, y_train = permute_xyz_dims(x_train, y_train)

                # x_train[:, :, 1:] = x_train[:, :, 1:] / 6378137
                # y_train = normalize_statenp(y_train)
                # y_train = y_train / 6378137
                # _, _, _, _, mean_y, std_y = normalize_statenp(copy.copy(x_train[:, :, :4]), copy.copy(y_uvw))

                # max_pos = 50000
                # max_vel = 500
                # max_acc = 50
                # max_jer = 10
                #
                # mean_y = np.array([max_pos, max_pos, max_pos,
                #                    max_vel, max_vel, max_vel,
                #                    max_acc, max_acc, max_acc,
                #                    max_jer, max_jer, max_jer])

                # mean_y2 = np.array([max_pos, max_pos, max_pos,
                #                     max_vel, max_vel, max_vel])

                # if shuffle_data:
                #     shuf = np.arange(x_train.shape[0])
                #     np.random.shuffle(shuf)
                #     x_train = x_train[shuf]
                #     y_train = y_train[shuf]

                s_train = x_train

                # n_train_batches = int(x_train.shape[0] / self.batch_size_np)
                # print("Batch Number: {0:2d} out of {1:2d}".format(batch_number, total_batches))

                x0, y0, meta0, prev_y, prev_x, prev_time, prev_meta, initial_meas, initial_state_truth, initial_time = prepare_batch(0, x_train, y_train, s_train,
                                                                                                                 seq_len=self.max_seq, batch_size=self.batch_size_np,
                                                                                                                 new_batch=True, window_mode=self.window_mode, pad_front=self.pad_front)

                count, _, _, _, _, _, prev_cov, prev_Q, prev_R, q_plot, q_plott, k_plot, out_plot_X, out_plot_F, out_plot_P, time_vals, \
                meas_plot, truth_plot, Q_plot, R_plot, maha_plot, x, y, meta = initialize_run_variables(self.batch_size_np, self.max_seq, self.num_state, x0, y0, meta0)

                feed_dict = {}

                time_plotter = np.zeros([self.batch_size_np, int(x.shape[1]), 1])

                # windows = np.ceil(x.shape[1] / self.max_seq)
                # total = int(windows * self.max_seq)
                # actual = x.shape[1]

                windows2 = int((x.shape[1]) / self.max_seq)
                # windows2 = int(x.shape[1])
                # max_seq = windows2
                time_plotter = np.zeros([self.batch_size_np, int(x.shape[1]), 1])
                plt.close()
                # for tstep in range(0, windows2):
                for tstep in range(0, 1):

                    r1 = tstep * self.max_seq
                    r2 = r1 + self.max_seq

                    prev_state = copy.copy(prev_y)
                    prev_meas = copy.copy(prev_x)

                    current_x, current_y, current_time, current_meta = \
                        get_feed_time_asynch(x, y, meta, prev_x, prev_y, prev_time, prev_meta, self.max_seq, tstep, self.num_state, self.window_mode)

                    if np.all(current_x == 0):
                        print('skipping')
                        continue

                    # if testing is True:
                    #     self.evaluate(current_x, current_y, current_time, initial_meas, initial_time, prev_cov[:, -1, :, :], prev_Q[:, -1, :, :], prev_R[:, -1, :, :], prev_x, prev_y, prev_time, ecef_ref, lla_data, sensor_onehots, epoch, minibatch_index, step)
                    #     continue

                    seqlen = np.ones(shape=[self.batch_size_np, ])
                    int_time = np.zeros(shape=[self.batch_size_np, self.max_seq])

                    seqweight = np.zeros(shape=[self.batch_size_np, self.max_seq])

                    for i in range(self.batch_size_np):
                        current_yt = current_y[i, :, :3]
                        m = ~(current_yt == 0).all(1)
                        yf = current_yt[m]
                        seq = yf.shape[0]
                        seqlen[i] = seq
                        int_time[i, :] = range(r1, r2)
                        seqweight[i, :] = m.astype(int)

                    cur_time = x[:, 0, 0]

                    time_plotter[:, 0, :] = cur_time[:, np.newaxis]
                    max_t = np.max(time_plotter[0, :, 0])
                    count += 1
                    step += 1
                    idxi = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
                    idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
                    # idxi2 = [0, 3, 1, 4, 2, 5]

                    R = initial_meas[:, :, 0, np.newaxis]
                    A = initial_meas[:, :, 1, np.newaxis]
                    E = initial_meas[:, :, 2, np.newaxis]

                    east = (R * np.sin(A) * np.cos(E))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
                    north = (R * np.cos(E) * np.cos(A))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
                    up = (R * np.sin(E))  # * ((tf.exp(tf.negative(tf.pow(se, 2) / 2))))

                    lat = lla_datar[:, 0, np.newaxis, np.newaxis]
                    lon = lla_datar[:, 1, np.newaxis, np.newaxis]

                    cosPhi = np.cos(lat)
                    sinPhi = np.sin(lat)
                    cosLambda = np.cos(lon)
                    sinLambda = np.sin(lon)

                    tv = cosPhi * up - sinPhi * north
                    wv = sinPhi * up + cosPhi * north
                    uv = cosLambda * tv - sinLambda * east
                    vv = sinLambda * tv + cosLambda * east

                    initial_meas_uvw = np.concatenate([uv, vv, wv], axis=2)

                    dtn = np.sum(np.diff(initial_time, axis=1), axis=1)
                    pos = initial_meas_uvw[:, 2, :]
                    vel = (initial_meas_uvw[:, 2, :] - initial_meas_uvw[:, 0, :]) / (2 * dtn)

                    R1 = np.linalg.norm(initial_meas + ecef_ref[:, np.newaxis, :], axis=2, keepdims=True)
                    R1 = np.mean(R1, axis=1)
                    R1 = np.where(np.less(R1, np.ones_like(R1) * self.RE), np.ones_like(R1) * self.RE, R1)
                    rad_temp = np.power(R1, 3)
                    GMt1 = np.divide(self.GM, rad_temp)
                    acc = get_legendre_np(GMt1, pos + ecef_ref, R1)

                    # acc = (initial_meas[:, 2, :] - 2*initial_meas[:, 1, :] + initial_meas[:, 0, :]) / dtn**2

                    initial_state = np.expand_dims(np.concatenate([pos, vel, acc, np.random.normal(loc=np.zeros_like(acc), scale=1)], axis=1), 1)
                    initial_cov = prev_cov[:, -1, :, :]
                    initial_Q = prev_Q[:, -1, :, :]
                    initial_R = prev_R[:, -1, :, :]
                    initial_state = initial_state[:, :, idxi]

                    dt0 = prev_time - initial_time[:, -1:, :]
                    dt1 = current_time[:, 0, :] - prev_time[:, 0, :]
                    current_state, covariance_out, converted_meas, pred_state, pred_covariance = \
                        unscented_kalman_np(self.batch_size_np, prev_meas.shape[1], initial_state[:, -1, :], prev_cov[:, -1, :, :], prev_meas, prev_time, dt0, dt1, lla_datar)

                    # initial_state = initial_state[:, :, idxo]

                    update = False

                    # mean_y = mean_y[idxi]
                    # prev_y = prev_y[:, :, idxi]
                    current_y = current_y[:, :, idxi]
                    prev_state = prev_state[:, :, idxi]
                    # initial_state = initial_state[:, :, idxi]

                    # pidx = 1
                    # plt.plot(initial_time[pidx, :, 0], initial_state_truth[pidx, :, 0], 'r')
                    # plt.scatter(prev_time[pidx, 0, 0], converted_meas[pidx, 0, 0], c='b')
                    # plt.scatter(prev_time[pidx, 0, 0], current_state[pidx, 0, 0], c='m')
                    # plt.plot(current_time[pidx, :5, :], current_y[pidx, :5, 0], 'r')
                    # plt.scatter(current_time[pidx, 0, 0], pred_state[pidx, 0, 0], c='k')

                    feed_dict.update({self.measurement[t]: current_x[:, t, :].reshape(-1, self.num_meas) for t in range(self.max_seq)})
                    feed_dict.update({self.prev_measurement: prev_meas.reshape(-1, self.num_meas)})
                    feed_dict.update({self.prev_covariance_estimate: prev_cov[:, -1, :, :]})
                    feed_dict.update({self.truth_state[t]: current_y[:, t, :].reshape(-1, self.num_state) for t in range(self.max_seq)})
                    feed_dict.update({self.prev_state_truth: prev_state.reshape(-1, self.num_state)})
                    feed_dict.update({self.prev_state_estimate: initial_state.reshape(-1, self.num_state)})
                    feed_dict.update({self.sensor_ecef: ecef_ref})
                    feed_dict.update({self.sensor_lla: lla_datar})
                    feed_dict.update({self.sensor_onehots: sensor_onehots})
                    feed_dict.update({self.seqlen: seqlen})
                    feed_dict.update({self.int_time: int_time})
                    feed_dict.update({self.update_condition: update})
                    feed_dict.update({self.is_training: True})
                    feed_dict.update({self.seqweightin: seqweight})
                    feed_dict.update({self.P_inp: covariance_out[-1]})
                    feed_dict.update({self.Q_inp: initial_Q})
                    feed_dict.update({self.R_inp: initial_R})
                    feed_dict.update({self.state_input: current_state.reshape(-1, self.num_state)})
                    feed_dict.update({self.prev_time: prev_time[:, :, 0]})
                    feed_dict.update({self.current_timei[t]: current_time[:, t, :].reshape(-1, 1) for t in range(self.max_seq)})
                    feed_dict.update({self.drop_rate: 1.0})

                    std = 0.3

                    feed_dict.update({self.init_c_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                    feed_dict.update({self.init_h_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                    feed_dict.update({self.init_c_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                    feed_dict.update({self.init_h_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                    feed_dict.update({self.init_c_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                    feed_dict.update({self.init_h_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

                    ee = minibatch_index

                    lr = 1e-3

                    if e < 1 and ee < 25:
                        # lr = 1e-2
                        train_op1 = self.train_op
                        stateful = True
                    elif e < 1 and ee < 50 and ee >= 25:
                        # lr = 5e-3
                        stateful = True
                        train_op1 = self.train_op
                    else:
                        # lr = 1e-3
                        stateful = True
                        train_op1 = self.train_op

                    feed_dict.update({self.learning_rate_inp: lr})

                    pred_output0, pred_output00, pred_output1, q_out_t, q_out, _, rmsp, rmsv, rmsa, rmsj, LR, \
                    cov_pos_loss, cov_vel_loss, kalman_cov_loss, maha_loss, MD, trace_loss, rl, \
                    entropy, qt_out, rt_out, at_out, trans_out, q_loss, state_fwf, state_fwf2, state_fwf3, new_meas, \
                    y_t_resh, Cz_t, trans1, trans2, MDP, mvn_inv, state_error, num_el, summary_str = \
                        self.sess.run([self.final_state_filter,
                                       self.final_state_pred,
                                       self.final_state_pred,
                                       self.final_cov_filter,
                                       self.final_cov_pred,
                                       train_op1,
                                       self.rmse_pos,
                                       self.rmse_vel,
                                       self.rmse_acc,
                                       self.rmse_jer,
                                       self.learning_rate,
                                       self.error_loss_pos,
                                       self.error_loss_vel,
                                       self.error_loss_full,
                                       self.maha_loss,
                                       self.maha_out,
                                       self.trace_loss,
                                       self.rl,
                                       self.entropy,
                                       self.qo_list,
                                       self.ro_list,
                                       self.ao_list,
                                       self.to_list,
                                       self.error_loss_Q,
                                       self.state_fwf,
                                       self.state_fwf2,
                                       self.state_fwf3,
                                       self.new_meas,
                                       self.y_t_resh,
                                       self.Cz_t,
                                       self.trans1,
                                       self.trans2,
                                       self.MDP,
                                       self.mvn_inv,
                                       self.state_error,
                                       self.num_el,
                                       summary],
                                      feed_dict)

                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    # if tstep % 10 == 0 or tstep <= self.max_seq or tstep > int(x.shape[1] - 10):
                    print("Epoch: {0:2d} MB: {1:1d} Time: {2:3d} "
                          "RMSP: {3:2.2e} RMSV: {4:2.2e} RMSA: {5:2.2e} RMSJ: {6:2.2e} "
                          "LR: {7:1.2e} ST: {8:1.2f} CPL: {9:1.2f} "
                          "CVL: {10:1.2f} EN: {11:1.2f} QL: {12:1.2f} "
                          "MD: {13:1.2f} RL: {14:1.2f} COV {15:1.2f} ".format(epoch, minibatch_index, tstep,
                                                                              rmsp, rmsv, rmsa, rmsj,
                                                                              LR, max_t, cov_pos_loss,
                                                                              cov_vel_loss, entropy, q_loss,
                                                                              MD, rl, kalman_cov_loss))

                    # mean_y = mean_y[idxo]
                    current_y = current_y[:, :, idxo]
                    # prev_state = prev_state[:, :, idxo]

                    pred_output0 = pred_output0[:, :, idxo]
                    pred_output00 = pred_output00[:, :, idxo]
                    pred_output1 = pred_output1[:, :, idxo]

                    if stateful is True:
                        feed_dict.update({self.init_c_fwf: state_fwf[0]})
                        feed_dict.update({self.init_h_fwf: state_fwf[1]})
                        feed_dict.update({self.init_c_fwf2: state_fwf2[0]})
                        feed_dict.update({self.init_h_fwf2: state_fwf2[1]})
                        feed_dict.update({self.init_c_fwf3: state_fwf3[0]})
                        feed_dict.update({self.init_h_fwf3: state_fwf3[1]})

                    else:
                        std = 0.1
                        feed_dict.update({self.init_c_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        feed_dict.update({self.init_h_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        feed_dict.update({self.init_c_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        feed_dict.update({self.init_h_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

                    prop_output = np.array(pred_output0)
                    if len(prop_output.shape) < 3:
                        prop_output = np.expand_dims(prop_output, axis=1)

                    pred_output = np.array(pred_output1)
                    if len(pred_output.shape) < 3:
                        pred_output = np.expand_dims(pred_output, axis=1)

                    full_final_output = np.array(pred_output00)
                    if len(full_final_output.shape) < 3:
                        full_final_output = np.expand_dims(full_final_output, axis=1)

                    idx = -1
                    prev_meas = np.concatenate([prev_meas, current_x], axis=1)
                    prev_meas = prev_meas[:, idx, np.newaxis, :]

                    # err0 = np.sum(np.abs(temp_prev_y - temp_pred0))
                    # err1 = np.sum(np.abs(temp_prev_y - temp_pred1))
                    # if err1 < err0 and e > 5:
                    #     new_prev = temp_pred0
                    # else:
                    #     new_prev = temp_pred1

                    # initial_state = prop_output[:, idx, np.newaxis, :]
                    # initial_cov = q_out_t[:, idx, :, :]
                    #
                    # prev_state = np.concatenate([prev_state, current_y], axis=1)
                    # prev_state = prev_state[:, idx, np.newaxis, :]
                    #
                    # prev_time = np.concatenate([prev_time, current_time], axis=1)
                    # prev_time = prev_time[:, idx, np.newaxis, :]
                    #
                    # prev_cov = np.concatenate([prev_cov[:, -1:, :, :], q_out_t], axis=1)
                    # prev_cov = prev_cov[:, idx, np.newaxis, :, :]
                    #
                    # prev_y = copy.copy(prev_state)
                    # prev_x = copy.copy(prev_meas)

                    if tstep == 0:
                        out_plot_F = full_final_output[0, np.newaxis, :, :]
                        out_plot_X = pred_output[0, np.newaxis, :, :]
                        out_plot_P = prop_output[0, np.newaxis, :, :]

                        q_plott = q_out_t[0, np.newaxis, :, :, :]
                        q_plot = q_out[0, np.newaxis, :, :, :]
                        qt_plot = qt_out[0, np.newaxis, :, :]
                        rt_plot = rt_out[0, np.newaxis, :, :]
                        at_plot = at_out[0, np.newaxis, :, :]
                        trans_plot = trans_out[0, np.newaxis, :, :]

                        time_vals = current_time[0, np.newaxis, :, :]
                        meas_plot = new_meas[0, np.newaxis, :, :]
                        truth_plot = current_y[0, np.newaxis, :, :]

                    else:
                        new_vals_F = full_final_output[0, :, :]  # current step
                        new_vals_X = pred_output[0, :, :]
                        new_vals_P = prop_output[0, :, :]
                        new_q = q_out[0, :, :, :]
                        new_qt = q_out_t[0, :, :, :]
                        new_qtt = qt_out[0, :, :, :]
                        new_rtt = rt_out[0, :, :, :]
                        new_att = at_out[0, :, :, :]

                        new_time = current_time[0, :, 0]
                        new_meas = new_meas[0, :, :]
                        new_truth = current_y[0, :, :]

                    if tstep > 0:
                        out_plot_F = np.concatenate([out_plot_F, new_vals_F[np.newaxis, :, :]], axis=1)
                        out_plot_X = np.concatenate([out_plot_X, new_vals_X[np.newaxis, :, :]], axis=1)
                        out_plot_P = np.concatenate([out_plot_P, new_vals_P[np.newaxis, :, :]], axis=1)
                        meas_plot = np.concatenate([meas_plot, new_meas[np.newaxis, :, :]], axis=1)
                        truth_plot = np.concatenate([truth_plot, new_truth[np.newaxis, :, :]], axis=1)
                        time_vals = np.concatenate([time_vals, new_time[np.newaxis, :, np.newaxis]], axis=1)
                        q_plot = np.concatenate([q_plot, new_q[np.newaxis, :, :, :]], axis=1)
                        q_plott = np.concatenate([q_plott, new_qt[np.newaxis, :, :, :]], axis=1)
                        qt_plot = np.concatenate([qt_plot, new_qtt[np.newaxis, :, :, :]], axis=1)
                        rt_plot = np.concatenate([rt_plot, new_rtt[np.newaxis, :, :, :]], axis=1)
                        at_plot = np.concatenate([at_plot, new_att[np.newaxis, :, :, :]], axis=1)
                        trans_plot = np.concatenate([trans_plot, new_att[np.newaxis, :, :, :]], axis=1)

                    if minibatch_index % 5 == 0:
                        plotpath = self.plot_dir + '/epoch_' + str(epoch) + '_B_' + str(batch_number) + '_step_' + str(step)
                        if os.path.isdir(plotpath):
                            print('folder exists')
                        else:
                            os.mkdir(plotpath)
                        # plot_all2(out_plot_X, out_plot_F, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, mean_y)
                        try:
                            comparison_plot(out_plot_X, out_plot_F, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, qt_plot, rt_plot, trans_plot)
                            plt.close()
                        except:
                            plt.close()
                            print('Coudl not generated plots for this epoch...')
                            pass

                    if minibatch_index % 250 == 0 and minibatch_index != 0:
                        if os.path.isdir(self.checkpoint_dir):
                            print('filter Checkpoint Directory Exists')
                        else:
                            os.mkdir(self.checkpoint_dir)
                        print("Saving filter Weights for epoch" + str(epoch))
                        save_path = self.saver.save(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(epoch) + '_' + str(step) + ".ckpt", global_step=step)
                        print("Checkpoint saved at: ", save_path)

                e += 1

    def evaluate(self, current_x, current_y, current_time, initial_meas, initial_time, initial_cov, initial_Q, initial_R, prev_x, prev_y, prev_time, ecef_ref, lla_data, sensor_onehots, epoch, minibatch_index, step):

        lla_datar = copy.copy(lla_data)
        lla_datar[:, 0] = lla_datar[:, 0] * np.pi / 180
        lla_datar[:, 1] = lla_datar[:, 1] * np.pi / 180

        seqlen = np.ones(shape=[self.batch_size_np, ])
        int_time = np.zeros(shape=[self.batch_size_np, self.max_seq])

        feed_dict = {}

        time_plotter = np.zeros([self.batch_size_np, int(current_x.shape[1]), 1])

        tstep = 0
        r1 = tstep * self.max_seq
        r2 = r1 + self.max_seq

        prev_state = copy.copy(prev_y)
        prev_meas = copy.copy(prev_x)

        seqweight = np.zeros(shape=[self.batch_size_np, self.max_seq])

        for i in range(self.batch_size_np):
            current_yt = current_y[i, :, :3]
            m = ~(current_yt == 0).all(1)
            yf = current_yt[m]
            seq = yf.shape[0]
            seqlen[i] = seq
            int_time[i, :] = range(r1, r2)
            seqweight[i, :] = m.astype(int)

        cur_time = current_x[:, 0, 0]

        time_plotter[:, 0, :] = cur_time[:, np.newaxis]
        max_t = np.max(time_plotter[0, :, 0])
        idxi = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        idxi2 = [0, 3, 1, 4, 2, 5]

        R = initial_meas[:, :, 0, np.newaxis]
        A = initial_meas[:, :, 1, np.newaxis]
        E = initial_meas[:, :, 2, np.newaxis]

        east = (R * np.sin(A) * np.cos(E))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        north = (R * np.cos(E) * np.cos(A))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        up = (R * np.sin(E))  # * ((tf.exp(tf.negative(tf.pow(se, 2) / 2))))

        lat = lla_datar[:, 0, np.newaxis, np.newaxis]
        lon = lla_datar[:, 1, np.newaxis, np.newaxis]

        cosPhi = np.cos(lat)
        sinPhi = np.sin(lat)
        cosLambda = np.cos(lon)
        sinLambda = np.sin(lon)

        tv = cosPhi * up - sinPhi * north
        wv = sinPhi * up + cosPhi * north
        uv = cosLambda * tv - sinLambda * east
        vv = sinLambda * tv + cosLambda * east

        initial_meas_uvw = np.concatenate([uv, vv, wv], axis=2)

        dtn = np.sum(np.diff(initial_time, axis=1), axis=1)
        pos = initial_meas_uvw[:, 2, :]
        vel = (initial_meas_uvw[:, 2, :] - initial_meas_uvw[:, 0, :]) / (2 * dtn)

        R1 = np.linalg.norm(initial_meas + ecef_ref[:, np.newaxis, :], axis=2, keepdims=True)
        R1 = np.mean(R1, axis=1)
        R1 = np.where(np.less(R1, np.ones_like(R1) * self.RE), np.ones_like(R1) * self.RE, R1)
        rad_temp = np.power(R1, 3)
        GMt1 = np.divide(self.GM, rad_temp)
        acc = get_legendre_np(GMt1, pos + ecef_ref, R1)

        # acc = (initial_meas[:, 2, :] - 2*initial_meas[:, 1, :] + initial_meas[:, 0, :]) / dtn**2

        initial_state = np.expand_dims(np.concatenate([pos, vel, acc, np.random.normal(loc=np.zeros_like(acc), scale=10.)], axis=1), 1)

        # current_state, covariance_out = unscented_kalman_np(self.batch_size_np, prev_meas.shape[1], initial_state[:, -1, :], initial_cov, prev_meas, prev_time)

        current_y = current_y[:, :, idxi]
        prev_state = prev_state[:, :, idxi]
        initial_state = initial_state[:, :, idxi]

        feed_dict.update({self.measurement[t]: current_x[:, t, :].reshape(-1, self.num_meas) for t in range(self.max_seq)})
        feed_dict.update({self.prev_measurement: prev_meas.reshape(-1, self.num_meas)})
        feed_dict.update({self.truth_state[t]: current_y[:, t, :].reshape(-1, self.num_state) for t in range(self.max_seq)})
        feed_dict.update({self.prev_state_truth: prev_state.reshape(-1, self.num_state)})
        feed_dict.update({self.prev_state_estimate: initial_state.reshape(-1, self.num_state)})
        feed_dict.update({self.sensor_ecef: ecef_ref})
        feed_dict.update({self.sensor_lla: lla_datar})
        feed_dict.update({self.sensor_onehots: sensor_onehots})
        feed_dict.update({self.seqlen: seqlen})
        feed_dict.update({self.int_time: int_time})
        feed_dict.update({self.seqweightin: seqweight})
        feed_dict.update({self.P_inp: initial_cov})
        feed_dict.update({self.Q_inp: initial_Q})
        feed_dict.update({self.R_inp: initial_R})
        feed_dict.update({self.state_input: initial_state.reshape(-1, self.num_state)})
        feed_dict.update({self.prev_time: prev_time[:, :, 0]})
        feed_dict.update({self.current_timei[t]: current_time[:, t, :].reshape(-1, 1) for t in range(self.max_seq)})
        feed_dict.update({self.drop_rate: 1.0})

        std = 0.0

        feed_dict.update({self.init_c_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
        feed_dict.update({self.init_h_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
        feed_dict.update({self.init_c_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
        feed_dict.update({self.init_h_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
        feed_dict.update({self.init_c_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
        feed_dict.update({self.init_h_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

        feed_dict.update({self.is_training: False})
        feed_dict.update({self.deterministic: True})
        stateful = True

        pred_output0, pred_output00, pred_output1, q_out_t, q_out, rmsp, rmsv, rmsa, rmsj, LR, \
        cov_pos_loss, cov_vel_loss, kalman_cov_loss, maha_loss, MD, trace_loss, rl, \
        entropy, qt_out, rt_out, at_out, trans_out, q_loss, \
        state_fwf, state_fwf2, new_meas, y_t_resh, Cz_t, trans1, trans2 = \
            self.sess.run([self.final_state_filter,
                           self.final_state_pred,
                           self.final_state_filter,
                           self.final_cov_filter,
                           self.final_cov_pred,
                           self.rmse_pos,
                           self.rmse_vel,
                           self.rmse_acc,
                           self.rmse_jer,
                           self.learning_rate,
                           self.error_loss_pos,
                           self.error_loss_vel,
                           self.error_loss_full,
                           self.maha_loss,
                           self.maha_out,
                           self.trace_loss,
                           self.rl,
                           self.entropy,
                           self.qo_list,
                           self.ro_list,
                           self.ao_list,
                           self.to_list,
                           self.error_loss_Q,
                           self.state_fwf,
                           self.state_fwf2,
                           self.new_meas,
                           self.y_t_resh,
                           self.Cz_t,
                           self.trans1,
                           self.trans2
                           ],
                          feed_dict)

        print("Epoch: {0:2d} MB: {1:1d} Time: {2:3d} "
              "RMSP: {3:2.2e} RMSV: {4:2.2e} RMSA: {5:2.2e} RMSJ: {6:2.2e} "
              "LR: {7:1.2e} ST: {8:1.2f} CPL: {9:1.2f} "
              "CVL: {10:1.2f} EN: {11:1.2f} QL: {12:1.2f} "
              "MD: {13:1.2f} RL: {14:1.2f} COV {15:1.2f} ".format(epoch, minibatch_index, tstep,
                                                                  rmsp, rmsv, rmsa, rmsj,
                                                                  LR, max_t, cov_pos_loss,
                                                                  cov_vel_loss, entropy, q_loss,
                                                                  MD, rl, kalman_cov_loss))

        if stateful is True:
            if self.state_type != 'GRU':
                feed_dict.update({self.init_c_fwf: state_fwf[0]})
                feed_dict.update({self.init_h_fwf: state_fwf[1]})
                feed_dict.update({self.init_c_fwf2: state_fwf2[0]})
                feed_dict.update({self.init_h_fwf2: state_fwf2[1]})

        current_y = current_y[:, :, idxo]
        prev_state = prev_state[:, :, idxo]

        pred_output0 = pred_output0[:, :, idxo]
        pred_output00 = pred_output00[:, :, idxo]
        pred_output1 = pred_output1[:, :, idxo]

        prop_output = np.array(pred_output0)
        if len(prop_output.shape) < 3:
            prop_output = np.expand_dims(prop_output, axis=1)

        pred_output = np.array(pred_output1)
        if len(pred_output.shape) < 3:
            pred_output = np.expand_dims(pred_output, axis=1)

        full_final_output = np.array(pred_output00)
        if len(full_final_output.shape) < 3:
            full_final_output = np.expand_dims(full_final_output, axis=1)

        if tstep == 0:
            out_plot_F = full_final_output[0, np.newaxis, :, :]
            out_plot_X = pred_output[0, np.newaxis, :, :]
            out_plot_P = prop_output[0, np.newaxis, :, :]

            q_plott = q_out_t[0, np.newaxis, :, :, :]
            q_plot = q_out[0, np.newaxis, :, :, :]
            qt_plot = qt_out[0, np.newaxis, :, :]
            rt_plot = rt_out[0, np.newaxis, :, :]
            at_plot = at_out[0, np.newaxis, :, :]
            trans_plot = trans_out[0, np.newaxis, :, :]

            time_vals = current_time[0, np.newaxis, :, :]
            meas_plot = new_meas[0, np.newaxis, :, :]
            truth_plot = current_y[0, np.newaxis, :, :]

        else:
            new_vals_F = full_final_output[0, :, :]  # current step
            new_vals_X = pred_output[0, :, :]
            new_vals_P = prop_output[0, :, :]
            new_q = q_out[0, :, :, :]
            new_qt = q_out_t[0, :, :, :]
            new_qtt = qt_out[0, :, :, :]
            new_rtt = rt_out[0, :, :, :]
            new_att = at_out[0, :, :, :]

            new_time = current_time[0, :, 0]
            new_meas = new_meas[0, :, :]
            new_truth = current_y[0, :, :]

        if tstep > 0:
            out_plot_F = np.concatenate([out_plot_F, new_vals_F[np.newaxis, :, :]], axis=1)
            out_plot_X = np.concatenate([out_plot_X, new_vals_X[np.newaxis, :, :]], axis=1)
            out_plot_P = np.concatenate([out_plot_P, new_vals_P[np.newaxis, :, :]], axis=1)
            meas_plot = np.concatenate([meas_plot, new_meas[np.newaxis, :, :]], axis=1)
            truth_plot = np.concatenate([truth_plot, new_truth[np.newaxis, :, :]], axis=1)
            time_vals = np.concatenate([time_vals, new_time[np.newaxis, :, np.newaxis]], axis=1)
            q_plot = np.concatenate([q_plot, new_q[np.newaxis, :, :, :]], axis=1)
            q_plott = np.concatenate([q_plott, new_qt[np.newaxis, :, :, :]], axis=1)
            qt_plot = np.concatenate([qt_plot, new_qtt[np.newaxis, :, :, :]], axis=1)
            rt_plot = np.concatenate([rt_plot, new_rtt[np.newaxis, :, :, :]], axis=1)
            # at_plot = np.concatenate([at_plot, new_att[np.newaxis, :, :, :]], axis=1)
            trans_plot = np.concatenate([trans_plot, new_att[np.newaxis, :, :, :]], axis=1)

        plotpath = self.plot_dir + '/epoch_' + str(epoch) + '_eval_B_' + str(minibatch_index) + '_step_' + str(step)
        if os.path.isdir(plotpath):
            print('folder exists')
        else:
            os.mkdir(plotpath)

        # plot_all2(out_plot_X, out_plot_F, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, mean_y)
        comparison_plot(out_plot_X, out_plot_F, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, qt_plot, rt_plot, trans_plot)

        # if minibatch_index % 50 == 0 and minibatch_index != 0:
        if os.path.isdir(self.checkpoint_dir):
            print('filter Checkpoint Directory Exists')
        else:
            os.mkdir(self.checkpoint_dir)
        print("Saving filter Weights for epoch" + str(epoch))
        save_path = self.saver.save(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(epoch) + '_' + str(step) + ".ckpt", global_step=step)
        print("Checkpoint saved at: ", save_path)

    def test(self, data_rate, max_exp_seq):

        shuffle_data = False
        self.data_rate = data_rate
        self.max_exp_seq = max_exp_seq
        # initialize all variables
        tf.global_variables_initializer().run()

        save_files = os.listdir(self.checkpoint_dir + '/')
        save_files = natsorted(save_files, reverse=True)
        recent = str.split(save_files[1], '_')
        start_epoch = recent[2]
        step = str.split(recent[3], '.')[0]
        print("Resuming run from epoch " + str(start_epoch) + ' and step ' + str(step))
        step = int(step)

        print('Loading filter...')
        self.saver = tf.train.import_meta_graph(self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step) + '.meta')
        self.saver.restore(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step))
        # self.saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step) + '.meta'))
        print("filter restored.")

        ds = DataServerLive(self.meas_dir, self.state_dir)

        # plot_count = 0
        # train_writer = tf.summary.FileWriter('./log/0/train/', self.sess.graph)

        n_train_batches = int(ds.num_train / self.batch_size_np)

        for minibatch_index in range(n_train_batches):

            testing = True
            print('Testing filter for batch ' + str(minibatch_index))

            # Data is unnormalized at this point
            x_train, y_train, batch_number, total_batches, ecef_ref, lla_data = ds.load(batch_size=self.batch_size_np, constant=self.constant, test=testing,
                                                                                        max_seq_len=self.max_exp_seq, HZ=self.data_rate)

            lla_datar = copy.copy(lla_data)
            lla_datar[:, 0] = lla_datar[:, 0] * np.pi / 180
            lla_datar[:, 1] = lla_datar[:, 1] * np.pi / 180
            # print("Batch Number: {0:2d} out of {1:2d}".format(batch_number, total_batches))

            x_train = np.concatenate([x_train[:, :, 0, np.newaxis], x_train[:, :, 4:7]], axis=2)  # rae measurements

            y_uvw = y_train[:, :, :3] - np.ones_like(y_train[:, :, :3]) * ecef_ref[:, np.newaxis, :]
            # y_enu = np.zeros_like(y_uvw)
            # y_rae = np.zeros_like(y_uvw)
            zero_rows = (y_train[:, :, :3] == 0).all(2)
            for i in range(y_train.shape[0]):
                zz = zero_rows[i, :, np.newaxis]
                y_uvw[i, :, :] = np.where(zz, np.zeros_like(y_uvw[i, :, :]), y_uvw[i, :, :])

            y_train = np.concatenate([y_uvw, y_train[:, :, 3:]], axis=2)

            # if shuffle_data:
            #     shuf = np.arange(x_train.shape[0])
            #     np.random.shuffle(shuf)
            #     x_train = x_train[shuf]
            #     y_train = y_train[shuf]

            s_train = x_train

            # n_train_batches = int(x_train.shape[0] / self.batch_size_np)
            # print("Batch Number: {0:2d} out of {1:2d}".format(batch_number, total_batches))

            x0, y0, meta0, prev_y, prev_x, prev_time, prev_meta, initial_meas, initial_state_truth, initial_time = prepare_batch(0, x_train, y_train, s_train,
                                                                                                                                 seq_len=self.max_seq, batch_size=self.batch_size_np,
                                                                                                                                 new_batch=True, window_mode=self.window_mode, pad_front=self.pad_front)

            count, _, _, _, _, _, prev_cov, prev_Q, prev_R, q_plot, q_plott, k_plot, out_plot_X, out_plot_F, out_plot_P, time_vals, \
            meas_plot, truth_plot, Q_plot, R_plot, maha_plot, x, y, meta = initialize_run_variables(self.batch_size_np, self.max_seq, self.num_state, x0, y0, meta0)

            prev_cov = prev_cov[:, -1:, :, :]

            # print('Resetting Feed Dict')
            feed_dict = {}

            time_plotter = np.zeros([self.batch_size_np, int(x.shape[1]), 1])

            tstep = 0
            r1 = tstep * self.max_seq
            r2 = r1 + self.max_seq

            prev_state = copy.copy(prev_y)
            prev_meas = copy.copy(prev_x)

            current_x, current_y, current_time, current_meta = \
                get_feed_time_asynch(x, y, meta, prev_x, prev_y, prev_time, prev_meta, self.max_seq, tstep, self.num_state, self.window_mode)

            seqlen = np.ones(shape=[self.batch_size_np, ])
            int_time = np.zeros(shape=[self.batch_size_np, x0.shape[1]])

            seqweight = np.zeros(shape=[self.batch_size_np, x0.shape[1]])

            for i in range(self.batch_size_np):
                current_yt = y_data[i, :, :3]
                m = ~(current_yt == 0).all(1)
                yf = current_yt[m]
                seq = yf.shape[0]
                seqlen[i] = seq
                int_time[i, :] = range(r1, r2)
                seqweight[i, :] = m.astype(int)

            cur_time = x[:, 0, 0]

            time_plotter[:, 0, :] = cur_time[:, np.newaxis]
            max_t = np.max(time_plotter[0, :, 0])
            idxi = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
            idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
            # idxi2 = [0, 3, 1, 4, 2, 5]

            R = initial_meas[:, :, 0, np.newaxis]
            A = initial_meas[:, :, 1, np.newaxis]
            E = initial_meas[:, :, 2, np.newaxis]

            east = (R * np.sin(A) * np.cos(E))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
            north = (R * np.cos(E) * np.cos(A))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
            up = (R * np.sin(E))  # * ((tf.exp(tf.negative(tf.pow(se, 2) / 2))))

            lat = lla_datar[:, 0, np.newaxis, np.newaxis]
            lon = lla_datar[:, 1, np.newaxis, np.newaxis]

            cosPhi = np.cos(lat)
            sinPhi = np.sin(lat)
            cosLambda = np.cos(lon)
            sinLambda = np.sin(lon)

            tv = cosPhi * up - sinPhi * north
            wv = sinPhi * up + cosPhi * north
            uv = cosLambda * tv - sinLambda * east
            vv = sinLambda * tv + cosLambda * east

            initial_meas_uvw = np.concatenate([uv, vv, wv], axis=2)

            dtn = np.sum(np.diff(initial_time, axis=1), axis=1)
            pos = initial_meas_uvw[:, 2, :]
            vel = (initial_meas_uvw[:, 2, :] - initial_meas_uvw[:, 0, :]) / (2 * dtn)

            R1 = np.linalg.norm(initial_meas_uvw + ecef_ref[:, np.newaxis, :], axis=2, keepdims=True)
            R1 = np.mean(R1, axis=1)
            R1 = np.where(np.less(R1, np.ones_like(R1) * self.RE), np.ones_like(R1) * self.RE, R1)
            rad_temp = np.power(R1, 3)
            GMt1 = np.divide(self.GM, rad_temp)
            acc = get_legendre_np(GMt1, pos + ecef_ref, R1)

            # acc = (initial_meas[:, 2, :] - 2*initial_meas[:, 1, :] + initial_meas[:, 0, :]) / dtn**2

            std = 0.0

            feed_dict.update({self.init_c_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            feed_dict.update({self.init_h_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            feed_dict.update({self.init_c_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            feed_dict.update({self.init_h_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            feed_dict.update({self.init_c_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            feed_dict.update({self.init_h_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

            feed_dict.update({self.is_training: False})
            feed_dict.update({self.deterministic: True})
            stateful = True

            initial_state = np.expand_dims(np.concatenate([pos, vel, acc, np.random.normal(loc=np.zeros_like(acc), scale=1)], axis=1), 1)
            initial_cov = prev_cov[:, -1, :, :]
            initial_Q = prev_Q[:, -1, :, :]
            initial_R = prev_R[:, -1, :, :]
            initial_state = initial_state[:, :, idxi]

            dt0 = prev_time - initial_time[:, -1:, :]
            dt1 = current_time[:, 0, :] - prev_time[:, 0, :]
            current_state, covariance_out, converted_meas, pred_state, pred_covariance = \
                unscented_kalman_np(self.batch_size_np, prev_meas.shape[1], initial_state[:, -1, :], prev_cov[:, -1, :, :], prev_meas, prev_time, dt0, dt1, lla_datar)
            steps = x_data.shape[1]

            for step in range(0, steps):

                current_y = y_data[:, step, idxi]
                prev_state = prev_state[:, :, idxi]

                current_x = x_data[:, step, :]
                initial_state = initial_state[:, :, idxi]

                if np.all(current_x == 0):
                    continue

                current_time = time_data[:, step, :]
                current_int = int_time[:, step, tf.newaxis]
                current_weight = seqweight[:, step, tf.newaxis]

                feed_dict.update({self.measurement[t]: current_x[:, t, :].reshape(-1, self.num_meas) for t in range(self.max_seq)})
                feed_dict.update({self.prev_measurement: prev_meas.reshape(-1, self.num_meas)})
                feed_dict.update({self.prev_covariance_estimate: prev_cov[:, -1, :, :]})
                feed_dict.update({self.truth_state[t]: current_y[:, t, :].reshape(-1, self.num_state) for t in range(self.max_seq)})
                feed_dict.update({self.prev_state_truth: prev_state.reshape(-1, self.num_state)})
                feed_dict.update({self.prev_state_estimate: initial_state.reshape(-1, self.num_state)})
                feed_dict.update({self.sensor_ecef: ecef_ref})
                feed_dict.update({self.sensor_lla: lla_datar})
                feed_dict.update({self.sensor_onehots: sensor_onehots})
                feed_dict.update({self.seqlen: seqlen})
                feed_dict.update({self.int_time: int_time})
                feed_dict.update({self.update_condition: update})
                feed_dict.update({self.is_training: True})
                feed_dict.update({self.seqweightin: seqweight})
                feed_dict.update({self.P_inp: covariance_out[-1]})
                feed_dict.update({self.Q_inp: initial_Q})
                feed_dict.update({self.R_inp: initial_R})
                feed_dict.update({self.state_input: current_state.reshape(-1, self.num_state)})
                feed_dict.update({self.prev_time: prev_time[:, :, 0]})
                feed_dict.update({self.current_timei[t]: current_time[:, t, :].reshape(-1, 1) for t in range(self.max_seq)})
                feed_dict.update({self.drop_rate: 1.0})

                pred_output0, pred_output00, pred_output1, q_out_t, q_out, rmsp, rmsv, rmsa, rmsj, LR, \
                cov_pos_loss, cov_vel_loss, kalman_cov_loss, maha_loss, MD, trace_loss, rl, \
                entropy, qt_out, rt_out, at_out, trans_out, q_loss, \
                state_fwf, state_fwf2, new_meas, y_t_resh, Cz_t, trans1, trans2 = \
                    self.sess.run([self.final_state_filter,
                                   self.final_state_pred,
                                   self.final_state_filter,
                                   self.final_cov_pred,
                                   self.final_cov_pred,
                                   self.rmse_pos,
                                   self.rmse_vel,
                                   self.rmse_acc,
                                   self.rmse_jer,
                                   self.learning_rate,
                                   self.error_loss_pos,
                                   self.error_loss_vel,
                                   self.error_loss_full,
                                   self.maha_loss,
                                   self.maha_out,
                                   self.trace_loss,
                                   self.rl,
                                   self.entropy,
                                   self.qo_list,
                                   self.ro_list,
                                   self.ao_list,
                                   self.to_list,
                                   self.error_loss_Q,
                                   self.state_fwf,
                                   self.state_fwf2,
                                   self.new_meas,
                                   self.y_t_resh,
                                   self.Cz_t,
                                   self.trans1,
                                   self.trans2
                                   ],
                                  feed_dict)

                print("Test: {0:2d} MB: {1:1d} Time: {2:3d} "
                      "RMSP: {3:2.2e} RMSV: {4:2.2e} RMSA: {5:2.2e} RMSJ: {6:2.2e} "
                      "LR: {7:1.2e} ST: {8:1.2f} CPL: {9:1.2f} "
                      "CVL: {10:1.2f} EN: {11:1.2f} QL: {12:1.2f} "
                      "MD: {13:1.2f} RL: {14:1.2f} COV {15:1.2f} ".format(0, minibatch_index, tstep,
                                                                          rmsp, rmsv, rmsa, rmsj,
                                                                          LR, max_t, cov_pos_loss,
                                                                          cov_vel_loss, entropy, q_loss,
                                                                          MD, rl, kalman_cov_loss))

                if stateful is True:
                    feed_dict.update({self.init_c_fwf: state_fwf[0]})
                    feed_dict.update({self.init_h_fwf: state_fwf[1]})
                    feed_dict.update({self.init_c_fwf2: state_fwf2[0]})
                    feed_dict.update({self.init_h_fwf2: state_fwf2[1]})

                current_y = current_y[:, np.newaxis, :]
                # prev_state = prev_state[:, np.newaxis, :]

                current_y = current_y[:, :, idxo]
                prev_state = prev_state[:, :, idxo]

                pred_output0 = pred_output0[:, :, idxo]
                pred_output00 = pred_output00[:, :, idxo]
                pred_output1 = pred_output1[:, :, idxo]

                prop_output = np.array(pred_output0)
                # if len(prop_output.shape) < 3:
                #     prop_output = np.expand_dims(prop_output, axis=1)

                pred_output = np.array(pred_output1)
                # if len(pred_output.shape) < 3:
                #     pred_output = np.expand_dims(pred_output, axis=1)

                full_final_output = np.array(pred_output00)
                # if len(full_final_output.shape) < 3:
                #     full_final_output = np.expand_dims(full_final_output, axis=1)

                idx = -1

                temp_prev_x = current_x[:, np.newaxis, :]
                temp_prev_y = current_y
                temp_prev_time = current_time[:, np.newaxis, :]
                temp_prev_cov = q_out_t
                temp_new_state = pred_output0[:, -1, np.newaxis, :]
                # temp_pred0 = np.expand_dims(prop_output[:, idx, :], axis=1)
                # temp_pred1 = np.expand_dims(pred_output[:, idx, :], axis=1)

                if step == 0:
                    # prev_y = prev_y[:, :, idxo]

                    new_vals_F = full_final_output[0, -1, np.newaxis, :]
                    out_plot_F = new_vals_F[np.newaxis, :, :]
                    new_vals_X = pred_output[0, -1, np.newaxis, :]
                    out_plot_X = new_vals_X[np.newaxis, :, :]
                    new_vals_P = prop_output[0, -1, np.newaxis, :]
                    out_plot_P = new_vals_P[np.newaxis, :, :]

                    new_q = q_out[0, :, :, :]
                    new_qt = q_out_t[0, :, :, :]

                    q_plot = new_q[np.newaxis, :, :, :]
                    q_plott = new_qt[np.newaxis, :, :, :]

                    qt_plot = qt_out[0, np.newaxis, :, :, :]
                    rt_plot = rt_out[0, np.newaxis, :, :, :]
                    at_plot = at_out[0, np.newaxis, :, :, :]

                    new_time = current_time[0, :]
                    new_meas = new_meas[0, :, :]
                    new_truth = current_y[0, :, :]

                    # qt_plot = np.tile(qt_out[0, np.newaxis, np.newaxis, :, :], [1, self.max_seq, 1, 1])
                    # rt_plot = np.tile(rt_out[0, np.newaxis, np.newaxis, :, :], [1, self.max_seq, 1, 1])
                    # at_plot = np.tile(at_out[0, np.newaxis, np.newaxis, :, :], [1, self.max_seq, 1, 1])

                    # new_time = current_time[0, -1:, 0]
                    time_vals = new_time[np.newaxis, :]
                    # new_meas = current_x[0, -1, :]
                    meas_plot = new_meas[np.newaxis, :, :]
                    # new_truth = current_y[0, -1, :]
                    truth_plot = new_truth[np.newaxis, :, :]

                else:
                    new_vals_F = full_final_output[0, -1, :]  # current step
                    new_vals_X = pred_output[0, -1, :]
                    new_vals_P = prop_output[0, -1, :]
                    new_q = q_out[0, :, :]
                    new_qt = q_out_t[0, :, :]
                    new_qtt = qt_out[0, :, :]
                    new_rtt = rt_out[0, :, :]
                    new_att = at_out[0, :, :]

                    new_time = current_time[0, -1]
                    new_meas = new_meas[0, -1, :]
                    new_truth = current_y[0, -1, :]

                if step > 0:
                    out_plot_F = np.concatenate([out_plot_F, new_vals_F[np.newaxis, np.newaxis, :]], axis=1)
                    out_plot_X = np.concatenate([out_plot_X, new_vals_X[np.newaxis, np.newaxis, :]], axis=1)
                    out_plot_P = np.concatenate([out_plot_P, new_vals_P[np.newaxis, np.newaxis, :]], axis=1)
                    meas_plot = np.concatenate([meas_plot, new_meas[np.newaxis, np.newaxis, :]], axis=1)
                    truth_plot = np.concatenate([truth_plot, new_truth[np.newaxis, np.newaxis, :]], axis=1)
                    time_vals = np.concatenate([time_vals, new_time[np.newaxis, np.newaxis]], axis=1)
                    q_plot = np.concatenate([q_plot, new_q[np.newaxis, :, :, :]], axis=1)
                    q_plott = np.concatenate([q_plott, new_qt[np.newaxis, :, :, :]], axis=1)
                    qt_plot = np.concatenate([qt_plot, new_qtt[np.newaxis, :, :, :]], axis=1)
                    rt_plot = np.concatenate([rt_plot, new_rtt[np.newaxis, :, :, :]], axis=1)
                    at_plot = np.concatenate([at_plot, new_att[np.newaxis, :, :, :]], axis=1)

                # err0 = np.sum(np.abs(temp_prev_y - temp_pred0))
                # err1 = np.sum(np.abs(temp_prev_y - temp_pred1))
                # if err1 < err0 and e > 5:
                #     new_prev = temp_pred0
                # else:
                #     new_prev = temp_pred1

                initial_state = np.concatenate([initial_state, temp_new_state], axis=1)
                initial_state = initial_state[:, 1:, :]

                # prev_state3 = np.concatenate([prev_state3, temp_pred1], axis=1)
                # prev_state3 = prev_state3[:, 1:, :]

                prev_state = np.concatenate([prev_state, temp_prev_y], axis=1)
                prev_state = prev_state[:, 1:, :]

                prev_time = np.concatenate([prev_time, temp_prev_time], axis=1)
                prev_time = prev_time[:, 1:, :]

                initial_cov = np.concatenate([initial_cov, temp_prev_cov], axis=1)
                initial_cov = initial_cov[:, 1:, :, :]

                prev_meas = np.concatenate([prev_meas, temp_prev_x], axis=1)
                prev_meas = prev_meas[:, 1:, :]

                # prev_y = copy.copy(prev_state)
                # prev_x = copy.copy(prev_meas)

            trans_plot = np.zeros_like(out_plot_X)

            plotpath = self.plot_dir + '/epoch_' + str(9999) + '_TEST_B_' + str(minibatch_index)
            if os.path.isdir(plotpath):
                print('folder exists')
            else:
                os.mkdir(plotpath)

            comparison_plot(out_plot_X, out_plot_F, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, qt_plot, rt_plot, trans_plot)