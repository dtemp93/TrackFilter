from propagation_utils import *
from helper3 import *
from helper2 import *
from load_all_data_4 import DataServerLive
from modules import *
import math
from plotting import *
from old.attention import *

import tensorflow as tf
import tensorflow.contrib as tfc
from tensorflow.contrib.layers import fully_connected as FCL
import numpy as np
import tensorflow_probability as tfp

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
                 data_dir='', filter_name='', plot_dir='', save_dir='',
                 F_hidden=12, R_hidden=12, num_state=12, num_meas=3, max_seq=2,
                 max_epoch=10000, RE=6378137, GM=398600441890000, batch_size=10,
                 window_mode=False, pad_front=False, constant=False):

        self.sess = sess
        self.mode = mode
        self.max_seq = max_seq
        self.num_mixtures = 5
        self.max_sj = 250
        self.min_sj = 1e-3
        self.max_at = 1
        self.min_at = 1e-3
        self.train_init_state = trainable_state
        self.F_hidden = F_hidden
        self.R_hidden = R_hidden
        self.num_state = num_state
        self.num_meas = num_meas
        self.plot_dir = plot_dir
        self.checkpoint_dir = save_dir
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

        self.Ql = [None] * plen
        self.Qlp = [None] * plen
        self.Ql2 = [None] * plen
        self.Ql3 = [None] * plen
        self.Ql4 = [None] * plen
        self.Sl = [None] * plen
        self.Sl2 = [None] * plen
        self.Al = [None] * plen
        self.Bl = [None] * plen
        self.Pl = [None] * plen
        self.Atl = [None] * plen
        self.state_fwc = [None] * plen
        self.state_bwc = [None] * plen
        self.state_fwc2 = [None] * plen
        self.state_bwc2 = [None] * plen
        self.state_fws = [None] * plen
        self.state_bws = [None] * plen
        self.state_fws2 = [None] * plen
        self.state_bws2 = [None] * plen
        self.state_fwf = [None] * plen
        self.state_bwf = [None] * plen
        self.source_track_out_fwf = [None for _ in range(plen)]
        self.source_track_out_fws = [None for _ in range(plen)]
        self.source_track_out_bws = [None for _ in range(plen)]
        self.source_track_out_fws2 = [None for _ in range(plen)]
        self.source_track_out_bws2 = [None for _ in range(plen)]
        self.source_track_out_fwc = [None for _ in range(plen)]
        self.source_track_out_bwc = [None for _ in range(plen)]
        self.source_track_out_fwc2 = [None for _ in range(plen)]
        self.source_track_out_bwc2 = [None for _ in range(plen)]
        self.predicted_state_update_temp = [None for _ in range(plen)]
        self.predicted_state_update_temp1 = [None for _ in range(plen)]
        self.predicted_state_update_temp2 = [None for _ in range(plen)]
        self.predicted_state_update_temp3 = [None for _ in range(plen)]

        self.al = list()
        self.al2 = list()
        self.bl = list()
        self.cl = list()
        self.mul = list()
        self.mult = list()
        self.sigl = list()
        self.siglt = list()
        self.sf1 = list()
        self.sjl = list()
        self.new_measl = list()
        self.nl = list()
        self.soutl = list()

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
            var_reuse = False;
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

    def project(self, x, y_width, name='', dtype=tf.float32):
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
            return y, w1

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

    def filter_measurement(self, prev_state):
        alpha = 1e-3 * tf.ones([self.batch_size, 1], dtype=tf.float64)
        beta = 2. * tf.ones([self.batch_size, 1], dtype=tf.float64)
        k = 0. * tf.ones([self.batch_size, 1], dtype=tf.float64)

        L = tf.cast(self.num_state, dtype=tf.float64)
        lam = alpha * (L + k) - L
        c1 = L + lam
        tmat = tf.ones([1, 2 * self.num_state], dtype=tf.float64)
        Wm = tf.concat([(lam / c1), (0.5 / c1) * tmat], axis=1)
        Wc1 = tf.expand_dims(copy.copy(Wm[:, 0]), axis=1) + (tf.ones_like(alpha, dtype=tf.float64) - (alpha) + beta)
        Wc = tf.concat([Wc1, copy.copy(Wm[:, 1:])], axis=1)
        c = tf.sqrt(c1)

        print('Building UKF')

        all_states = tf.stack(prev_state, axis=1)
        all_meas = tf.stack(self.prev_measurement, axis=1)
        # all_time = tf.stack(self.prev_time, axis=1) / 200

        meanv = tf.ones_like(all_states) * self.meanv
        # stdv = tf.ones_like(final_state_gs) * self.stdv

        all_states = all_states / meanv

        pos_m = tf.concat([meanv[:, :, 0, tf.newaxis], meanv[:, :, 4, tf.newaxis], meanv[:, :, 8, tf.newaxis]], axis=2)

        all_meas = all_meas / pos_m

        pr0 = all_meas - tf.squeeze(tf.matmul(tf.tile(self.meas_mat[:, tf.newaxis, :, :], [1, self.max_seq, 1, 1]), all_states[:, :, :, tf.newaxis]), -1)

        h = tf.concat([all_meas, all_states, pr0], axis=2)

        # rnn_inp03, _ = self.filterh2(h[: 1:], y_width=self.F_hidden, name='init', dtype=self.vdtype)
        rnn_inp03 = FCL(h, self.F_hidden, activation_fn=tf.nn.elu, scope='init', reuse=tf.AUTO_REUSE)

        cov_est0 = tf.reshape(self.P_inp, [self.batch_size, self.num_state, self.num_state])

        state_list = list()
        for q in range(self.max_seq):

            if q == self.max_seq - 1:
                inp = (self.current_timei, rnn_inp03[:, q, :])
            else:
                inp = (self.prev_time[q], rnn_inp03[:, q, :])

            if q == 0:
                with tf.variable_scope('Source_Track_Forward/state'):
                    # dynamic_out = tf.nn.dynamic_rnn(self.source_fwf, rnn_inp03, seq_length=self.seqlen, initial_state=self.state_fw_in_state, dtype=self.vdtype)
                    (self.source_track_out_fwf, self.state_fwf) = self.source_fwf(inp, state=self.state_fw_in_state)
            else:
                with tf.variable_scope('Source_Track_Forward/state'):
                    tf.get_variable_scope().reuse_variables()
                    # dynamic_out = tf.nn.dynamic_rnn(self.source_fwf, rnn_inp03, seq_length=self.seqlen, initial_state=self.state_fw_in_state, dtype=self.vdtype)
                    (self.source_track_out_fwf, self.state_fwf) = self.source_fwf(inp, state=self.state_fwf)

            state_list.append(tf.concat(self.state_fwf, axis=1))

        out_states = tf.stack(state_list, axis=1)

        with tf.variable_scope('attn/state'):
            weighted_output = attention_time(out_states, self.F_hidden, dtype=self.vdtype)
            weighted_output = normalize(weighted_output, scope='ln1', dtype=self.vdtype)

        # alpha_mix = self.source_track_out_fwf[:, 25:50]
        # sjl = FCL(self.source_track_out_fwf[:, 50:], activation_fn=tf.nn.relu, scope='sjvar', reuse=tf.AUTO_REUSE)
        # alpha_mix = FCL(alpha_mix, self.num_mixtures, activation_fn=tf.nn.softmax, scope='alpha_var', reuse=tf.AUTO_REUSE)

        # u_part = self.source_track_out_fwf[:, 50:]
        # u_part = FCL(u_part, 6, activation_fn=None, scope='u/state', reuse=tf.AUTO_REUSE)

        # sjaj0 = FCL(r_part, self.F_hidden, activation_fn=tf.nn.elu, scope='0/measr', reuse=tf.AUTO_REUSE)

        rm0 = FCL(weighted_output, 6, activation_fn=None, scope='1/state', weights_initializer=tfc.layers.variance_scaling_initializer(), reuse=tf.AUTO_REUSE)
        rm = FCL(rm0, 6, activation_fn=None, scope='2/state', weights_initializer=tfc.layers.variance_scaling_initializer(), reuse=tf.AUTO_REUSE) + rm0
        rdd = FCL(rm[:, :6], 6, activation_fn=tf.nn.softplus, scope='3/state', weights_initializer=tfc.layers.variance_scaling_initializer(), reuse=tf.AUTO_REUSE)

        # self.rd = rdd + tf.ones_like(rdd) * 1e-6
        self.rd = tril_with_diag_softplus_and_shift(rdd, diag_shift=0.00001, name='4/state')
        rdist = tfd.MultivariateNormalTriL(loc=None, scale_tril=self.rd, name='rdist')
        # rdist = tfd.MultivariateNormalDiag(loc=None, scale_diag=self.rd)

        cur_meas_temp = self.measurement
        # sj = self.om[:, :, 0] * 250

        # sjx = FCL(rm[:, -3, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, scope='sjx/state', weights_initializer=tfc.layers.variance_scaling_initializer(), reuse=tf.AUTO_REUSE) * 100
        # sjx = sjx + self.om[:, :, 0] * 1
        #
        # sjy = FCL(rm[:, -2, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, scope='sjy/state', weights_initializer=tfc.layers.variance_scaling_initializer(), reuse=tf.AUTO_REUSE) * 100
        # sjy = sjy + self.om[:, :, 0] * 1
        #
        # sjz = FCL(rm[:, -1, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, scope='sjz/state', weights_initializer=tfc.layers.variance_scaling_initializer(), reuse=tf.AUTO_REUSE) * 100
        # sjz = sjz + self.om[:, :, 0] * 1

        pstate_est_temp = copy.copy(prev_state[-1])

        dt = self.current_timei - self.prev_time[-1]
        dt = tf.where(dt <= 1 / 100, tf.ones_like(dt) * 1 / 25, dt)

        self.Qt, self.At, self.Bt, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                                              dimension=int(self.num_state / 3),
                                              sjix=self.om[:, :, 0] * 100 ** 2,
                                              sjiy=self.om[:, :, 0] * 100 ** 2,
                                              sjiz=self.om[:, :, 0] * 100 ** 2,
                                              aji=self.om[:, :, 0] * 1.0)

        qcholr = tf.cholesky(tf.cast(cov_est0, tf.float64))
        self.Rt = rdist.covariance()

        Am = tf.expand_dims(c, axis=2) * tf.cast(qcholr, tf.float64)
        Y = tf.tile(tf.expand_dims(pstate_est_temp, axis=2), [1, 1, self.num_state])
        X = tf.concat([tf.expand_dims(pstate_est_temp, axis=2), Y + Am, Y - Am], axis=2)
        X = tf.transpose(X, [0, 2, 1])

        x1, X1, P1, X2 = ut_state_batch(X, Wm, Wc, self.Qt, self.num_state, self.batch_size, self.At)
        z1, Z1, P2, Z2 = ut_meas(X1, Wm, Wc, self.Rt, self.meas_mat, self.batch_size)

        P12 = tf.matmul(tf.matmul(X2, tf.matrix_diag(Wc)), Z2, transpose_b=True)

        gain = tf.matmul(P12, tf.matrix_inverse(P2))
        pos_res2 = cur_meas_temp[:, :, tf.newaxis] - tf.matmul(self.meas_mat, x1)
        x = x1 + tf.matmul(gain, pos_res2)

        cov_est_t0 = P1 - tf.matmul(gain, P12, transpose_b=True)
        cov_est_t = (cov_est_t0 + tf.transpose(cov_est_t0, [0, 2, 1])) / 2

        final_state = x[:, :, 0]
        final_cov = cov_est_t

        self.mvn_smooth = tfd.MultivariateNormalTriL(loc=final_state, scale_tril=tf.cholesky(final_cov))

        self.ro_list = self.Rt
        self.qo_list = self.Qt
        self.rd_list = self.rd
        self.ao_list = self.At

        print('Completed UKF')

        return final_state, final_cov

    def filter_measurement_set(self, prev_state):

        print('Building UKF')

        # prev_state = tf.stack(prev_state, axis=1)
        # prev_meas = tf.stack(self.prev_measurement, axis=1)
        # prev_time = tf.stack(self.prev_time, axis=1) / 200

        # cur_states = tf.stack(prev_state, axis=1)
        # cur_meas = tf.stack(self.measurement, axis=1)
        # cur_time = tf.stack(self.current_timei, axis=1) / 200

        # meanv = tf.ones_like(prev_state) * self.meanv
        # stdv = tf.ones_like(final_state_gs) * self.stdv

        # all_states = all_states / meanv

        # pos_m = tf.concat([meanv[:, 0, tf.newaxis], meanv[:, 4, tf.newaxis], meanv[:, 8, tf.newaxis]], axis=1)

        # all_meas = cur_meas / pos_m[:, tf.newaxis, :]

        # pr0 = all_meas - tf.squeeze(tf.matmul(tf.tile(self.meas_mat[:, tf.newaxis, :, :], [1, self.max_seq, 1, 1]), all_states[:, :, :, tf.newaxis]), -1)

        # h = tf.concat([cur_time], axis=2)

        fl = list()
        cov_list = list()
        rolist = list()
        qolist = list()
        rdlist = list()
        aolist = list()
        state_list = list()
        meas_list = list()
        for q in range(self.max_seq):

            if q == 0:
                pstate_est = prev_state
                cov_est0 = tf.reshape(self.P_inp, [self.batch_size, self.num_state, self.num_state])
                prev_time = self.prev_time
            else:
                pstate_est = x[:, :, 0]
                cov_est0 = cov_est_t

            dt = self.current_timei[q] - prev_time
            prev_time = self.current_timei[q]
            dt = tf.where(dt <= 1 / 100, tf.ones_like(dt) * 1 / 25, dt)

            meas_rae = self.measurement[q]

            lat = self.sensor_lla[:, 0, tf.newaxis]
            lon = self.sensor_lla[:, 1, tf.newaxis]
            alt = self.sensor_lla[:, 2, tf.newaxis]

            R = meas_rae[:, 0, tf.newaxis]
            A = meas_rae[:, 1, tf.newaxis]
            E = meas_rae[:, 2, tf.newaxis]

            temp_input = tf.concat([self.current_timei[q], dt, pstate_est, meas_rae, tf.matrix_diag_part(cov_est0)], axis=1)

            # temp_input = tfc.layers.batch_norm(temp_input, updates_collections=None, is_training=self.is_training)
            temp_input = normalize(temp_input, epsilon=1e-15, scope='layernorm', reuse=tf.AUTO_REUSE, dtype=self.vdtype)

            # rnn_inp03 = FCL(temp_input, self.F_hidden, activation_fn=tf.nn.elu, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='input/state', reuse=tf.AUTO_REUSE)

            rnn_inp03, _ = self.filterh(temp_input, y_width=self.F_hidden * 2, name='init', dtype=self.vdtype)

            inp = (self.current_timei[q], rnn_inp03[:, :self.F_hidden])

            if q == 0:
                with tf.variable_scope('Source_Track_Forward/state'):
                    # dynamic_out = tf.nn.dynamic_rnn(self.source_fwf, rnn_inp03, seq_length=self.seqlen, initial_state=self.state_fw_in_state, dtype=self.vdtype)
                    (self.source_track_out_fwf, self.state_fwf) = self.source_fwf(inp, state=self.state_fw_in_state)
            else:
                with tf.variable_scope('Source_Track_Forward/state'):
                    tf.get_variable_scope().reuse_variables()
                    # dynamic_out = tf.nn.dynamic_rnn(self.source_fwf, rnn_inp03, seq_length=self.seqlen, initial_state=self.state_fw_in_state, dtype=self.vdtype)
                    (self.source_track_out_fwf, self.state_fwf) = self.source_fwf(inp, state=self.state_fwf)

            if q == 0:
                with tf.variable_scope('Source_Track_Forward2/state'):
                    # dynamic_out = tf.nn.dynamic_rnn(self.source_fwf, rnn_inp03, seq_length=self.seqlen, initial_state=self.state_fw_in_state, dtype=self.vdtype)
                    (self.source_track_out_fwf2, self.state_fwf2) = self.source_fwf2((self.current_timei[q], rnn_inp03[:, self.F_hidden:]), state=self.state_fw_in_state2)
            else:
                with tf.variable_scope('Source_Track_Forward2/state'):
                    tf.get_variable_scope().reuse_variables()
                    # dynamic_out = tf.nn.dynamic_rnn(self.source_fwf, rnn_inp03, seq_length=self.seqlen, initial_state=self.state_fw_in_state, dtype=self.vdtype)
                    (self.source_track_out_fwf2, self.state_fwf2) = self.source_fwf2((self.current_timei[q], rnn_inp03[:, self.F_hidden:]), state=self.state_fwf2)

            state_list.append(tf.concat(self.state_fwf, axis=1))

            r_part = tf.concat(self.state_fwf, axis=1)
            q_part = tf.concat(self.state_fwf2, axis=1)
            # r_part = tf.concat([self.source_track_out_fwf, self.source_track_out_fwf2], axis=1)
            # r_part = self.source_track_out_fwf
            # q_part = self.source_track_out_fwf2
            # alpha_mix = self.source_track_out_fwf[:, 25:50]
            # sjl = FCL(self.source_track_out_fwf[:, 50:], activation_fn=tf.nn.relu, scope='sjvar', reuse=tf.AUTO_REUSE)
            # alpha_mix = FCL(alpha_mix, self.num_mixtures, activation_fn=tf.nn.softmax, scope='alpha_var', reuse=tf.AUTO_REUSE)

            # u_part = self.source_track_out_fwf[:, 50:]
            # u_part = FCL(u_part, 6, activation_fn=None, scope='u/state', reuse=tf.AUTO_REUSE)

            # sjaj0 = FCL(r_part, self.F_hidden, activation_fn=tf.nn.elu, scope='0/measr', reuse=tf.AUTO_REUSE)

            rm0 = FCL(r_part, 9, activation_fn=None, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='1/state', reuse=tf.AUTO_REUSE)
            rm = FCL(rm0, 9, activation_fn=None, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='2/state', reuse=tf.AUTO_REUSE)

            sr = FCL(rm[:, 0, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='sr', reuse=tf.AUTO_REUSE) * 10
            sa = FCL(rm[:, 1, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='sa', reuse=tf.AUTO_REUSE) * 1e-3
            se = FCL(rm[:, 2, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='sz', reuse=tf.AUTO_REUSE) * 1e-3

            # sr = sr + tf.ones_like(sr)*1e-3
            # se = se + tf.ones_like(sr)*1e-9
            # sa = sa + tf.ones_like(sr)*1e-9

            apx = tf.pow(tf.sin(A), 2) * tf.sinh(tf.pow(sa, 2)) + tf.pow(tf.cos(A), 2) * tf.cosh(tf.pow(sa, 2))
            apy = tf.pow(tf.sin(A), 2) * tf.cosh(tf.pow(sa, 2)) + tf.pow(tf.cos(A), 2) * tf.sinh(tf.pow(sa, 2))
            apz = tf.pow(tf.sin(E), 2) * tf.cosh(tf.pow(se, 2)) + tf.pow(tf.cos(E), 2) * tf.sinh(tf.pow(se, 2))
            apxy = tf.pow(tf.sin(E), 2) * tf.sinh(tf.pow(se, 2)) + tf.pow(tf.cos(E), 2) * tf.cosh(tf.pow(se, 2))

            bpx = tf.pow(tf.sin(A), 2) * tf.sinh(tf.pow(2 * sa, 2)) + tf.pow(tf.cos(A), 2) * tf.cosh(tf.pow(2 * sa, 2))
            bpy = tf.pow(tf.sin(A), 2) * tf.cosh(tf.pow(2 * sa, 2)) + tf.pow(tf.cos(A), 2) * tf.sinh(tf.pow(2 * sa, 2))
            bpz = tf.pow(tf.sin(E), 2) * tf.cosh(tf.pow(2 * se, 2)) + tf.pow(tf.cos(E), 2) * tf.sinh(tf.pow(2 * se, 2))
            bpxy = tf.pow(tf.sin(E), 2) * tf.sinh(tf.pow(2 * se, 2)) + tf.pow(tf.cos(E), 2) * tf.cosh(tf.pow(2 * se, 2))

            rxx = (tf.pow(R, 2) * (bpx * bpxy - apx * apxy) + tf.pow(sr, 2) * (2 * bpx * bpxy - apx * apxy)) * (tf.exp(tf.negative(2 * tf.pow(sa, 2))) * tf.exp(tf.negative(2 * tf.pow(se, 2))))

            ryy = (tf.pow(R, 2) * (bpy * bpxy - apy * apxy) + tf.pow(sr, 2) * (2 * bpy * bpxy - apy * apxy)) * (tf.exp(tf.negative(2 * tf.pow(sa, 2))) * tf.exp(tf.negative(2 * tf.pow(se, 2))))

            rzz = (tf.pow(R, 2) * (bpz - apz) + tf.pow(sr, 2) * (2 * bpz - apz)) * tf.exp(tf.negative(2 * tf.pow(se, 2)))

            rxy = (tf.pow(R, 2) * (bpxy - apxy * tf.exp(tf.pow(sa, 2))) + tf.pow(sr, 2) * (2 * bpxy - apx * tf.pow(sa, 2))) * tf.sin(A) * tf.cos(A) * tf.exp(tf.negative(4 * tf.pow(sa, 2))) * tf.exp(tf.negative(2 * tf.pow(se, 2)))

            rxz = (tf.pow(R, 2) * (tf.ones_like(R) - tf.exp(tf.pow(se, 2))) + tf.pow(sr, 2) * (tf.ones_like(R) * 2 - tf.exp(tf.pow(se, 2)))) * tf.cos(A) * tf.cos(E) * tf.cos(E) * tf.exp(tf.negative(tf.pow(sa, 2))) * tf.exp(
                tf.negative(4 * tf.pow(se, 2)))

            ryz = (tf.pow(R, 2) * (tf.ones_like(R) - tf.exp(tf.pow(se, 2))) + tf.pow(sr, 2) * (tf.ones_like(R) * 2 - tf.exp(tf.pow(se, 2)))) * tf.sin(A) * tf.sin(E) * tf.cos(E) * tf.exp(tf.negative(tf.pow(sa, 2))) * tf.exp(
                tf.negative(4 * tf.pow(se, 2)))

            east = (R * tf.sin(A) * tf.cos(E)) * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
            north = (R * tf.cos(E) * tf.cos(A)) * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
            up = (R * tf.sin(E)) * ((tf.exp(tf.negative(tf.pow(se, 2) / 2))))

            cosPhi = tf.cos(lat)
            sinPhi = tf.sin(lat)
            cosLambda = tf.cos(lon)
            sinLambda = tf.sin(lon)

            tv = cosPhi * up - sinPhi * north
            wv = sinPhi * up + cosPhi * north
            uv = cosLambda * tv - sinLambda * east
            vv = sinLambda * tv + cosLambda * east

            meas_uvw = tf.concat([uv, vv, wv], axis=1)

            # sr = tf.ones_like(rm[:, 0, tf.newaxis]) * 1
            # sa = tf.ones_like(sr) * 1e-6
            # se = tf.ones_like(sa) * 1e-6

            # lb = tf.exp(-tf.pow(sa, 2))
            # lb2 = tf.exp(-2*tf.pow(sa, 2))
            # le = tf.exp(-tf.pow(se, 2))
            # le2 = tf.exp(-2*tf.pow(se, 2))

            # rxx = (-tf.pow(lb, 2)*tf.pow(le, 2)*tf.pow(R, 2)*tf.pow(tf.cos(A), 2)*tf.pow(tf.cos(E), 2)) + 0.25*(tf.pow(R, 2) + tf.pow(sr, 2))
            # rxx = ((tf.pow(sr, 2)*tf.pow(tf.cos(tf.pow(E, 2)), 2)) + tf.pow(R, 2)) * (tf.pow(se, 2)*tf.sin(tf.pow(E, 2))*tf.cos(tf.pow(A, 2)) + tf.pow(sa, 2)*tf.cos(tf.pow(E, 2))*tf.sin(tf.pow(A, 2)))
            #
            # ryy = ((tf.pow(sr, 2)*tf.cos(tf.pow(E, 2))*tf.sin(tf.pow(A, 2))) + tf.pow(R, 2)) * (tf.pow(se, 2)*tf.sin(tf.pow(E, 2))*tf.sin(tf.pow(A, 2)) + tf.pow(sa, 2)*tf.cos(tf.pow(E, 2))*tf.cos(tf.pow(A, 2)))
            #
            # rzz = (tf.pow(sr, 2)*tf.sin(tf.pow(E, 2))) + tf.pow(R, 2)*tf.pow(se, 2)*tf.cos(tf.pow(E, 2))
            #
            # rxy = (1/2) * (tf.pow(sr, 2)*tf.cos(tf.pow(E, 2))*tf.sin(2*A) + tf.pow(R, 2)*(tf.pow(se, 2)*tf.sin(tf.pow(E, 2))*tf.sin(2*A) - tf.pow(sa, 2)*tf.cos(tf.pow(E, 2))*tf.sin(2*A)))
            #
            # rxz = (1/2) * (tf.pow(sr, 2)*tf.sin(2*E)*tf.cos(A) - tf.pow(R, 2)*tf.pow(se, 2)*tf.sin(2*E)*tf.cos(A))
            #
            # ryz = (1/2) * (tf.pow(sr, 2)*tf.sin(2*E)*tf.sin(A) - tf.pow(R, 2)*tf.pow(se, 2)*tf.sin(2*E)*tf.sin(A))
            #
            # rzer = tf.zeros_like(rxx[:, tf.newaxis, tf.newaxis])

            self.rd1 = tf.concat([tf.concat([rxx[:, tf.newaxis, tf.newaxis], rxy[:, tf.newaxis, tf.newaxis], rxz[:, tf.newaxis, tf.newaxis]], axis=2),
                                  tf.concat([rxy[:, tf.newaxis, tf.newaxis], ryy[:, tf.newaxis, tf.newaxis], ryz[:, tf.newaxis, tf.newaxis]], axis=2),
                                  tf.concat([rxz[:, tf.newaxis, tf.newaxis], ryz[:, tf.newaxis, tf.newaxis], rzz[:, tf.newaxis, tf.newaxis]], axis=2)], axis=1)

            self.rd1 = tf.squeeze(self.rd1, -1)

            # self.rd1 = self.rd
            qm = FCL(q_part, 3, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='qpart', reuse=tf.AUTO_REUSE) * 25
            # self.rd = FCL(rm[:, 3:], 3, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='rd', reuse=tf.AUTO_REUSE) * 50 + tf.ones_like(rm[:, :3]) * 1e-6
            self.rd = tril_with_diag_softplus_and_shift(rm[:, 3:], diag_shift=0.00001, name='3/state')
            rdist = tfd.MultivariateNormalTriL(loc=None, scale_tril=self.rd)
            # rdist = tfd.MultivariateNormalDiag(loc=None, scale_diag=self.rd)
            # sj = self.om[:, :, 0] * 250

            # sjx = FCL(qm[:, -3, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='sjx', reuse=tf.AUTO_REUSE) * 100
            sjx = qm[:, -3, tf.newaxis] + self.om[:, :, 0] * 0.1

            # sjx = tf.clip_by_value(sjx, clip_value_min=1e-3, clip_value_max=1.)

            # sjy = FCL(qm[:, -2, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='sjy', reuse=tf.AUTO_REUSE) * 100
            sjy = qm[:, -2, tf.newaxis] + self.om[:, :, 0] * 0.1

            # sjz = FCL(qm[:, -1, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='sjz', reuse=tf.AUTO_REUSE) * 100
            sjz = qm[:, -1, tf.newaxis] + self.om[:, :, 0] * 0.1

            self.Qt, self.At, self.Bt, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                                                  dimension=int(self.num_state / 3),
                                                  sjix=self.om[:, :, 0] * sjx ** 2,
                                                  sjiy=self.om[:, :, 0] * sjy ** 2,
                                                  sjiz=self.om[:, :, 0] * sjz ** 2,
                                                  aji=self.om[:, :, 0] * 1.0)

            qcholr = tf.cholesky(tf.cast(cov_est0, self.vdtype))
            self.Rt = rdist.covariance()

            # self.Rt = tf.cast(self.I_3 * 5, tf.float64)

            Am = tf.expand_dims(self.c, axis=2) * qcholr
            Y = tf.tile(tf.expand_dims(pstate_est, axis=2), [1, 1, self.num_state])
            X = tf.concat([tf.expand_dims(pstate_est, axis=2), Y + Am, Y - Am], axis=2)
            X = tf.transpose(X, [0, 2, 1])

            x1, X1, P1, X2 = ut_state_batch(X, self.Wm, self.Wc, self.Qt, self.num_state, self.batch_size, self.At, dt)
            z1, Z1, P2, Z2 = ut_meas(X1, self.Wm, self.Wc, self.Rt, self.meas_mat, self.batch_size)

            P12 = tf.matmul(tf.matmul(X2, tf.matrix_diag(self.Wc)), Z2, transpose_b=True)

            gain = tf.matmul(P12, tf.matrix_inverse(P2))
            pos_res2 = meas_uvw[:, :, tf.newaxis] - tf.matmul(self.meas_mat, x1)
            # pos_res2 = tf.where(cur_meas_temp[:, :, tf.newaxis] == 0, tf.zeros_like(pos_res2), pos_res2)
            x = x1 + tf.matmul(gain, pos_res2)

            cov_est_t0 = P1 - tf.matmul(gain, P12, transpose_b=True)
            cov_est_t = (cov_est_t0 + tf.transpose(cov_est_t0, [0, 2, 1])) / 2

            fl.append(x[:, :, 0])
            cov_list.append(cov_est_t)
            rolist.append(self.Rt)
            qolist.append(self.Qt)
            aolist.append(self.At)
            meas_list.append(meas_uvw)

            self.Atl[q] = self.At
            self.Pl[q] = self.Qt
            self.cl.append(self.meas_mat)

        final_state = tf.stack(fl, axis=1)
        final_cov = tf.stack(cov_list, axis=1)

        self.mvn = tfd.MultivariateNormalTriL(loc=final_state, scale_tril=tf.cholesky(final_cov))

        self.ro_list = tf.stack(rolist, axis=1)
        self.qo_list = tf.stack(qolist, axis=1)
        self.ao_list = tf.stack(aolist, axis=1)
        self.new_meas = tf.stack(meas_list, axis=1)
        print('Completed UKF')

        return final_state, final_cov

    # def smooth(self, final_state, final_cov):
    #     j = [None] * self.max_seq
    #
    #     # all_states = tf.concat([self.prev_state2[:, tf.newaxis, :], final_state], axis=1)
    #     # all_covs = tf.concat([tf.reshape(self.prev_covariance, [self.batch_size, 1, self.num_state, self.num_state]), final_cov], axis=1)
    #
    #     xtemp = copy.copy(tf.unstack(tf.expand_dims(final_state, 3), axis=1))
    #     Ptemp = copy.copy(tf.unstack(final_cov, axis=1))
    #
    #     for q in range(self.max_seq - 2, -1, -1):
    #         if q >= 0:
    #             P_pred = tf.matmul(tf.matmul(self.Atl[q], Ptemp[q]), self.Atl[q], transpose_b=True) + self.Pl[q]
    #             j[q] = tf.matmul(tf.matmul(Ptemp[q], self.Atl[q], transpose_b=True), tf.matrix_inverse(P_pred))
    #             xtemp[q] += tf.matmul(j[q], xtemp[q + 1] - tf.matmul(self.Atl[q], xtemp[q]))
    #             Ptemp[q] += tf.matmul(tf.matmul(j[q], Ptemp[q + 1] - P_pred), j[q], transpose_b=True)
    #
    #     # self.final_state_update = tf.squeeze(tf.stack(xtemp, axis=1), -1)[:, :-1, :]
    #     # self.final_state = tf.squeeze(xtemp[-1], -1)
    #
    #     final_state_smooth = tf.squeeze(tf.stack(xtemp, axis=1), -1)
    #     final_cov_smooth = tf.stack(Ptemp, axis=1)
    #
    #     self.mvn_smooth = tfd.MultivariateNormalTriL(loc=final_state_smooth, scale_tril=tf.cholesky(final_cov_smooth))
    #
    #     return final_state_smooth, final_cov

    def alpha(self, current_time, int_time, dt, pstate, meas_rae, diag_cov_est0, LLA, state1=None, state2=None):

        lat = LLA[:, 0, tf.newaxis]
        lon = LLA[:, 1, tf.newaxis]
        alt = LLA[:, 2, tf.newaxis]

        R = meas_rae[:, 0, tf.newaxis]
        A = meas_rae[:, 1, tf.newaxis]
        E = meas_rae[:, 2, tf.newaxis]

        temp_input = tf.concat([current_time, pstate, meas_rae, diag_cov_est0], axis=1)

        # temp_input = tfc.layers.batch_norm(temp_input, updates_collections=None, is_training=self.is_training)
        # temp_input = normalize(temp_input, epsilon=1e-15, scope='layernorm', reuse=tf.AUTO_REUSE, dtype=self.vdtype)

        rnn_inp03 = FCL(temp_input, 48, activation_fn=tf.nn.elu, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='input/state', reuse=tf.AUTO_REUSE)

        # rnn_inp03, _ = self.filterh(temp_input, y_width=self.F_hidden * 2, name='init', dtype=self.vdtype)

        if self.state_type == 'PLSTM':
            inp1 = (int_time, rnn_inp03[:, :24])
            inp2 = (int_time, rnn_inp03[:, 24:])
        else:
            inp1 = rnn_inp03[:, :24]
            inp2 = rnn_inp03[:, 24:]

        with tf.variable_scope('Source_Track_Forward/state', reuse=tf.AUTO_REUSE):
            # dynamic_out = tf.nn.dynamic_rnn(self.source_fwf, rnn_inp03, seq_length=self.seqlen, initial_state=self.state_fw_in_state, dtype=self.vdtype)
            (out1, state1) = self.source_fwf(inp1, state=state1)

        with tf.variable_scope('Source_Track_Forward2/state', reuse=tf.AUTO_REUSE):
            # dynamic_out = tf.nn.dynamic_rnn(self.source_fwf, rnn_inp03, seq_length=self.seqlen, initial_state=self.state_fw_in_state, dtype=self.vdtype)
            (out2, state2) = self.source_fwf2(inp2, state=state2)

            # r_part = tf.concat(state1, axis=1)
            # q_part = tf.concat(state2, axis=1)

        rm0 = FCL(out1, 8, activation_fn=None, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='r1/state', reuse=tf.AUTO_REUSE)
        rm = FCL(rm0, 8, activation_fn=None, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='r2/state', reuse=tf.AUTO_REUSE)

        qm0 = FCL(out2, 3, activation_fn=None, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='q1/state', reuse=tf.AUTO_REUSE)
        qm = FCL(qm0, 3, activation_fn=tf.nn.softplus, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='q2/state', reuse=tf.AUTO_REUSE) * 50

        # sr = FCL(rm[:, 0, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='sr', reuse=tf.AUTO_REUSE) * 10
        sa = FCL(rm[:, 1, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='sa', reuse=tf.AUTO_REUSE) * 1e-3
        se = FCL(rm[:, 2, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='sz', reuse=tf.AUTO_REUSE) * 1e-3

        east = (R * tf.sin(A) * tf.cos(E)) * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        north = (R * tf.cos(E) * tf.cos(A))  * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        up = (R * tf.sin(E)) * ((tf.exp(tf.negative(tf.pow(se, 2) / 2))))

        cosPhi = tf.cos(lat)
        sinPhi = tf.sin(lat)
        cosLambda = tf.cos(lon)
        sinLambda = tf.sin(lon)

        tv = cosPhi * up - sinPhi * north
        wv = sinPhi * up + cosPhi * north
        uv = cosLambda * tv - sinLambda * east
        vv = sinLambda * tv + cosLambda * east

        meas_uvw = tf.concat([uv, vv, wv], axis=1)

        # qm = FCL(out2, 3, activation_fn=tf.nn.softplus, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='qpart', reuse=tf.AUTO_REUSE)
        # self.rd = FCL(rm[:, 3:], 3, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='rd', reuse=tf.AUTO_REUSE) * 50 + tf.ones_like(rm[:, :3]) * 1e-6
        rd = tril_with_diag_softplus_and_shift(rm[:, -6:], diag_shift=0.1, name='r3/state')
        rdist = tfd.MultivariateNormalTriL(loc=None, scale_tril=rd)

        # sjx = FCL(qm[:, -3, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='sjx', reuse=tf.AUTO_REUSE) * 100
        sjx = qm[:, -3, tf.newaxis] + self.om[:, :, 0] * 1

        # sjy = FCL(qm[:, -2, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='sjy', reuse=tf.AUTO_REUSE) * 100
        sjy = qm[:, -2, tf.newaxis] + self.om[:, :, 0] * 1

        # sjz = FCL(qm[:, -1, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, weights_initializer=tfc.layers.variance_scaling_initializer(), scope='sjz', reuse=tf.AUTO_REUSE) * 100
        sjz = qm[:, -1, tf.newaxis] + self.om[:, :, 0] * 1

        Qt, At, Bt, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                                              dimension=int(self.num_state / 3),
                                              sjix=self.om[:, :, 0] * sjx ** 2,
                                              sjiy=self.om[:, :, 0] * sjy ** 2,
                                              sjiz=self.om[:, :, 0] * sjz ** 2,
                                              aji=self.om[:, :, 0] * 1.0)

        Rt = rdist.covariance()

        return meas_uvw, Qt, At, Rt, state1, state2

    def forward_step_fn(self, params, inputs):
        """
        Forward step over a batch, to be used in tf.scan
        :param params:
        :param inputs: (batch_size, variable dimensions)
        :return:
        """
        # mu_pred, Sigma_pred, _, _, alpha, u, state, buffer, _, _, _ = params

        # inputs = tf.concat([meas_time, meas_rae, sensor_lla], axis=2)
        current_time = inputs[:, 0, tf.newaxis]
        prev_time = inputs[:, 1, tf.newaxis]
        int_time = inputs[:, 2, tf.newaxis]
        meas_rae = inputs[:, 3:6]
        LLA = inputs[:, -3:]

        mu_pred, Sigma_pred, _, _, meas_uvw, state1, state2, Q, R, A = params

        dt = current_time - prev_time
        dt = tf.where(dt <= 1 / 100, tf.ones_like(dt) * 1 / 25, dt)

        meas_uvw, Qt, At, Rt, state1, state2 = self.alpha(current_time, int_time, dt, mu_pred, meas_rae, tf.matrix_diag_part(Sigma_pred), LLA, state1=state1, state2=state2)  # (bs, k)

        # Am = tf.expand_dims(self.c, axis=2) * tf.cholesky(tf.cast(Sigma_pred, self.vdtype))
        # Y = tf.tile(tf.expand_dims(mu_pred, axis=2), [1, 1, self.num_state])
        # X = tf.concat([tf.expand_dims(mu_pred, axis=2), Y + Am, Y - Am], axis=2)
        # X = tf.transpose(X, [0, 2, 1])

        pos_res2 = meas_uvw[:, :, tf.newaxis] - tf.matmul(self.meas_mat, mu_pred[:, :, tf.newaxis])
        pos_res2 = tf.where(meas_uvw[:, :, tf.newaxis] == 0, tf.zeros_like(pos_res2), pos_res2)
        # x1, X1, P1, X2 = ut_state_batch_no_prop(X, self.Wm, self.Wc, Qt, self.num_state, self.batch_size, At, dt)
        # z1, Z1, P2, Z2 = ut_meas(X1, self.Wm, self.Wc, Rt, self.meas_mat, self.batch_size)

        # P12 = tf.matmul(tf.matmul(X2, tf.matrix_diag(self.Wc)), Z2, transpose_b=True)
        S = tf.matmul(tf.matmul(self.meas_mat, Sigma_pred), self.meas_mat, transpose_b=True) + Rt

        S_inv = tf.matrix_inverse(S)
        gain = tf.matmul(tf.matmul(Sigma_pred, self.meas_mat, transpose_b=True), S_inv)

        # gain = tf.matmul(P12, tf.matrix_inverse(P2))

        mu_t = mu_pred + tf.matmul(gain, pos_res2)
        mu_t = mu_t[:, :, 0]

        I_KC = self.I_12 - tf.matmul(gain, self.meas_mat)  # (bs, dim_z, dim_z)
        Sigma_t = tf.matmul(tf.matmul(I_KC, Sigma_pred), I_KC, transpose_b=True) + self._sast(Rt, gain)
        Sigma_t = (Sigma_t + tf.transpose(Sigma_t, [0, 2, 1])) / 2

        # Am = tf.expand_dims(self.c, axis=2) * tf.cholesky(tf.cast(Sigma_t, self.vdtype))
        # Y = tf.tile(tf.expand_dims(mu_t, axis=2), [1, 1, self.num_state])
        # X = tf.concat([tf.expand_dims(mu_t, axis=2), Y + Am, Y - Am], axis=2)
        # X = tf.transpose(X, [0, 2, 1])
        #
        # mu_pred, _, Sigma_pred, _ = ut_state_batch(X, self.Wm, self.Wc, Qt, self.num_state, self.batch_size, At, dt)
        #
        # mu_pred = mu_pred[:, :, 0]

        return mu_pred, Sigma_pred, mu_t, Sigma_t, meas_uvw, state1, state2, Qt, Rt, At

    def backward_step_fn(self, params, inputs):
        """
        Backwards step over a batch, to be used in tf.scan
        :param params:
        :param inputs: (batch_size, variable dimensions)
        :return:
        """
        mu_back, Sigma_back = params
        mu_pred_tp1, Sigma_pred_tp1, mu_filt_t, Sigma_filt_t, A = inputs

        # J_t = tf.matmul(tf.reshape(tf.transpose(tf.matrix_inverse(Sigma_pred_tp1), [0, 2, 1]), [-1, self.dim_z]),
        #                 self.A)
        # J_t = tf.transpose(tf.reshape(J_t, [-1, self.dim_z, self.dim_z]), [0, 2, 1])
        J_t = tf.matmul(tf.transpose(A, [0, 2, 1]), tf.matrix_inverse(Sigma_pred_tp1))
        J_t = tf.matmul(Sigma_filt_t, J_t)

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

        # To make sure we are not accidentally using the real outputs in the steps with missing values, set them to 0.
        # y_masked = tf.multiply(tf.expand_dims(self.mask, 2), self.y)

        # inputs = tf.concat([y_masked, self.u, tf.expand_dims(self.mask, 2)], axis=2)

        all_time = tf.concat([self.prev_time[:, tf.newaxis], tf.stack(self.current_timei, axis=1)], axis=1)
        # meas_rae = tf.concat([self.prev_measurement[:, tf.newaxis], tf.stack(self.measurement, axis=1)], axis=1)
        meas_rae = tf.stack(self.measurement, axis=1)
        # meas_rae = meas_rae[:, :-1, :]
        meas_time = all_time[:, 1:, :]
        prev_time = all_time[:, :-1, :]

        dt0 = meas_time[:, 0, :] - prev_time[:, 0, :]

        int_time = self.int_time

        sensor_lla = tf.expand_dims(self.sensor_lla, axis=1)
        sensor_lla = tf.tile(sensor_lla, [1, meas_rae.shape[1], 1])

        inputs = tf.concat([meas_time, int_time[:, :, tf.newaxis], prev_time, meas_rae, sensor_lla], axis=2)

        cd_input = tf.matrix_diag_part(tf.reshape(self.prev_covariance, [self.batch_size, self.num_state, self.num_state]))

        prev_meas_uvw, Qt, At, Rt, _, _ = self.alpha(prev_time[:, 0, :], int_time[:, 0], dt0, self.prev_state2, self.prev_measurement, cd_input,
                                                sensor_lla[:, 0, :], state1=self.state_fw_in_state, state2=self.state_fw_in_state2)

        state1 = self.state_fw_in_state
        state2 = self.state_fw_in_state2

        init_Q = tf.ones([self.batch_size, 12, 12], self.vdtype)
        init_R = tf.ones([self.batch_size, 3, 3], self.vdtype)
        init_A = tf.ones([self.batch_size, 12, 12], self.vdtype)
        # meas_uvw = tf.zeros([self.batch_size, 3], self.vdtype)

        # init_Q = Qt
        # init_R = Rt
        # init_A = At

        self.mu = self.state_input
        self.Sigma = tf.reshape(self.P_inp, [self.batch_size, self.num_state, self.num_state])

        forward_states = tf.scan(self.forward_step_fn, tf.transpose(inputs, [1, 0, 2]),
                                 initializer=(self.mu, self.Sigma, self.mu, self.Sigma, prev_meas_uvw, state1, state2,
                                              init_Q, init_R, init_A),
                                 parallel_iterations=1, name='forward')
        return forward_states

    def compute_backwards(self, forward_states):
        mu_pred, Sigma_pred, mu_filt, Sigma_filt, meas_uvw, state1, state2, Q, R, A = forward_states

        mu_pred = tf.expand_dims(mu_pred, 3)
        mu_filt = tf.expand_dims(mu_filt, 3)
        # The tf.scan below that does the smoothing is initialized with the filtering distribution at time T.
        # following the derivarion in Murphy's book, we then need to discard the last time step of the predictive
        # (that will then have t=2,..T) and filtering distribution (t=1:T-1)
        states_scan = [mu_pred[:-1, :, :, :],
                       Sigma_pred[:-1, :, :, :],
                       mu_filt[:-1, :, :, :],
                       Sigma_filt[:-1, :, :, :],
                       A[:-1]]

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

        return backward_states, Q, R, A, meas_uvw, state1, state2

    def filter(self):
        mu_pred, Sigma_pred, mu_filt, Sigma_filt, meas_uvw, state1, state2, Q, R, A = forward_states = \
            self.compute_forwards()

        state1c = tf.transpose(state1[0], [1, 0, 2])
        state1h = tf.transpose(state1[1], [1, 0, 2])
        state1_out = tf.contrib.rnn.LSTMStateTuple(state1c[:, -2, :], state1h[:, -2, :])

        state2c = tf.transpose(state2[0], [1, 0, 2])
        state2h = tf.transpose(state2[1], [1, 0, 2])
        state2_out = tf.contrib.rnn.LSTMStateTuple(state2c[:, -2, :], state2h[:, -2, :])

        forward_states = [mu_filt, Sigma_filt]
        # Swap batch dimension and time dimension
        forward_states[0] = tf.transpose(forward_states[0], [1, 0, 2])
        forward_states[1] = tf.transpose(forward_states[1], [1, 0, 2, 3])
        return tuple(forward_states), tf.transpose(A, [1, 0, 2, 3]), tf.transpose(Q, [1, 0, 2, 3]), \
               tf.transpose(R, [1, 0, 2, 3]), tf.transpose(meas_uvw, [1, 0, 2]), state1_out, state2_out

    def smooth(self):
        backward_states, Q, R, A, meas_uvw, state1, state2 = self.compute_backwards(self.compute_forwards())

        state1c = tf.transpose(state1[0], [1, 0, 2])
        state1h = tf.transpose(state1[1], [1, 0, 2])
        state1_out = tf.contrib.rnn.LSTMStateTuple(state1c[:, -1, :], state1h[:, -1, :])

        state2c = tf.transpose(state2[0], [1, 0, 2])
        state2h = tf.transpose(state2[1], [1, 0, 2])
        state2_out = tf.contrib.rnn.LSTMStateTuple(state2c[:, -1, :], state2h[:, -1, :])

        # Swap batch dimension and time dimension
        backward_states[0] = tf.transpose(backward_states[0], [1, 0, 2])
        backward_states[1] = tf.transpose(backward_states[1], [1, 0, 2, 3])
        return tuple(backward_states), tf.transpose(A, [1, 0, 2, 3]), tf.transpose(Q, [1, 0, 2, 3]), \
               tf.transpose(R, [1, 0, 2, 3]), tf.transpose(meas_uvw, [1, 0, 2]), state1_out, state2_out

    def _sast(self, a, s):
        _, dim_1, dim_2 = s.get_shape().as_list()
        sast = tf.matmul(tf.reshape(s, [-1, dim_2]), a, transpose_b=True)
        sast = tf.transpose(tf.reshape(sast, [-1, dim_1, dim_2]), [0, 2, 1])
        sast = tf.matmul(s, sast)
        return sast

    def sample_generative_tf(self, backward_states, n_steps, deterministic=True, init_fixed_steps=1):
        """
        Get a sample from the generative model
        """
        # Get states from the Kalman filter to get the initial state
        mu_z, sigma_z = backward_states
        # z = tf.contrib.distributions.MultivariateNormalTriL(mu_z[seq_idx, 0], sigma_z[seq_idx, 0]).sample()

        if init_fixed_steps > 0:
            z = mu_z[:, 0]
            z = tf.expand_dims(z, 2)
        else:
            raise("Prior sampling from z not implemented")

        if not deterministic:
            # Pre-compute samples of noise
            noise_trans = tfd.MultivariateNormalTriL(tf.zeros((self.dim_z,)), tf.cholesky(self.Q))
            epsilon = noise_trans.sample((z.get_shape()[0].value, n_steps))
            noise_emiss = tfd.MultivariateNormalTriL(tf.zeros((self.dim_y,)), tf.cholesky(self.R))
            delta = noise_emiss.sample((z.get_shape()[0].value, n_steps))
        else:
            epsilon = tf.zeros((z.get_shape()[0], n_steps, self.dim_z))
            delta = tf.zeros((z.get_shape()[0], n_steps, self.dim_y))

        y_prev = tf.expand_dims(self.y_0, 0)  # (1, dim_y)
        y_prev = tf.tile(y_prev, (tf.shape(self.mu)[0], 1))  # (bs, dim_y)
        alpha, state, u, buffer = self.alpha(y_prev, self.state, self.u[:, 0], reuse=True, init_buffer=True)

        y_samples = list()
        z_samples = list()
        alpha_samples = list()
        for n in range(n_steps):

            # Mixture of C
            C = tf.matmul(alpha, tf.reshape(self.C, [-1, self.dim_y*self.dim_z]))  # (bs, k) x (k, dim_y*dim_z)
            C = tf.reshape(C, [-1, self.dim_y, self.dim_z])  # (bs, dim_y, dim_z)

            # Output for the current time step
            y = tf.matmul(C, z) + tf.expand_dims(delta[:, n], 2)
            y = tf.squeeze(y, 2)

            # Store current state and output at time t
            z_samples.append(tf.squeeze(z, 2))
            y_samples.append(y)

            # Compute the mixture of A
            alpha, state, u, buffer = self.alpha(y, state, self.u[:, n], buffer, reuse=True)
            alpha_samples.append(alpha)
            A = tf.matmul(alpha, tf.reshape(self.A, [-1, self.dim_z * self.dim_z]))
            A = tf.reshape(A, [-1, self.dim_z, self.dim_z])

            # Mixture of B
            B = tf.matmul(alpha, tf.reshape(self.B, [-1, self.dim_z*self.dim_u]))  # (bs, k) x (k, dim_y*dim_z)
            B = tf.reshape(B, [-1, self.dim_z, self.dim_u])  # (bs, dim_y, dim_z)

            # Get new state z_{t+1}
            # z = tf.matmul(A, z) + tf.matmul(B,  tf.expand_dims(self.u[:, n],2)) + tf.expand_dims(epsilon[:, n], 2)
            if (n + 1) >= init_fixed_steps:
                z = tf.matmul(A, z) + tf.matmul(B,  tf.expand_dims(u, 2)) + tf.expand_dims(epsilon[:, n], 2)
            else:
                z = mu_z[:, n+1]
                z = tf.expand_dims(z, 2)

        return tf.stack(y_samples, 1), tf.stack(z_samples, 1), tf.stack(alpha_samples, 1)

    def get_elbo(self, backward_states):

        mu_smooth = backward_states[0]
        Sigma_smooth = backward_states[1]

        all_truth = tf.stack(self.truth_state, axis=1)
        # Sample from smoothing distribution
        # jitter = 1e-2 * tf.eye(tf.shape(Sigma_smooth)[-1], batch_shape=tf.shape(Sigma_smooth)[0:-2])
        # mvn_smooth = tf.contrib.distributions.MultivariateNormalTriL(mu_smooth, Sigma_smooth + jitter)
        mvn_smooth = tfd.MultivariateNormalTriL(mu_smooth, tf.cholesky(Sigma_smooth))
        mvn_smooth_error = tfd.MultivariateNormalTriL(None, tf.cholesky(Sigma_smooth))
        z_smooth = mvn_smooth.sample()

        ## Transition distribution \prod_{t=2}^T p(z_t|z_{t-1}, u_{t})
        # We need to evaluate N(z_t; Az_tm1 + Bu_t, Q), where Q is the same for all the elements
        # z_tm1 = tf.reshape(z_smooth[:, :-1, :], [-1, self.dim_z])
        # Az_tm1 = tf.transpose(tf.matmul(self.A, tf.transpose(z_tm1)))
        Az_tm1 = tf.matmul(self.ao_list[:, :-1], tf.expand_dims(all_truth[:, :-1], 3))

        # Remove the first input as our prior over z_1 does not depend on it
        # u_t_resh = tf.reshape(u, [-1, self.dim_u])
        # Bu_t = tf.transpose(tf.matmul(self.B, tf.transpose(u_t_resh)))
        # Bu_t = tf.reshape(tf.matmul(B[:, :-1], tf.expand_dims(self.u[:, 1:], 3)), [-1, self.dim_z])
        mu_transition = Az_tm1[:, :, :, 0]
        z_t_transition = all_truth[:, 1:, :]

        # MultivariateNormalTriL supports broadcasting only for the inputs, not for the covariance
        # To exploit this we then write N(z_t; Az_tm1 + Bu_t, Q) as N(z_t - Az_tm1 - Bu_t; 0, Q)
        trans_centered = z_t_transition - mu_transition
        mvn_transition = tfd.MultivariateNormalTriL(tf.zeros(self.num_state, dtype=self.vdtype), tf.cholesky(self.qo_list[:, :-1]))
        log_prob_transition = mvn_transition.log_prob(trans_centered) * self.seqweightin[:, :-1]

        ## Emission distribution \prod_{t=1}^T p(y_t|z_t)
        # We need to evaluate N(y_t; Cz_t, R). We write it as N(y_t - Cz_t; 0, R)
        # z_t_emission = tf.reshape(z_smooth, [-1, self.dim_z])
        # Cz_t = tf.transpose(tf.matmul(self.C, tf.transpose(z_t_emission)))
        # Cz_t = tf.matmul(self.meas_mat, tf.expand_dims(z_smooth, 3))
        # Cz_t = tf.concat([z_smooth[:, :, 0, tf.newaxis], z_smooth[:, :, 4, tf.newaxis], z_smooth[:, :, 8, tf.newaxis]], axis=2)
        Cz_t = tf.concat([all_truth[:, :, 0, tf.newaxis], all_truth[:, :, 4, tf.newaxis], all_truth[:, :, 8, tf.newaxis]], axis=2)
        y_t_resh = self.new_meas
        emiss_centered = y_t_resh - Cz_t
        mvn_emission = tfd.MultivariateNormalTriL(tf.zeros(self.num_meas, dtype=self.vdtype), tf.cholesky(self.ro_list))
        # mask_flat = tf.reshape(self.mask, (-1, ))
        log_prob_emission = mvn_emission.log_prob(emiss_centered)
        log_prob_emission = log_prob_emission * self.seqweightin

        # ## Distribution of the initial state p(z_1|z_0)
        # z_0 = z_smooth[:, 0, :]
        # mvn_0 = tfd.MultivariateNormalTriL(self.mu, tf.cholesky(self.Sigma))
        # log_prob_0 = mvn_0.log_prob(z_0)

        # mvn = tfd.MultivariateNormalTriL(tf.stack(self.truth_state, axis=1), tf.cholesky(Sigma_smooth))
        error_state = all_truth - mu_smooth
        log_prob_0 = mvn_smooth_error.log_prob(error_state)

        # Entropy log(\prod_{t=1}^T p(z_t|y_{1:T}, u_{1:T}))
        entropy = -mvn_smooth.log_prob(z_smooth) * self.seqweightin

        # Compute terms of the lower bound
        # We compute the log-likelihood *per frame*
        num_el = tf.reduce_sum(self.seqweightin)
        log_probs = [tf.truediv(tf.reduce_sum(log_prob_transition), (num_el*12*1e12)),
                     tf.truediv(tf.reduce_sum(log_prob_0), (num_el*12)),
                     tf.truediv(tf.reduce_sum(log_prob_emission), num_el*3),
                     tf.truediv(tf.reduce_sum(entropy), num_el*12)]

        # tf.truediv(tf.reduce_sum(log_prob_transition), num_el),

        self.num_el = num_el

        kf_elbo = tf.reduce_sum(log_probs)

        self.error_loss_Q = log_probs[0]
        self.rl = log_probs[2]
        self.error_loss_full = log_probs[1]
        self.entropy = log_probs[3]

        self.error_loss_pos = tf.reduce_mean(0)
        self.error_loss_vel = tf.reduce_mean(0)

        return kf_elbo, log_probs, z_smooth

    def build_loss(self, final_state, final_cov, final_state_smooth, final_cov_smooth):

        final_state_smooth = tf.cast(final_state_smooth, self.vdtype)  # / meanv[:, 0, :]
        final_state = tf.cast(final_state, self.vdtype)  # / meanv[:, 0, :]
        if self.window_mode:
            _y = tf.stack(self.truth_state, axis=1)
            # final_state_smootht = final_state_smooth[:, 1:, :]
            all_states = final_state_smooth
        else:
            _y = self.truth_state
            final_state_smooth = tf.expand_dims(final_state_smooth, axis=1)  # / self.meanv
            final_state = tf.expand_dims(final_state, axis=1)  # / self.meanv
            _y = tf.expand_dims(_y, axis=1)
            prev_truth = tf.stack(self.prev_truth, axis=1)
            _y = tf.concat([prev_truth, _y], axis=1)
            all_states = tf.concat([tf.stack(self.prev_state3, axis=1), final_state], axis=1)

            final_state_smootht = final_state_smooth

        # pos_m = tf.concat([self.meanv[:, 0, tf.newaxis], self.meanv[:, 4, tf.newaxis], self.meanv[:, 8, tf.newaxis]], axis=1)
        # pos_m = tf.squeeze(pos_m, 0)

        self.mvn = tfd.MultivariateNormalTriL(final_state, tf.cholesky(final_cov))

        loss_func = weighted_mape_tf

        z_smooth = self.mvn.sample()

        # self.final_state2 = z_smooth

        total_weight = tf.cast(self.seqweightin, self.vdtype)

        print('Building Loss')
        tot = tf.cast(self.max_seq, self.vdtype)
        self.den = self.max_seq

        state_loss_pos100 = 0
        state_loss_pos200 = 0
        state_loss_pos300 = 0
        state_loss_vel100 = 0
        state_loss_vel200 = 0
        state_loss_vel300 = 0
        state_loss_acc100 = 0
        state_loss_acc200 = 0
        state_loss_acc300 = 0
        state_loss_j100 = 0
        state_loss_j200 = 0
        state_loss_j300 = 0

        state_loss_pos100 += loss_func(_y[:, :, 0], all_states[:, :, 0], total_weight, tot)
        state_loss_pos200 += loss_func(_y[:, :, 4], all_states[:, :, 4], total_weight, tot)
        state_loss_pos300 += loss_func(_y[:, :, 8], all_states[:, :, 8], total_weight, tot)
        state_loss_vel100 += loss_func(_y[:, :, 1], all_states[:, :, 1], total_weight, tot) / 1000
        state_loss_vel200 += loss_func(_y[:, :, 5], all_states[:, :, 5], total_weight, tot) / 1000
        state_loss_vel300 += loss_func(_y[:, :, 9], all_states[:, :, 9], total_weight, tot) / 1000
        state_loss_acc100 += loss_func(_y[:, :, 2], all_states[:, :, 2], total_weight, tot) / 10000
        state_loss_acc200 += loss_func(_y[:, :, 6], all_states[:, :, 6], total_weight, tot) / 10000
        state_loss_acc300 += loss_func(_y[:, :, 10], all_states[:, :, 10], total_weight, tot) / 10000
        state_loss_j100 += loss_func(_y[:, :, 3], all_states[:, :, 3], total_weight, tot) / 100000
        state_loss_j200 += loss_func(_y[:, :, 7], all_states[:, :, 7], total_weight, tot) / 100000
        state_loss_j300 += loss_func(_y[:, :, 11], all_states[:, :, 11], total_weight, tot) / 100000

        state_loss_pos10 = 0
        state_loss_pos20 = 0
        state_loss_pos30 = 0
        state_loss_vel10 = 0
        state_loss_vel20 = 0
        state_loss_vel30 = 0
        state_loss_acc10 = 0
        state_loss_acc20 = 0
        state_loss_acc30 = 0
        state_loss_j10 = 0
        state_loss_j20 = 0
        state_loss_j30 = 0

        state_loss_pos10 += loss_func(_y[:, :, 0], final_state[:, :, 0], total_weight, tot)
        state_loss_pos20 += loss_func(_y[:, :, 4], final_state[:, :, 4], total_weight, tot)
        state_loss_pos30 += loss_func(_y[:, :, 8], final_state[:, :, 8], total_weight, tot)
        state_loss_vel10 += loss_func(_y[:, :, 1], final_state[:, :, 1], total_weight, tot) / 1e3
        state_loss_vel20 += loss_func(_y[:, :, 5], final_state[:, :, 5], total_weight, tot) / 1e3
        state_loss_vel30 += loss_func(_y[:, :, 9], final_state[:, :, 9], total_weight, tot) / 1e3
        state_loss_acc10 += loss_func(_y[:, :, 2], final_state[:, :, 2], total_weight, tot) / 1e5
        state_loss_acc20 += loss_func(_y[:, :, 6], final_state[:, :, 6], total_weight, tot) / 1e5
        state_loss_acc30 += loss_func(_y[:, :, 10], final_state[:, :, 10], total_weight, tot) / 1e5
        state_loss_j10 += loss_func(_y[:, :, 3], final_state[:, :, 3], total_weight, tot) / 1e6
        state_loss_j20 += loss_func(_y[:, :, 7], final_state[:, :, 7], total_weight, tot) / 1e6
        state_loss_j30 += loss_func(_y[:, :, 11], final_state[:, :, 11], total_weight, tot) / 1e6

        print('Completed Loss')

        # if self.window_mode is False:
        #     sweight = self.seqweightin[:, 0]
        # else:
        sweight = self.seqweightin

        print('Building Covariance Loss')
        truth_state = copy.copy(_y)

        if self.window_mode:
            # truth_cov_norm_pos = tf.concat([truth_cov_norm_6[:, :, 0, tf.newaxis], truth_cov_norm_6[:, :, 2, tf.newaxis], truth_cov_norm_6[:, :, 4, tf.newaxis]], axis=1)
            # truth_cov_norm_vel = tf.concat([truth_cov_norm_6[:, :, 1, tf.newaxis], truth_cov_norm_6[:, :, 3, tf.newaxis], truth_cov_norm_6[:, :, 5, tf.newaxis]], axis=1)
            truth_pos = tf.concat([truth_state[:, :, 0, tf.newaxis], truth_state[:, :, 4, tf.newaxis], truth_state[:, :, 8, tf.newaxis]], axis=2)
            truth_vel = tf.concat([truth_state[:, :, 1, tf.newaxis], truth_state[:, :, 5, tf.newaxis], truth_state[:, :, 9, tf.newaxis]], axis=2)
            cov_pos = tf.concat([tf.concat([final_cov[:, :, 0, 0, tf.newaxis, tf.newaxis], final_cov[:, :, 0, 4, tf.newaxis, tf.newaxis], final_cov[:, :, 0, 8, tf.newaxis, tf.newaxis]], axis=3),
                                 tf.concat([final_cov[:, :, 4, 0, tf.newaxis, tf.newaxis], final_cov[:, :, 4, 4, tf.newaxis, tf.newaxis], final_cov[:, :, 4, 8, tf.newaxis, tf.newaxis]], axis=3),
                                 tf.concat([final_cov[:, :, 8, 0, tf.newaxis, tf.newaxis], final_cov[:, :, 8, 4, tf.newaxis, tf.newaxis], final_cov[:, :, 8, 8, tf.newaxis, tf.newaxis]], axis=3)],
                                axis=2)

            cov_vel = tf.concat([tf.concat([final_cov[:, :, 1, 1, tf.newaxis, tf.newaxis], final_cov[:, :, 1, 5, tf.newaxis, tf.newaxis], final_cov[:, :, 1, 9, tf.newaxis, tf.newaxis]], axis=3),
                                 tf.concat([final_cov[:, :, 5, 1, tf.newaxis, tf.newaxis], final_cov[:, :, 5, 5, tf.newaxis, tf.newaxis], final_cov[:, :, 5, 9, tf.newaxis, tf.newaxis]], axis=3),
                                 tf.concat([final_cov[:, :, 9, 1, tf.newaxis, tf.newaxis], final_cov[:, :, 9, 5, tf.newaxis, tf.newaxis], final_cov[:, :, 9, 9, tf.newaxis, tf.newaxis]], axis=3)],
                                axis=2)

            # train_cov00 = tfd.MultivariateNormalFullCovariance(loc=truth_state, covariance_matrix=final_cov)
            train_cov_pos = tfd.MultivariateNormalFullCovariance(loc=truth_pos, covariance_matrix=cov_pos)
            train_cov_vel = tfd.MultivariateNormalFullCovariance(loc=truth_vel, covariance_matrix=cov_vel)

        else:
            # truth_cov_norm_6 = truth_cov_norm_6[:, -1, :]
            # truth_cov_norm_pos = tf.concat([truth_cov_norm_6[:, 0, tf.newaxis], truth_cov_norm_6[:, 2, tf.newaxis], truth_cov_norm_6[:, 4, tf.newaxis]], axis=1)
            # truth_cov_norm_vel = tf.concat([truth_cov_norm_6[:, 1, tf.newaxis], truth_cov_norm_6[:, 3, tf.newaxis], truth_cov_norm_6[:, 5, tf.newaxis]], axis=1)
            truth_pos = tf.concat([truth_state[:, :, 0, tf.newaxis], truth_state[:, :, 4, tf.newaxis], truth_state[:, :, 8, tf.newaxis]], axis=2)
            truth_vel = tf.concat([truth_state[:, :, 1, tf.newaxis], truth_state[:, :, 5, tf.newaxis], truth_state[:, :, 9, tf.newaxis]], axis=2)
            train_cov00 = tfd.MultivariateNormalFullCovariance(loc=truth_state, covariance_matrix=final_cov)

            # cov_pos = tf.concat([tf.concat([final_cov[:, 0, 0, tf.newaxis, tf.newaxis], final_cov[:, 0, 4, tf.newaxis, tf.newaxis], final_cov[:, 0, 8, tf.newaxis, tf.newaxis]], axis=2),
            #                      tf.concat([final_cov[:, 4, 0, tf.newaxis, tf.newaxis], final_cov[:, 4, 4, tf.newaxis, tf.newaxis], final_cov[:, 4, 8, tf.newaxis, tf.newaxis]], axis=2),
            #                      tf.concat([final_cov[:, 8, 0, tf.newaxis, tf.newaxis], final_cov[:, 8, 4, tf.newaxis, tf.newaxis], final_cov[:, 8, 8, tf.newaxis, tf.newaxis]], axis=2)], axis=1)
            #
            # cov_vel = tf.concat([tf.concat([final_cov[:, 1, 1, tf.newaxis, tf.newaxis], final_cov[:, 1, 5, tf.newaxis, tf.newaxis], final_cov[:, 1, 9, tf.newaxis, tf.newaxis]], axis=2),
            #                      tf.concat([final_cov[:, 5, 1, tf.newaxis, tf.newaxis], final_cov[:, 5, 5, tf.newaxis, tf.newaxis], final_cov[:, 5, 9, tf.newaxis, tf.newaxis]], axis=2),
            #                      tf.concat([final_cov[:, 9, 1, tf.newaxis, tf.newaxis], final_cov[:, 9, 5, tf.newaxis, tf.newaxis], final_cov[:, 9, 8, tf.newaxis, tf.newaxis]], axis=2)], axis=1)

            # train_cov_pos = tfd.MultivariateNormalDiag(loc=truth_pos[:, 0, :], scale_diag=self.Ql2_pos)
            # train_cov_vel = tfd.MultivariateNormalDiag(loc=truth_vel[:, 0, :], scale_diag=self.Ql2_vel)

            train_cov_pos = tfd.MultivariateNormalTriL(loc=None, scale_tril=tf.cholesky(self.kde_pos.covariance()))
            train_cov_vel = tfd.MultivariateNormalTriL(loc=None, scale_tril=tf.cholesky(self.kde_vel.covariance()))

            # train_cov_pos = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=cov_pos)
            # train_cov_vel = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=cov_vel)

        # mvn_smooth = tfd.MultivariateNormalFullCovariance(loc=final_state, covariance_matrix=final_cov)

        if self.window_mode:
            z_current = tf.squeeze(tf.matmul(self.ao_list[:, :-1], z_smooth[:, :-1][:, :, :, tf.newaxis]), -1)
            # meas_error = tf.squeeze(tf.stack(self.measurement, axis=1)[:, :, :, tf.newaxis] - tf.matmul(tf.tile(self.meas_mat[:, tf.newaxis, :, :], [1, self.max_seq, 1, 1]), _y[:, :, :, tf.newaxis]),-1)
            meas_error = self.new_meas - truth_pos
            emission_prob = tfd.MultivariateNormalTriL(loc=None, scale_tril=tf.cholesky(self.ro_list))
            # emission_prob = tfd.MultivariateNormalDiag(loc=None, scale_diag=tf.sqrt(tf.matrix_diag_part(self.ro_list)))
            rlt = emission_prob.log_prob(meas_error)
            self.rl = tf.losses.compute_weighted_loss(tf.negative(rlt), weights=sweight) / (tf.cast(self.den, self.vdtype))
        else:
            z_current = tf.squeeze(tf.matmul(self.At, self.prev_truth[-1][:, :, tf.newaxis]), -1)
            meas_error = tf.squeeze(self.measurement[:, :, tf.newaxis] - tf.matmul(self.meas_mat, tf.transpose(_y[:, -1:, :], [0, 2, 1])), -1)
            # meas_error = tf.where((meas_error < 1.0 and meas_error > 0.0), tf.sqrt(meas_error), meas_error)
            emission_prob = tfd.MultivariateNormalTriL(loc=None, scale_tril=self.rd)
            # emission_prob = tfd.MultivariateNormalDiag(loc=None, scale_diag=self.rd)
            rlt = emission_prob.log_prob(meas_error)
            self.rl = tf.losses.compute_weighted_loss(tf.negative(rlt), weights=sweight, reduction="weighted_mean") / (tf.cast(self.den, self.vdtype))

        trans_centered = z_smooth[:, 1:] - z_current
        # mvn_transition = tfd.MultivariateNormalTriL(loc=None, scale_tril=tf.cholesky(self.qo_list[:, 1:]))
        mvn_transition = tfd.MultivariateNormalDiag(loc=None, scale_diag=tf.sqrt(tf.matrix_diag_part((self.qo_list[:, 1:]))))
        log_prob_transition = mvn_transition.log_prob(trans_centered)
        self.error_loss_Q = tf.losses.compute_weighted_loss(tf.negative(log_prob_transition), weights=sweight[:, 1:]) / (tf.cast(self.den, self.vdtype))
        # self.error_loss_Q = tf.cast(tf.negative(tf.reduce_mean(0.0)), self.vdtype)

        delta_12 = final_state_smooth - truth_state
        delta2 = tf.expand_dims(delta_12, 3)

        self.error_loss_full = self.mvn.log_prob(_y)

        self.error_loss_pos = train_cov_pos.log_prob(truth_pos)
        self.error_loss_vel = train_cov_vel.log_prob(truth_vel)

        inv_cov = tf.matrix_inverse(final_cov_smooth)
        self.cov_out = self.final_cov

        # z_0 = z_smooth[:, 0, :]
        # mvn_0 = tfd.MultivariateNormalTriL(self.final_state[:, 0, :], tf.cholesky(self.final_cov[:, 0, :, :]))
        # self.error_loss_full = tf.losses.compute_weighted_loss(tf.negative(mvn_0.log_prob(z_0)), weights=sweight[:, 0])

        self.error_loss_full = tf.losses.compute_weighted_loss(tf.negative(self.error_loss_full), weights=sweight) / (tf.cast(self.den, self.vdtype))
        self.error_loss_pos = tf.losses.compute_weighted_loss(tf.negative(self.error_loss_pos), weights=sweight) / (tf.cast(self.den, self.vdtype))
        self.error_loss_vel = tf.losses.compute_weighted_loss(tf.negative(self.error_loss_vel), weights=sweight) / (tf.cast(self.den, self.vdtype))

        M1 = tf.matmul(delta2, inv_cov, transpose_a=True)
        M2 = tf.sqrt(tf.square(tf.matmul(M1, delta2)))
        MD = tf.squeeze(tf.sqrt(M2 / 12))

        MD0 = 0
        # MD0 += tf.losses.huber_loss(tf.ones_like(MD) / 12, MD, weights=sweight)
        self.MD0 = tf.cast(tf.losses.compute_weighted_loss(MD, weights=sweight), self.vdtype) / tf.cast(self.den, self.vdtype)

        # self.MD0 = tf.reduce_mean(0)

        # delta_pos = tf.reduce_mean(tf.transpose(self.kde_pos.sample((1000)), [1, 0, 2]) - tf.tile(truth_pos[:, -1, tf.newaxis, :], [1, 1000, 1]), axis=1)
        # delta_vel = tf.reduce_mean(tf.transpose(self.kde_vel.sample((1000)), [1, 0, 2]) - tf.tile(truth_vel[:, -1, tf.newaxis, :], [1, 1000, 1]), axis=1)
        #
        # # sweight2 = tf.ones_like(delta_pos) * sweight[:, tf.newaxis, tf.newaxis]
        # delta2 = tf.expand_dims(delta_pos, 2)
        # M1 = tf.matmul(delta2, tf.matrix_inverse(self.kde_pos.covariance()), transpose_a=True)
        # M2 = tf.sqrt(tf.square(tf.matmul(M1, delta2)))
        # MD = tf.squeeze(tf.sqrt(M2 / 3))
        # self.MD0 += tf.losses.huber_loss(tf.ones_like(MD) / 3, MD, weights=sweight)
        # # MD0 += tf.reduce_sum(tf.losses.compute_weighted_loss(MD, weights=sweight))
        #
        # delta2 = tf.expand_dims(delta_vel, 2)
        # M1 = tf.matmul(delta2, tf.matrix_inverse(self.kde_vel.covariance()), transpose_a=True)
        # M2 = tf.sqrt(tf.square(tf.matmul(M1, delta2)))
        # MD = tf.squeeze(tf.sqrt(M2 / 3))
        # self.MD0 += tf.losses.huber_loss(tf.ones_like(MD) / 3, MD, weights=sweight)
        # MD0 += tf.reduce_sum(tf.losses.compute_weighted_loss(MD, weights=sweight))

        # Build output covariance
        # zmat = tf.zeros_like(self.Ql4_pos)
        # self.cov_out = tf.concat([tf.concat([self.Ql4_pos, zmat], axis=2), tf.concat([zmat, self.Ql4_vel], axis=2)], axis=1)
        # trace_pos = tf.reduce_sum(tf.sqrt(tf.pow(tf.matrix_diag_part(self.Ql4_pos), 2)))  # * tf.tile(tf.cast(total_weight[:, q, tf.newaxis, tf.newaxis], self.vdtype), [1, 12, 12]))
        # trace_vel = tf.reduce_sum(tf.sqrt(tf.pow(tf.matrix_diag_part(self.Ql4_vel), 2)))  # * tf.tile(tf.cast(total_weight[:, q, tf.newaxis, tf.newaxis], self.vdtype), [1, 12, 12]))
        # TL = tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4_pos, 2)))

        entropy1 = tf.losses.compute_weighted_loss(self.mvn.log_prob(z_smooth), weights=sweight) / (tf.cast(self.den, self.vdtype))
        # entropy2 = tf.negative(self.kde_pos.log_prob(self.kde_pos.sample()))
        # entropy3 = tf.negative(self.kde_vel.log_prob(self.kde_vel.sample()))
        self.entropy = tf.reduce_mean(entropy1)
        trace = tf.reduce_mean(tf.sqrt(tf.pow(tf.matrix_diag_part(final_cov_smooth), 2)))

        print('Completed Covariance Loss')
        self.kalman_cov_loss = tf.cast(self.error_loss_full, self.vdtype)

        self.SLPf1 = state_loss_pos100 + state_loss_pos200 + state_loss_pos300
        self.SLVf1 = state_loss_vel100 + state_loss_vel200 + state_loss_vel300
        self.SLAf1 = state_loss_acc100 + state_loss_acc200 + state_loss_acc300
        self.SLJf1 = state_loss_j100 + state_loss_j200 + state_loss_j300

        self.SLPf2 = state_loss_pos10 + state_loss_pos20 + state_loss_pos30
        self.SLVf2 = state_loss_vel10 + state_loss_vel20 + state_loss_vel30
        self.SLAf2 = state_loss_acc10 + state_loss_acc20 + state_loss_acc30
        self.SLJf2 = state_loss_j10 + state_loss_j20 + state_loss_j30

        self.rmse_pos = self.SLPf1
        self.rmse_vel = self.SLVf1
        self.rmse_acc = self.SLAf1
        self.rmse_jer = self.SLJf1
        self.cov_pos_loss = self.error_loss_pos
        self.cov_vel_loss = self.error_loss_vel
        self.maha_loss = tf.reduce_mean(self.MD0)
        self.maha_out = tf.reduce_mean(MD)
        self.trace_loss = tf.reduce_mean(trace)
        self.dout = tf.reduce_mean(self.rl)
        self.saver = tf.train.Saver(save_relative_paths=True)

    def build_model(self):
        self.DROPOUT = tf.placeholder(self.vdtype)
        self.update_condition = tf.placeholder(tf.bool, name='update_condition')
        self.meanv = tf.placeholder(self.vdtype, shape=(1, self.num_state), name='meanv')
        # self.stdv = tf.placeholder(self.vdtype, shape=(1, self.num_state), name='stdv')

        self.grad_clip = tf.placeholder(self.vdtype, name='grad_clip')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.measurement = [tf.placeholder(self.vdtype, shape=(None, self.num_meas), name="meas_uvw_{}".format(t)) for t in range(self.max_seq)]

        self.sensor_ecef = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name='sen_ecef')
        self.sensor_lla = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name='sen_lla')

        self.prev_measurement = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name="px")
        self.prev_covariance = tf.placeholder(self.vdtype, shape=(None, self.num_state ** 2), name="pcov")
        self.prev_time = tf.placeholder(self.vdtype, shape=(None, 1), name="ptime")
        self.prev_truth = tf.placeholder(self.vdtype, shape=(None, self.num_state), name="ptruth")
        self.prev_state2 = tf.placeholder(self.vdtype, shape=(None, self.num_state), name="py2")

        self.current_timei = [tf.placeholder(self.vdtype, shape=(None, 1), name="current_time_{}".format(t)) for t in range(self.max_seq)]
        self.P_inp = tf.placeholder(self.vdtype, shape=(None, self.num_state ** 2), name="yc")
        self.state_input = tf.placeholder(self.vdtype, shape=(None, self.num_state), name="ycc")
        self.truth_state = [tf.placeholder(self.vdtype, shape=(None, self.num_state), name="y_truth_{}".format(t)) for t in range(self.max_seq)]
        self.seqweightin = tf.placeholder(self.vdtype, [None, self.max_seq])

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
                self.source_fwf = cell_type(self.F_hidden, period_init_max=1500.0)

        with tf.variable_scope('Source_Track_Forward2/state'):
            if use_dropout:
                self.source_fwf2 = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
                                                          input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
            else:
                self.source_fwf2 = cell_type(self.F_hidden, period_init_max=1500.0)

        # with tf.variable_scope('Source_Track_Forward/cov'):
        #     if use_dropout:
        #         self.source_fwc = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
        #                                                  input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
        #     else:
        #         self.source_fwc = cell_type(self.F_hidden)
        #
        # with tf.variable_scope('Source_Track_Forward2/cov'):
        #     if use_dropout:
        #         self.source_fwc2 = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
        #                                                   input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
        #     else:
        #         self.source_fwc2 = cell_type(self.F_hidden)
        #
        # with tf.variable_scope('Source_Track_Forward3/cov'):
        #     if use_dropout:
        #         self.source_fwc3 = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
        #                                                   input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
        #     else:
        #         self.source_fwc3 = cell_type(self.F_hidden)

        if self.state_type != 'GRU':

            self.init_c_fwc = tf.placeholder(name='init_c_fw/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.init_h_fwc = tf.placeholder(name='init_h_fw/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_fw_in_cov = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwc, self.init_h_fwc)

            self.init_c_fwc2 = tf.placeholder(name='init_c_fw2/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.init_h_fwc2 = tf.placeholder(name='init_h_fw2/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_fw_in_cov2 = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwc2, self.init_h_fwc2)

            self.init_c_fwc3 = tf.placeholder(name='init_c_fw3/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.init_h_fwc3 = tf.placeholder(name='init_h_fw3/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_fw_in_cov3 = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwc3, self.init_h_fwc3)

            self.init_c_fwf = tf.placeholder(name='init_c_fwf/state', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.init_h_fwf = tf.placeholder(name='init_h_fwf/state', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_fw_in_state = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwf, self.init_h_fwf)

            self.init_c_fwf2 = tf.placeholder(name='init_c_fwf2/state', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.init_h_fwf2 = tf.placeholder(name='init_h_fwf2/state', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_fw_in_state2 = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwf2, self.init_h_fwf2)


        else:

            self.init_c_bws = tf.placeholder(name='init_c_bw/state', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_bw_in_state = self.init_c_bws

            self.init_c_fwc = tf.placeholder(name='init_c_fw/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_fw_in_cov = self.init_c_fwc

            self.init_c_bwc = tf.placeholder(name='init_c_bw/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_bw_in_cov = self.init_c_bwc

        self.I_3 = tf.scalar_mul(1.0, tf.eye(3, batch_shape=[self.batch_size], dtype=self.vdtype))
        self.I_12 = tf.scalar_mul(1.0, tf.eye(12, batch_shape=[self.batch_size], dtype=self.vdtype))
        self.I_3z = tf.scalar_mul(0.0, tf.eye(3, batch_shape=[self.batch_size], dtype=self.vdtype))
        self.I_4z = tf.scalar_mul(0.0, tf.eye(4, batch_shape=[self.batch_size], dtype=self.vdtype))
        self.I_6z = tf.scalar_mul(0.0, tf.eye(6, batch_shape=[self.batch_size], dtype=self.vdtype))
        self.om = tf.ones([self.batch_size, 1, 1], dtype=self.vdtype)
        self.zb = tf.zeros([self.batch_size, 4, 2], dtype=self.vdtype)
        self.zm = tf.zeros([self.batch_size, 1, 1], dtype=self.vdtype)
        omp = np.ones([1, 1], self.vdp_np)
        zmp = np.zeros([1, 1], self.vdp_np)

        m1 = np.concatenate([omp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp], axis=1).astype(self.vdp_np)
        m2 = np.concatenate([zmp, zmp, zmp, zmp, omp, zmp, zmp, zmp, zmp, zmp, zmp, zmp], axis=1).astype(self.vdp_np)
        m3 = np.concatenate([zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, omp, zmp, zmp, zmp], axis=1).astype(self.vdp_np)
        self.meas_mat = tf.tile(tf.expand_dims(tf.concat([m1, m2, m3], axis=0), axis=0), [self.batch_size, 1, 1])

        # final_state_truth, final_cov_truth, hidden_states_truth1 = self.filter_measurement(self.prev_truth)
        # final_state2_truth, hidden_states_truth3 = self.estimate_covariance(final_state_truth, final_cov_truth, self.prev_truth)

        # smooth, A, Q, R, meas_uvw, state1_out, state2_out = self.smooth()

        smooth, A, Q, R, meas_uvw, state1_out, state2_out = self.filter()

        self.ao_list = A
        self.qo_list = Q
        self.ro_list = R
        self.new_meas = meas_uvw
        self.final_state = smooth[0]
        self.final_state2 = smooth[0]
        self.final_cov = smooth[1]
        self.cov_out = smooth[1]
        self.state_fwf = state1_out
        self.state_fwf2 = state2_out

        _y = tf.stack(self.truth_state, axis=1)

        elbo_kf, kf_log_probs, z_smooth = self.get_elbo(smooth)

        elbo_kf = tf.where(tf.is_nan(elbo_kf), tf.zeros_like(elbo_kf), elbo_kf)

        state_loss_pos100 = 0
        state_loss_pos200 = 0
        state_loss_pos300 = 0
        state_loss_vel100 = 0
        state_loss_vel200 = 0
        state_loss_vel300 = 0
        state_loss_acc100 = 0
        state_loss_acc200 = 0
        state_loss_acc300 = 0
        state_loss_j100 = 0
        state_loss_j200 = 0
        state_loss_j300 = 0

        total_weight = tf.cast(self.seqweightin, self.vdtype)
        tot = tf.cast(self.max_seq, self.vdtype)
        loss_func = weighted_mape_tf
        state_loss_pos100 += loss_func(_y[:, :, 0], self.final_state2[:, :, 0], total_weight, tot)
        state_loss_pos200 += loss_func(_y[:, :, 4], self.final_state2[:, :, 4], total_weight, tot)
        state_loss_pos300 += loss_func(_y[:, :, 8], self.final_state2[:, :, 8], total_weight, tot)
        state_loss_vel100 += loss_func(_y[:, :, 1], self.final_state2[:, :, 1], total_weight, tot) / 1000
        state_loss_vel200 += loss_func(_y[:, :, 5], self.final_state2[:, :, 5], total_weight, tot) / 1000
        state_loss_vel300 += loss_func(_y[:, :, 9], self.final_state2[:, :, 9], total_weight, tot) / 1000
        state_loss_acc100 += loss_func(_y[:, :, 2], self.final_state2[:, :, 2], total_weight, tot) / 10000
        state_loss_acc200 += loss_func(_y[:, :, 6], self.final_state2[:, :, 6], total_weight, tot) / 10000
        state_loss_acc300 += loss_func(_y[:, :, 10], self.final_state2[:, :, 10], total_weight, tot) / 10000
        state_loss_j100 += loss_func(_y[:, :, 3], self.final_state2[:, :, 3], total_weight, tot) / 100000
        state_loss_j200 += loss_func(_y[:, :, 7], self.final_state2[:, :, 7], total_weight, tot) / 100000
        state_loss_j300 += loss_func(_y[:, :, 11], self.final_state2[:, :, 11], total_weight, tot) / 100000

        self.SLPf1 = state_loss_pos100 + state_loss_pos200 + state_loss_pos300
        self.SLVf1 = state_loss_vel100 + state_loss_vel200 + state_loss_vel300
        self.SLAf1 = state_loss_acc100 + state_loss_acc200 + state_loss_acc300
        self.SLJf1 = state_loss_j100 + state_loss_j200 + state_loss_j300

        self.rmse_pos = self.SLPf1
        self.rmse_vel = self.SLVf1
        self.rmse_acc = self.SLAf1
        self.rmse_jer = self.SLJf1
        self.maha_loss = tf.reduce_mean(0.0)
        self.maha_out = tf.reduce_mean(0.0)
        self.trace_loss = tf.reduce_mean(0.0)
        self.saver = tf.train.Saver(save_relative_paths=True)

        # if self.window_mode:
        #     final_state0, final_cov = self.filter_measurement_set(self.prev_state2)
        #
        #     # final_state_smooth, final_cov_smooth = self.smooth(final_state0, final_cov)
        #
        #     # self.estimate_covariance(final_state_smooth, final_cov_smooth, self.prev_state2)
        #
        #     # final_state_smooth, final_cov_smooth = final_state0, final_cov
        # else:
        #     with tf.variable_scope('UKF'):
        #         final_state0, final_cov = self.filter_measurement(self.prev_state2)
        #
        #     with tf.variable_scope('COV'):
        #         final_state = self.estimate_covariance(final_state0, final_cov, self.prev_state2)
        #
        #     final_state_smooth, final_cov_smooth = final_state, final_cov
        #
        #     # final_state2, hidden_states_cov = self.estimate_covariance(final_state, final_cov, self.prev_state2)

        print('Building Loss')
        # self.build_loss(self.final_state, self.final_cov, self.final_state2, self.final_cov)

        # self.learning_rate = self.learning_rate_inp
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_inp, global_step=self.global_step, decay_steps=5000, decay_rate=0.8, staircase=True)
        # # int(5 * (1500 / self.max_seq) * (500 / self.batch_size))
        # max_lr = self.learning_rate
        # base_lr = max_lr / 4
        # step_size = int(1500 * 1.5)
        # stepi = tf.cast(self.global_step, self.vdtype)
        # cycle = tf.floor(1 + stepi / (2 * step_size))
        # xi = tf.abs(stepi / step_size - 2 * cycle + 1)
        # self.learning_rate = base_lr + (max_lr - base_lr) * tf.maximum(tf.cast(0., self.vdtype), 1. - xi)

        # all_vars = tf.trainable_variables()
        # discrim_vars = [var for var in all_vars if 'discriminator' in var.name]
        # cov_vars = [var for var in all_vars if 'cov' in var.name]
        # state_vars = [var for var in all_vars if 'state' in var.name]
        # not_d_vars = [var for var in all_vars if 'discriminator' not in var.name]

        with tf.variable_scope("TrainOps"):
            print('cov_update gradients...')

            # tfc.opt.MomentumWOptimizer(learning_rate=self.learning_rate, momentum=0.9, weight_decay=1e-10, name='r3')
            # tfc.opt.AdamWOptimizer(weight_decay=1e-10, learning_rate=self.learning_rate, name='r3')

            opt1 = tfc.opt.MultitaskOptimizerWrapper(tfc.opt.AdamWOptimizer(weight_decay=1e-10, learning_rate=self.learning_rate, name='r3'))

            # gradvars1 = opt1.compute_gradients((self.rl * 1 + self.error_loss_full*0 + self.SLPf2 + self.entropy + self.error_loss_Q*0 + self.MD0))
            gradvars1 = opt1.compute_gradients(-elbo_kf + self.SLPf1)

            gradvars1, _ = tfc.opt.clip_gradients_by_global_norm(gradvars1, 1.)
            self.train_g3 = opt1.apply_gradients(gradvars1, global_step=self.global_step)

            # opt2 = tfc.opt.MultitaskOptimizerWrapper(tfc.opt.AdamWOptimizer(weight_decay=1e-10, learning_rate=self.learning_rate, name='r4'))
            #
            # gradvars2 = opt2.compute_gradients(-elbo_kf)
            #
            # gradvars2, _ = tfc.opt.clip_gradients_by_global_norm(gradvars2, 1.)
            # self.train_g4 = opt2.apply_gradients(gradvars2, global_step=self.global_step)

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Total traininable network parameters:: ' + str(total_parameters))

    def train(self, data_rate, max_exp_seq):

        # rho0 = 1.22  # kg / m**3
        # k0 = 0.14141e-3
        # area = 0.25  # / self.RE  # meter squared
        # cd = 0.03  # unitless
        # gmn = self.GM / (self.RE ** 3)

        shuffle_data = False
        self.data_rate = data_rate
        self.max_exp_seq = max_exp_seq
        # initialize all variables
        tf.global_variables_initializer().run()

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
        if start_epoch != 0 or step != 0:
            print('Loading filter...')
            try:
                self.saver = tf.train.import_meta_graph(self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step) + '.meta')
                self.saver.restore(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step))
                # self.saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step) + '.meta'))
                print("filter restored.")
            except:
                start_epoch = 0
                step = 0
                print("Could not restore filter")

        e = int(start_epoch)

        ds = DataServerLive(self.meas_dir, self.state_dir)

        plot_count = 0
        # train_writer = tf.summary.FileWriter('./log/0/train/', self.sess.graph)

        for epoch in range(int(start_epoch), self.max_epoch):

            n_train_batches = int(ds.num_train / self.batch_size_np)

            for minibatch_index in range(n_train_batches):

                if (epoch % 2 == 0 or self.mode == 'testing') and epoch != 0 and minibatch_index % 5 == 0:
                    testing = True
                    print('Testing filter for epoch ' + str(epoch))
                else:
                    testing = False
                    print('Training filter for epoch ' + str(epoch))

                # Data is unnormalized at this point
                x_train, y_train, batch_number, total_batches, ecef_ref, lla_data = ds.load(batch_size=self.batch_size_np, constant=self.constant, test=testing,
                                                                                            max_seq_len=self.max_exp_seq, HZ=self.data_rate)

                # if shuffle_data:
                # shuf = np.arange(x_train.shape[0])
                # np.random.shuffle(shuf)
                # x_train = x_train[shuf]
                # y_train = y_train[shuf]

                # x_train = x_train[:, -self.max_seq:, :]
                # y_train = y_train[:, -self.max_seq:, :]

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

                    # Ti2e = np.zeros(shape=[3, 3])
                    # Ti2e[0, 0] = -np.sin(lla_datar[i, 1])
                    # Ti2e[0, 1] = np.cos(lla_datar[i, 1])
                    # Ti2e[1, 0] = -np.sin(lla_datar[i, 0]) * np.cos(lla_datar[i, 1])
                    # Ti2e[1, 1] = -np.sin(lla_datar[i, 0]) * np.sin(lla_datar[i, 1])
                    # Ti2e[1, 2] = np.cos(lla_datar[i, 0])
                    # Ti2e[2, 0] = np.cos(lla_datar[i, 0]) * np.cos(lla_datar[i, 1])
                    # Ti2e[2, 1] = np.cos(lla_datar[i, 0]) * np.sin(lla_datar[i, 1])
                    # Ti2e[2, 2] = np.sin(lla_datar[i, 0])
                    #
                    # for ii in range(y_train.shape[1]):
                    #     y_enu[i, ii, :] = np.squeeze(np.matmul(Ti2e, y_uvw[i, ii, np.newaxis, :].T), -1)
                    #     y_rae[i, ii, 0] = np.sqrt(y_enu[i, ii, 0] * y_enu[i, ii, 0] + y_enu[i, ii, 1] * y_enu[i, ii, 1] + y_enu[i, ii, 2] * y_enu[i, ii, 2])
                    #     y_rae[i, ii, 1] = np.arctan2(y_enu[i, ii, 0], y_enu[i, ii, 1])
                    #     if y_rae[i, ii, 1] < 0:
                    #         y_rae[i, ii, 1] = (2*np.pi) + y_rae[i, ii, 1]
                    #     y_rae[i, ii, 2] = np.arcsin(y_enu[i, ii, 2] / y_rae[i, ii, 0])
                    #
                    # y_enu[i, :, :] = np.where(zz, np.zeros_like(y_enu[i, :, :]), y_enu[i, :, :])
                    # y_rae[i, :, :] = np.where(zz, np.zeros_like(y_rae[i, :, :]), y_rae[i, :, :])

                y_train = np.concatenate([y_uvw, y_train[:, :, 3:]], axis=2)

                # a = y_train[0, :, :]
                # aa = x_train[0, :, :]

                permute_dims = False
                if permute_dims:
                    rn = np.random.rand()
                    if rn < 0.333:
                        perm = [0, 1, 2]
                    elif rn >= 0.333 and rn < 0.6667:
                        perm = [1, 0, 2]
                    else:
                        perm = [2, 1, 0]

                    m1 = copy.copy(x_train[:, :, 1, np.newaxis])
                    m2 = copy.copy(x_train[:, :, 2, np.newaxis])
                    m3 = copy.copy(x_train[:, :, 3, np.newaxis])

                    y1 = copy.copy(y_train[:, :, 0, np.newaxis])
                    y2 = copy.copy(y_train[:, :, 1, np.newaxis])
                    y3 = copy.copy(y_train[:, :, 2, np.newaxis])
                    y4 = copy.copy(y_train[:, :, 3, np.newaxis])
                    y5 = copy.copy(y_train[:, :, 4, np.newaxis])
                    y6 = copy.copy(y_train[:, :, 5, np.newaxis])
                    y7 = copy.copy(y_train[:, :, 6, np.newaxis])
                    y8 = copy.copy(y_train[:, :, 7, np.newaxis])
                    y9 = copy.copy(y_train[:, :, 8, np.newaxis])
                    y10 = copy.copy(y_train[:, :, 9, np.newaxis])
                    y11 = copy.copy(y_train[:, :, 10, np.newaxis])
                    y12 = copy.copy(y_train[:, :, 11, np.newaxis])

                    x_train[:, :, 1 + perm[0], np.newaxis] = m1
                    x_train[:, :, 1 + perm[1], np.newaxis] = m2
                    x_train[:, :, 1 + perm[2], np.newaxis] = m3

                    y_train[:, :, perm[0], np.newaxis] = y1
                    y_train[:, :, perm[1], np.newaxis] = y2
                    y_train[:, :, perm[2], np.newaxis] = y3
                    y_train[:, :, 3 + perm[0], np.newaxis] = y4
                    y_train[:, :, 3 + perm[1], np.newaxis] = y5
                    y_train[:, :, 3 + perm[2], np.newaxis] = y6
                    y_train[:, :, 6 + perm[0], np.newaxis] = y7
                    y_train[:, :, 6 + perm[1], np.newaxis] = y8
                    y_train[:, :, 6 + perm[2], np.newaxis] = y9
                    y_train[:, :, 9 + perm[0], np.newaxis] = y10
                    y_train[:, :, 9 + perm[1], np.newaxis] = y11
                    y_train[:, :, 9 + perm[2], np.newaxis] = y12

                # x_train[:, :, 1:] = x_train[:, :, 1:] / 6378137
                # y_train = normalize_statenp(y_train)
                # y_train = y_train / 6378137
                # _, _, _, _, mean_y, std_y = normalize_statenp(copy.copy(x_train[:, :, :4]), copy.copy(y_uvw))

                max_pos = 50000
                max_vel = 500
                max_acc = 50
                max_jer = 10

                mean_y = np.array([max_pos, max_pos, max_pos,
                                   max_vel, max_vel, max_vel,
                                   max_acc, max_acc, max_acc,
                                   max_jer, max_jer, max_jer])

                mean_y2 = np.array([max_pos, max_pos, max_pos,
                                    max_vel, max_vel, max_vel])

                # if shuffle_data:
                #     shuf = np.arange(x_train.shape[0])
                #     np.random.shuffle(shuf)
                #     x_train = x_train[shuf]
                #     y_train = y_train[shuf]

                s_train = x_train

                # n_train_batches = int(x_train.shape[0] / self.batch_size_np)
                print("Batch Number: {0:2d} out of {1:2d}".format(batch_number, total_batches))

                x0, y0, meta0, prev_y, prev_x, prev_time, prev_meta, initial_meas, initial_time = prepare_batch(0, x_train, y_train, s_train,
                                                                                                                seq_len=self.max_seq, batch_size=self.batch_size_np,
                                                                                                                new_batch=True, window_mode=self.window_mode, pad_front=self.pad_front)

                count, _, _, _, _, _, prev_cov, _, q_plot, q_plott, k_plot, out_plot_X, out_plot_F, out_plot_P, time_vals, \
                meas_plot, truth_plot, Q_plot, R_plot, maha_plot, x, y, meta = initialize_run_variables(self.batch_size_np, self.max_seq, self.num_state, x0, y0, meta0)

                # print('Resetting Feed Dict')
                feed_dict = {}

                # windows = np.ceil(x.shape[1] / self.max_seq)
                # total = int(windows * self.max_seq)
                # actual = x.shape[1]

                windows2 = int((x.shape[1]) / self.max_seq)
                # windows2 = int(x.shape[1])
                # max_seq = windows2
                time_plotter = np.zeros([self.batch_size_np, int(x.shape[1]), 1])

                if self.window_mode:
                    mstep = windows2
                else:
                    if self.pad_front:
                        mstep = x.shape[1]
                    else:
                        mstep = x.shape[1] - self.max_seq

                for tstep in range(0, mstep):

                    # merge = tf.summary.merge_all()

                    # if self.window_mode:
                    r1 = tstep * self.max_seq
                    r2 = r1 + self.max_seq

                    if tstep == 0:
                        prev_state = copy.copy(prev_y)
                        prev_meas = copy.copy(prev_x)

                    current_x, current_y, current_time, current_meta = \
                        get_feed_time_asynch(x, y, meta, prev_x, prev_y, prev_time, prev_meta, self.max_seq, tstep, self.num_state, self.window_mode)

                    if np.all(current_x == 0):
                        continue

                    seqlen = np.ones(shape=[self.batch_size_np, ])
                    int_time = np.zeros(shape=[self.batch_size_np, self.max_seq])

                    if self.window_mode:
                        seqweight = np.zeros(shape=[self.batch_size_np, self.max_seq])
                    else:
                        seqweight = np.zeros(shape=[self.batch_size_np, 1])

                    for i in range(self.batch_size_np):
                        current_yt = current_y[i, :, :3]
                        m = ~(current_yt == 0).all(1)
                        yf = current_yt[m]
                        seq = yf.shape[0]
                        seqlen[i] = seq
                        int_time[i, :] = range(r1, r2)
                        seqweight[i, :] = m.astype(int)

                    cur_time = x[:, tstep, 0]

                    time_plotter[:, tstep, :] = cur_time[:, np.newaxis]
                    max_t = np.max(time_plotter[0, :, 0])
                    count += 1
                    step += 1
                    idxi = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
                    idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
                    idxi2 = [0, 3, 1, 4, 2, 5]

                    if tstep == 0:
                        pos = initial_meas[:, 2, :]
                        vel = (initial_meas[:, 2, :] - initial_meas[:, 0, :]) / np.sum(np.diff(initial_time, axis=1), axis=1)

                        R1 = np.linalg.norm(initial_meas + ecef_ref[:, np.newaxis, :], axis=2, keepdims=True)
                        R1 = np.mean(R1, axis=1)
                        R1 = np.where(np.less(R1, np.ones_like(R1) * self.RE), np.ones_like(R1) * self.RE, R1)
                        rad_temp = np.power(R1, 3)
                        GMt1 = np.divide(self.GM, rad_temp)
                        acc = get_legendre_np(GMt1, pos + ecef_ref, R1)
                        initial_state = np.expand_dims(np.concatenate([pos, vel, acc, np.random.normal(loc=np.zeros_like(acc), scale=10.)], axis=1), 1)

                        initial_state = initial_state[:, :, idxi]

                        current_state, covariance_out = unscented_kalman_np(self.batch_size_np, prev_meas.shape[1], initial_state[:, -1, :], prev_cov[:, -1, :, :], prev_meas, prev_time)

                        # initial_state = initial_state[:, :, idxo]
                        prev_state2 = initial_state[:, :, idxo]  # initial state guess
                        prev_state3 = copy.copy(prev_state2)
                        # current_covariance = prev_covariance[-1]

                        prev_cov = prev_cov[:, -1:, :, :]
                        # current_covariance = covariance_out[-1]
                        # current_covariance2 = prev_cov2[:, -1, :, :]
                        # else:
                        #     current_covariance = covariance_out[-1]
                        #     current_covariance2 = prev_cov2[:, -1, :, :]
                        #     prev_cov = np.concatenate([prev_cov[:, -1, np.newaxis, :, :], np.stack(covariance_out, axis=1)[:, :-1, :, :]], axis=1)
                            # prev_state2 = np.concatenate([initial_state, prev_state2[:, :-1, :]], axis=1)
                            # prev_state3 = copy.copy(prev_state2)

                    update = False

                    mean_y = mean_y[idxi]
                    mean_y2 = mean_y2[idxi2]
                    # std_y = std_y[idxi]
                    prev_y = prev_y[:, :, idxi]
                    current_state = current_state[:, :, idxi]
                    current_y = current_y[:, :, idxi]
                    prev_state = prev_state[:, :, idxi]
                    prev_state2 = prev_state2[:, :, idxi]
                    prev_state3 = prev_state3[:, :, idxi]

                    # a1 = np.diag(current_covariance[0,:,:])
                    # a2 = prev_y[0,:,:]
                    # a3 = prev_x[0, :, :]
                    # a4 = current_x[:, 0, :]
                    # a5 = current_y[:, 0, :]
                    # a6 = prev_state2[0, :, :]

                    feed_dict.update({self.measurement[t]: current_x[:, t, :].reshape(-1, self.num_meas) for t in range(self.max_seq)})
                    feed_dict.update({self.prev_measurement: prev_x.reshape(-1, self.num_meas)})
                    feed_dict.update({self.prev_covariance: prev_cov.reshape(-1, self.num_state ** 2)})
                    feed_dict.update({self.truth_state[t]: current_y[:, t, :].reshape(-1, self.num_state) for t in range(self.max_seq)})
                    feed_dict.update({self.prev_truth: prev_y.reshape(-1, self.num_state)})
                    feed_dict.update({self.prev_state2: prev_state2.reshape(-1, self.num_state)})
                    # feed_dict.update({self.prev_state3: prev_state3[:, -1, :].reshape(-1, self.num_state)})
                    feed_dict.update({self.sensor_ecef: ecef_ref})
                    feed_dict.update({self.sensor_lla: lla_datar})
                    feed_dict.update({self.seqlen: seqlen})
                    feed_dict.update({self.int_time: int_time})
                    feed_dict.update({self.update_condition: update})
                    feed_dict.update({self.is_training: True})
                    # feed_dict.update({self.meanv: mean_y[np.newaxis, :]})
                    # feed_dict.update({self.stdv: std_y[np.newaxis, :]})
                    feed_dict.update({self.seqweightin: seqweight})
                    # feed_dict.update({self.maneuverin[t]: prev_meta[:, t, :].reshape(-1, 1) for t in range(self.max_seq)})
                    feed_dict.update({self.P_inp: prev_cov.reshape(-1, self.num_state ** 2)})
                    feed_dict.update({self.state_input: prev_state2.reshape(-1, self.num_state)})
                    # feed_dict.update({self.P_inp2: current_covariance2.reshape(-1, 6 ** 2)})
                    feed_dict.update({self.prev_time: prev_time[:, :, 0]})
                    feed_dict.update({self.current_timei[t]: current_time[:, t, :].reshape(-1, 1) for t in range(self.max_seq)})
                    feed_dict.update({self.batch_step: tstep})
                    feed_dict.update({self.drop_rate: 1.0})

                    if tstep == 0:
                        # print("Resetting LSTM States")
                        if testing is True:
                            std = 0.0
                        else:
                            std = 0.05

                        feed_dict.update({self.init_c_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        feed_dict.update({self.init_h_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        feed_dict.update({self.init_c_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        feed_dict.update({self.init_h_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        # feed_dict.update({self.init_c_fwc: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        # feed_dict.update({self.init_h_fwc: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        # feed_dict.update({self.init_c_fwc2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        # feed_dict.update({self.init_h_fwc2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        # feed_dict.update({self.init_c_fwc3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        # feed_dict.update({self.init_h_fwc3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

                    if testing is False:
                        if e < 1:
                            lr = 5e-3
                            train_op1 = self.train_g3
                            stateful = True
                        elif e >= 1 and e < 10:
                            lr = 2e-3
                            stateful = True
                            train_op1 = self.train_g3
                        else:
                            lr = 5e-4
                            stateful = True
                            train_op1 = self.train_g3

                        feed_dict.update({self.learning_rate_inp: lr})

                        pred_output0, pred_output00, pred_output1, q_out_t, q_out, _, rmsp, rmsv, rmsa, rmsj, LR, \
                        cov_pos_loss, cov_vel_loss, kalman_cov_loss, maha_loss, MD, trace_loss, rl, \
                        entropy, qt_out, rt_out, at_out, q_loss, state_fwf, state_fwf2, new_meas, num_el = \
                            self.sess.run([self.final_state,
                                           self.final_state2,
                                           self.final_state2,
                                           self.final_cov,
                                           self.cov_out,
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
                                           self.error_loss_Q,
                                           self.state_fwf,
                                           self.state_fwf2,
                                           self.new_meas,
                                           self.num_el],
                                          feed_dict)

                    else:
                        feed_dict.update({self.is_training: False})
                        feed_dict.update({self.deterministic: True})
                        feed_dict.update({self.drop_rate: 1.0})
                        stateful = True
                        pred_output0, pred_output00, pred_output1, q_out_t, q_out, rmsp, rmsv, rmsa, rmsj, LR, \
                        cov_pos_loss, cov_vel_loss, kalman_cov_loss, maha_loss, MD, trace_loss, rl, \
                        entropy, qt_out, rt_out, at_out, q_loss, \
                        state_fwf, state_fwf2, new_meas = \
                            self.sess.run([self.final_state,
                                           self.final_state2,
                                           self.final_state2,
                                           self.final_cov,
                                           self.cov_out,
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
                                           self.error_loss_Q,
                                           self.state_fwf,
                                           self.state_fwf2,
                                           self.new_meas],
                                          feed_dict)

                    if tstep % 10 == 0 or tstep <= self.max_seq or tstep > int(x.shape[1] - 10):
                        print("Epoch: {0:2d} MB: {1:1d} Time: {2:3d} "
                              "RMSP: {3:2.2e} RMSV: {4:2.2e} RMSA: {5:2.2e} RMSJ: {6:2.2e} "
                              "LR: {7:1.2e} ST: {8:1.2f} CPL: {9:1.2f} "
                              "CVL: {10:1.2f} EN: {11:1.2f} QL: {12:1.2f} "
                              "MD: {13:1.2f} RL: {14:1.2f} COV {15:1.2f} ".format(epoch, minibatch_index, tstep,
                                                                                  rmsp, rmsv, rmsa, rmsj,
                                                                                  LR, max_t, cov_pos_loss,
                                                                                  num_el, entropy, q_loss,
                                                                                  MD, rl, kalman_cov_loss))

                    mean_y = mean_y[idxo]
                    # std_y = std_y[idxo]
                    # prev_y = prev_y[:, :, idxo]
                    current_y = current_y[:, :, idxo]
                    prev_state = prev_state[:, :, idxo]
                    current_state = current_state[:, :, idxo]
                    prev_state2 = prev_state2[:, :, idxo]
                    prev_state3 = prev_state3[:, :, idxo]

                    pred_output0 = pred_output0[:, :, idxo]
                    pred_output00 = pred_output00[:, :, idxo]
                    pred_output1 = pred_output1[:, :, idxo]

                    if stateful is True:
                        if self.state_type != 'GRU':
                            feed_dict.update({self.init_c_fwf: state_fwf[0]})
                            feed_dict.update({self.init_h_fwf: state_fwf[1]})
                            feed_dict.update({self.init_c_fwf2: state_fwf2[0]})
                            feed_dict.update({self.init_h_fwf2: state_fwf2[1]})
                            # feed_dict.update({self.init_c_fwc: state_fwc[0]})
                            # feed_dict.update({self.init_h_fwc: state_fwc[1]})
                            # feed_dict.update({self.init_c_fwc2: state_fwc2[0]})
                            # feed_dict.update({self.init_h_fwc2: state_fwc2[1]})
                            # feed_dict.update({self.init_c_fwc3: state_fwc3[0]})
                            # feed_dict.update({self.init_h_fwc3: state_fwc3[1]})
                            # else:
                            #     feed_dict.update({self.init_c_fw3: state_fwf})
                            # feed_dict.update({self.init_c_fws: drnn1f[0]})
                            # feed_dict.update({self.init_c_fwc: drnn2f[0]})

                    else:
                        if testing is True:
                            std = 0.0
                        else:
                            std = 0.05
                        feed_dict.update({self.init_c_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        feed_dict.update({self.init_h_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        feed_dict.update({self.init_c_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        feed_dict.update({self.init_h_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        # feed_dict.update({self.init_c_fwc: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        # feed_dict.update({self.init_h_fwc: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        # feed_dict.update({self.init_c_fwc2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        # feed_dict.update({self.init_h_fwc2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        # feed_dict.update({self.init_c_fwc3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        # feed_dict.update({self.init_h_fwc3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

                    prop_output = np.array(pred_output0)
                    if len(prop_output.shape) < 3:
                        prop_output = np.expand_dims(prop_output, axis=1)
                    # if prop_output.shape[1] != self.max_seq:
                    #     prop_output = np.transpose(prop_output, [1, 0, 2])

                    pred_output = np.array(pred_output1)
                    if len(pred_output.shape) < 3:
                        pred_output = np.expand_dims(pred_output, axis=1)
                    # if pred_output.shape[1] != self.max_seq:
                    #     pred_output = np.transpose(pred_output, [1, 0, 2])

                    full_final_output = np.array(pred_output00)
                    if len(full_final_output.shape) < 3:
                        full_final_output = np.expand_dims(full_final_output, axis=1)

                    # Single time step plotting
                    if self.window_mode is False:

                        idx = -1

                        temp_prev_x = current_x
                        temp_prev_y = current_y
                        temp_prev_time = current_time

                        temp_pred0 = np.expand_dims(prop_output[:, idx, :], axis=1)
                        temp_pred1 = np.expand_dims(pred_output[:, idx, :], axis=1)

                        if tstep == 0:
                            prev_y = prev_y[:, :, idxo]

                            new_vals_F = full_final_output[0, -1, :]
                            out_plot_F = prev_state2[0, np.newaxis, :, :]
                            new_vals_X = pred_output[0, -1, :]
                            out_plot_X = prev_state2[0, np.newaxis, :, :]
                            new_vals_P = prop_output[0, -1, :]
                            out_plot_P = prev_state2[0, np.newaxis, :, :]

                            new_q = q_out[0, :, :]
                            # q_plot = np.tile(new_q[np.newaxis, np.newaxis, :, :], [1, self.max_seq, 1, 1])

                            q_plott = np.stack(covariance_out, axis=1)
                            q_plott = q_plott[0, :, :, :]

                            # q_initial = np.tile(np.eye(6, 6)[np.newaxis, :, :], [self.max_seq, 1, 1])
                            # q_initial[:, 0, 0] = q_plott[:, 0, 0]
                            # q_initial[:, 1, 1] = q_plott[:, 1, 1]
                            # q_initial[:, 2, 2] = q_plott[:, 2, 2]
                            # q_initial[:, 3, 3] = q_plott[:, 3, 3]
                            # q_initial[:, 4, 4] = q_plott[:, 4, 4]
                            # q_initial[:, 5, 5] = q_plott[:, 5, 5]
                            # q_plot = np.concatenate([q_initial[np.newaxis, :, :, :], new_q[np.newaxis, np.newaxis, :, :]], axis=1)

                            q_plott = q_plott[np.newaxis, :, :, :]
                            # q_plot = np.concatenate([q_plott, new_q[np.newaxis, np.newaxis, :, :]], axis=1)
                            # q_plott = q_plot
                            q_plot = q_plott

                            qt_plot = np.tile(qt_out[0, np.newaxis, np.newaxis, :, :], [1, self.max_seq, 1, 1])
                            rt_plot = np.tile(rt_out[0, np.newaxis, np.newaxis, :, :], [1, self.max_seq, 1, 1])
                            at_plot = np.tile(at_out[0, np.newaxis, np.newaxis, :, :], [1, self.max_seq, 1, 1])

                            new_time = current_time[0, -1:, 0]
                            time_vals = prev_time[0, np.newaxis, :, 0, np.newaxis]
                            new_meas = current_x[0, -1, :]
                            meas_plot = prev_x[0, np.newaxis, :, :]
                            new_truth = current_y[0, -1, :]
                            truth_plot = prev_y[0, np.newaxis, :, :]

                        else:
                            new_vals_F = full_final_output[0, -1, :]  # current step
                            # update_F = full_final_output[0, :-1, :]
                            new_vals_X = pred_output[0, -1, :]
                            # update_X = pred_output[0, :-1, :]
                            new_vals_P = prop_output[0, -1, :]
                            # update_P = prop_output[0, :-1, :]
                            new_q = q_out[0, :, :]
                            # update_q = q_out[0, :-1, :, :]
                            new_qt = q_out_t[0, :, :]
                            # update_qt = q_truth[0, :-1, :, :]
                            new_qtt = qt_out[0, :, :]
                            new_rtt = rt_out[0, :, :]
                            new_att = at_out[0, :, :]

                            new_time = current_time[0, -1, 0]
                            new_meas = current_x[0, -1, :]
                            new_truth = current_y[0, -1, :]

                        if tstep > 0:
                            out_plot_F = np.concatenate([out_plot_F, new_vals_F[np.newaxis, np.newaxis, :]], axis=1)
                            out_plot_X = np.concatenate([out_plot_X, new_vals_X[np.newaxis, np.newaxis, :]], axis=1)
                            out_plot_P = np.concatenate([out_plot_P, new_vals_P[np.newaxis, np.newaxis, :]], axis=1)
                            meas_plot = np.concatenate([meas_plot, new_meas[np.newaxis, np.newaxis, :]], axis=1)
                            truth_plot = np.concatenate([truth_plot, new_truth[np.newaxis, np.newaxis, :]], axis=1)
                            time_vals = np.concatenate([time_vals, new_time[np.newaxis, np.newaxis, np.newaxis]], axis=1)
                            q_plot = np.concatenate([q_plot, new_q[np.newaxis, np.newaxis, :, :]], axis=1)
                            q_plott = np.concatenate([q_plott, new_qt[np.newaxis, np.newaxis, :, :]], axis=1)
                            qt_plot = np.concatenate([qt_plot, new_qtt[np.newaxis, np.newaxis, :, :]], axis=1)
                            rt_plot = np.concatenate([rt_plot, new_rtt[np.newaxis, np.newaxis, :, :]], axis=1)
                            at_plot = np.concatenate([at_plot, new_att[np.newaxis, np.newaxis, :, :]], axis=1)

                        # err0 = np.sum(np.abs(temp_prev_y - temp_pred0))
                        # err1 = np.sum(np.abs(temp_prev_y - temp_pred1))
                        # if err1 < err0 and e > 5:
                        #     new_prev = temp_pred0
                        # else:
                        #     new_prev = temp_pred1

                        prev_state2 = np.concatenate([prev_state2, temp_pred0], axis=1)
                        prev_state2 = prev_state2[:, 1:, :]

                        prev_state3 = np.concatenate([prev_state3, temp_pred1], axis=1)
                        prev_state3 = prev_state3[:, 1:, :]

                        prev_state = np.concatenate([prev_state, temp_prev_y], axis=1)
                        prev_state = prev_state[:, 1:, :]

                        prev_time = np.concatenate([prev_time, temp_prev_time], axis=1)
                        prev_time = prev_time[:, 1:, :]

                        prev_cov = np.concatenate([prev_cov, q_out_t[:, np.newaxis, :, :]], axis=1)
                        prev_cov = prev_cov[:, 1:, :, :]

                        prev_meas = np.concatenate([prev_meas, temp_prev_x], axis=1)
                        prev_meas = prev_meas[:, 1:, :]

                        prev_y = copy.copy(prev_state)
                        prev_x = copy.copy(prev_meas)

                    else:
                        
                        idx = -1
                        prev_meas = np.concatenate([prev_meas, current_x], axis=1)
                        prev_meas = prev_meas[:, idx, np.newaxis, :]

                        # err0 = np.sum(np.abs(temp_prev_y - temp_pred0))
                        # err1 = np.sum(np.abs(temp_prev_y - temp_pred1))
                        # if err1 < err0 and e > 5:
                        #     new_prev = temp_pred0
                        # else:
                        #     new_prev = temp_pred1

                        current_state = prop_output[:, idx, np.newaxis, :]

                        prev_state2 = np.concatenate([prev_state2, prop_output], axis=1)
                        prev_state2 = prev_state2[:, idx, np.newaxis, :]

                        prev_state3 = np.concatenate([prev_state3, pred_output], axis=1)
                        prev_state3 = prev_state3[:, idx, np.newaxis, :]

                        prev_state = np.concatenate([prev_state, current_y], axis=1)
                        prev_state = prev_state[:, idx, np.newaxis, :]

                        prev_time = np.concatenate([prev_time, current_time], axis=1)
                        prev_time = prev_time[:, idx, np.newaxis, :]

                        prev_cov = np.concatenate([prev_cov[:, -1:, :, :], q_out_t], axis=1)
                        prev_cov = prev_cov[:, idx, np.newaxis, :, :]

                        prev_y = copy.copy(prev_state)
                        prev_x = copy.copy(prev_meas)

                        if tstep == 0:
                            out_plot_F = full_final_output[0, np.newaxis, :, :]
                            out_plot_X = pred_output[0, np.newaxis, :, :]
                            out_plot_P = prop_output[0, np.newaxis, :, :]

                            q_plott = q_out_t[0, np.newaxis, :, :, :]
                            q_plot = q_out[0, np.newaxis, :, :, :]
                            qt_plot = qt_out[0, np.newaxis, :, :]
                            rt_plot = rt_out[0, np.newaxis, :, :]
                            at_plot = at_out[0, np.newaxis, :, :]

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

                # if tstep == int((mstep - 1)) or tstep % 1000000 == 0 and tstep != 0 and (step - plot_count) > 1:
                # plt.show()
                # plt.close()
                if minibatch_index % 25 == 0:
                    if testing is False:
                        plotpath = self.plot_dir + '/epoch_' + str(epoch) + '_B_' + str(batch_number) + '_step_' + str(step)
                    else:
                        plotpath = self.plot_dir + '/epoch_' + str(epoch) + '_test_B_' + str(batch_number) + '_step_' + str(step)
                    if os.path.isdir(plotpath):
                        print('folder exists')
                    else:
                        os.mkdir(plotpath)

                    # plot_all2(out_plot_X, out_plot_F, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, mean_y)
                    comparison_plot(out_plot_X, out_plot_F, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, mean_y, qt_plot, rt_plot)

                    plot_count = step

                # if e % 25 == 0 and e != 0 and minibatch_index == n_train_batches - 1:
                if minibatch_index % 50 == 0 and minibatch_index != 0:
                    if os.path.isdir(self.checkpoint_dir):
                        print('filter Checkpoint Directory Exists')
                    else:
                        os.mkdir(self.checkpoint_dir)
                    print("Saving filter Weights for epoch" + str(epoch))
                    save_path = self.saver.save(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(epoch) + '_' + str(step) + ".ckpt", global_step=step)
                    print("Checkpoint saved at: ", save_path)

            e += 1
