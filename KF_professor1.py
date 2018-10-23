from propagation_utils import *
from load_all_data_4 import DataServerLive
from modules import *
import math
from plotting import *

import tensorflow.contrib as tfc
from tensorflow.contrib.layers import fully_connected as FCL
import numpy as np
import tensorflow_probability as tfp
from tensorflow.python.ops import init_ops

tfd = tfp.distributions
tfb = tfp.bijectors

filter_name = 'KF_iw10s'
plot_dir = './' + filter_name
save_dir = 'D:/Checkpoints/' + filter_name

if os.path.isdir(plot_dir):
    print('Plotting Folder Exists')
else:
    print('Making plot directory')
    os.mkdir(plot_dir)

if os.path.isdir(save_dir):

    print('Save Directory Exists')
else:
    print('Making save directory')
    os.mkdir(save_dir)

# # sensor_locations
# [0, 0.0449, 8]
# [-0.18, 0.0449, 8]
# [0.45, 0.0449, 8]

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', '100', """batch size""")
tf.app.flags.DEFINE_integer('max_seq', '5', """number of measurements included""")
tf.app.flags.DEFINE_integer('start_seq', '1', """number of measurements included""")
tf.app.flags.DEFINE_integer('num_state', '12', """number of state variables""")
tf.app.flags.DEFINE_integer('num_meas', '3', """number of measurement variables""")
tf.app.flags.DEFINE_float('max_epoch', 9999, """Radius of Earth plus an altitude""")
tf.app.flags.DEFINE_float('RE', 6378137, """Radius of Earth plus an altitude""")
tf.app.flags.DEFINE_float('GM', 398600441890000, """GM""")
tf.app.flags.DEFINE_string('log_dir', './log/', """Directory where to write event logs """)
tf.app.flags.DEFINE_string('plot_dir', plot_dir, """Path to the state directory.""")
tf.app.flags.DEFINE_string('save_dir', save_dir, """Path to the state directory.""")
tf.app.flags.DEFINE_integer('F_hidden', '12', """F hidden layers""")
tf.app.flags.DEFINE_integer('R_hidden', '48', """R hidden layers""")

setattr(tfc.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)


def create_dist(loc, scale, scope):
    # qmatrix = FCL(tf.cast(scale, tf.float32), self.num_state, activation_fn=tf.nn.tanh, scope=scope)
    # qmatrix = tf.reshape(qmatrix, [self.batch_size, self.num_state, self.num_state])
    # qloc = FCL(tf.cast(loc, tf.float64), self.num_state, activation_fn=None, scope='mu')
    # qchol = tfp.distributions.matrix_diag_transform(tf.cast(qmatrix, tf.float32), tf.nn.softplus)
    # qchol = tf.linalg.LinearOperatorLowerTriangular(qchol).to_dense()
    x = tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)
    return x


class Filter(object):

    def __init__(self, sess, trainable_state=False, state_type='GRU', mode='training'):

        self.sess = sess
        self.mode = mode
        self.max_seq = FLAGS.max_seq
        self.train_init_state = trainable_state
        self.F_hidden = FLAGS.F_hidden
        self.R_hidden = FLAGS.R_hidden
        self.num_state = FLAGS.num_state
        self.num_meas = FLAGS.num_meas
        self.plot_dir = FLAGS.plot_dir
        self.save_dir = FLAGS.save_dir
        self.GM = FLAGS.GM
        self.max_epoch = FLAGS.max_epoch
        self.RE = FLAGS.RE
        self.state_type = state_type

        # tf.set_random_seed(1)

        print('Using Advanced Datagen 2 ')
        self.meas_dir = 'D:/TrackFilterData/Delivery_13/5k25Hz_oop_broad_data/NoiseRAE/'
        self.state_dir = 'D:/TrackFilterData/Delivery_13/5k25Hz_oop_broad_data/Translate/'

        self.log_dir = FLAGS.log_dir + '/' + filter_name
        # summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

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

        self.vdtype = tf.float64
        self.vdp_np = np.float64
        # fading = tf.Variable(0.9, trainable=False) * tf.pow(0.99, (global_step / (1500*40/self.max_seq))) + 0.05

    def filter_measurement(self, prev_state):
        alpha = 1e-1 * tf.ones([self.batch_size, 1], dtype=tf.float64)
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
        all_time = tf.stack(self.prev_time, axis=1) / 2000

        meanv = tf.ones_like(all_states) * self.meanv
        # stdv = tf.ones_like(final_state_gs) * self.stdv

        all_states = all_states / meanv

        pos_m = tf.concat([meanv[:, :, 0, tf.newaxis], meanv[:, :, 4, tf.newaxis], meanv[:, :, 8, tf.newaxis]], axis=2)

        all_meas = all_meas / pos_m

        pr0 = all_meas - tf.squeeze(tf.matmul(tf.tile(self.meas_mat[:, tf.newaxis, :, :], [1, self.max_seq, 1, 1]), all_states[:, :, :, tf.newaxis]), -1)

        h = tf.concat([all_meas, all_states, pr0, all_time], axis=2)

        rnn_inp02 = FCL(h, self.F_hidden, activation_fn=None, scope='input/state', reuse=tf.AUTO_REUSE)

        inp1 = tf.unstack(rnn_inp02, axis=1)

        inp1t = self.prev_time

        sf = list()
        for q in range(self.max_seq):
            temp_input = (inp1t[q], inp1[q])
            if q == 0:
                # hp = state_fw2_in[1] + noise_sample
                # state_fw2_in = tf.contrib.rnn.LSTMStateTuple(state_fw2_in[0], hp)
                with tf.variable_scope('Source_Track_Forward/state'):
                    (self.source_track_out_fws[q], self.state_fws[q]) = self.source_fws(temp_input, state=self.state_fw_in_state)
            else:
                # hp = self.state_fw2[q - 1][1] + noise_sample
                # self.state_fw2[q - 1] = tf.contrib.rnn.LSTMStateTuple(self.state_fw2[q - 1][0], hp)
                with tf.variable_scope('Source_Track_Forward/state'):
                    tf.get_variable_scope().reuse_variables()
                    (self.source_track_out_fws[q], self.state_fws[q]) = self.source_fws(temp_input, state=self.state_fws[q - 1])
            sf.append(self.source_track_out_fws[q])

        states_fwso = self.state_fws[-1]
        self.final_drnn1_statef = states_fwso
        rnn_output = tf.stack(sf, axis=1)

        both_states = tf.concat(states_fwso, axis=1)
        tmp = tf.tile(both_states[:, tf.newaxis, :], [1, self.max_seq, 1])
        out = tf.concat([rnn_output, tmp], axis=2)

        attended_inputs = multihead_attention(queries=out, keys=out, num_units=out.shape[2], num_heads=out.shape[2],
                                              dropout_rate=self.drop_rate, is_training=self.is_training, scope='attention1/state', dtype=self.vdtype, reuse=tf.AUTO_REUSE)

        with tf.variable_scope('encoder1/state', reuse=tf.AUTO_REUSE):
            stddev = 1.0 / (attended_inputs.shape[2].value * self.max_seq)
            Ue = tf.Variable(dtype=self.vdtype,
                             initial_value=tf.truncated_normal(shape=[attended_inputs.shape[2].value, 1],
                                                               mean=0.0, stddev=stddev, dtype=self.vdtype), name='Ue/state')

        var = tf.tile(tf.expand_dims(Ue, 0), [self.batch_size, 1, 1])  # (b,T,T)
        weighted_output = tf.squeeze(tf.matmul(attended_inputs, var), -1) / self.max_seq

        pstate_est = prev_state[-1]
        cov_est0 = tf.reshape(self.P_inp, [self.batch_size, self.num_state, self.num_state])

        if self.state_type == 'PLSTM':
            weighted_output = (self.current_time, weighted_output)

        with tf.variable_scope('Source_Track_Forward3/state'):
            (self.source_track_out_fwf, self.state_fwf) = self.source_fw_filter(weighted_output, state=self.state_fw_in_filter)

        sjaj0 = FCL(self.source_track_out_fwf, 9, activation_fn=None, scope='0/measr', reuse=tf.AUTO_REUSE)

        rm = FCL(sjaj0, 9, activation_fn=None, scope='1/state', reuse=tf.AUTO_REUSE)
        # rd = tf.nn.relu(rm[:, :3])*50 + tf.ones_like(rm[:, :3])*0.1
        self.rd = tril_with_diag_softplus_and_shift(rm[:, :6], diag_shift=0.1, name='2/state')
        rdist = tfp.distributions.MultivariateNormalTriL(loc=None, scale_tril=self.rd)
        # rdist = tfp.distributions.MultivariateNormalDiag(loc=None, scale_diag=rd)

        cur_meas_temp = self.measurement
        sjx = tf.nn.sigmoid(rm[:, -3, tf.newaxis]) * 1000
        sjx = sjx + self.om[:, :, 0] * 1

        sjy = tf.nn.sigmoid(rm[:, -2, tf.newaxis]) * 1000
        sjy = sjy + self.om[:, :, 0] * 1

        sjz = tf.nn.sigmoid(rm[:, -1, tf.newaxis]) * 1000
        sjz = sjz + self.om[:, :, 0] * 1

        pstate_est_temp = copy.copy(pstate_est)

        dt = self.current_time - self.prev_time[-1]
        dt = tf.where(dt <= 1 / 100, tf.ones_like(dt) * 1 / 25, dt)

        self.Qt, At, Bt, At2 = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                                   dimension=int(self.num_state / 3),
                                   sjix=self.om[:, :, 0] * sjx ** 2,
                                   sjiy=self.om[:, :, 0] * sjy ** 2,
                                   sjiz=self.om[:, :, 0] * sjz ** 2,
                                   aji=self.om[:, :, 0] * 1.)

        qcholr = tf.cholesky(tf.cast(cov_est0, tf.float64))
        self.Rt = rdist.covariance()

        Am = tf.expand_dims(c, axis=2) * tf.cast(qcholr, tf.float64)
        Y = tf.tile(tf.expand_dims(pstate_est_temp, axis=2), [1, 1, self.num_state])
        X = tf.concat([tf.expand_dims(pstate_est_temp, axis=2), Y + Am, Y - Am], axis=2)
        X = tf.transpose(X, [0, 2, 1])

        x1, X1, P1, X2 = ut_state_batch(X, Wm, Wc, self.Qt, self.num_state, self.batch_size, At)
        z1, Z1, P2, Z2 = ut_meas(X1, Wm, Wc, self.Rt, self.meas_mat, self.batch_size)

        P12 = tf.matmul(tf.matmul(X2, tf.matrix_diag(Wc)), Z2, transpose_b=True)

        gain = tf.matmul(P12, tf.matrix_inverse(P2))
        pos_res2 = cur_meas_temp[:, :, tf.newaxis] - tf.matmul(self.meas_mat, x1)
        x = x1 + tf.matmul(gain, pos_res2)

        cov_est_t0 = P1 - tf.matmul(gain, P12, transpose_b=True)
        cov_est_t = (cov_est_t0 + tf.transpose(cov_est_t0, [0, 2, 1])) / 2

        final_state = x[:, :, 0]
        final_cov = cov_est_t
        # self.al.append(At)
        # self.al2.append(At2)
        # self.bl.append(Bt)
        # self.cl.append(self.meas_mat)
        # self.mul.append(pstate_est_temp)
        # self.mult.append(x[:, :, 0])
        # self.sigl.append(cov_est0)
        # self.siglt.append(cov_est_t0)
        # self.new_measl.append(cur_meas_temp)

        print('Completed UKF')

        return final_state, final_cov, self.state_fwf, states_fwso

    # def smooth(self, all_states, all_covs):
    #     # Smoothing
    #     j = [None] * self.max_seq
    #     all_states = tf.concat([tf.stack(self.prev_state2, axis=1), self.final_state[:, tf.newaxis, :]], axis=1)
    #     all_covs = tf.concat([tf.reshape(tf.stack(self.prev_covariance, axis=1), [self.batch_size, self.max_seq, self.num_state, self.num_state]), self.Ql[:, tf.newaxis, :, :]], axis=1)
    #
    #     xtemp = copy.copy(tf.unstack(tf.expand_dims(all_states, 3), axis=1))
    #     Ptemp = copy.copy(tf.unstack(all_covs, axis=1))
    #
    #     for q in range(self.max_seq - 2, -1, -1):
    #         if q >= 0:
    #             P_pred = tf.matmul(tf.matmul(At, Ptemp[q]), At, transpose_b=True) + self.Pl[q]
    #             j[q] = tf.matmul(tf.matmul(Ptemp[q], At, transpose_b=True), tf.matrix_inverse(P_pred))
    #             xtemp[q] += tf.matmul(j[q], xtemp[q + 1] - tf.matmul(At, xtemp[q]))
    #             Ptemp[q] += tf.matmul(tf.matmul(j[q], Ptemp[q + 1] - P_pred), j[q], transpose_b=True)
    #
    #     # self.final_state_update = tf.squeeze(tf.stack(xtemp, axis=1), -1)[:, :-1, :]
    #     self.final_state = tf.squeeze(xtemp[-1], -1)

    def estimate_covariance(self, final_state, final_cov, prev_state):

        all_covs_flat = tf.concat([tf.stack(self.prev_covariance, axis=1), tf.reshape(final_cov[:, tf.newaxis, :, :], [self.batch_size, 1, self.num_state ** 2])], axis=1)
        final_state_gs = tf.stop_gradient(final_state)

        all_states = tf.concat([tf.stack(prev_state, axis=1), final_state_gs[:, tf.newaxis, :]], axis=1)
        all_meas = tf.concat([tf.stack(self.prev_measurement, axis=1), self.measurement[:, tf.newaxis, :]], axis=1)
        all_time = tf.concat([tf.stack(self.prev_time, axis=1), self.current_time[:, tf.newaxis, :]], axis=1) / 2000

        meanv = tf.ones_like(all_states) * self.meanv
        all_states = all_states / meanv
        pos_m = tf.concat([meanv[:, :, 0, tf.newaxis], meanv[:, :, 4, tf.newaxis], meanv[:, :, 8, tf.newaxis]], axis=2)
        all_meas = all_meas / pos_m
        fsgs1 = final_state_gs / meanv[:, -1, :]
        pr0 = all_meas - tf.squeeze(tf.matmul(tf.tile(self.meas_mat[:, tf.newaxis, :, :], [1, self.max_seq + 1, 1, 1]), all_states[:, :, :, tf.newaxis]), -1)
        h = tf.concat([all_meas, all_states, all_covs_flat, pr0], axis=2)

        rnn_inp1 = FCL(h, self.F_hidden, activation_fn=None, scope='input/cov', reuse=tf.AUTO_REUSE)

        inp1 = tf.unstack(rnn_inp1, axis=1)
        inp1t = tf.unstack(all_time, axis=1)

        sf = list()
        for q in range(self.max_seq):
            temp_input = (inp1t[q], inp1[q])
            if q == 0:
                # hp = state_fw2_in[1] + noise_sample
                # state_fw2_in = tf.contrib.rnn.LSTMStateTuple(state_fw2_in[0], hp)
                with tf.variable_scope('Source_Track_Forward/cov'):
                    (self.source_track_out_fwc[q], self.state_fwc[q]) = self.source_fwc(temp_input, state=self.state_fw_in_state)
            else:
                # hp = self.state_fw2[q - 1][1] + noise_sample
                # self.state_fw2[q - 1] = tf.contrib.rnn.LSTMStateTuple(self.state_fw2[q - 1][0], hp)
                with tf.variable_scope('Source_Track_Forward/cov'):
                    tf.get_variable_scope().reuse_variables()
                    (self.source_track_out_fwc[q], self.state_fwc[q]) = self.source_fwc(temp_input, state=self.state_fwc[q - 1])
            sf.append(self.source_track_out_fwc[q])

        states_fwco = self.state_fwc[-1]
        rnn_output = tf.stack(sf, axis=1)
        both_states = tf.concat(states_fwco, axis=1)
        tmp = tf.tile(both_states[:, tf.newaxis, :], [1, self.max_seq, 1])
        out = tf.concat([rnn_output, tmp], axis=2)
        attended_inputs = multihead_attention(queries=out, keys=out, num_units=out.shape[2], num_heads=out.shape[2],
                                              dropout_rate=self.drop_rate, is_training=self.is_training,
                                              scope='attention1/cov', dtype=self.vdtype, reuse=tf.AUTO_REUSE)

        with tf.variable_scope('encoder1/cov', reuse=tf.AUTO_REUSE):
            stddev = 1.0 / (attended_inputs.shape[2].value * self.max_seq)
            Ue = tf.Variable(dtype=self.vdtype,
                             initial_value=tf.truncated_normal(shape=[attended_inputs.shape[2].value, 1],
                                                               mean=0.0, stddev=stddev, dtype=self.vdtype), name='Ue/cov')

        var = tf.tile(tf.expand_dims(Ue, 0), [self.batch_size, 1, 1])  # (b,T,T)
        weighted_output = tf.squeeze(tf.matmul(attended_inputs, var), -1) / self.max_seq

        n_cov = 6
        rnn_out30 = FCL(weighted_output, 12 + 36, activation_fn=None, scope='rnn_out30/cov', reuse=tf.AUTO_REUSE)

        inp1c2_pos = FCL(rnn_out30[:, :6], n_cov, activation_fn=None, scope='proj1/cov', reuse=tf.AUTO_REUSE)
        qmat_pos = tril_with_diag_softplus_and_shift(inp1c2_pos, diag_shift=0., name='r2cov/cov')

        inp1c2_vel = FCL(rnn_out30[:, 6:12], n_cov, activation_fn=None, scope='proj2/cov', reuse=tf.AUTO_REUSE)
        qmat_vel = tril_with_diag_softplus_and_shift(inp1c2_vel, diag_shift=0., name='r2cov2/cov')

        gain = FCL(rnn_out30[:, -36:, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, scope='n1v/cov', reuse=tf.AUTO_REUSE)
        gain = tf.reshape(gain, [self.batch_size, 12, 3])
        # act = tf.nn.sigmoid
        # pos1pv = FCL(rnn_out30[:, -6, tf.newaxis], 1, activation_fn=act, scope='n1v/cov', reuse=tf.AUTO_REUSE)  # * 5 / 200000
        # pos2pv = FCL(rnn_out30[:, -5, tf.newaxis], 1, activation_fn=act, scope='n2v/cov', reuse=tf.AUTO_REUSE)  # * 5 / 200000
        # pos3pv = FCL(rnn_out30[:, -4, tf.newaxis], 1, activation_fn=act, scope='n3v/cov', reuse=tf.AUTO_REUSE)  # * 5 / 200000
        # vel1pv = FCL(rnn_out30[:, -3, tf.newaxis], 1, activation_fn=act, scope='n4v/cov', reuse=tf.AUTO_REUSE)  # * 5 / 3000
        # vel2pv = FCL(rnn_out30[:, -2, tf.newaxis], 1, activation_fn=act, scope='n5v/cov', reuse=tf.AUTO_REUSE)  # * 5 / 3000
        # vel3pv = FCL(rnn_out30[:, -1, tf.newaxis], 1, activation_fn=act, scope='n6v/cov', reuse=tf.AUTO_REUSE)  # * 5 / 3000

        # oph1 = tf.zeros_like(vel1pv)
        # residual_estimate = tf.concat([pos1pv, vel1pv, oph1, oph1,
        #                                pos2pv, vel2pv, oph1, oph1,
        #                                pos3pv, vel3pv, oph1, oph1], axis=1)

        pr1 = self.measurement[:, :, tf.newaxis] - tf.matmul(self.meas_mat, final_state[:, :, tf.newaxis])
        final_state2 = tf.squeeze(fsgs1[:, :, tf.newaxis] * tf.matmul(gain, pr1), -1)

        estimate_cov_norm = final_state2 * meanv[:, -1, :]
        self.estimate_cov_norm_6 = tf.concat([estimate_cov_norm[:, 0, tf.newaxis], estimate_cov_norm[:, 1, tf.newaxis], estimate_cov_norm[:, 4, tf.newaxis],
                                              estimate_cov_norm[:, 5, tf.newaxis], estimate_cov_norm[:, 8, tf.newaxis], estimate_cov_norm[:, 9, tf.newaxis]], axis=1)

        self.estimate_cov_norm_pos = tf.concat([estimate_cov_norm[:, 0, tf.newaxis], estimate_cov_norm[:, 4, tf.newaxis], estimate_cov_norm[:, 8, tf.newaxis]], axis=1)
        self.estimate_cov_norm_vel = tf.concat([estimate_cov_norm[:, 1, tf.newaxis], estimate_cov_norm[:, 5, tf.newaxis], estimate_cov_norm[:, 9, tf.newaxis]], axis=1)

        q_dist_pos = tfp.distributions.MultivariateNormalTriL(loc=self.estimate_cov_norm_pos, scale_tril=qmat_pos)
        q_dist_vel = tfp.distributions.MultivariateNormalTriL(loc=self.estimate_cov_norm_vel, scale_tril=qmat_vel)

        final_state2 = final_state2 * meanv[:, -1, :]

        self.Ql4_pos = tf.cast(q_dist_pos.covariance(), tf.float64)
        self.Ql2_pos = qmat_pos

        self.Ql4_vel = tf.cast(q_dist_vel.covariance(), tf.float64)
        self.Ql2_vel = qmat_vel

        self.final_drnn2_statef = states_fwco

        return final_state2, states_fwco

    def discriminator(self, hidden_states, reuse=False):

        # for q in range(self.max_seq):
        #     if q == 0:
        #         # hp = state_fw2_in[1] + noise_sample
        #         # state_fw2_in = tf.contrib.rnn.LSTMStateTuple(state_fw2_in[0], hp)
        with tf.variable_scope('discriminator', reuse=reuse):
            (output_discrim, self.state_discrim) = self.discrim_cell(tf.concat(hidden_states, axis=1), state=self.state_discrim)
            # else:
            #     # hp = self.state_fw2[q - 1][1] + noise_sample
            #     # self.state_fw2[q - 1] = tf.contrib.rnn.LSTMStateTuple(self.state_fw2[q - 1][0], hp)
            #     with tf.variable_scope('discriminator'):
            #         tf.get_variable_scope().reuse_variables()
            #         (output_discrim, self.state_discrim) = self.discrim_cell(tf.concat(hidden_states, axis=1), state=self.state_discrim)

        layer1 = FCL(output_discrim, 128, activation_fn=tf.nn.relu, scope='l1/discriminator', reuse=tf.AUTO_REUSE)
        layer2 = FCL(layer1, 64, activation_fn=tf.nn.relu, scope='l2/discriminator', reuse=tf.AUTO_REUSE)
        logits = FCL(layer2, 1, activation_fn=None, scope='l3/discriminator', reuse=tf.AUTO_REUSE)

        return logits

    def build_loss(self, final_state, final_cov, final_state2):
        _Y000 = tf.cast(final_state2, self.vdtype)  # / meanv[:, 0, :]
        _Y00 = tf.cast(final_state, self.vdtype)  # / meanv[:, 0, :]
        # _Y0 = tf.cast(self.final_prop, self.vdtype) / meanv[:, 0, :]
        # _Y0 = tf.cast(s_est, self.vdtype) / meanv
        _y = self.truth_state  # / meanv[:, 0, :]

        _Y000 = tf.expand_dims(_Y000, axis=1)
        _Y00 = tf.expand_dims(_Y00, axis=1)
        _y = tf.expand_dims(_y, axis=1)

        # loss_func = msec
        loss_func = normed_mse

        pos_valst = tf.concat([_y[:, :, 0, tf.newaxis], _y[:, :, 4, tf.newaxis], _y[:, :, 8, tf.newaxis]], axis=2)
        vel_valst = tf.concat([_y[:, :, 1, tf.newaxis], _y[:, :, 5, tf.newaxis], _y[:, :, 9, tf.newaxis]], axis=2)
        acc_valst = tf.concat([_y[:, :, 2, tf.newaxis], _y[:, :, 6, tf.newaxis], _y[:, :, 10, tf.newaxis]], axis=2)
        jer_valst = tf.concat([_y[:, :, 3, tf.newaxis], _y[:, :, 7, tf.newaxis], _y[:, :, 11, tf.newaxis]], axis=2)

        pos_vals = tf.concat([_Y00[:, :, 0, tf.newaxis], _Y00[:, :, 4, tf.newaxis], _Y00[:, :, 8, tf.newaxis]], axis=2)
        vel_vals = tf.concat([_Y00[:, :, 1, tf.newaxis], _Y00[:, :, 5, tf.newaxis], _Y00[:, :, 9, tf.newaxis]], axis=2)
        acc_vals = tf.concat([_Y00[:, :, 2, tf.newaxis], _Y00[:, :, 6, tf.newaxis], _Y00[:, :, 10, tf.newaxis]], axis=2)
        jer_vals = tf.concat([_Y00[:, :, 3, tf.newaxis], _Y00[:, :, 7, tf.newaxis], _Y00[:, :, 11, tf.newaxis]], axis=2)

        pos_total = tf.reduce_mean(tf.sqrt(tf.square(pos_valst - pos_vals)), axis=1, keepdims=True)
        vel_total = tf.reduce_mean(tf.sqrt(tf.square(vel_valst - vel_vals)), axis=1, keepdims=True)
        acc_total = tf.reduce_mean(tf.sqrt(tf.square(acc_valst - acc_vals)), axis=1, keepdims=True)
        jer_total = tf.reduce_mean(tf.sqrt(tf.square(jer_valst - jer_vals)), axis=1, keepdims=True)

        total_weight = tf.cast(self.seqweightin, self.vdtype)

        print('Building Loss')
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

        state_loss_pos10 += loss_func(_y[:, :, 0], _Y000[:, :, 0], total_weight, pos_total)
        state_loss_pos20 += loss_func(_y[:, :, 4], _Y000[:, :, 4], total_weight, pos_total)
        state_loss_pos30 += loss_func(_y[:, :, 8], _Y000[:, :, 8], total_weight, pos_total)
        state_loss_vel10 += loss_func(_y[:, :, 1], _Y000[:, :, 1], total_weight, vel_total)
        state_loss_vel20 += loss_func(_y[:, :, 5], _Y000[:, :, 5], total_weight, vel_total)
        state_loss_vel30 += loss_func(_y[:, :, 9], _Y000[:, :, 9], total_weight, vel_total)
        state_loss_acc10 += loss_func(_y[:, :, 2], _Y000[:, :, 2], total_weight, acc_total)
        state_loss_acc20 += loss_func(_y[:, :, 6], _Y000[:, :, 6], total_weight, acc_total)
        state_loss_acc30 += loss_func(_y[:, :, 10], _Y000[:, :, 10], total_weight, acc_total)
        state_loss_j10 += loss_func(_y[:, :, 3], _Y000[:, :, 3], total_weight, jer_total)
        state_loss_j20 += loss_func(_y[:, :, 7], _Y000[:, :, 7], total_weight, jer_total)
        state_loss_j30 += loss_func(_y[:, :, 11], _Y000[:, :, 11], total_weight, jer_total)

        state_loss_pos100 += loss_func(_y[:, :, 0], _Y00[:, :, 0], total_weight, pos_total)
        state_loss_pos200 += loss_func(_y[:, :, 4], _Y00[:, :, 4], total_weight, pos_total)
        state_loss_pos300 += loss_func(_y[:, :, 8], _Y00[:, :, 8], total_weight, pos_total)
        state_loss_vel100 += loss_func(_y[:, :, 1], _Y00[:, :, 1], total_weight, vel_total)
        state_loss_vel200 += loss_func(_y[:, :, 5], _Y00[:, :, 5], total_weight, vel_total)
        state_loss_vel300 += loss_func(_y[:, :, 9], _Y00[:, :, 9], total_weight, vel_total)
        state_loss_acc100 += loss_func(_y[:, :, 2], _Y00[:, :, 2], total_weight, acc_total)
        state_loss_acc200 += loss_func(_y[:, :, 6], _Y00[:, :, 6], total_weight, acc_total)
        state_loss_acc300 += loss_func(_y[:, :, 10], _Y00[:, :, 10], total_weight, acc_total)
        state_loss_j100 += loss_func(_y[:, :, 3], _Y00[:, :, 3], total_weight, jer_total)
        state_loss_j200 += loss_func(_y[:, :, 7], _Y00[:, :, 7], total_weight, jer_total)
        state_loss_j300 += loss_func(_y[:, :, 11], _Y00[:, :, 11], total_weight, jer_total)

        print('Completed Loss')

        sweight = self.seqweightin[:, 0]

        C = 0
        MD0 = 0

        print('Building Covariance Loss')
        truth_cov_norm = copy.copy(_y)
        truth_cov_norm_6 = tf.concat([truth_cov_norm[:, :, 0, tf.newaxis], truth_cov_norm[:, :, 1, tf.newaxis], truth_cov_norm[:, :, 4, tf.newaxis],
                                      truth_cov_norm[:, :, 5, tf.newaxis], truth_cov_norm[:, :, 8, tf.newaxis], truth_cov_norm[:, :, 9, tf.newaxis]], axis=2)

        truth_cov_norm_6 = truth_cov_norm_6[:, 0, :]
        truth_cov_norm_pos = tf.concat([truth_cov_norm_6[:, 0, tf.newaxis], truth_cov_norm_6[:, 2, tf.newaxis], truth_cov_norm_6[:, 4, tf.newaxis]], axis=1)
        truth_cov_norm_vel = tf.concat([truth_cov_norm_6[:, 1, tf.newaxis], truth_cov_norm_6[:, 3, tf.newaxis], truth_cov_norm_6[:, 5, tf.newaxis]], axis=1)

        # z_current = tf.squeeze(tf.matmul(self.At, self.prev_state2[-1][:, :, tf.newaxis]), -1)

        meas_error = tf.squeeze(self.measurement[:, :, tf.newaxis] - tf.matmul(self.meas_mat, self.truth_state[:, :, tf.newaxis]), -1)

        emission_prob = tfp.distributions.MultivariateNormalTriL(loc=None, scale_tril=self.rd)
        # emission_prob = tfp.distributions.MultivariateNormalDiag(loc=None, scale_diag=rd)
        rlt = emission_prob.log_prob(meas_error)
        rl = tf.losses.compute_weighted_loss(tf.negative(rlt), weights=sweight)

        # trans_centered = self.final_state - z_current
        # mvn_transition = tfp.distributions.MultivariateNormalTriL(loc=None, scale_tril=tf.cholesky(self.Qt))
        # log_prob_transition = mvn_transition.log_prob(trans_centered)
        # qll0 = tf.losses.compute_weighted_loss(tf.negative(log_prob_transition), weights=sweight)

        # Entropy log(\prod_{t=1}^T p(z_t|y_{1:T}, u_{1:T}))
        # entropy = tf.reduce_sum(-mvn_smooth.log_prob(z_smooth))

        delta = self.estimate_cov_norm_6 - truth_cov_norm_6  # / pos_vel_m[:, -1, :]

        # delta_sq = tf.sqrt(tf.matmul(delta[:, :, tf.newaxis], delta[:, :, tf.newaxis], transpose_b=True))

        delta_12 = final_state - truth_cov_norm[:, 0, :]  # / meanv[:, -1, :]

        inv_cov_pos = tf.matrix_inverse(self.Ql4_pos)
        inv_cov_vel = tf.matrix_inverse(self.Ql4_vel)

        train_cov0 = tfp.distributions.MultivariateNormalFullCovariance(loc=None, covariance_matrix=final_cov)

        train_covp = tfp.distributions.MultivariateNormalTriL(loc=self.estimate_cov_norm_pos, scale_tril=self.Ql2_pos)
        train_covv = tfp.distributions.MultivariateNormalTriL(loc=self.estimate_cov_norm_vel, scale_tril=self.Ql2_vel)
        # train_cov2 = tfp.distributions.MultivariateNormalTriL(loc=None, scale_tril=self.Ql2)

        error_loss0 = train_cov0.log_prob(delta_12)
        error_lossp = train_covp.log_prob(truth_cov_norm_pos)
        error_lossv = train_covv.log_prob(truth_cov_norm_vel)
        # error_loss2 = train_cov2.log_prob(delta)

        error_loss0 = tf.where(tf.is_nan(error_loss0), tf.ones_like(error_loss0) * -9999, error_loss0)
        error_lossp = tf.where(tf.is_nan(error_lossp), tf.ones_like(error_lossp) * -9999, error_lossp)
        error_lossv = tf.where(tf.is_nan(error_lossv), tf.ones_like(error_lossv) * -9999, error_lossv)

        C += tf.losses.compute_weighted_loss(tf.negative(error_lossp), weights=sweight)
        C += tf.losses.compute_weighted_loss(tf.negative(error_lossv), weights=sweight)

        qll = tf.losses.compute_weighted_loss(tf.negative(error_loss0), weights=sweight)

        delta_pos = tf.reduce_mean(tf.transpose(train_covp.sample((1000)), [1, 0, 2]) - tf.tile(truth_cov_norm_pos[:, tf.newaxis, :], [1, 1000, 1]), axis=1)
        delta_vel = tf.reduce_mean(tf.transpose(train_covv.sample((1000)), [1, 0, 2]) - tf.tile(truth_cov_norm_vel[:, tf.newaxis, :], [1, 1000, 1]), axis=1)

        # sweight2 = tf.ones_like(delta_pos) * sweight[:, tf.newaxis, tf.newaxis]
        delta2 = tf.expand_dims(delta_pos, 2)
        M1 = tf.matmul(delta2, inv_cov_pos, transpose_a=True)
        M2 = tf.sqrt(tf.square(tf.matmul(M1, delta2)))
        MD = tf.squeeze(tf.sqrt(M2 / 3))
        MD0 += tf.reduce_sum(tf.losses.compute_weighted_loss(MD, weights=sweight))

        delta2 = tf.expand_dims(delta_vel, 2)
        M1 = tf.matmul(delta2, inv_cov_vel, transpose_a=True)
        M2 = tf.sqrt(tf.square(tf.matmul(M1, delta2)))
        MD = tf.squeeze(tf.sqrt(M2 / 3))
        # MD0 += tf.losses.huber_loss(tf.ones_like(MD) / 3, MD, weights=sweight)
        MD0 += tf.reduce_sum(tf.losses.compute_weighted_loss(MD, weights=sweight))

        # Build output covariance
        zmat = tf.zeros_like(self.Ql4_pos)
        self.cov_out = tf.concat([tf.concat([self.Ql4_pos, zmat], axis=2), tf.concat([zmat, self.Ql4_vel], axis=2)], axis=1)
        trace = tf.reduce_sum(tf.sqrt(tf.pow(tf.matrix_diag_part(self.Ql4_pos), 2)))  # * tf.tile(tf.cast(total_weight[:, q, tf.newaxis, tf.newaxis], self.vdtype), [1, 12, 12]))

        # ods = 0
        #
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 1, 0], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 2, 0], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 3, 0], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 4, 0], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 5, 0], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 2, 1], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 3, 1], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 4, 1], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 5, 1], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 3, 2], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 4, 2], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 5, 2], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 4, 3], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 5, 3], 2)))
        # ods += tf.reduce_sum(tf.sqrt(tf.pow(self.Ql4[:, 5, 4], 2)))
        #
        # TL += ods

        print('Completed Covariance Loss')
        self.nllo = tf.cast(rl, self.vdtype)
        self.sout = self.Ql4
        dlist = delta

        # self.truth_future2 = tf.stack(self.truth_future, axis=1)
        # self.all_truth = tf.concat([self.truth, self.truth_future2], 1)

        _cov_loss = tf.cast(C, tf.float64)

        SLPf = state_loss_pos10 + state_loss_pos20 + state_loss_pos30
        SLVf = state_loss_vel10 + state_loss_vel20 + state_loss_vel30
        SLAf = state_loss_acc10 + state_loss_acc20 + state_loss_acc30
        SLJf = state_loss_j10 + state_loss_j20 + state_loss_j30

        # SLPf0 = state_loss_pos100 + state_loss_pos200 + state_loss_pos300
        # SLVf0 = tf.log(state_loss_vel100 + state_loss_vel200 + state_loss_vel300)
        # SLAf0 = tf.log(state_loss_acc100 + state_loss_acc200 + state_loss_acc300)
        # SLJf0 = tf.log(state_loss_j100 + state_loss_j200 + state_loss_j300)

        # rmse_acc2 = tf.minimum(5 * rmse_pos, rmse_acc)
        # rmse_jer2 = tf.minimum(5 * rmse_pos, rmse_jer)

        # SLPf00 = state_loss_pos1000 + state_loss_pos2000 + state_loss_pos3000
        # SLVf00 = state_loss_vel1000 + state_loss_vel2000 + state_loss_vel3000
        # SLAf00 = state_loss_acc1000 + state_loss_acc2000 + state_loss_acc3000
        # SLJf00 = state_loss_j1000 + state_loss_j2000 + state_loss_j3000

        filter_covariance_loss = qll
        filter_meas_cov_loss = rl

        self.rmse_pos = SLPf
        self.rmse_vel = SLVf
        self.rmse_acc = SLAf
        self.rmse_jer = SLJf
        self.covariance_loss = tf.reduce_mean(C)
        self.maha_loss = tf.reduce_mean(MD)
        self.maha_out = tf.reduce_mean(MD)
        self.trace_loss = tf.reduce_mean(trace)
        self.dout = dlist
        self.saver = tf.train.Saver()

        return SLPf, SLVf, SLAf, SLJf, filter_covariance_loss, filter_meas_cov_loss, _cov_loss

    def build_model(self):
        self.DROPOUT = tf.placeholder(tf.float64)
        self.seqlen = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.shape(self.seqlen)[0]
        self.maneuverin = [tf.placeholder(tf.int32, shape=(None, 1), name="maneuver".format(t)) for t in range(self.max_seq)]
        self.update_condition = tf.placeholder(tf.bool, name='update_condition')
        self.meanv = tf.placeholder(tf.float64, shape=(1, self.num_state), name='meanv')
        self.stdv = tf.placeholder(tf.float64, shape=(1, self.num_state), name='stdv')

        self.grad_clip = tf.placeholder(self.vdtype, name='grad_clip')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.measurement = tf.placeholder(tf.float64, shape=(None, self.num_meas), name="meas_uvw")

        self.sensor_ecef = tf.placeholder(tf.float64, shape=(None, self.num_meas), name='sen_ecef')
        self.sensor_lla = tf.placeholder(tf.float64, shape=(None, self.num_meas), name='sen_lla')

        self.prev_measurement = [tf.placeholder(tf.float64, shape=(None, self.num_meas), name="px_{}".format(t)) for t in range(self.max_seq)]
        self.prev_covariance = [tf.placeholder(tf.float64, shape=(None, self.num_state ** 2), name="pcov_{}".format(t)) for t in range(self.max_seq)]
        self.prev_time = [tf.placeholder(tf.float64, shape=(None, 1), name="ptime".format(t)) for t in range(self.max_seq)]
        self.prev_truth = [tf.placeholder(tf.float64, shape=(None, self.num_state), name="ptruth_{}".format(t)) for t in range(self.max_seq)]
        self.prev_state2 = [tf.placeholder(tf.float64, shape=(None, self.num_state), name="py2_{}".format(t)) for t in range(self.max_seq)]
        self.prev_state3 = [tf.placeholder(tf.float64, shape=(None, self.num_state), name="py3_{}".format(t)) for t in range(self.max_seq)]

        self.current_time = tf.placeholder(tf.float64, shape=(None, 1), name="current_time")
        self.P_inp = tf.placeholder(tf.float64, shape=(None, self.num_state ** 2), name="yc")
        self.truth_state = tf.placeholder(tf.float64, shape=(None, self.num_state), name="y_truth")
        self.seqweightin = tf.placeholder(tf.float64, [None, 1])
        self.rho_ref = tf.placeholder(dtype=tf.float32, shape=[], name='rho_ref')

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
                self.source_fwc = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True, input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
            else:
                self.source_fwc = cell_type(self.F_hidden)

        with tf.variable_scope('Source_Track_Backward/state'):
            if use_dropout:
                self.source_bwc = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True, input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
            else:
                self.source_bwc = cell_type(self.F_hidden)

        with tf.variable_scope('Source_Track_Forward/cov'):
            if use_dropout:
                self.source_fws = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True, input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
            else:
                self.source_fws = cell_type(self.F_hidden)

        with tf.variable_scope('Source_Track_Backward/cov'):
            if use_dropout:
                self.source_bws = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True, input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
            else:
                self.source_bws = cell_type(self.F_hidden)

        with tf.variable_scope('Source_Track_Forward3/measr'):
            if use_dropout:
                self.source_fw_filter = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True, input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
            else:
                self.source_fw_filter = cell_type(self.F_hidden)

        with tf.variable_scope('discriminator'):
            if use_dropout:
                self.discrim_cell = tfc.rnn.DropoutWrapper(tfc.rnn.LSTMCell(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True, input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
            else:
                self.discrim_cell = tfc.rnn.LSTMCell(self.F_hidden)

        if self.train_init_state is False:

            if self.state_type != 'GRU':
                self.init_c_fws = tf.placeholder(name='init_c_fw/state', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.init_h_fws = tf.placeholder(name='init_h_fw/state', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_fw_in_state = tf.contrib.rnn.LSTMStateTuple(self.init_c_fws, self.init_h_fws)

                self.init_c_bws = tf.placeholder(name='init_c_bw/state', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.init_h_bws = tf.placeholder(name='init_h_bw/state', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_bw_in_state = tf.contrib.rnn.LSTMStateTuple(self.init_c_bws, self.init_h_bws)

                self.init_c_fwc = tf.placeholder(name='init_c_fw/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.init_h_fwc = tf.placeholder(name='init_h_fw/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_fw_in_cov = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwc, self.init_h_fwc)

                self.init_c_bwc = tf.placeholder(name='init_c_bw/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.init_h_bwc = tf.placeholder(name='init_h_bw/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_bw_in_cov = tf.contrib.rnn.LSTMStateTuple(self.init_c_bwc, self.init_h_bwc)

                self.init_c_fw3 = tf.placeholder(name='init_c_fw3/measr', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.init_h_fw3 = tf.placeholder(name='init_h_fw3/measr', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_fw_in_filter = tf.contrib.rnn.LSTMStateTuple(self.init_c_fw3, self.init_h_fw3)

                self.init_c_d = tf.placeholder(name='init_c/discriminator', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.init_h_d = tf.placeholder(name='init_h/discriminator', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_discrim = tf.contrib.rnn.LSTMStateTuple(self.init_c_d, self.init_h_d)

            else:
                self.init_c_fws = tf.placeholder(name='init_c_fw/state', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_fw_in_state = self.init_c_fws

                self.init_c_bws = tf.placeholder(name='init_c_bw/state', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_bw_in_state = self.init_c_bws

                self.init_c_fwc = tf.placeholder(name='init_c_fw/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_fw_in_cov = self.init_c_fwc

                self.init_c_bwc = tf.placeholder(name='init_c_bw/cov', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_bw_in_cov = self.init_c_bwc

                self.init_c_fw3 = tf.placeholder(name='init_c_fw3/measr', shape=[None, self.F_hidden * 4], dtype=self.vdtype)
                self.state_fw_in_filter = self.init_c_fw3

        self.I_4z = tf.scalar_mul(0.0, tf.eye(4, batch_shape=[self.batch_size], dtype=tf.float64))
        self.I_6z = tf.scalar_mul(0.0, tf.eye(6, batch_shape=[self.batch_size], dtype=tf.float64))
        self.I_3z = tf.scalar_mul(0.0, tf.eye(3, batch_shape=[self.batch_size], dtype=tf.float64))
        self.zb = tf.zeros([self.batch_size, 4, 2], dtype=tf.float64)
        self.om = tf.ones([self.batch_size, 1, 1], dtype=tf.float64)
        self.zm = tf.zeros([self.batch_size, 1, 1], dtype=tf.float64)
        omp = np.ones([1, 1], self.vdp_np)
        zmp = np.zeros([1, 1], self.vdp_np)

        m1 = np.concatenate([omp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp], axis=1).astype(np.float64)
        m2 = np.concatenate([zmp, zmp, zmp, zmp, omp, zmp, zmp, zmp, zmp, zmp, zmp, zmp], axis=1).astype(np.float64)
        m3 = np.concatenate([zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, omp, zmp, zmp, zmp], axis=1).astype(np.float64)
        self.meas_mat = tf.tile(tf.expand_dims(tf.concat([m1, m2, m3], axis=0), axis=0), [self.batch_size, 1, 1])

        final_state_truth, final_cov_truth, hidden_states_truth1, hidden_states_truth2 = self.filter_measurement(self.prev_truth)
        final_state2_truth, hidden_states_truth3 = self.estimate_covariance(final_state_truth, final_cov_truth, self.prev_truth)

        final_state, final_cov, hidden_states1, hidden_states2 = self.filter_measurement(self.prev_state2)
        final_state2, hidden_states3 = self.estimate_covariance(final_state, final_cov, self.prev_state2)

        hidden_truth = tf.concat([tf.concat([hidden_states_truth1], axis=0), tf.concat([hidden_states_truth2], axis=0), tf.concat([hidden_states_truth3], axis=0)], axis=2)
        hidden_truth = tf.reshape(hidden_truth, [self.batch_size, hidden_truth.shape[0]*hidden_truth.shape[2]])

        hidden_real = tf.concat([tf.concat([hidden_states1], axis=0), tf.concat([hidden_states2], axis=0), tf.concat([hidden_states3], axis=0)], axis=2)
        hidden_real = tf.reshape(hidden_real, [self.batch_size, hidden_real.shape[0] * hidden_real.shape[2]])

        fake_logits = self.discriminator(hidden_truth, reuse=False)
        real_logits = self.discriminator(hidden_real, reuse=True)

        discrim_loss = discriminator_loss(real_logits, fake_logits)

        generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logits), logits=fake_logits))

        self.final_state = final_state
        self.final_state2 = final_state2
        self.final_state2_truth = final_state2_truth
        self.final_cov = final_cov
        self.discrim_loss = discrim_loss
        self.generator_loss = generator_loss

        print('Building Loss')
        SLPf1, SLVf1, SLAf1, SLJf1, filter_covariance_loss1, filter_meas_cov_loss1, _cov_loss1 = self.build_loss(final_state, final_cov, final_state2)

        SLPf2, SLVf2, SLAf2, SLJf2, filter_covariance_loss2, filter_meas_cov_loss2, _cov_loss2 = self.build_loss(final_state, final_cov, final_state2)

        SLPf = SLPf1 + SLPf2
        SLVf = SLVf1 + SLVf2
        SLAf = SLAf1 + SLAf2
        SLJf = SLJf1 + SLJf2
        filter_covariance_loss = filter_covariance_loss1 + filter_covariance_loss2
        filter_meas_cov_loss = filter_meas_cov_loss1 + filter_meas_cov_loss2
        _cov_loss = _cov_loss1 + _cov_loss2

        self.learning_rate = tf.train.exponential_decay(self.learning_rate_inp, global_step=self.global_step, decay_steps=200000, decay_rate=0.8, staircase=True)
        # int(5 * (1500 / self.max_seq) * (500 / self.batch_size))
        max_lr = self.learning_rate
        base_lr = max_lr / 4
        step_size = int(1500 * 1.5)
        stepi = tf.cast(self.global_step, self.vdtype)
        cycle = tf.floor(1 + stepi / (2 * step_size))
        xi = tf.abs(stepi / step_size - 2 * cycle + 1)
        self.learning_rate = base_lr + (max_lr - base_lr) * tf.maximum(tf.cast(0., self.vdtype), 1. - xi)

        all_vars = tf.trainable_variables()
        discrim_vars = [var for var in all_vars if 'discriminator' in var.name]
        cov_vars = [var for var in all_vars if 'cov' in var.name]
        state_vars = [var for var in all_vars if 'state' in var.name]
        not_d_vars = [var for var in all_vars if 'discriminator' not in var.name]

        with tf.variable_scope("TrainOps"):
            print('cov_update gradients...')
            self.train_g1 = tfc.layers.optimize_loss(loss=filter_covariance_loss1 + filter_meas_cov_loss1 + generator_loss1,
                                                    global_step=self.global_step,
                                                    learning_rate=self.learning_rate,
                                                    optimizer=tfc.opt.AdamWOptimizer(1e-3, name='r1'),
                                                    clip_gradients=1.0,
                                                    variables=not_d_vars,
                                                    name='qll')

            self.train_g2 = tfc.layers.optimize_loss(loss=filter_covariance_loss1 + filter_meas_cov_loss1 + _cov_loss1 + generator_loss1,
                                                    global_step=self.global_step,
                                                    learning_rate=self.learning_rate,
                                                    optimizer=tfc.opt.AdamWOptimizer(1e-3, name='r2'),
                                                    clip_gradients=1.0,
                                                    variables=not_d_vars,
                                                    name='qll_cov')

            self.train_g3 = tfc.layers.optimize_loss(loss=filter_covariance_loss1 + _cov_loss1 + filter_meas_cov_loss1 + SLPf1 + SLVf1 + generator_loss1,
                                                    global_step=self.global_step,
                                                    learning_rate=self.learning_rate,
                                                    optimizer=tfc.opt.AdamWOptimizer(1e-3, name='r3'),
                                                    clip_gradients=1.0,
                                                    variables=not_d_vars,
                                                    name='all_updates')

            self.train_discrim = tfc.layers.optimize_loss(loss=discrim_loss,
                                                    global_step=self.global_step,
                                                    learning_rate=self.learning_rate,
                                                    optimizer=tfc.opt.AdamWOptimizer(1e-3, name='r4'),
                                                    clip_gradients=1.0,
                                                    variables=discrim_vars,
                                                    name='d_updates')

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Total traininable network parameters:: ' + str(total_parameters))

    def train(self):

        rho0 = 1.22  # kg / m**3
        k0 = 0.14141e-3
        area = 0.25  # / FLAGS.RE  # meter squared
        cd = 0.03  # unitless
        gmn = FLAGS.GM / (FLAGS.RE ** 3)
        
        # initialize all variables
        tf.global_variables_initializer().run()

        try:
            save_files = os.listdir(self.save_dir + '/')
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
        if start_epoch != 0 and step != 0:
            print('Loading filter...')
            try:
                imported_meta = tf.train.import_meta_graph(FLAGS.save_dir + '/' + filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step) + '.meta')
                imported_meta.restore(self.sess, tf.train.latest_checkpoint(r'D:/Checkpoints/' + filter_name + '/'))
                print("filter restored.")
            except:
                start_epoch = 0
                step = 0
                print("Could not restore filter")

        e = int(start_epoch)

        ds = DataServerLive(self.meas_dir, self.state_dir)
        # sensor_locations = ds.sensor_list
        # sensor_nums = list(range(0, len(sensor_locations)))
        # test_list = [19]
        # sampled = list()

        # BD = OrderedDict.fromkeys(batch_list0)
        plot_count = 0
        lr = 0.01
        for epoch in range(int(start_epoch), 9999):

            if (epoch % 25 == 0 or self.mode == 'testing') and epoch != 0:
                testing = True
                print('Testing filter for epoch ' + str(epoch))
            else:
                testing = False
                print('Training filter for epoch ' + str(epoch))

            # Data is unnormalized at this point
            x_train, y_train, batch_number, total_batches, ecef_ref, lla_data = ds.load(batch_size=FLAGS.batch_size, constant=False, test=testing, max_seq_len=3500, HZ=25)

            print(str(ds._index_in_epoch))

            x_train = np.concatenate([x_train[:, :, 0, np.newaxis], x_train[:, :, -3:]], axis=2)  # uvw measurements

            y_uvw = y_train[:, :, :3] - np.ones_like(y_train[:, :, :3]) * ecef_ref[:, np.newaxis, :]
            zero_rows = (y_train[:, :, :3] == 0).all(2)
            for i in range(y_train.shape[0]):
                zz = zero_rows[i, :, np.newaxis]
                y_uvw[i, :, :] = np.where(zz, np.zeros_like(y_uvw[i, :, :]), y_uvw[i, :, :])

            y_train = np.concatenate([y_uvw, y_train[:, :, 3:]], axis=2)

            # x_train[:, :, 1:] = x_train[:, :, 1:] / 6378137
            # y_train = normalize_statenp(y_train)
            # y_train = y_train / 6378137
            # _, _, _, _, mean_y, std_y = normalize_statenp(copy.copy(x_train[:, :, :4]), copy.copy(y_uvw))

            max_pos = 200000
            max_vel = 3000
            max_acc = 200
            max_jer = 200

            mean_y = np.array([max_pos, max_pos, max_pos,
                               max_vel, max_vel, max_vel,
                               max_acc, max_acc, max_acc,
                               max_jer, max_jer, max_jer])

            mean_y2 = np.array([max_pos, max_pos, max_pos,
                                max_vel, max_vel, max_vel])

            # pos range = -1.1: 1.1 RE
            # vel range = -0.00157 : 0.00157 RE
            # acc range = -3.136e-5 : 3.136e-5 RE
            # jer range = -3.136e-5 : 3.136e-5 RE
            # idxi = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

            # BD.update({batch_number: FND})

            # x_train = x_train[:50]
            # y_train = y_train[:50]
            # s_train = s_train[:50]

            # shuf = np.arange(x_train.shape[0])
            # np.random.shuffle(shuf)
            # x_train = x_train[shuf]
            # y_train = y_train[shuf]
            s_train = x_train

            n_train_batches = int(x_train.shape[0] / FLAGS.batch_size)
            print("Batch Number: {0:2d} out of {1:2d}".format(batch_number, total_batches))
            for minibatch_index in range(n_train_batches):

                # x0n, y0n, _, prev_y0n, prev_x0n, _, _ = prepare_batch(minibatch_index, x_trainn, y_trainn, s_train, batch_size=FLAGS.batch_size, new_batch=True)
                x0, y0, meta0, prev_y, prev_x, prev_time, prev_meta, initial_meas, initial_time = prepare_batch(minibatch_index, x_train, y_train, s_train, seq_len=FLAGS.max_seq, batch_size=FLAGS.batch_size, new_batch=True)

                count, _, _, _, _, _, prev_cov, q_plot, q_plott, k_plot, out_plot_X, out_plot_F, out_plot_P, time_vals, meas_plot, truth_plot, Q_plot, R_plot, maha_plot, x, y, meta = \
                    initialize_run_variables(FLAGS.batch_size, FLAGS.max_seq, FLAGS.num_state, x0, y0, meta0)

                print('Resetting Feed Dict')
                feed_dict = {}
                time_plotter = np.zeros([FLAGS.batch_size, int(x.shape[1]), 1])
                batch_loss = 0.0
                mstep = x.shape[1] - FLAGS.max_seq

                for tstep in range(0, mstep):

                    if tstep == 0:
                        prev_state = copy.copy(prev_y)
                        prev_meas = copy.copy(prev_x)

                    current_x, current_y, current_time, current_meta = \
                        get_feed_time_asynch(x, y, meta, prev_x, prev_y, prev_time, prev_meta, FLAGS.max_seq, tstep, FLAGS.num_state)

                    seqlen = np.ones(shape=[FLAGS.batch_size, ])
                    seqweight = np.zeros(shape=[FLAGS.batch_size, 1])
                    for i in range(FLAGS.batch_size):
                        current_yt = current_y[i, :, :3]
                        m = ~(current_yt == 0).all(1)
                        yf = current_yt[m]
                        seq = yf.shape[0]
                        seqlen[i] = seq
                        seqweight[i, :] = m.astype(int)

                    cur_time = x[:, tstep, 0]

                    time_plotter[:, tstep, :] = cur_time[:, np.newaxis]
                    max_t = np.max(time_plotter[0, :, 0])
                    count += 1
                    step += 1

                    if tstep == 0:
                        pos = initial_meas[:, 2, :]
                        vel = (initial_meas[:, 2, :] - initial_meas[:, 0, :]) / np.sum(np.diff(initial_time, axis=1), axis=1)

                        R1 = np.linalg.norm(initial_meas + ecef_ref[:, np.newaxis, :], axis=2, keepdims=True)
                        R1 = np.mean(R1, axis=1)
                        R1 = np.where(np.less(R1, np.ones_like(R1) * 6378137), np.ones_like(R1) * 6378137, R1)
                        rad_temp = np.power(R1, 3)
                        GMt1 = np.divide(FLAGS.GM, rad_temp)
                        acc = get_legendre_np(GMt1, pos + ecef_ref, R1)
                        initial_state = np.expand_dims(np.concatenate([pos, vel, acc, np.zeros_like(acc)], axis=1), 1)

                        prev_state2, prev_covariance = unscented_kalman_np(FLAGS.batch_size, FLAGS.max_seq, initial_state[:, 0, :], prev_cov[:, -1, :, :], prev_meas, prev_time)

                        current_covariance = prev_covariance[-1]
                        prev_state3 = copy.copy(prev_state2)

                    update = False

                    idxi = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
                    idxi2 = [0, 3, 1, 4, 2, 5]
                    mean_y = mean_y[idxi]
                    mean_y2 = mean_y2[idxi2]
                    # std_y = std_y[idxi]
                    prev_y = prev_y[:, :, idxi]
                    current_y = current_y[:, :, idxi]
                    prev_state = prev_state[:, :, idxi]
                    prev_state2 = prev_state2[:, :, idxi]
                    prev_state3 = prev_state3[:, :, idxi]

                    feed_dict.update({self.measurement: current_x[:, 0, :].reshape(-1, FLAGS.num_meas)})
                    feed_dict.update({self.prev_measurement[t]: prev_x[:, t, :].reshape(-1, FLAGS.num_meas) for t in range(FLAGS.max_seq)})
                    feed_dict.update({self.prev_covariance[t]: prev_cov[:, t, :].reshape(-1, FLAGS.num_state ** 2) for t in range(FLAGS.max_seq)})
                    feed_dict.update({self.truth_state: current_y[:, 0, :].reshape(-1, FLAGS.num_state)})
                    feed_dict.update({self.prev_truth[t]: prev_y[:, t, :].reshape(-1, FLAGS.num_state) for t in range(FLAGS.max_seq)})
                    feed_dict.update({self.prev_state2[t]: prev_state2[:, t, :].reshape(-1, FLAGS.num_state) for t in range(FLAGS.max_seq)})
                    feed_dict.update({self.prev_state3[t]: prev_state3[:, t, :].reshape(-1, FLAGS.num_state) for t in range(FLAGS.max_seq)})
                    feed_dict.update({self.sensor_ecef: ecef_ref})
                    feed_dict.update({self.sensor_lla: lla_data})
                    feed_dict.update({self.seqlen: seqlen})
                    feed_dict.update({self.update_condition: update})
                    feed_dict.update({self.is_training: True})
                    feed_dict.update({self.meanv: mean_y[np.newaxis, :]})
                    # feed_dict.update({self.stdv: std_y[np.newaxis, :]})
                    feed_dict.update({self.seqweightin: seqweight})
                    # feed_dict.update({self.maneuverin[t]: prev_meta[:, t, :].reshape(-1, 1) for t in range(FLAGS.max_seq)})
                    feed_dict.update({self.P_inp: current_covariance.reshape(-1, FLAGS.num_state ** 2)})
                    feed_dict.update({self.prev_time[t]: prev_time[:, t, 0].reshape(-1, 1) for t in range(FLAGS.max_seq)})
                    feed_dict.update({self.current_time: current_time[:, 0, :].reshape(-1, 1)})
                    feed_dict.update({self.batch_step: tstep})
                    feed_dict.update({self.drop_rate: 0.5})

                    if tstep == 0:
                        print("Resetting LSTM States")
                        if testing is True:
                            std = 0.0
                        else:
                            std = 0.3

                        feed_dict.update({self.init_c_fw3: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_h_fw3: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_c_fws: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_h_fws: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_c_fws: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_h_fwc: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_c_d: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_h_d: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})

                    if testing is False:
                        if e < 5:
                            train_op = self.train_1
                            stateful = False
                            iters = 2
                        elif e >= 5 and e < 20:
                            lr = 1e-3
                            stateful = True
                            train_op = self.train_2
                            iters = 2
                        else:
                            lr = 1e-4
                            stateful = True
                            train_op = self.train_3
                            if e < 100:
                                iters = 5
                            else:
                                iters = 1

                        feed_dict.update({self.learning_rate_inp: lr})

                        d_loss, _ = self.sess.run([self.discrim_loss, self.train_discrim], feed_dict)

                        pred_output0, pred_output00, pred_output1, q_out, q_out_t, _, rmsp, rmsv, rmsa, rmsj, LR, \
                        cov_loss, maha_loss, MD, trace_loss, \
                        dout, rdout, gen_loss, nllo, qt_out, rt_out, drnn1f, drnn2f, new_state_fw3 = \
                            self.sess.run([self.final_state,
                                      self.final_state2_truth,
                                      self.final_state2,
                                      self.cov_out,
                                      self.final_cov,
                                      train_op,
                                      self.rmse_pos,
                                      self.rmse_vel,
                                      self.rmse_acc,
                                      self.rmse_jer,
                                      self.learning_rate,
                                      self.covariance_loss,
                                      self.maha_loss,
                                      self.maha_out,
                                      self.trace_loss,
                                      self.dout,
                                      self.rd,
                                      self.generator_loss,
                                      self.nllo,
                                      self.Qt,
                                      self.Rt,
                                      self.final_drnn1_statef,
                                      self.final_drnn2_statef,
                                      self.state_fwf],
                                     feed_dict)
                    else:
                        feed_dict.update({self.is_training: False})
                        feed_dict.update({self.deterministic: True})
                        feed_dict.update({self.drop_rate: 1.0})
                        stateful = True
                        pred_output0, pred_output00, pred_output1, q_out, q_out_t, rmsp, rmsv, rmsa, rmsj, LR, \
                        cov_loss, maha_loss, MD, trace_loss, dout, rdout, gen_loss, nllo, qt_out, rt_out, \
                        drnn1f, drnn2f, new_state_fw3 = \
                            self.sess.run([self.final_state,
                                      self.final_state2_truth,
                                      self.final_state2,
                                      self.cov_out,
                                      self.final_cov,
                                      self.rmse_pos,
                                      self.rmse_vel,
                                      self.rmse_acc,
                                      self.rmse_jer,
                                      self.learning_rate,
                                      self.covariance_loss,
                                      self.maha_loss,
                                      self.maha_out,
                                      self.trace_loss,
                                      self.dout,
                                      self.rd,
                                      self.generator_loss,
                                      self.nllo,
                                      self.Qt,
                                      self.Rt,
                                      self.final_drnn1_statef,
                                      self.final_drnn2_statef,
                                      self.state_fwf],
                                     feed_dict)

                    # q_out = q_out * cov_scale[np.newaxis, np.newaxis, :, :]
                    batch_loss += (rmsp + rmsv + rmsa)

                    q_truth = q_out_t
                    current_covariance = q_out_t
                    idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
                    mean_y = mean_y[idxo]
                    # std_y = std_y[idxo]
                    prev_y = prev_y[:, :, idxo]
                    current_y = current_y[:, :, idxo]
                    prev_state = prev_state[:, :, idxo]
                    prev_state2 = prev_state2[:, :, idxo]
                    prev_state3 = prev_state3[:, :, idxo]
                    # prev_state = prev_state[:, :, idxo]

                    pred_output0 = pred_output0[:, idxo]
                    pred_output00 = pred_output00[:, idxo]
                    pred_output1 = pred_output1[:, idxo]

                    # a = pred_output0[0, :]
                    # pred_output0 = (pred_output0 * std_y) + mean_y
                    # pred_output00 = (pred_output00 * std_y) + mean_y
                    # pred_output1 = (pred_output1 * std_y) + mean_y

                    # if 1 == 0 and tstep % 10 == 0:
                    #     if not os.path.isdir('./covariance_output/' + str(epoch)):
                    #         os.mkdir('./covariance_output/' + str(epoch))
                    #
                    #     # csamp = sample_out[0, -1, :, :].T
                    #     # ssamp = sout[0, -1, :, :].T
                    #     ssamp = q_out[0, -1, :, :] * cov_scale
                    #     # oo = np.concatenate([csamp, ssamp], axis=1)
                    #     oo = pd.DataFrame(ssamp)
                    #     oo.to_csv('./covariance_output/' + str(epoch) + '/' + str(tstep) + '.csv', float_format='%g')

                    if stateful is True:
                        if self.state_type != 'GRU':
                            feed_dict.update({self.init_c_fw3: new_state_fw3[0]})
                            feed_dict.update({self.init_h_fw3: new_state_fw3[1]})
                            feed_dict.update({self.init_c_fws: drnn1f[0]})
                            feed_dict.update({self.init_h_fws: drnn1f[1]})
                            feed_dict.update({self.init_c_fwc: drnn2f[0]})
                            feed_dict.update({self.init_h_fwc: drnn2f[1]})
                            feed_dict.update({self.init_c_d: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, 0.3)})
                            feed_dict.update({self.init_h_d: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, 0.3)})
                        else:
                            feed_dict.update({self.init_c_fw3: new_state_fw3})
                            feed_dict.update({self.init_c_fws: drnn1f[0]})
                            feed_dict.update({self.init_c_fwc: drnn2f[0]})

                    else:
                        if testing is True:
                            std = 0.0
                        else:
                            std = 0.3
                        feed_dict.update({self.init_c_fw3: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_h_fw3: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_c_fws: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_h_fws: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_c_fwc: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_h_fwc: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_c_d: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})
                        feed_dict.update({self.init_h_d: get_zero_state(1, FLAGS.F_hidden, FLAGS.batch_size, 4, std)})

                    prop_output = np.array(pred_output0)
                    if len(prop_output.shape) < 3:
                        prop_output = np.expand_dims(prop_output, axis=1)
                    # if prop_output.shape[1] != FLAGS.max_seq:
                    #     prop_output = np.transpose(prop_output, [1, 0, 2])

                    pred_output = np.array(pred_output1)
                    if len(pred_output.shape) < 3:
                        pred_output = np.expand_dims(pred_output, axis=1)
                    # if pred_output.shape[1] != FLAGS.max_seq:
                    #     pred_output = np.transpose(pred_output, [1, 0, 2])

                    full_final_output = np.array(pred_output00)
                    if len(full_final_output.shape) < 3:
                        full_final_output = np.expand_dims(full_final_output, axis=1)

                    idx = -1

                    # temp_prev_x = np.expand_dims(prev_x[:, idx, :], axis=1)
                    temp_prev_x = current_x
                    # temp_prev_y = np.expand_dims(prev_y[:, idx, :], axis=1)
                    temp_prev_y = current_y
                    # temp_prev_time = np.expand_dims(prev_time[:, idx, :], axis=1)
                    temp_prev_time = current_time

                    temp_pred0 = np.expand_dims(prop_output[:, idx, :], axis=1)
                    temp_pred1 = np.expand_dims(pred_output[:, idx, :], axis=1)
                    temp_pred2 = np.expand_dims(full_final_output[:, idx, :], axis=1)

                    prev_meas = np.concatenate([prev_meas, temp_prev_x], axis=1)
                    prev_meas = prev_meas[:, 1:, :]

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

                    prev_cov = np.concatenate([prev_cov, current_covariance[:, np.newaxis, :, :]], axis=1)
                    prev_cov = prev_cov[:, 1:, :, :]

                    prev_y = copy.copy(prev_state)
                    prev_x = copy.copy(prev_meas)

                    # prev_time_in = np.concatenate([prev_time, temp_prev_time], axis=1)
                    # prev_time_in = prev_time_in[:, -1:, :]

                    # Single time step plotting
                    if tstep == 0:
                        new_vals_F = full_final_output[0, -1, :]
                        out_plot_F = prev_state2[0, np.newaxis, :, :]
                        new_vals_X = pred_output[0, -1, :]
                        out_plot_X = prev_state2[0, np.newaxis, :, :]
                        new_vals_P = prop_output[0, -1, :]
                        out_plot_P = prev_state2[0, np.newaxis, :, :]

                        new_q = q_out[0, :, :]
                        # q_plot = np.tile(new_q[np.newaxis, np.newaxis, :, :], [1, FLAGS.max_seq, 1, 1])

                        q_plott = np.stack(prev_covariance, axis=1)
                        q_plott = q_plott[0, :, :, :]

                        q_initial = np.tile(np.eye(6, 6)[np.newaxis, :, :], [FLAGS.max_seq, 1, 1])
                        q_initial[:, 0, 0] = q_plott[:, 0, 0]
                        q_initial[:, 1, 1] = q_plott[:, 1, 1]
                        q_initial[:, 2, 2] = q_plott[:, 4, 4]
                        q_initial[:, 3, 3] = q_plott[:, 5, 5]
                        q_initial[:, 4, 4] = q_plott[:, 8, 8]
                        q_initial[:, 5, 5] = q_plott[:, 9, 9]

                        q_plott = q_plott[np.newaxis, :, :, :]

                        q_plot = np.concatenate([q_initial[np.newaxis, :-1, :, :], new_q[np.newaxis, np.newaxis, :, :]], axis=1)

                        qt_plot = np.tile(qt_out[0, np.newaxis, np.newaxis, :, :], [1, FLAGS.max_seq, 1, 1])
                        rt_plot = np.tile(rt_out[0, np.newaxis, np.newaxis, :, :], [1, FLAGS.max_seq, 1, 1])

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
                        new_qt = q_truth[0, :, :]
                        # update_qt = q_truth[0, :-1, :, :]
                        new_qtt = qt_out[0, :, :]
                        new_rtt = rt_out[0, :, :]

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

                        avg_plot = copy.copy(out_plot_F)
                        # maha_plot = copy.copy()

                    # if tstep > 0:
                    #     out_plot_F[0, -FLAGS.max_seq, :] = fs_update[0, :, 0]
                    #     out_plot_X[0, -FLAGS.max_seq, :] = fs_update[0, :, 0]
                    #     out_plot_P[0, -FLAGS.max_seq, :] = fs_update[0, :, 0]
                    #     q_plot[0, -FLAGS.max_seq:-1, :, :] = update_q
                    #     q_plott[0, -FLAGS.max_seq:-1, :, :] = update_qt

                    accuracy = 0.0
                    if tstep % 10 == 0 or tstep <= FLAGS.max_seq or tstep > int(x.shape[1] - 10):
                        print("Epoch: {0:2d} MB: {1:1d} Time: {2:3d} "
                              "RMSP: {3:2.2e} RMSV: {4:2.2e} RMSA: {5:2.2e} RMSJ: {6:2.2e} "
                              "LR: {7:1.2e} ST: {8:1.2f} PL: {9:1.2e} "
                              "NEF: {10:1.2f} Trace: {11:1.2e} MD: {12:1.2e} "
                              "DL: {13:1.2e} EL: {14:1.2e} PL {15:1.2e} ".format(epoch, minibatch_index, tstep,
                                                                                 rmsp, rmsv, rmsa, rmsj, LR, max_t,
                                                                                 cov_loss, accuracy, trace_loss,
                                                                                 MD, d_loss, gen_loss, nllo))

                    # if np.any(np.isnan([epoch, minibatch_index, tstep, rmsp, rmsv, rmsa, LR, max_t, cov_loss, L_loss, maha_loss])):
                    #     pdb.set_trace()
                    #     pass

                    # print(r2)
                    if tstep == int((mstep - 1)) or tstep % 1000000 == 0 and tstep != 0 and (step - plot_count) > 1:
                        # plt.show()
                        # plt.close()
                        if testing is False:
                            plotpath = FLAGS.plot_dir + '/epoch_' + str(epoch) + '_B_' + str(batch_number) + '_step_' + str(step)
                        else:
                            plotpath = FLAGS.plot_dir + '/epoch_' + str(epoch) + '_test_B_' + str(batch_number) + '_step_' + str(step)
                        if os.path.isdir(plotpath):
                            print('folder exists')
                        else:
                            os.mkdir(plotpath)
                        try:
                            # plot_all2(out_plot_X, out_plot_F, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, mean_y)
                            comparison_plot(out_plot_X, out_plot_F, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, mean_y, qt_plot, rt_plot)
                        except:
                            pdb.set_trace()
                            pass

                        plot_count = step

                if e % 25 == 0 and e != 0 and minibatch_index == n_train_batches - 1:
                    if os.path.isdir(FLAGS.save_dir):
                        print('filter Checkpoint Directory Exists')
                    else:
                        os.mkdir(FLAGS.save_dir)
                    print("Saving filter Weights for epoch" + str(epoch))
                    save_path = self.saver.save(self.sess, FLAGS.save_dir + '/' + filter_name + '_' + str(epoch) + '_' + str(step) + ".ckpt", global_step=step)
                    print("Checkpoint saved at: ", save_path)

                e += 1
