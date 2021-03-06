import math
import time
import random

from tensorflow.contrib.layers import fully_connected as FCL

from slim_helper_1 import *
from slim_helper_2 import *
from slim_data_loaders import DataServerPrePro, DataServerLive
from plotting import *
from slim_propagation_utils import *

from natsort import natsorted

tfd = tfp.distributions
tfna = tf.newaxis

setattr(tfc.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)


class Filter(object):
    def __init__(self, sess, trainable_state=False, state_type='GRU', mode='training',
                 data_dir='', filter_name='', save_dir='',
                 F_hidden=12, num_state=12, num_meas=3, max_seq=2, num_mixtures=4,
                 max_epoch=10000, RE=6378137, GM=398600441890000, batch_size=10, learning_rate=1e-2,
                 constant=False, decimate_data=False):

        self.sess = sess
        self.mode = mode
        self.max_seq = max_seq
        self.num_mixtures = num_mixtures
        self.train_init_state = trainable_state
        self.F_hidden = F_hidden
        self.num_state = num_state
        self.num_meas = num_meas
        self.plot_dir = save_dir + '/plots/'
        self.plot_eval_dir = save_dir + '/plots_eval/'
        self.plot_test_dir = save_dir + '/plots_test/'
        self.checkpoint_dir = save_dir + '/checkpoints/'
        self.log_dir = save_dir + '/logs/'
        self.GM = GM
        self.max_epoch = max_epoch
        self.RE = RE
        self.state_type = state_type
        self.filter_name = filter_name
        self.constant = constant
        self.learning_rate_main = learning_rate

        self.batch_size_np = batch_size
        self.meas_dir = 'NoiseRAE/'
        self.state_dir = 'Translate/'

        self.root = 'D:/TrackFilterData'
        self.data_dir = data_dir

        self.train_dir = self.root + '/OOPBroad/5k25hz_oop_broad_data'
        self.test_dir = self.root + '/OOPBroad/5k25hz_oop_broad_data'

        self.preprocessed = False
        self.plot_interval = 5
        self.decimate_data = decimate_data

        # self.train_dir = self.root + '/AdvancedBroad_preprocessed/Train/'
        # self.test_dir = self.root + '/AdvancedBroad_preprocessed/Test/'

        self.global_step = tf.Variable(initial_value=0, name="global_step", trainable=False, collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES], dtype=tf.int32)
        # self.batch_step = tf.Variable(0.0, trainable=False)

        # self.deterministic = tf.constant(False)

        self.idxi = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        self.idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

        # Meta Variables
        self.plen = int(self.max_seq)
        self.pi_val = tf.constant(math.pi, dtype=tf.float64)

        self.decay_steps = (30000 / self.batch_size_np) * (1500 / self.max_seq)

        self.filters = {}
        self.hiways = {}
        self.projects = {}
        self.vdtype = tf.float64
        self.vdp_np = np.float64
        # self.fading = tf.Variable(0.9, trainable=False, dtype=self.vdtype) * tf.pow(0.99, (self.global_step / self.decay_steps) + 0.05)

        self.seqlen = tf.placeholder(tf.int32, [None])
        self.int_time = tf.placeholder(tf.float64, [None, self.max_seq])
        self.batch_size = tf.shape(self.seqlen)[0]

    def railroad(self, railroad_input, skip=None, width=1, name='', act=tf.nn.elu, dtype=tf.float32):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            if skip is not None:
                railroad_input = FCL(railroad_input, width, activation_fn=act, weights_initializer=tf.initializers.variance_scaling, scope=name+'_input', reuse=tf.AUTO_REUSE) + skip
            else:
                railroad_input = FCL(railroad_input, width, activation_fn=act, weights_initializer=tf.initializers.variance_scaling, scope=name + '_input', reuse=tf.AUTO_REUSE)

            rnn_inp1o = FCL(railroad_input, width, activation_fn=act,
                            weights_initializer=tf.initializers.truncated_normal(stddev=0.01),
                            biases_initializer=tf.initializers.truncated_normal(stddev=0.2 / self.num_state),
                            scope=name+'_y', reuse=tf.AUTO_REUSE)
            rnn_inp2o = FCL(railroad_input, width, activation_fn=tf.nn.sigmoid,
                            weights_initializer=tf.initializers.truncated_normal(stddev=0.01),
                            biases_initializer=tf.initializers.constant(-2.0),
                            scope=name+'_t', reuse=tf.AUTO_REUSE)
            junction = tf.ones_like(rnn_inp2o) - rnn_inp2o
            y = rnn_inp1o * rnn_inp2o + railroad_input * junction

            return y

    def alpha(self, int_time, dt, pstate, meas_rae, cov_est, Q_est, R_est, LLA, sensor_onehot, sensor_noise, states):

        with tf.variable_scope('alpha'):

            state1 = states[0]
            state2 = states[1]
            state3 = states[2]

            lat = LLA[:, 0, tfna]
            lon = LLA[:, 1, tfna]

            R = meas_rae[:, 0, tfna]
            A = meas_rae[:, 1, tfna]
            E = meas_rae[:, 2, tfna]

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

            meas_uvw = tf.concat([uv, vv, wv], axis=1)

            _, At, _, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                                 dimension=int(self.num_state / 3),
                                 sjix=self.om[:, :, 0] * 0.0001 ** 2,
                                 sjiy=self.om[:, :, 0] * 0.0001 ** 2,
                                 sjiz=self.om[:, :, 0] * 0.0001 ** 2,
                                 aji=self.om[:, :, 0] * 1.0)

            prop_state = tf.matmul(At, pstate[:, :, tfna])
            pos_part = tf.matmul(self.meas_mat, prop_state)
            vel_part = tf.concat([prop_state[:, 1, tfna], prop_state[:, 5, tfna], prop_state[:, 9, tfna]], axis=1)
            acc_part = tf.concat([prop_state[:, 2, tfna], prop_state[:, 6, tfna], prop_state[:, 10, tfna]], axis=1)
            jer_part = tf.concat([prop_state[:, 3, tfna], prop_state[:, 7, tfna], prop_state[:, 11, tfna]], axis=1)
            pre_residual = meas_uvw[:, :, tfna] - pos_part

            cov_diag = tf.matrix_diag_part(cov_est)
            Q_diag = tf.matrix_diag_part(Q_est)
            R_diag = tf.matrix_diag_part(R_est)

            cov_diag_n = cov_diag / tf.ones_like(cov_diag)
            prop_state_n = prop_state / (tf.ones_like(prop_state) * 10000)
            pos_part_n = pos_part[:, :, 0] / (tf.ones_like(pos_part[:, :, 0]) * 10000)
            vel_part_n = vel_part[:, :, 0] / (tf.ones_like(vel_part[:, :, 0]) * 1000)
            acc_part_n = acc_part[:, :, 0] / (tf.ones_like(acc_part[:, :, 0]) * 100)
            jer_part_n = jer_part[:, :, 0] / (tf.ones_like(jer_part[:, :, 0]) * 50)

            Q_diag_n = Q_diag / tf.ones_like(Q_diag)
            R_diag_n = R_diag / tf.ones_like(R_diag)
            pre_res_n = pre_residual[:, :, 0] / tf.ones_like(pre_residual[:, :, 0])
            meas_uvw_n = meas_uvw / (tf.ones_like(meas_uvw) * 10000)

            rnn_inpa = tf.concat([dt, pre_res_n, cov_diag_n, pos_part_n, vel_part_n, acc_part_n, jer_part_n], axis=1)
            # rnn_inpb = tf.concat([dt, pre_res_n, cov_diag_n, pos_part_n, vel_part_n, acc_part_n, jer_part_n], axis=1)

            inp_width = rnn_inpa.shape[1].value

            rnn_inpa0 = FCL(rnn_inpa, inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpa0/q_cov', reuse=tf.AUTO_REUSE)
            rnn_inpa1 = FCL(tf.concat([rnn_inpa0], axis=1), inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpa1/q_cov', reuse=tf.AUTO_REUSE)
            rnn_inpa2 = FCL(tf.concat([rnn_inpa1], axis=1), inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpa2/q_cov', reuse=tf.AUTO_REUSE)

            # rnn_inpb0 = FCL(rnn_inpb, inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpb0/r_cov', reuse=tf.AUTO_REUSE)
            # rnn_inpb1 = FCL(tf.concat([rnn_inpb0], axis=1), inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpb1/r_cov', reuse=tf.AUTO_REUSE)
            # rnn_inpb2 = FCL(tf.concat([rnn_inpb1], axis=1), inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpb2/r_cov', reuse=tf.AUTO_REUSE)

            # rnn_inpc0 = FCL(rnn_inp, self.F_hidden, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpc0/time', reuse=tf.AUTO_REUSE)
            # rnn_inpc1 = FCL(tf.concat([rnn_inpc0], axis=1), self.F_hidden, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpc1/time', reuse=tf.AUTO_REUSE)

            rnn_inp = tfc.layers.dropout(rnn_inpa2, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_inputsa/q_cov')
            # rnn_inpb = tfc.layers.dropout(rnn_inpb2, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_inputsb/r_cov')
            # rnn_inpc = tfc.layers.dropout(rnn_inpc1, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_inputsc/time')

            with tf.variable_scope('Cell_1/q_cov', reuse=tf.AUTO_REUSE):
                (outa, state1) = self.source_fwf(rnn_inp, state=state1)

            with tf.variable_scope('Cell_2/t_cov', reuse=tf.AUTO_REUSE):
                (outb, state2) = self.source_fwf2(rnn_inp, state=state2)

            with tf.variable_scope('Cell_3/r_cov', reuse=tf.AUTO_REUSE):
                (outc, state3) = self.source_fwf3(rnn_inp, state=state3)

            # out = tf.concat([outa, outb, outc], axis=1)
            # out = outa + outb + outc

            nv = outb.shape[1].value // 3

            rm0 = FCL(tf.concat([outb], axis=1), outb.shape[1].value, activation_fn=tf.nn.elu, scope='r_cov/1', reuse=tf.AUTO_REUSE)
            # rm1 = FCL(rm0, outb.shape[1].value, activation_fn=tf.nn.elu, scope='r_cov/2', reuse=tf.AUTO_REUSE)
            # d_mult = (tf.nn.sigmoid(rm1[:, -1:], 'r_cov/d_mult') * 50) + tf.ones_like(rm1[:, -1:]) * 1
            # rd = tril_with_diag_softplus_and_shift(rm1[:, :6], diag_shift=0.01, diag_mult=None, name='r_cov/tril')
            asr = FCL(rm0[:, :nv], self.num_mixtures, activation_fn=tf.nn.softmax, scope='r_cov/alphar', reuse=tf.AUTO_REUSE)
            asa = FCL(rm0[:, nv:2*nv], self.num_mixtures, activation_fn=tf.nn.softmax, scope='r_cov/alphaa', reuse=tf.AUTO_REUSE)
            ase = FCL(rm0[:, 2*nv:], self.num_mixtures, activation_fn=tf.nn.softmax, scope='r_cov/alphae', reuse=tf.AUTO_REUSE)

            sr_list = list()
            sa_list = list()
            se_list = list()

            srl = [0.1, 0.5, 1.0, 2.0]

            for ppp in range(self.num_mixtures):
                sr_list.append(tf.ones([1, 1], dtype=self.vdtype) * srl[ppp])
                sa_list.append(tf.ones([1, 1], dtype=self.vdtype) * srl[ppp])
                se_list.append(tf.ones([1, 1], dtype=self.vdtype) * srl[ppp])

            sr_vals = tf.squeeze(tf.stack(sr_list, axis=1), 0)
            sa_vals = tf.squeeze(tf.stack(sa_list, axis=1), 0)
            se_vals = tf.squeeze(tf.stack(se_list, axis=1), 0)

            sr = tf.matmul(asr, sr_vals)
            sa = tf.matmul(asa, sa_vals)
            se = tf.matmul(ase, se_vals)

            ones = tf.ones([self.batch_size, 1], self.vdtype)
            rd_mult = tf.concat([ones*sr, ones * tf.sqrt(R)*sa, ones * tf.sqrt(R)*se], axis=1)
            rd = sensor_noise * rd_mult

            # rdist = tfd.MultivariateNormalTriL(loc=None, scale_tril=rd)
            rdist = tfd.MultivariateNormalDiag(loc=None, scale_diag=rd)
            Rt = rdist.covariance()

            u = tf.zeros([self.batch_size, 3], self.vdtype)

            nv = outa.shape[1].value // 3
            qm0 = FCL(outa, outa.shape[1].value, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='q_cov/1', reuse=tf.AUTO_REUSE)
            # qm1 = FCL(qm0, outa.shape[1].value, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='q_cov/2', reuse=tf.AUTO_REUSE)

            qmx = FCL(qm0[:, :nv], self.num_mixtures, activation_fn=tf.nn.softmax, weights_initializer=tf.initializers.variance_scaling, scope='q_cov/x', reuse=tf.AUTO_REUSE)
            qmy = FCL(qm0[:, nv:2*nv], self.num_mixtures, activation_fn=tf.nn.softmax, weights_initializer=tf.initializers.variance_scaling, scope='q_cov/y', reuse=tf.AUTO_REUSE)
            qmz = FCL(qm0[:, 2*nv:], self.num_mixtures, activation_fn=tf.nn.softmax, weights_initializer=tf.initializers.variance_scaling, scope='q_cov/z', reuse=tf.AUTO_REUSE)

            sjx_l = list()
            sjy_l = list()
            sjz_l = list()
            sjxl = [0.0001, 1.0, 50, 500]
            sjyl = [0.0001, 1.0, 50, 500]
            sjzl = [0.0001, 1.0, 50, 500]

            for ppp in range(self.num_mixtures):
                sjx_l.append(tf.ones([1, 1], dtype=self.vdtype) * sjxl[ppp])
                sjy_l.append(tf.ones([1, 1], dtype=self.vdtype) * sjyl[ppp])
                sjz_l.append(tf.ones([1, 1], dtype=self.vdtype) * sjzl[ppp])

            sjx_vals = tf.squeeze(tf.stack(sjx_l, axis=1), 0)
            sjy_vals = tf.squeeze(tf.stack(sjy_l, axis=1), 0)
            sjz_vals = tf.squeeze(tf.stack(sjz_l, axis=1), 0)

            sjx = tf.matmul(qmx, sjx_vals)
            sjy = tf.matmul(qmy, sjy_vals)
            sjz = tf.matmul(qmz, sjz_vals)

            am0 = FCL(tf.concat([outc], axis=1), outc.shape[1].value, activation_fn=tf.nn.elu, scope='time/1', reuse=tf.AUTO_REUSE)
            # rm1 = FCL(rm0, 3, activation_fn=None, scope='r_cov/2', reuse=tf.AUTO_REUSE)
            # d_mult = (tf.nn.sigmoid(rm1[:, -1:], 'r_cov/d_mult') * 50) + tf.ones_like(rm1[:, -1:]) * 1
            # rd = tril_with_diag_softplus_and_shift(rm1[:, :6], diag_shift=0.01, diag_mult=None, name='r_cov/tril')
            aar = FCL(am0, self.num_mixtures, activation_fn=tf.nn.softmax, scope='time/alphar', reuse=tf.AUTO_REUSE)

            time_list = list()

            srl = [0.1, 0.5, 0.75, 1.0]
            for ppp in range(self.num_mixtures):
                time_list.append(tf.ones([1, 1], dtype=self.vdtype) * srl[ppp])

            time_vals = tf.squeeze(tf.stack(time_list, axis=1), 0)
            time_constant = tf.matmul(aar, time_vals)

            Qt, At, Bt, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                                   dimension=int(self.num_state / 3),
                                   sjix=self.om[:, :, 0] * sjx ** 2,
                                   sjiy=self.om[:, :, 0] * sjy ** 2,
                                   sjiz=self.om[:, :, 0] * sjz ** 2,
                                   aji=self.om[:, :, 0] * time_constant)

            # xcov_ut = tf.concat(values=[xcov_prev[:, 0, :], xcov_prev[:, 1, 1:], xcov_prev[:, 2, 2:]], axis=1)
            Qt_diag = tf.where(tf.matrix_diag_part(Qt) <= tf.ones_like(tf.matrix_diag_part(Qt)) * 1e-16, tf.ones_like(tf.matrix_diag_part(Qt)), tf.matrix_diag_part(Qt))
            Qt = tf.matrix_set_diag(Qt, Qt_diag)
            # Q_rr = self.railroad(tf.sqrt(tf.matrix_diag_part(Qt)), skip=tf.matrix_diag_part(Q_est_chol))
            qdist = tfd.MultivariateNormalTriL(loc=None, scale_tril=tf.cholesky(Qt))
            # qdist = tfd.MultivariateNormalDiag(loc=None, scale_diag=tf.sqrt(tf.matrix_diag_part(Qt)))
            Qt = qdist.covariance()

            states = (state1, state2, state3)

            return meas_uvw, Qt, At, Rt, Bt, u, states

    def refinement(self, int_time, dt, meas_uvw, mu_t0, Sigma_t0, At, Rt, sensor_onehot, cur_weight, states):
        with tf.variable_scope('refinement'):
            weight = cur_weight[:, :, tf.newaxis]
            inv_weight = tf.ones_like(weight) - weight

            state1 = states[0]
            state2 = states[1]
            state3 = states[2]

            # layer_input = tf.concat([dt, pos_res_uvw, mu_t0, tf.matrix_diag_part(Sigma_t0), sensor_onehot], axis=1)
            # rnn_inp0 = tf.concat([layer_input], axis=1)
            # rnn_inp1 = FCL(rnn_inp0, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp1_beta', reuse=tf.AUTO_REUSE)
            # rnn_inp2 = FCL(rnn_inp1, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp2_beta', reuse=tf.AUTO_REUSE)
            # rnn_inp3 = FCL(rnn_inp2, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp3_beta', reuse=tf.AUTO_REUSE)
            # rnn_inp = tfc.layers.dropout(rnn_inp3, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_inputs_beta')
            #
            # with tf.variable_scope('Source_Track_Forward/q_cov', reuse=tf.AUTO_REUSE):
            #     # (outa, state1) = self.source_fwf((int_time, rnn_inpa), state=state1)
            #     (out1, state1) = self.source_fwf(rnn_inp, state=state1)

            n_samples = 25

            jerk_scale = tf.ones([self.batch_size, 3], dtype=self.vdtype) * 500 ** 2
            jerk_dist = tfp.distributions.MultivariateNormalDiag(loc=None, scale_diag=tf.sqrt(jerk_scale), name='jerk_distribution')

            # cov_jer = tf.concat([tf.concat([Sigma_pred[:, 3, 3, tf.newaxis, tf.newaxis], Sigma_pred[:, 3, 7, tf.newaxis, tf.newaxis], Sigma_pred[:, 3, 11, tf.newaxis, tf.newaxis]], axis=2),
            #                      tf.concat([Sigma_pred[:, 7, 3, tf.newaxis, tf.newaxis], Sigma_pred[:, 7, 7, tf.newaxis, tf.newaxis], Sigma_pred[:, 7, 11, tf.newaxis, tf.newaxis]], axis=2),
            #                      tf.concat([Sigma_pred[:, 11, 3, tf.newaxis, tf.newaxis], Sigma_pred[:, 11, 7, tf.newaxis, tf.newaxis], Sigma_pred[:, 11, 11, tf.newaxis, tf.newaxis]], axis=2)],
            #                     axis=1)

            # smooth_jer = tf.concat([mu_pred[:, 3, tf.newaxis], mu_pred[:, 7, tf.newaxis], mu_pred[:, 11, tf.newaxis]], axis=1)

            # n_samples = 10
            # jerk_dist = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=cov_jer)
            jerk_samples = jerk_dist.sample(n_samples)
            jerk_samples = tf.transpose(jerk_samples, [1, 0, 2])

            ts = tf.tile(mu_t0[:, tf.newaxis, :], [1, n_samples, 1])

            # state_t1 = tf.squeeze(tf.matmul(At, mu_t0[:, :, tf.newaxis]), -1)
            # states_t1 = tf.tile(state_t1[:, tf.newaxis, :], [1, n_samples, 1])
            # residual_t1 = tf.squeeze(tf.matmul(self.meas_mat, state_t1[:, :, tf.newaxis]), -1) - meas_uvw

            states_t0 = tf.concat([ts[:, :, 0, tf.newaxis], ts[:, :, 1, tf.newaxis], ts[:, :, 2, tf.newaxis], jerk_samples[:, :, 0, tf.newaxis],
                                   ts[:, :, 4, tf.newaxis], ts[:, :, 5, tf.newaxis], ts[:, :, 6, tf.newaxis], jerk_samples[:, :, 1, tf.newaxis],
                                   ts[:, :, 8, tf.newaxis], ts[:, :, 9, tf.newaxis], ts[:, :, 10, tf.newaxis], jerk_samples[:, :, 2, tf.newaxis]], axis=2)

            # states_t1 = tf.transpose(tf.matmul(At, states_t0, transpose_b=True), [0, 2, 1])  # states at current time

            state_l = list()
            for _ in range(n_samples):
                state_l.append(tf.matmul(At, states_t0[:, _, :, tf.newaxis]))

            states_t1 = tf.squeeze(tf.stack(state_l, axis=1), -1)
            meas_tile = tf.tile(meas_uvw[:, tf.newaxis, :], [1, n_samples, 1])  # measurements at current time

            # state_measurement_pos = tf.matmul(states_t1, self.meas_mat, transpose_b=True)
            state_measurement_pos = tf.concat([states_t1[:, :, 0, tf.newaxis], states_t1[:, :, 4, tf.newaxis], states_t1[:, :, 8, tf.newaxis]], axis=2)

            # meas_err_mag = tf.linalg.norm(state_measurement_pos - meas_tile, axis=2, keepdims=True) * weight

            meas_err = tf.pow((state_measurement_pos - meas_tile), -1) * weight  # + (tf.ones_like(meas_tile) * 1e-3) * inv_weight
            meas_err_mag = tf.linalg.norm(meas_err, axis=2, keepdims=False) * cur_weight
            meas_err_alphas = tfc.seq2seq.hardmax(meas_err_mag)
            # Rt = Rt * weight
            # Rt = tf.tile(Rt[:, tf.newaxis, :, :], [1, n_samples, 1, 1])

            # meas_err_log_prob = tfd.MultivariateNormalDiag(loc=None, scale_diag=tf.sqrt(tf.matrix_diag_part(Rt))).log_prob(meas_err)
            # meas_err_log_prob = tfd.MultivariateNormalTriL(loc=None, scale_tril=tf.cholesky(Rt)).log_prob(meas_err)

            # state_attention = multihead_attention2(states_t1, meas_err_mag, num_units=self.num_state, num_heads=1,
            #                                        dropout_rate=self.drop_rate, is_training=self.is_training, dtype=self.vdtype)
            #
            # state_attention = tf.reduce_mean(state_attention, axis=1)

            # state_attention_pos = tf.concat([state_attention[:, 0, tf.newaxis], state_attention[:, 4, tf.newaxis], state_attention[:, 8, tf.newaxis]], axis=1)
            # state_attention_error = state_attention_pos - meas_uvw

            # meas_err_alphas = tf.nn.softmax(meas_err_log_prob)
            # meas_err_alphas = meas_err_log_prob / tf.reduce_sum(meas_err_log_prob)
            # meas_err_alphas = meas_err_alphas[:, :, 0]

            meas_err_weighted = tf.reduce_sum(meas_err_alphas[:, :, tf.newaxis] * meas_err, axis=1)
            state_measurement = tf.reduce_sum(meas_err_alphas[:, :, tf.newaxis] * states_t1, axis=1)

            # meas_err_weighted = tf.squeeze(tf.matmul(meas_err_alphas[:, tf.newaxis, :], meas_err), axis=1)
            # state_measurement = tf.squeeze(tf.matmul(meas_err_alphas[:, tf.newaxis, :], states_t1), axis=1)

            # meas_err_weighted = residual_t1
            # state_measurement = tf.reduce_mean(states_t1, axis=1)
            # state_measurement = state_t1

            # state_measurement_error = state_measurement_pos - meas_uvw

            # state_difference = state_attention - state_measurement
            # error_difference = state_attention_error - state_measurement_error

            layer_inputb = tf.concat([dt, meas_err_weighted * cur_weight], axis=1)
            rnn_inp0b = tf.concat([layer_inputb], axis=1)
            rnn_inp1b = FCL(rnn_inp0b, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp1b_beta', reuse=tf.AUTO_REUSE)
            # rnn_inp2b = FCL(rnn_inp1b, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp2b_beta', reuse=tf.AUTO_REUSE)
            # rnn_inp3b = FCL(rnn_inp2b, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp3b_beta', reuse=tf.AUTO_REUSE)
            # rnn_inpb = tfc.layers.dropout(rnn_inp3b, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_inputsb_beta')

            with tf.variable_scope('Source_Track_Forward2/q_cov', reuse=tf.AUTO_REUSE):
                # (outa, state1) = self.source_fwf((int_time, rnn_inpa), state=state1)
                (out2, state2) = self.source_fwf2(rnn_inp1b, state=state2)

            layer_outa = tf.concat([dt, meas_err_weighted * cur_weight, out2], axis=1)
            rnn_out0a = tf.concat([layer_outa], axis=1)
            rnn_out1a = FCL(rnn_out0a, 3, activation_fn=tf.nn.elu, scope='rnn_out1a_beta', reuse=tf.AUTO_REUSE)
            rnn_out2a = FCL(rnn_out1a, 3, activation_fn=tf.nn.elu, scope='rnn_out2a_beta', reuse=tf.AUTO_REUSE)
            gain = FCL(rnn_out2a, 3, activation_fn=tf.nn.sigmoid, scope='rnn_out3a_beta', reuse=tf.AUTO_REUSE)
            # gain = tf.reshape(gain, [self.batch_size, 12, 3])

            # final_state = state_measurement + state_diff
            final_pos = tf.concat([state_measurement[:, 0, tf.newaxis], state_measurement[:, 4, tf.newaxis], state_measurement[:, 8, tf.newaxis]], axis=1) + (gain * 0)

            final_state = tf.concat([final_pos[:, 0, tf.newaxis], state_measurement[:, 1, tf.newaxis], state_measurement[:, 2, tf.newaxis], state_measurement[:, 3, tf.newaxis],
                                     final_pos[:, 1, tf.newaxis], state_measurement[:, 5, tf.newaxis], state_measurement[:, 6, tf.newaxis], state_measurement[:, 7, tf.newaxis],
                                     final_pos[:, 2, tf.newaxis], state_measurement[:, 9, tf.newaxis], state_measurement[:, 10, tf.newaxis], state_measurement[:, 11, tf.newaxis]], axis=1)

            # final_state = state_measurement + tf.squeeze(tf.matmul(gain, meas_err_weighted[:, :, tf.newaxis]), -1)
            final_pos = tf.concat([state_measurement[:, 0, tf.newaxis], state_measurement[:, 4, tf.newaxis], state_measurement[:, 8, tf.newaxis]], axis=1)
            final_residual = (final_pos - meas_err_weighted) * cur_weight

            all_diff = tf.transpose(states_t1 - tf.tile(final_state[:, tf.newaxis, :], [1, n_samples, 1]), [0, 2, 1])
            covariance = tf.matmul(all_diff, tf.transpose(all_diff, [0, 2, 1]))

            layer_inpc = tf.concat([dt, final_residual, tf.reshape(covariance, [self.batch_size, 144])], axis=1)
            rnn_inp0c = tf.concat([layer_inpc], axis=1)
            rnn_inp1c = FCL(rnn_inp0c, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp1c_beta', reuse=tf.AUTO_REUSE)
            rnn_inp2c = FCL(rnn_inp1c, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp2c_beta', reuse=tf.AUTO_REUSE)
            rnn_inp3c = FCL(rnn_inp2c, self.F_hidden, activation_fn=tf.nn.elu, scope='rnn_inp3c_beta', reuse=tf.AUTO_REUSE)
            rnn_inpc = tfc.layers.dropout(rnn_inp3c, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_inpc_beta')

            with tf.variable_scope('Source_Track_Forward3/q_cov', reuse=tf.AUTO_REUSE):
                # (outa, state1) = self.source_fwf((int_time, rnn_inpa), state=state1)
                (out3, state3) = self.source_fwf3(rnn_inpc, state=state3)

            flat_cov = tf.reshape(covariance, [self.batch_size, 144])
            layer_outb = tf.concat([dt, final_residual, flat_cov], axis=1)
            rnn_out0b = tf.concat([layer_outb], axis=1)
            rnn_out1b = FCL(rnn_out0b, 78, activation_fn=None, scope='rnn_out1b_beta', reuse=tf.AUTO_REUSE)
            rnn_out2b = FCL(rnn_out1b, 78, activation_fn=None, scope='rnn_out2b_beta', reuse=tf.AUTO_REUSE)
            rnn_out3b = FCL(rnn_out2b, 78, activation_fn=None, scope='rnn_out3b_beta', reuse=tf.AUTO_REUSE)
            rnn_outb = tfc.layers.dropout(rnn_out3b, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_outb_beta')

            covariance_out = tril_with_diag_softplus_and_shift(rnn_outb, diag_shift=1, name='cov_dist')
            # covariance_out = multivariate_normal_tril(rnn_outb, dims=12, diag_shift=0.001, name='cov_dist') + tf.nn.sigmoid(covariance)

            # # rnn_out = self.filter_layer(tf.concat([out3, rnn_inp], axis=1), name='rnn_out_beta')
            # beta0 = FCL(out3, n_samples, activation_fn=None, scope='rnn_out2_beta0', reuse=tf.AUTO_REUSE)
            # beta1 = FCL(beta0, n_samples, activation_fn=None, scope='rnn_out2_beta1', reuse=tf.AUTO_REUSE)
            # beta = FCL(beta1, n_samples, activation_fn=tf.nn.softmax, scope='rnn_out2_beta', reuse=tf.AUTO_REUSE)
            # beta = beta[:, tf.newaxis, :]
            #
            # mu_out = tf.squeeze(tf.matmul(beta, all_states), 1)
            # rnn_out2 = FCL(out3, 36, activation_fn=tf.nn.sigmoid, scope='rnn_out2_beta0', reuse=tf.AUTO_REUSE)
            # gain = tf.reshape(rnn_out2, [self.batch_size, 12, 3])
            # gain = tf.where(tf.equal(cur_weight[:, :, tf.newaxis] * gain, tf.zeros_like(gain)), tf.zeros_like(gain), gain)
            # pos_pred0 = tf.concat([mu_t[:, 0, tf.newaxis], mu_t[:, 4, tf.newaxis], mu_t[:, 8, tf.newaxis]], axis=1)
            # pos_pred = pos_pred0 + gain * pos_res
            # mu_pred = mu_t + tf.squeeze(tf.matmul(gain, pos_res[:, :, tf.newaxis]), -1)

            # mu_pred = tf.concat([pos_pred[:, 0, tf.newaxis], mu_pred[:, 1, tf.newaxis], mu_pred[:, 2, tf.newaxis], mu_pred[:, 3, tf.newaxis],
            #                      pos_pred[:, 1, tf.newaxis], mu_pred[:, 5, tf.newaxis], mu_pred[:, 6, tf.newaxis], mu_pred[:, 7, tf.newaxis],
            #                      pos_pred[:, 2, tf.newaxis], mu_pred[:, 9, tf.newaxis], mu_pred[:, 10, tf.newaxis], mu_pred[:, 11, tf.newaxis]], axis=1)

            states = (state1, state2, state3)

            return final_state, covariance_out, states, states_t1

    def forward_step_fn(self, params, inputs):

        with tf.variable_scope('forward_step_fn'):
            current_time = inputs[:, 0, tfna]
            prev_time = inputs[:, 1, tfna]
            int_time = inputs[:, 2, tfna]
            meas_rae = inputs[:, 3:6]
            cur_weight = inputs[:, 6, tfna]
            LLA = inputs[:, 7:10]
            sensor_onehot = inputs[:, 10:13]
            sensor_noise = inputs[:, 13:]

            weight = cur_weight[:, :, tfna]

            _, _, mu_t0, Sigma_t0, _, state1, state2, state3, Qt0, Rt0, _, _, _, _, _ = params

            states = (state1, state2, state3)

            dt = current_time - prev_time
            dt = tf.where(dt <= 1 / 100, tf.ones_like(dt) * 1 / 25, dt)

            meas_uvw, Qt, At, Rt, Bt, u, states = self.alpha(int_time, dt, mu_t0, meas_rae,
                                                             Sigma_t0, Qt0, Rt0, LLA,
                                                             sensor_onehot, sensor_noise, states)

            mu_pred = tf.squeeze(tf.matmul(At, tf.expand_dims(mu_t0, 2)), -1) + tf.squeeze(tf.matmul(Bt, u[:, :, tfna]), -1)
            Sigma_pred = tf.matmul(tf.matmul(At, Sigma_t0), At, transpose_b=True) + Qt

            mu_pred_uvw = tf.matmul(self.meas_mat, mu_pred[:, :, tfna])
            pos_res_uvw = meas_uvw[:, :, tfna] - mu_pred_uvw

            lat = LLA[:, 0, tfna, tfna]
            lon = LLA[:, 1, tfna, tfna]

            uvw_to_enu = uvw2enu_tf(lat, lon)
            enu_to_uvw = tf.transpose(uvw_to_enu, [0, 2, 1])

            y_enu = tf.squeeze(tf.matmul(uvw_to_enu, mu_pred_uvw), -1)

            rae_to_enu = rae2enu_tf(y_enu, self.pi_val)

            Rt = Rt * (tf.ones_like(Rt) * weight)

            enu_cov = tf.matmul(tf.matmul(rae_to_enu, Rt), rae_to_enu, transpose_b=True)

            Rt = tf.matmul(tf.matmul(enu_to_uvw, enu_cov), enu_to_uvw, transpose_b=True)

            sp1 = tf.matmul(tf.matmul(self.meas_mat, Sigma_pred), self.meas_mat, transpose_b=True)
            S = sp1 + Rt

            S_inv = tf.matrix_inverse(S)
            gain = tf.matmul(tf.matmul(Sigma_pred, self.meas_mat, transpose_b=True), S_inv)
            gain = tf.where(tf.equal(weight * gain, tf.zeros_like(gain)), tf.zeros_like(gain), gain)

            mu_t = mu_pred[:, :, tfna] + tf.matmul(gain, pos_res_uvw)
            mu_t = mu_t[:, :, 0]

            I_KC = self.I_12 - tf.matmul(gain, self.meas_mat)  # (bs, dim_z, dim_z)
            Sigma_t = tf.matmul(tf.matmul(I_KC, Sigma_pred), I_KC, transpose_b=True) + tf.matmul(tf.matmul(gain, Rt), gain, transpose_b=True)
            Sigma_t = (Sigma_t + tf.transpose(Sigma_t, [0, 2, 1])) / 2

            mu_pred = tf.squeeze(tf.matmul(At, tf.expand_dims(mu_t, 2)), -1) + tf.squeeze(tf.matmul(Bt, u[:, :, tfna]), -1)
            Sigma_pred = tf.matmul(tf.matmul(At, Sigma_t), At, transpose_b=True) + Qt

            return mu_pred, Sigma_pred, mu_t, Sigma_t, meas_uvw, state1, state2, state3, Qt, Rt, At, Bt, S_inv, weight, u

    @staticmethod
    def backward_step_fn(params, inputs):
        with tf.variable_scope('backward_step_fn'):
            mu_back, Sigma_back = params
            mu_pred_tp1, Sigma_pred_tp1, mu_filt_t, Sigma_filt_t, A, weight = inputs

            J_t = tf.matmul(tf.transpose(A, [0, 2, 1]), tf.matrix_inverse(Sigma_pred_tp1))
            J_t = tf.matmul(Sigma_filt_t, J_t)

            mu_back = mu_filt_t + tf.matmul(J_t, mu_back - mu_pred_tp1)
            Sigma_back = Sigma_filt_t + tf.matmul(J_t, tf.matmul(Sigma_back - Sigma_pred_tp1, J_t, adjoint_b=True))

            return mu_back, Sigma_back

    def compute_forwards(self):

        with tf.variable_scope('Forward'):
            self.mu = self.state_input
            self.Sigma = tf.reshape(self.P_inp, [self.batch_size, self.num_state, self.num_state])

            all_time = tf.concat([self.prev_time[:, tfna], tf.stack(self.current_timei, axis=1)], axis=1)
            meas_rae = tf.concat([self.prev_measurement[:, tfna], tf.stack(self.measurement, axis=1)], axis=1)

            current_time = all_time[:, 1:, :]
            prev_time = all_time[:, :-1, :]

            int_time = self.int_time

            sensor_lla = tf.expand_dims(self.sensor_lla, axis=1)
            sensor_lla = tf.tile(sensor_lla, [1, meas_rae.shape[1], 1])

            sensor_onehot = tf.expand_dims(self.sensor_vector, axis=1)
            sensor_onehot = tf.tile(sensor_onehot, [1, meas_rae.shape[1], 1])

            sensor_noise = tf.expand_dims(self.R_static, axis=1)
            sensor_noise = tf.tile(sensor_noise, [1, meas_rae.shape[1], 1])

            inputs = tf.concat([current_time, prev_time, int_time[:, :, tfna], meas_rae[:, 1:, :],
                                self.seqweightin[:, :, tfna], sensor_lla[:, 1:, :],
                                sensor_onehot[:, 1:, :], sensor_noise[:, 1:, :]], axis=2)

            init_Q = self.Q_inp
            init_R = self.R_inp

            state1 = self.cell_state1
            state2 = self.cell_state2
            state3 = self.cell_state3

            init_Si = tf.ones([self.batch_size, 3, 3], self.vdtype)
            init_A = tf.ones([self.batch_size, 12, 12], self.vdtype)
            init_B = tf.ones([self.batch_size, 12, 3], self.vdtype)
            meas_uvw = tf.zeros([self.batch_size, 3], self.vdtype)
            init_weight = tf.ones([self.batch_size, 1, 1], self.vdtype)
            init_u = tf.zeros([self.batch_size, 3], self.vdtype)

            forward_states = tf.scan(self.forward_step_fn, tf.transpose(inputs, [1, 0, 2]),
                                     initializer=(self.mu, self.Sigma, self.mu, self.Sigma, meas_uvw,
                                                  state1, state2, state3,
                                                  init_Q, init_R, init_A, init_B, init_Si,
                                                  init_weight, init_u),
                                     parallel_iterations=1, name='forward')
            return forward_states

    def compute_backwards(self, forward_states):
        with tf.variable_scope('Backward'):
            mu_pred, Sigma_pred, mu_filt, Sigma_filt, meas_uvw, state1, state2, state3, Q, R, A, B, S_inv, weights, u = forward_states

            forward_states_filter = [mu_filt, Sigma_filt]
            forward_states_pred = [mu_pred, Sigma_pred]

            # Swap batch dimension and time dimension
            forward_states_filter[0] = tf.transpose(forward_states_filter[0], [1, 0, 2])
            forward_states_filter[1] = tf.transpose(forward_states_filter[1], [1, 0, 2, 3])

            forward_states_pred[0] = tf.transpose(forward_states_pred[0], [1, 0, 2])
            forward_states_pred[1] = tf.transpose(forward_states_pred[1], [1, 0, 2, 3])

            mu_pred = tf.expand_dims(mu_pred, 3)
            mu_filt = tf.expand_dims(mu_filt, 3)

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

            return backward_states, forward_states_filter, forward_states_pred, Q, R, A, B, S_inv, u, meas_uvw, state1, state2, state3

    def filter(self):
        mu_pred, Sigma_pred, mu_filt, Sigma_filt, meas_uvw, state1, state2, state3, Q, R, A, B, S_inv, weights, u = forward_states = \
            self.compute_forwards()

        state1_out = tf.transpose(state1, [1, 0, 2])
        state2_out = tf.transpose(state2, [1, 0, 2])
        state3_out = tf.transpose(state3, [1, 0, 2])

        forward_states_filter = [mu_filt, Sigma_filt]
        forward_states_pred = [mu_pred, Sigma_pred]

        # Swap batch dimension and time dimension
        forward_states_filter[0] = tf.transpose(forward_states_filter[0], [1, 0, 2])
        forward_states_filter[1] = tf.transpose(forward_states_filter[1], [1, 0, 2, 3])

        forward_states_pred[0] = tf.transpose(forward_states_pred[0], [1, 0, 2])
        forward_states_pred[1] = tf.transpose(forward_states_pred[1], [1, 0, 2, 3])

        return tuple(forward_states_filter), tf.transpose(A, [1, 0, 2, 3]), tf.transpose(Q, [1, 0, 2, 3]), \
               tf.transpose(R, [1, 0, 2, 3]), tf.transpose(B, [1, 0, 2, 3]), tf.transpose(S_inv, [1, 0, 2, 3]), \
               tf.transpose(u, [1, 0, 2]), tf.transpose(meas_uvw, [1, 0, 2]), tuple(forward_states_pred), \
               state1_out, state2_out, state3_out

    def smooth(self):

        backward_states, forward_states_filter, forward_states_pred, Q, R, A, B, S_inv, u, meas_uvw, state1, state2, state3 = self.compute_backwards(self.compute_forwards())

        state1_out = tf.transpose(state1, [1, 0, 2])
        state2_out = tf.transpose(state2, [1, 0, 2])
        state3_out = tf.transpose(state3, [1, 0, 2])

        # Swap batch dimension and time dimension
        backward_states[0] = tf.transpose(backward_states[0], [1, 0, 2])
        backward_states[1] = tf.transpose(backward_states[1], [1, 0, 2, 3])

        return tuple(backward_states), tuple(forward_states_filter), tuple(forward_states_pred), tf.transpose(A, [1, 0, 2, 3]), tf.transpose(Q, [1, 0, 2, 3]), \
               tf.transpose(R, [1, 0, 2, 3]), tf.transpose(B, [1, 0, 2, 3]), tf.transpose(S_inv, [1, 0, 2, 3]), \
               tf.transpose(u, [1, 0, 2]), tf.transpose(meas_uvw, [1, 0, 2]), \
               state1_out, state2_out, state3_out

    def refine(self, filtered_state, filtered_covariance, meas_uvw, Q, R):

        with tf.variable_scope('refine'):
            filtered_state = tf.stop_gradient(filtered_state)
            filtered_covariance = tf.stop_gradient(filtered_covariance)
            meas_uvw = tf.stop_gradient(meas_uvw)
            # Q = tf.stop_gradient(Q)
            # R = tf.stop_gradient(R)

            all_time = tf.concat([self.prev_time[:, tfna], tf.stack(self.current_timei, axis=1)], axis=1)
            # meas_rae = tf.concat([self.prev_measurement[:, tfna], tf.stack(self.measurement, axis=1)], axis=1)

            current_time = all_time[:, 1:, :]
            prev_time = all_time[:, :-1, :]

            dt = current_time - prev_time

            # int_time = self.int_time

            # sensor_lla = tf.expand_dims(self.sensor_lla, axis=1)
            # sensor_lla = tf.tile(sensor_lla, [1, meas_rae.shape[1], 1])

            # sensor_onehot = tf.expand_dims(self.sensor_vector, axis=1)
            # sensor_onehot = tf.tile(sensor_onehot, [1, meas_rae.shape[1], 1])

            # sensor_noise = tf.expand_dims(self.R_static, axis=1)
            # sensor_noise = tf.tile(sensor_noise, [1, meas_rae.shape[1], 1])

            # inputs = tf.concat([current_time, prev_time, int_time[:, :, tfna], meas_rae[:, 1:, :],
            #                     self.seqweightin[:, :, tfna], sensor_lla[:, 1:, :],
            #                     sensor_onehot[:, 1:, :], sensor_noise[:, 1:, :]], axis=2)

            state_measurement_pos = tf.concat([filtered_state[:, :, 0, tf.newaxis], filtered_state[:, :, 4, tf.newaxis], filtered_state[:, :, 8, tf.newaxis]], axis=2)

            pos_res_uvw = state_measurement_pos - meas_uvw

            # layer_input = tf.concat([dt, pos_res_uvw, tf.matrix_diag_part(filtered_covariance), sensor_onehot[:, 1:, :]], axis=2)
            layer_input = tf.concat([dt, pos_res_uvw], axis=2)
            inp_width = layer_input.shape[2].value
            rnn_inp0 = tf.concat([layer_input], axis=1)
            rnn_inp1 = FCL(rnn_inp0, inp_width*4, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inp1', reuse=tf.AUTO_REUSE)
            # rnn_inp2 = FCL(rnn_inp1, inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inp2', reuse=tf.AUTO_REUSE)
            # rnn_inp3 = FCL(rnn_inp2, inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inp3', reuse=tf.AUTO_REUSE)
            rnn_inp = tfc.layers.dropout(rnn_inp1, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_inputs')

            rnn_outputs, cell_state = tf.nn.dynamic_rnn(self.source_fwf3, rnn_inp, sequence_length=self.seqlen, initial_state=self.cell_state3, dtype=self.vdtype, scope='dynamic_1')
            tile_rnn_state = tf.tile(cell_state[:, tfna, :], [1, self.max_seq, 1])

            # cell_state = self.cell_state3
            # state_list = list()
            # output_list = list()
            # for iii in range(self.max_seq):
            #     with tf.variable_scope('Cell_3', reuse=tf.AUTO_REUSE):
            #         (cell_output, cell_state) = self.source_fwf3(rnn_inp[:, iii, :], state=cell_state)
            #
            #         state_list.append(cell_state)
            #         output_list.append(cell_output)
            #
            # rnn_outputs = tf.stack(output_list, axis=1)
            # tile_rnn_state = tf.stack(state_list, axis=1)

            all_outputs = tf.concat([rnn_outputs, tile_rnn_state], axis=2)

            attended_state = multihead_attention2(pos_res_uvw, all_outputs, num_units=self.num_meas, num_heads=3, dropout_rate=self.drop_rate, is_training=self.is_training, scope='attenion',
                                                    reuse=tf.AUTO_REUSE)

            railroad_input = tf.concat([attended_state], axis=2)
            railroad_out1 = self.railroad(railroad_input, width=self.num_meas, act=None, name='railroad1a')
            railroad_out2 = self.railroad(railroad_out1, width=self.num_meas, act=None, name='railroad2a')
            railroad_out3 = self.railroad(railroad_out2, width=self.num_meas, act=None, name='railroad3a')

            # pos_vel_pred = tf.concat([filtered_state[:, :, 0, tf.newaxis], filtered_state[:, :, 1, tf.newaxis],
            #                           filtered_state[:, :, 4, tf.newaxis], filtered_state[:, :, 5, tf.newaxis],
            #                           filtered_state[:, :, 8, tf.newaxis], filtered_state[:, :, 9, tf.newaxis]], axis=2) + attended_state
            #
            # attended_state = tf.concat([pos_vel_pred[:, :, 0, tf.newaxis], pos_vel_pred[:, :, 1, tf.newaxis], filtered_state[:, :, 2, tf.newaxis], filtered_state[:, :, 3, tf.newaxis],
            #                             pos_vel_pred[:, :, 2, tf.newaxis], pos_vel_pred[:, :, 3, tf.newaxis], filtered_state[:, :, 6, tf.newaxis], filtered_state[:, :, 7, tf.newaxis],
            #                             pos_vel_pred[:, :, 4, tf.newaxis], pos_vel_pred[:, :, 5, tf.newaxis], filtered_state[:, :, 10, tf.newaxis], filtered_state[:, :, 11, tf.newaxis]], axis=2)

            pos_pred = tf.concat([filtered_state[:, :, 0, tf.newaxis], filtered_state[:, :, 4, tf.newaxis], filtered_state[:, :, 8, tf.newaxis]], axis=2) + railroad_out3

            attended_state = tf.concat([pos_pred[:, :, 0, tf.newaxis], filtered_state[:, :, 1, tf.newaxis], filtered_state[:, :, 2, tf.newaxis], filtered_state[:, :, 3, tf.newaxis],
                                        pos_pred[:, :, 1, tf.newaxis], filtered_state[:, :, 5, tf.newaxis], filtered_state[:, :, 6, tf.newaxis], filtered_state[:, :, 7, tf.newaxis],
                                        pos_pred[:, :, 2, tf.newaxis], filtered_state[:, :, 9, tf.newaxis], filtered_state[:, :, 10, tf.newaxis], filtered_state[:, :, 11, tf.newaxis]], axis=2)

            # attended_covariance = multihead_attention2(tf.sqrt(tf.matrix_diag_part(filtered_covariance)), tf.sqrt(tf.matrix_diag_part(filtered_covariance)),
            #                                            num_units=self.num_state, num_heads=12, dropout_rate=self.drop_rate, is_training=self.is_training,
            #                                            scope='attenion', reuse=tf.AUTO_REUSE)

            # attended_residual = tf.concat([attended_state[:, :, 0, tf.newaxis], attended_state[:, :, 4, tf.newaxis], attended_state[:, :, 8, tf.newaxis]], axis=2) - meas_uvw
            #
            # railroad_input = tf.concat([attended_residual, tf.sqrt(tf.matrix_diag_part(filtered_covariance))], axis=2)
            #
            # railroad_out1 = self.railroad(railroad_input, skip=tf.sqrt(tf.matrix_diag_part(filtered_covariance)), width=self.num_state, name='railroad1c')
            # railroad_out2 = self.railroad(railroad_out1, width=self.num_state, name='railroad2c')
            # railroad_out3 = self.railroad(railroad_out2, width=self.num_state, name='railroad3c')

            # rnn_inp3o = FCL(tf.concat([rnn_inp2o], axis=2), 12, activation_fn=tf.nn.softplus, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inp3o', reuse=tf.AUTO_REUSE)

            state_dist = tfd.MultivariateNormalDiag(loc=attended_state, scale_diag=tf.sqrt(tf.matrix_diag_part(filtered_covariance)))

            refined_state = state_dist.mean()
            refined_covariance = state_dist.covariance()

            # gain = tf.reshape(rnn_inp3o, [self.batch_size, self.max_seq, 12, 3])
            # gain = tf.reshape(rnn_inp3o, [self.batch_size, self.max_seq, 3])

            # refined_state = filtered_state + tf.squeeze(tf.matmul(gain, pos_res_uvw[:, :, :, tfna]), -1)

            # pos_pred = state_measurement_pos + gain

            # refined_state = tf.concat([pos_pred[:, :, 0, tf.newaxis], filtered_state[:, :, 1, tf.newaxis], filtered_state[:, :, 2, tf.newaxis], filtered_state[:, :, 3, tf.newaxis],
            #                            pos_pred[:, :, 1, tf.newaxis], filtered_state[:, :, 5, tf.newaxis], filtered_state[:, :, 6, tf.newaxis], filtered_state[:, :, 7, tf.newaxis],
            #                            pos_pred[:, :, 2, tf.newaxis], filtered_state[:, :, 9, tf.newaxis], filtered_state[:, :, 10, tf.newaxis], filtered_state[:, :, 11, tf.newaxis]], axis=2)

            # refined_covariance = filtered_covariance

            return refined_state, refined_covariance, cell_state

    @staticmethod
    def _sast(a, s):
        _, dim_1, dim_2 = s.get_shape().as_list()
        sastt = tf.matmul(tf.reshape(s, [-1, dim_2]), a, transpose_b=True)
        sastt = tf.transpose(tf.reshape(sastt, [-1, dim_1, dim_2]), [0, 2, 1])
        sastt = tf.matmul(s, sastt)
        return sastt

    def get_elbo(self, backward_states, name=''):
        with tf.variable_scope('ELBO' + name):

            num_el = tf.reduce_sum(self.seqweightin)  # / tf.cast(self.batch_size, self.vdtype)
            # num_el2 = num_el

            mu_smooth = backward_states[0]
            Sigma_smooth = backward_states[1]

            ssdiag = tf.matrix_diag_part(Sigma_smooth)
            ssdiag = tf.where(tf.less_equal(ssdiag, tf.zeros_like(ssdiag)), tf.ones_like(ssdiag) * 1, ssdiag)

            rodiag = tf.matrix_diag_part(self.ro_list)
            rodiag = tf.where(tf.less_equal(rodiag, tf.zeros_like(rodiag)), tf.ones_like(rodiag) * 1, rodiag)

            Sigma_smooth = tf.matrix_set_diag(Sigma_smooth, ssdiag)
            self.ro_list = tf.matrix_set_diag(self.ro_list, rodiag)

            all_truth = tf.stack(self.truth_state, axis=1)
            mvn_smooth = tfd.MultivariateNormalTriL(mu_smooth, tf.cholesky(Sigma_smooth))
            self.mvn_inv = tf.matrix_inverse(mvn_smooth.covariance())

            z_smooth = mvn_smooth.sample()
            z_0 = z_smooth[:, 0, :]
            mvn_0 = tfd.MultivariateNormalTriL(self.mu, tf.cholesky(self.Sigma))
            self.error_loss_initial = tf.truediv(tf.reduce_sum(tf.negative(mvn_0.log_prob(z_0) * self.seqweightin[:, 0])), (num_el * 144))

            num_el2 = tf.maximum(tf.log(self.error_loss_initial + 1), tf.ones_like(self.error_loss_initial)) * num_el

            # self.error_loss_initial = tf.truediv(self.error_loss_initial, num_el2)

            self.state_error = tf.sqrt(tf.square(all_truth - z_smooth))
            self.state_error = tf.where(self.state_error < 1e-12, tf.ones_like(self.state_error) * 1e-12, self.state_error)
            self.state_error = self.state_error[:, :, :, tfna]

            truth_pos = tf.concat([all_truth[:, :, 0, tfna], all_truth[:, :, 4, tfna], all_truth[:, :, 8, tfna]], axis=2)
            truth_vel = tf.concat([all_truth[:, :, 1, tfna], all_truth[:, :, 5, tfna], all_truth[:, :, 9, tfna]], axis=2)
            truth_acc = tf.concat([all_truth[:, :, 2, tfna], all_truth[:, :, 6, tfna], all_truth[:, :, 10, tfna]], axis=2)

            smooth_pos = tf.concat([z_smooth[:, :, 0, tfna], z_smooth[:, :, 4, tfna], z_smooth[:, :, 8, tfna]], axis=2)
            smooth_vel = tf.concat([z_smooth[:, :, 1, tfna], z_smooth[:, :, 5, tfna], z_smooth[:, :, 9, tfna]], axis=2)
            smooth_acc = tf.concat([z_smooth[:, :, 2, tfna], z_smooth[:, :, 6, tfna], z_smooth[:, :, 10, tfna]], axis=2)

            pos_error = truth_pos - smooth_pos
            vel_error = truth_vel - smooth_vel
            acc_error = truth_acc - smooth_acc

            if self.mode == 'training':
                Az_tm1 = tf.matmul(self.ao_list[:, :-1], tf.expand_dims(all_truth[:, :-1], 3))
                # Bz_tm1 = tf.matmul(self.bo_list[:, :-1], tf.expand_dims(self.uo_list[:, :-1], 3))
                mu_transition = Az_tm1[:, :, :, 0]  # + Bz_tm1[:, :, :, 0]
                z_t_transition = all_truth[:, 1:, :]
                trans_centered = z_t_transition - mu_transition
                Qdiag = tf.matrix_diag_part(self.qo_list[:, :-1, :, :])
                # log_prob_transition = mvn_transition.log_prob(trans_centered) * self.seqweightin[:, :-1]
                # trans_centered_j = tf.concat([trans_centered[:, :, 3, tfna], trans_centered[:, :, 7, tfna], trans_centered[:, :, 11, tfna]], axis=2)
                # Qdiag_j = tf.concat([Qdiag[:, :, 3, tfna], Qdiag[:, :, 7, tfna], Qdiag[:, :, 11, tfna]], axis=2)
                trans_centered_a = tf.concat([trans_centered[:, :, 2, tfna], trans_centered[:, :, 6, tfna], trans_centered[:, :, 10, tfna]], axis=2)
                Qdiag_a = tf.concat([Qdiag[:, :, 2, tfna], Qdiag[:, :, 6, tfna], Qdiag[:, :, 10, tfna]], axis=2)
                # mvn_transition = tfd.MultivariateNormalTriL(None, tf.cholesky(Qdiag_a))
                mvn_transition = tfd.MultivariateNormalDiag(None, tf.sqrt(Qdiag_a))
                log_prob_transition = mvn_transition.log_prob(trans_centered_a) * self.seqweightin[:, :-1]

            self.y_t_resh = tf.concat([all_truth[:, :, 0, tfna], all_truth[:, :, 4, tfna], all_truth[:, :, 8, tfna]], axis=2)
            self.Cz_t = self.new_meas
            emiss_centered = (self.Cz_t - self.y_t_resh)
            emiss_centered = emiss_centered + tf.ones_like(emiss_centered) * 1e-3
            mvn_emission = tfd.MultivariateNormalTriL(None, tf.cholesky(self.ro_list))
            # mvn_emission = tfd.MultivariateNormalDiag(None, tf.sqrt(tf.matrix_diag_part(self.ro_list)))

            cov_pos = tf.concat([tf.concat([Sigma_smooth[:, :, 0, 0, tfna, tfna], Sigma_smooth[:, :, 0, 4, tfna, tfna], Sigma_smooth[:, :, 0, 8, tfna, tfna]], axis=3),
                                 tf.concat([Sigma_smooth[:, :, 4, 0, tfna, tfna], Sigma_smooth[:, :, 4, 4, tfna, tfna], Sigma_smooth[:, :, 4, 8, tfna, tfna]], axis=3),
                                 tf.concat([Sigma_smooth[:, :, 8, 0, tfna, tfna], Sigma_smooth[:, :, 8, 4, tfna, tfna], Sigma_smooth[:, :, 8, 8, tfna, tfna]], axis=3)],
                                axis=2)

            cov_vel = tf.concat([tf.concat([Sigma_smooth[:, :, 1, 1, tfna, tfna], Sigma_smooth[:, :, 1, 5, tfna, tfna], Sigma_smooth[:, :, 1, 9, tfna, tfna]], axis=3),
                                 tf.concat([Sigma_smooth[:, :, 5, 1, tfna, tfna], Sigma_smooth[:, :, 5, 5, tfna, tfna], Sigma_smooth[:, :, 5, 9, tfna, tfna]], axis=3),
                                 tf.concat([Sigma_smooth[:, :, 9, 1, tfna, tfna], Sigma_smooth[:, :, 9, 5, tfna, tfna], Sigma_smooth[:, :, 9, 9, tfna, tfna]], axis=3)],
                                axis=2)

            cov_acc = tf.concat([tf.concat([Sigma_smooth[:, :, 2, 2, tfna, tfna], Sigma_smooth[:, :, 2, 6, tfna, tfna], Sigma_smooth[:, :, 2, 10, tfna, tfna]], axis=3),
                                 tf.concat([Sigma_smooth[:, :, 6, 2, tfna, tfna], Sigma_smooth[:, :, 6, 6, tfna, tfna], Sigma_smooth[:, :, 6, 10, tfna, tfna]], axis=3),
                                 tf.concat([Sigma_smooth[:, :, 10, 2, tfna, tfna], Sigma_smooth[:, :, 10, 6, tfna, tfna], Sigma_smooth[:, :, 10, 10, tfna, tfna]], axis=3)],
                                axis=2)

            M1P = tf.matmul(emiss_centered[:, :, :, tfna], tf.matrix_inverse(self.ro_list), transpose_a=True)
            M2P = tf.matmul(M1P, emiss_centered[:, :, :, tfna])

            self.MDP = tf.sqrt(tf.squeeze(M2P, -1))
            self.MDPi = tf.sqrt((tf.ones_like(self.MDP, self.vdtype) / tf.squeeze(M2P, -1)))
            self.maha_loss = tf.truediv(tf.reduce_sum((self.MDP * self.seqweightin[:, :, tfna] + self.MDPi * self.seqweightin[:, :, tfna])), num_el2)
            self.maha_out = tf.truediv(tf.reduce_sum(self.MDP * self.seqweightin[:, :, tfna]), num_el2)

            train_cov_full = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=Sigma_smooth)
            train_cov_pos = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=cov_pos)
            train_cov_vel = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=cov_vel)
            train_cov_acc = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=cov_acc)

            self.trace_loss = tf.log(tf.truediv(tf.reduce_sum(tf.sqrt(tf.pow(tf.matrix_diag_part(Sigma_smooth), 2))), num_el2))

            self.error_loss_pos = tf.truediv(tf.reduce_sum(tf.negative(train_cov_pos.log_prob(pos_error)) * self.seqweightin), num_el2 * 9)+1e-5
            self.error_loss_vel = tf.truediv(tf.reduce_sum(tf.negative(train_cov_vel.log_prob(vel_error)) * self.seqweightin), num_el2 * 9)+1e-5
            self.error_loss_acc = tf.truediv(tf.reduce_sum(tf.negative(train_cov_acc.log_prob(acc_error)) * self.seqweightin), num_el2 * 9)+1e-5

            self.error_loss_full = tf.truediv(tf.reduce_sum(tf.negative(train_cov_full.log_prob(self.state_error[:, :, :, 0])) * self.seqweightin), (num_el2 * 144))+1e-5

            self.entropy = tf.truediv(tf.reduce_sum(mvn_smooth.log_prob(z_smooth) * self.seqweightin), num_el2 * 12)

            # self.entropyp = tf.truediv(tf.reduce_sum(train_cov_pos.log_prob(smooth_pos) * self.seqweightin), num_el2 * 3)
            # self.entropyv = tf.truediv(tf.reduce_sum(train_cov_vel.log_prob(smooth_vel) * self.seqweightin), num_el2 * 3)
            # self.entropya = tf.truediv(tf.reduce_sum(train_cov_acc.log_prob(smooth_acc) * self.seqweightin), num_el2 * 3)

            self.rl = tf.truediv(tf.reduce_sum(tf.negative(mvn_emission.log_prob(emiss_centered)) * self.seqweightin), num_el2 * 9)

            # self.z_smooth = mu_smooth
            self.num_el = num_el
            self.num_el2 = num_el2
            # if self.mode == 'training':
            #     self.error_loss_Q = tf.truediv(tf.reduce_sum(tf.negative(log_prob_transition)), (num_el2 * 3))
            # else:
            self.error_loss_Q = tf.cast(tf.reduce_sum(0.0), self.vdtype)

    def get_regression_loss(self, _yhat, name=''):
        
        with tf.variable_scope('Regression_Loss_' + name):
            loss_func = weighted_mape_tf
    
            total_weight = tf.cast(self.seqweightin, self.vdtype)
            # tot = tf.cast(self.max_seq, self.vdtype)
            
            # Measurement Error
            pos1m_err = loss_func(self._y[:, :, 0], self.new_meas[:, :, 0], total_weight, name='merr1')
            pos2m_err = loss_func(self._y[:, :, 4], self.new_meas[:, :, 1], total_weight, name='merr1')
            pos3m_err = loss_func(self._y[:, :, 8], self.new_meas[:, :, 2], total_weight, name='merr3')
    
            # State Error
            pos1e_err = loss_func(self._y[:, :, 0], _yhat[:, :, 0], total_weight, name='serr1')
            pos2e_err = loss_func(self._y[:, :, 4], _yhat[:, :, 4], total_weight, name='serr2')
            pos3e_err = loss_func(self._y[:, :, 8], _yhat[:, :, 8], total_weight, name='serr3')
    
            total_err_state = pos1e_err + pos2e_err + pos3e_err
            total_err_meas = pos1m_err + pos2m_err + pos3m_err
    
            meas_error_ratio = total_err_state / total_err_meas
            # meas_err_use = meas_error_ratio + tf.log(1 + meas_error_ratio)
            meas_err_use = meas_error_ratio

            state_loss_pos100 = loss_func(self._y[:, :, 0], _yhat[:, :, 0], total_weight, name='pos1_err')
            state_loss_pos200 = loss_func(self._y[:, :, 4], _yhat[:, :, 4], total_weight, name='pos2_err')
            state_loss_pos300 = loss_func(self._y[:, :, 8], _yhat[:, :, 8], total_weight, name='pos3_err')
            state_loss_vel100 = loss_func(self._y[:, :, 1], _yhat[:, :, 1], total_weight, name='vel1_err')
            state_loss_vel200 = loss_func(self._y[:, :, 5], _yhat[:, :, 5], total_weight, name='vel2_err')
            state_loss_vel300 = loss_func(self._y[:, :, 9], _yhat[:, :, 9], total_weight, name='vel3_err')
            state_loss_acc100 = loss_func(self._y[:, :, 2], _yhat[:, :, 2], total_weight, name='acc1_err')
            state_loss_acc200 = loss_func(self._y[:, :, 6], _yhat[:, :, 6], total_weight, name='acc2_err')
            state_loss_acc300 = loss_func(self._y[:, :, 10], _yhat[:, :, 10], total_weight, name='acc3_err')
            state_loss_jer100 = loss_func(self._y[:, :, 3], _yhat[:, :, 3], total_weight, name='jer1_err')
            state_loss_jer200 = loss_func(self._y[:, :, 7], _yhat[:, :, 7], total_weight, name='jer2_err')
            state_loss_jer300 = loss_func(self._y[:, :, 11], _yhat[:, :, 11], total_weight, name='jer3_err')
    
            SLPf = state_loss_pos100 + state_loss_pos200 + state_loss_pos300
            SLVf = state_loss_vel100 + state_loss_vel200 + state_loss_vel300
            SLAf = state_loss_acc100 + state_loss_acc200 + state_loss_acc300
            SLJf = state_loss_jer100 + state_loss_jer200 + state_loss_jer300
    
            SLPf = tf.truediv(SLPf, self.num_el)
            SLVf = tf.truediv(SLVf, self.num_el)
            SLAf = tf.truediv(SLAf, self.num_el)
            SLJf = tf.truediv(SLJf, self.num_el)
    
            return SLPf, SLVf, SLAf, SLJf, meas_err_use

    def build_model(self, is_training):

        with tf.variable_scope('Input_Placeholders'):

            self.drop_rate = tf.Variable(0.5, trainable=False, dtype=tf.float64, name='dropout_rate')
            self.learning_rate_inp = tf.Variable(0.0, trainable=False, dtype=tf.float64, name='learning_rate_input')
            self.update_condition = tf.placeholder(tf.bool, name='update_condition')

            self.grad_clip = tf.placeholder(self.vdtype, name='grad_clip')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.measurement = [tf.placeholder(self.vdtype, shape=(None, self.num_meas), name="meas_uvw_{}".format(t)) for t in range(self.max_seq)]

            self.sensor_ecef = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name='sensor_ecef')
            self.sensor_lla = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name='sensor_lla')
            self.sensor_vector = tf.placeholder(self.vdtype, shape=(None, 3), name='sensor_vector')

            self.prev_measurement = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name="px")
            self.prev_covariance_estimate = tf.placeholder(self.vdtype, shape=(None, self.num_state, self.num_state), name="pcov")
            self.prev_time = tf.placeholder(self.vdtype, shape=(None, 1), name="ptime")
            self.prev_state_truth = tf.placeholder(self.vdtype, shape=(None, self.num_state), name="ptruth")
            self.prev_state_estimate = tf.placeholder(self.vdtype, shape=(None, self.num_state), name="prev_state_estimate")

            self.current_timei = [tf.placeholder(self.vdtype, shape=(None, 1), name="current_time_{}".format(t)) for t in range(self.max_seq)]
            self.P_inp = tf.placeholder(self.vdtype, shape=(None, self.num_state, self.num_state), name="p_inp")
            self.Q_inp = tf.placeholder(self.vdtype, shape=(None, self.num_state, self.num_state), name="q_inp")
            self.R_inp = tf.placeholder(self.vdtype, shape=(None, self.num_meas, self.num_meas), name="r_inp")
            self.R_static = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name="r_inp")
            self.state_input = tf.placeholder(self.vdtype, shape=(None, self.num_state), name="state_input")
            self.truth_state = [tf.placeholder(self.vdtype, shape=(None, self.num_state), name="y_truth_{}".format(t)) for t in range(self.max_seq)]
            self.seqweightin = tf.placeholder(self.vdtype, [None, self.max_seq], name='seqweight')

            if 'GRU' in self.state_type:
                self.cell_state1 = tf.placeholder(name='cell_state1', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.cell_state2 = tf.placeholder(name='cell_state2', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.cell_state3 = tf.placeholder(name='cell_state3', shape=[None, self.F_hidden], dtype=self.vdtype)

                # self.input_cell_states = [tf.placeholder(name="GRU_state_{}".format(t), shape=(None, self.F_hidden), dtype=self.vdtype) for t in self.num_cells]

            else:
                self.init_c_fwf = tf.placeholder(name='init_c_fwf', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.init_h_fwf = tf.placeholder(name='init_h_fwf', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_fw_in_state = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwf, self.init_h_fwf)

                self.init_c_fwf2 = tf.placeholder(name='init_c_fwf2', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.init_h_fwf2 = tf.placeholder(name='init_h_fwf2', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_fw_in_state2 = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwf2, self.init_h_fwf2)

                self.init_c_fwf3 = tf.placeholder(name='init_c_fwf3', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.init_h_fwf3 = tf.placeholder(name='init_h_fwf3', shape=[None, self.F_hidden], dtype=self.vdtype)
                self.state_fw_in_state3 = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwf3, self.init_h_fwf3)

            if self.state_type == 'INDYGRU':
                cell_type = tfc.rnn.IndyGRUCell
            elif self.state_type == 'LSTM':
                cell_type = tfc.rnn.IndyLSTMCell
            elif self.state_type == 'PLSTM':
                cell_type = PhasedLSTMCell
            elif self.state_type == 'GRU':
                cell_type = tfc.rnn.GRUCell
            else:
                cell_type = tfc.rnn.IndRNNCell

            with tf.variable_scope('Cell_1/q_cov'):
                self.source_fwf = cell_type(num_units=self.F_hidden)

            with tf.variable_scope('Cell_2/r_cov'):
                self.source_fwf2 = cell_type(num_units=self.F_hidden)

            with tf.variable_scope('Cell_3'):
                self.source_fwf3 = cell_type(num_units=self.F_hidden)

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

        if self.mode == 'training':
            smooth, filter_out, prediction, A, Q, R, B, S_inv, u, meas_uvw, state1_out, state2_out, state3_out = self.smooth()  # for plotting smoothed posterior
            # filter_out, A, Q, R, B, S_inv, u, meas_uvw, prediction, state1_out, state2_out, state3_out = self.filter()  # for plotting smoothed posterior
            # filter_out, _, Q, R, _, _, _, meas_uvw, prediction, _, _, _ = self.filter()  # for plotting smoothed posterior

        else:
            filter_out, A, Q, R, B, S_inv, u, meas_uvw, prediction, state1_out, state2_out, state3_out = self.filter()

        use_refinement = False

        if use_refinement is True:
            refined_state, refined_covariance, state3_out = self.refine(filter_out[0], filter_out[1], meas_uvw, Q, R)
        else:
            refined_state = filter_out[0]
            refined_covariance = filter_out[1]

        self.ao_list = A
        self.qo_list = Q
        self.ro_list = R
        self.uo_list = u
        self.bo_list = B
        self.si_list = S_inv

        self.new_meas = meas_uvw

        self.z_smooth = filter_out[0]
        self.final_state_filter = filter_out[0]
        self.final_state_prediction = refined_state

        if self.mode == 'training':
            self.final_state_smooth = smooth[0]
        else:
            self.final_state_smooth = filter_out[0]

        self.final_cov_filter = filter_out[1]
        self.final_cov_prediction = refined_covariance
        if self.mode == 'training':
            self.final_cov_smooth = smooth[1]
        else:
            self.final_cov_smooth = filter_out[1]

        self.state_fwf1 = state1_out
        self.state_fwf2 = state2_out
        self.state_fwf3 = state3_out

        self._y = tf.stack(self.truth_state, axis=1)

        if self.mode == 'training':
            self.get_elbo(filter_out)
        else:
            self.get_elbo(filter_out)

        with tf.variable_scope('regression_loss'):

            self.rmse_pos, self.rmse_vel, self.rmse_acc, self.rmse_jer, self.meas_err_use = self.get_regression_loss(self.z_smooth)

            if use_refinement is True:
                self.rmse_posr, self.rmse_velr, self.rmse_accr, self.rmse_jerr, self.meas_err_user = self.get_regression_loss(self.final_state_prediction, name='refinement')

            # state_error_refined = tf.stack(self.truth_state, axis=1) - self.final_state_prediction
            # pos_error_refined = tf.concat([state_error_refined[:, :, 0, tfna], state_error_refined[:, :, 4, tfna], state_error_refined[:, :, 8, tfna]], axis=2)
            # vel_error_refined = tf.concat([state_error_refined[:, :, 1, tfna], state_error_refined[:, :, 5, tfna], state_error_refined[:, :, 9, tfna]], axis=2)
            #
            # mvn_refined = tfd.MultivariateNormalDiag(loc=None, scale_diag=tf.sqrt(tf.matrix_diag_part(self.final_cov_prediction)))
            # # self.error_loss_full_refined = tf.truediv(tf.reduce_sum(tf.negative(mvn_refined.log_prob(state_error_refined)) * self.seqweightin), (self.num_el * 12))
            #
            # S2 = self.final_cov_prediction
            # cov_pos = tf.concat([tf.concat([S2[:, :, 0, 0, tfna, tfna], S2[:, :, 0, 4, tfna, tfna], S2[:, :, 0, 8, tfna, tfna]], axis=3),
            #                      tf.concat([S2[:, :, 4, 0, tfna, tfna], S2[:, :, 4, 4, tfna, tfna], S2[:, :, 4, 8, tfna, tfna]], axis=3),
            #                      tf.concat([S2[:, :, 8, 0, tfna, tfna], S2[:, :, 8, 4, tfna, tfna], S2[:, :, 8, 8, tfna, tfna]], axis=3)],
            #                     axis=2)
            #
            # cov_vel = tf.concat([tf.concat([S2[:, :, 1, 1, tfna, tfna], S2[:, :, 1, 5, tfna, tfna], S2[:, :, 1, 9, tfna, tfna]], axis=3),
            #                      tf.concat([S2[:, :, 5, 1, tfna, tfna], S2[:, :, 5, 5, tfna, tfna], S2[:, :, 5, 9, tfna, tfna]], axis=3),
            #                      tf.concat([S2[:, :, 9, 1, tfna, tfna], S2[:, :, 9, 5, tfna, tfna], S2[:, :, 9, 9, tfna, tfna]], axis=3)],
            #                     axis=2)
            #
            # mvn_refinedp = tfd.MultivariateNormalDiag(loc=None, scale_diag=tf.sqrt(tf.matrix_diag_part(cov_pos)))
            # self.error_loss_pos_refined = tf.truediv(tf.reduce_sum(tf.negative(mvn_refinedp.log_prob(pos_error_refined)) * self.seqweightin), (self.num_el * 9))
            #
            # mvn_refinedv = tfd.MultivariateNormalDiag(loc=None, scale_diag=tf.sqrt(tf.matrix_diag_part(cov_vel)))
            # self.error_loss_vel_refined = tf.truediv(tf.reduce_sum(tf.negative(mvn_refinedv.log_prob(vel_error_refined)) * self.seqweightin), (self.num_el * 9))
            #
            # self.entropy_refined = tf.truediv(tf.reduce_sum(mvn_refined.log_prob(mvn_refined.sample()) * self.seqweightin), self.num_el * 144)

            # self.vel_error_tempr = self.rmse_velr / self.rmse_vel
            # self.vel_error_refined = tf.sqrt(self.vel_error_tempr) + tf.log(1 + self.vel_error_tempr)
            #
            # self.vel_error_tempr = self.rmse_velr / self.rmse_vel
            # self.vel_error_refined = tf.sqrt(self.vel_error_tempr) + tf.log(1 + self.vel_error_tempr)

        with tf.variable_scope('learning_rate'):
            self.learning_rate = tf.train.exponential_decay(self.learning_rate_inp, global_step=self.global_step, decay_steps=self.decay_steps, decay_rate=0.8, staircase=True)

        with tf.variable_scope("TrainOps"):
            print('Updating Gradients')
            all_vars = tf.trainable_variables()
            filter_vars = [var for var in all_vars if 'refine' not in var.name]

            self.lower_bound = (tf.log(self.meas_err_use+1) + 0.4) * (tf.log(self.rmse_vel) + tf.log(self.rmse_acc) + self.rl + self.error_loss_pos + self.error_loss_vel)

            if use_refinement is True:
                refinement_error = self.meas_error_refined  # * (self.error_loss_pos_refined + self.error_loss_vel_refined + self.entropy_refined)

            opt1 = tfc.opt.MultitaskOptimizerWrapper(tfc.opt.AdamWOptimizer(weight_decay=1e-10, learning_rate=self.learning_rate, name='opt1'))
            gradvars1 = opt1.compute_gradients(self.lower_bound, filter_vars, colocate_gradients_with_ops=False)
            gradvars1, _ = tfc.opt.clip_gradients_by_global_norm(gradvars1, 1.0)
            self.train_1 = opt1.apply_gradients(gradvars1, global_step=self.global_step)

            if use_refinement is True:
                refine_vars = [var for var in all_vars if 'refine' in var.name]
                opt2 = tfc.opt.MultitaskOptimizerWrapper(tfc.opt.AdamWOptimizer(weight_decay=1e-10, learning_rate=self.learning_rate, name='opt2'))
                gradvars2 = opt2.compute_gradients(refinement_error, refine_vars, colocate_gradients_with_ops=False)
                gradvars2, _ = tfc.opt.clip_gradients_by_global_norm(gradvars2, 1.0)
                self.train_2 = opt1.apply_gradients(gradvars2, global_step=self.global_step)

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

            tf.summary.scalar("Total_Loss", self.lower_bound)
            tf.summary.scalar("Meas_Err_Norm", tf.log(self.meas_err_use+1) + 0.4)
            tf.summary.scalar("Trace", self.trace_loss)
            tf.summary.scalar("Cov_Meas", self.rl)
            tf.summary.scalar("Cov_Q", self.error_loss_Q)
            tf.summary.scalar("Entropy", self.entropy)
            # tf.summary.scalar("Entropyp", self.entropyp)
            # tf.summary.scalar("Entropyv", self.entropyv)
            # tf.summary.scalar("Entropya", self.entropya)
            tf.summary.scalar("Cov_Pos", self.error_loss_pos)
            tf.summary.scalar("Cov_Vel", self.error_loss_vel)
            tf.summary.scalar("Cov_Acc", self.error_loss_acc)
            tf.summary.scalar("Cov_Total", self.error_loss_full)
            tf.summary.scalar("Cov_Init", self.error_loss_initial)
            tf.summary.scalar("MahalanobisLoss", self.maha_loss)
            tf.summary.scalar("MahalanobisDistance", tf.truediv(tf.reduce_sum(self.MDP * self.seqweightin[:, :, tfna]), self.num_el))
            tf.summary.scalar("MahalanobisInverse", tf.truediv(tf.reduce_sum(self.MDPi * self.seqweightin[:, :, tfna]), self.num_el))
            tf.summary.scalar("RMSE_pos", self.rmse_pos + 1e-5)
            tf.summary.scalar("RMSE_vel", self.rmse_vel + 1e-5)
            tf.summary.scalar("RMSE_acc", self.rmse_acc + 1e-5)
            tf.summary.scalar("RMSE_jer", self.rmse_jer + 1e-5)
            tf.summary.scalar("Learning_Rate", self.learning_rate)

            # tf.summary.scalar("MeasurementRefined", self.meas_err_ratio)
            # tf.summary.scalar("VelocityRefined", self.vel_err_ratio)
            # tf.summary.scalar("Refine_Error", refinement_error)
            # tf.summary.scalar("PositionCovarianceR", self.error_loss_pos_refined)
            # tf.summary.scalar("VelocityCovarianceR", self.error_loss_vel_refined)
            # tf.summary.scalar("RMSE_pos_refined", self.rmse_posr)
            # tf.summary.scalar("RMSE_vel_refined", self.rmse_velr)
            # tf.summary.scalar("RMSE_acc_refined", self.rmse_accr)
            # tf.summary.scalar("RMSE_jer_refined", self.rmse_jerr)
            # tf.summary.scalar("EntropyR", self.entropy_refined)

    def train(self, data_rate, max_exp_seq):

        # rho0 = 1.22  # kg / m**3
        # k0 = 0.14141e-3
        # area = 0.25  # / self.RE  # meter squared
        # cd = 0.03  # unitless
        # gmn = self.GM / (self.RE ** 3)

        lr = self.learning_rate_main

        shuffle_data = False
        self.data_rate = data_rate
        self.max_exp_seq = max_exp_seq
        tf.global_variables_initializer().run()

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        try:
            save_files = os.listdir(self.checkpoint_dir)
            save_files = natsorted(save_files, reverse=True)
            recent = str.split(save_files[1], '_')
            start_epoch = recent[2]
            step = str.split(recent[3], '.')[0]
            print("Resuming run from epoch " + str(start_epoch) + ' and step ' + str(step))
            step = int(step)
        except:
            print("Beginning New Run ")
            print('Loading filter...')
            start_epoch = 0
            step = 0
        try:
            self.saver = tf.train.import_meta_graph(self.checkpoint_dir + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step) + '.meta')
            self.saver.restore(self.sess, self.checkpoint_dir + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step))
            print("filter restored.")
        except:
            self.saver = tf.train.Saver(save_relative_paths=True)
            print("Could not restore filter")
            start_epoch = 0
            step = 0

        if self.preprocessed is True:
            ds = DataServerPrePro(self.train_dir, self.test_dir)
        else:
            ds = DataServerLive(self.data_dir, self.meas_dir, self.state_dir, decimate_data=self.decimate_data)

        for epoch in range(int(start_epoch), self.max_epoch):

            n_train_batches = int(ds.num_examples_train / self.batch_size_np)

            for minibatch_index in range(n_train_batches):

                # if minibatch_index % n_train_batches // 4 == 0 and minibatch_index != 0:
                #     lr = lr * 0.98
                #
                # if lr < 1e-5:
                #     lr = 1e-5

                testing = False

                # plt_idx = random.randint(0, self.batch_size_np - 1)
                plt_idx = 0

                # if minibatch_index % 100 == 0 and minibatch_index != 0:
                #     testing = True
                #     print('Testing filter for epoch ' + str(epoch))
                # else:
                #     testing = False
                #     print('Training filter for epoch ' + str(epoch))

                # Data is unnormalized at this point
                if self.preprocessed is False:
                    x_data, y_data, batch_number, total_batches, ecef_ref, lla_data, sensor_vector, sensor_vector2, meas_list = ds.load(batch_size=self.batch_size_np, constant=self.constant,
                                                                                                                                        test=testing, max_seq_len=self.max_exp_seq, HZ=self.data_rate)
                    lla_datar = copy.copy(lla_data)
                    ecef_ref = np.ones([self.batch_size_np, y_data.shape[1], 3]) * ecef_ref[:, np.newaxis, :]

                else:
                    x_data, y_data, ecef_ref, lla_data = ds.load(batch_size=self.batch_size_np, constant=self.constant, test=testing)
                    lla_datar = copy.copy(lla_data[:, 0, :])

                if shuffle_data:
                    shuf = np.arange(x_data.shape[0])
                    np.random.shuffle(shuf)
                    x_data = x_data[shuf]
                    y_data = y_data[shuf]

                lla_datar[:, 0] = lla_datar[:, 0] * np.pi / 180
                lla_datar[:, 1] = lla_datar[:, 1] * np.pi / 180

                x_data = np.concatenate([x_data[:, :, 0, np.newaxis], x_data[:, :, 4:7]], axis=2)  # rae measurements

                y_uvw = y_data[:, :, :3] - ecef_ref
                zero_rows = (y_data[:, :, :3] == 0).all(2)
                for i in range(y_data.shape[0]):
                    zz = zero_rows[i, :, np.newaxis]
                    y_uvw[i, :, :] = np.where(zz, np.zeros_like(y_uvw[i, :, :]), y_uvw[i, :, :])

                y_data = np.concatenate([y_uvw, y_data[:, :, 3:]], axis=2)

                permute_dims = False
                if permute_dims:
                    x_data, y_data = permute_xyz_dims(x_data, y_data)

                s_data = x_data

                if testing is True:
                    print('Evaluating')
                    self.evaluate(x_data, y_data, ecef_ref, lla_datar, epoch, minibatch_index, step)
                    continue

                x, y, meta, prev_y, prev_x, prev_time, prev_meta, initial_meas, initial_state_truth, initial_time, max_length = prepare_batch(0, x_data, y_data, s_data,
                                                                                                                                              seq_len=self.max_seq, batch_size=self.batch_size_np,
                                                                                                                                              new_batch=True)

                count, _, _, _, _, _, q_plot, q_plott, k_plot, out_plot_filter, out_plot_F, out_plot_smooth, time_vals, \
                meas_plot, truth_plot, Q_plot, R_plot, maha_plot = initialize_run_variables(self.batch_size_np, self.max_seq, self.num_state)

                fd = {}

                windows = int((x.shape[1]) / self.max_seq)
                time_plotter = np.zeros([self.batch_size_np, int(x.shape[1]), 1])
                print(' ')
                for tstep in range(0, windows):

                    r1 = tstep * self.max_seq
                    r2 = r1 + self.max_seq

                    current_x, current_y, current_time, current_meta = \
                        get_feed_time_asynch(x, y, meta, prev_x, prev_y, prev_time, prev_meta, self.max_seq, tstep, self.num_state)

                    if np.all(current_x == 0):
                        print('Skipping: Empty Measurement Set')
                        continue

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
                        # if tstep == 0:
                        #     seqweight[i, :] = m.astype(int) * 0.1
                        seqweight[i, :] = m.astype(int)

                    cur_time = x[:, r1:r2, 0]
                    time_plotter[:, r1:r2, :] = cur_time[:, :, np.newaxis]
                    max_t = np.max(time_plotter[0, :, 0])
                    count += 1

                    if tstep == 0:
                        current_state_estimate, current_cov_estimate, prev_state_estimate, prev_covariance_estimate, initial_Q, initial_R = \
                            initialize_filter(self.batch_size_np, initial_time, initial_meas, prev_time, prev_x, current_time, lla_datar, sensor_vector2, ecef_ref)

                    update = False

                    prev_state_estimate = prev_state_estimate[:, :, self.idxi]
                    current_y = current_y[:, :, self.idxi]
                    prev_y = prev_y[:, :, self.idxi]
                    current_state_estimate = current_state_estimate[:, :, self.idxi]

                    # if tstep == 0:
                    #     pidx = 0
                    #     pidx2 = 4
                    #     plt.plot(initial_time[pidx, :, 0], initial_state_truth[pidx, :, 1], 'r')
                    #     plt.scatter(initial_time[pidx, :, 0], initial_meas_uvw[pidx, :, 1], c='b')
                    #     plt.scatter(prev_time[pidx, :, 0], converted_meas[pidx, :, 1], c='k')
                    #     plt.scatter(initial_time[pidx, -1, 0], prev_state_estimate[pidx, 0, pidx2], c='m')
                    #     plt.scatter(prev_time[pidx, 0, 0], current_state_estimate[pidx, 0, pidx2], c='y')
                    #     plt.plot(current_time[pidx, :, :], current_y[pidx, :, pidx2], 'r')
                    #     # plt.plot(current_time[pidx, :25, :], current_x[pidx, :25, 0], 'b')
                    #     # plt.scatter(current_time[pidx, 0, 0], pred_state[pidx, 0, 0], c='k')

                    if tstep == 0:
                        current_cov_estimate = prev_covariance_estimate[:, -1, :, :]

                        std = 0.3
                        fd.update({self.cell_state1: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.cell_state2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.cell_state3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

                    fd.update({self.measurement[t]: current_x[:, t, :].reshape(-1, self.num_meas) for t in range(self.max_seq)})
                    fd.update({self.prev_measurement: prev_x.reshape(-1, self.num_meas)})
                    fd.update({self.prev_covariance_estimate: prev_covariance_estimate[:, -1, :, :]})
                    fd.update({self.truth_state[t]: current_y[:, t, :].reshape(-1, self.num_state) for t in range(self.max_seq)})
                    fd.update({self.prev_state_truth: prev_y.reshape(-1, self.num_state)})
                    fd.update({self.prev_state_estimate: prev_state_estimate.reshape(-1, self.num_state)})
                    fd.update({self.sensor_ecef: ecef_ref[:, 0, :]})
                    fd.update({self.sensor_lla: lla_datar})
                    fd.update({self.sensor_vector: sensor_vector})
                    fd.update({self.seqlen: seqlen})
                    fd.update({self.int_time: int_time})
                    fd.update({self.update_condition: update})
                    fd.update({self.is_training: True})
                    fd.update({self.seqweightin: seqweight})
                    fd.update({self.P_inp: current_cov_estimate})
                    fd.update({self.Q_inp: initial_Q})
                    fd.update({self.R_inp: initial_R})
                    fd.update({self.R_static: sensor_vector2})
                    fd.update({self.state_input: current_state_estimate.reshape(-1, self.num_state)})
                    fd.update({self.prev_time: prev_time[:, :, 0]})
                    fd.update({self.current_timei[t]: current_time[:, t, :].reshape(-1, 1) for t in range(self.max_seq)})
                    fd.update({self.drop_rate: 1.0})

                    randn = random.random()
                    if randn > 0.9:
                        stateful = True
                    else:
                        stateful = False

                    fd.update({self.learning_rate_inp: lr})

                    t00 = time.time()

                    if epoch < 1 and minibatch_index < int(n_train_batches//4):
                        ittt = 1
                    else:
                        ittt = 1

                    for _ in range(ittt):
                        filter_output, smooth_output, refined_output, q_out_t, q_outs, q_out_refine, _, rmsp, rmsv, rmsa, rmsj, LR, \
                        cov_pos_loss, cov_vel_loss, kalman_cov_loss, maha_loss, MD, trace_loss, rl, \
                        entropy, qt_out, rt_out, at_out, q_loss, state_fwf1, state_fwf2, state_fwf3, new_meas, \
                        y_t_resh, Cz_t, MDP, mvn_inv, state_error, summary_str = \
                            self.sess.run([self.final_state_filter,
                                           self.final_state_smooth,
                                           self.final_state_prediction,
                                           self.final_cov_filter,
                                           self.final_cov_smooth,
                                           self.final_cov_prediction,
                                           self.train_1,
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
                                           self.state_fwf1,
                                           self.state_fwf2,
                                           self.state_fwf3,
                                           self.new_meas,
                                           self.y_t_resh,
                                           self.Cz_t,
                                           self.MDP,
                                           self.mvn_inv,
                                           self.state_error,
                                           summary],
                                          fd)

                    t01 = time.time()
                    sess_run_time = t01 - t00

                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    step += 1

                    print("Epoch: {0:2d} MB: {1:1d} Time: {2:3d} "
                          "RMSP: {3:2.2e} RMSV: {4:2.2e} RMSA: {5:2.2e} RMSJ: {6:2.2e} "
                          "LR: {7:1.2e} ST: {8:1.2f} CPL: {9:1.2f} "
                          "CVL: {10:1.2f} EN: {11:1.2f} QL: {12:1.2f} "
                          "MD: {13:1.2f} RL: {14:1.2f} COV {15:1.2f} ELE {16:1.2f} ".format(epoch, minibatch_index, tstep,
                                                                                            rmsp, rmsv, rmsa, rmsj,
                                                                                            LR, max_t, cov_pos_loss,
                                                                                            cov_vel_loss, entropy, q_loss,
                                                                                            MD, rl, kalman_cov_loss, sess_run_time))

                    current_y = current_y[:, :, self.idxo]
                    prev_y = prev_y[:, :, self.idxo]
                    filter_output = filter_output[:, :, self.idxo]
                    smooth_output = smooth_output[:, :, self.idxo]
                    refined_output = refined_output[:, :, self.idxo]

                    if stateful is True:
                        fd.update({self.cell_state1: state_fwf1[:, -1, :]})
                        fd.update({self.cell_state2: state_fwf2[:, -1, :]})
                        fd.update({self.cell_state3: state_fwf3[:, -1, :]})

                    else:
                        fd.update({self.cell_state1: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.cell_state2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.cell_state3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

                    idx = -1
                    prev_x = np.concatenate([prev_x, current_x], axis=1)
                    prev_x = prev_x[:, idx, np.newaxis, :]

                    prev_state_estimate = filter_output[:, -2, np.newaxis, :]
                    current_state_estimate = filter_output[:, -1, np.newaxis, :]

                    initial_Q = qt_out[:, -1, :, :]
                    initial_R = rt_out[:, -1, :, :]

                    prev_y = np.concatenate([prev_y, current_y], axis=1)
                    prev_y = prev_y[:, idx, np.newaxis, :]

                    prev_time = np.concatenate([prev_time, current_time], axis=1)
                    prev_time = prev_time[:, idx, np.newaxis, :]

                    # prev_cov = np.concatenate([prev_cov[:, -1, np.newaxis, :, :], q_out_t], axis=1)
                    # prev_cov = prev_cov[:, idx, np.newaxis, :, :]

                    prev_covariance_estimate = q_out_t[:, -2, np.newaxis, :, :]
                    current_cov_estimate = q_out_t[:, -1, :, :]

                    if tstep == 0:
                        # out_plot_F = full_final_output[0, np.newaxis, :, :]
                        out_plot_smooth = smooth_output[plt_idx, np.newaxis, :, :]
                        out_plot_filter = filter_output[plt_idx, np.newaxis, :, :]
                        out_plot_refined = refined_output[plt_idx, np.newaxis, :, :]

                        q_plott = q_out_t[plt_idx, np.newaxis, :, :, :]
                        q_plots = q_outs[plt_idx, np.newaxis, :, :, :]
                        q_plotr = q_out_refine[plt_idx, np.newaxis, :, :, :]
                        qt_plot = qt_out[plt_idx, np.newaxis, :, :]
                        rt_plot = rt_out[plt_idx, np.newaxis, :, :]
                        at_plot = at_out[plt_idx, np.newaxis, :, :]
                        # trans_plot = trans_out[0, np.newaxis, :, :]

                        time_vals = current_time[plt_idx, np.newaxis, :, :]
                        meas_plot = new_meas[plt_idx, np.newaxis, :, :]
                        truth_plot = current_y[plt_idx, np.newaxis, :, :]

                    else:
                        # new_vals_F = full_final_output[0, :, :]  # current step
                        new_vals_smooth = smooth_output[plt_idx, :, :]
                        new_vals_filter = filter_output[plt_idx, :, :]
                        new_vals_refined = refined_output[plt_idx, :, :]

                        new_qs = q_outs[plt_idx, :, :, :]
                        new_qt = q_out_t[plt_idx, :, :, :]
                        new_qr = q_out_refine[plt_idx, :, :, :]
                        new_qtt = qt_out[plt_idx, :, :, :]
                        new_rtt = rt_out[plt_idx, :, :, :]
                        new_att = at_out[plt_idx, :, :, :]
                        # new_trans = trans_out[0, :, :]

                        new_time = current_time[plt_idx, :, 0]
                        new_meas = new_meas[plt_idx, :, :]
                        new_truth = current_y[plt_idx, :, :]

                    if tstep > 0:
                        # out_plot_F = np.concatenate([out_plot_F, new_vals_F[np.newaxis, :, :]], axis=1)
                        out_plot_filter = np.concatenate([out_plot_filter, new_vals_filter[np.newaxis, :, :]], axis=1)
                        out_plot_smooth = np.concatenate([out_plot_smooth, new_vals_smooth[np.newaxis, :, :]], axis=1)
                        out_plot_refined = np.concatenate([out_plot_refined, new_vals_refined[np.newaxis, :, :]], axis=1)
                        meas_plot = np.concatenate([meas_plot, new_meas[np.newaxis, :, :]], axis=1)
                        truth_plot = np.concatenate([truth_plot, new_truth[np.newaxis, :, :]], axis=1)
                        time_vals = np.concatenate([time_vals, new_time[np.newaxis, :, np.newaxis]], axis=1)
                        q_plots = np.concatenate([q_plots, new_qs[np.newaxis, :, :, :]], axis=1)
                        q_plott = np.concatenate([q_plott, new_qt[np.newaxis, :, :, :]], axis=1)
                        q_plotr = np.concatenate([q_plotr, new_qr[np.newaxis, :, :, :]], axis=1)
                        qt_plot = np.concatenate([qt_plot, new_qtt[np.newaxis, :, :, :]], axis=1)
                        rt_plot = np.concatenate([rt_plot, new_rtt[np.newaxis, :, :, :]], axis=1)
                        at_plot = np.concatenate([at_plot, new_att[np.newaxis, :, :, :]], axis=1)
                        # trans_plot = np.concatenate([trans_plot, new_trans[np.newaxis, :, :]], axis=1)

                if minibatch_index % self.plot_interval == 0:
                    plotpath = self.plot_dir + 'epoch_' + str(epoch) + '_B_' + str(minibatch_index) + '_step_' + str(step)
                    if ~os.path.isdir(plotpath):
                        os.mkdir(plotpath)

                    output_plots(out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, q_plott, q_plots, q_plotr, time_plotter, plotpath, qt_plot, rt_plot, meas_list[plt_idx], self.max_seq)

                if minibatch_index % 50 == 0 and minibatch_index != 0:
                    print("Saving Weights for Epoch " + str(epoch))
                    save_path = self.saver.save(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(epoch) + '_' + str(step) + ".ckpt", global_step=step)
                    print("Checkpoint saved at :: ", save_path)

    def evaluate(self, x_data, y_data, ecef_ref, lla_datar, epoch, minibatch_index, step):

        x_data = np.concatenate([x_data[:, :, 0, np.newaxis], x_data[:, :, 4:7]], axis=2)  # rae measurements

        y_uvw = y_data[:, :, :3] - ecef_ref
        zero_rows = (y_data[:, :, :3] == 0).all(2)
        for i in range(y_data.shape[0]):
            zz = zero_rows[i, :, np.newaxis]
            y_uvw[i, :, :] = np.where(zz, np.zeros_like(y_uvw[i, :, :]), y_uvw[i, :, :])

        y_data = np.concatenate([y_uvw, y_data[:, :, 3:]], axis=2)

        permute_dims = False
        if permute_dims:
            x_data, y_data = permute_xyz_dims(x_data, y_data)

        s_data = x_data

        x, y, meta, prev_y, prev_x, prev_time, prev_meta, initial_meas, initial_state_truth, initial_time = prepare_batch(0, x_data, y_data, s_data,
                                                                                                                          seq_len=self.max_seq, batch_size=self.batch_size_np,
                                                                                                                          new_batch=True)

        count, _, _, _, _, _, prev_cov, prev_Q, prev_R, q_plot, q_plott, k_plot, out_plot_X, out_plot_F, out_plot_P, time_vals, \
        meas_plot, truth_plot, Q_plot, R_plot, maha_plot = initialize_run_variables(self.batch_size_np, self.max_seq, self.num_state)

        fd = {}

        windows = int((x.shape[1]) / self.max_seq)
        time_plotter = np.zeros([self.batch_size_np, int(x.shape[1]), 1])
        plt.close()
        for tstep in range(0, windows):

            r1 = tstep * self.max_seq
            r2 = r1 + self.max_seq

            current_x, current_y, current_time, current_meta = \
                get_feed_time_asynch(x, y, meta, prev_x, prev_y, prev_time, prev_meta, self.max_seq, tstep, self.num_state)

            if np.all(current_x == 0):
                print('skipping')
                continue

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

            if tstep == 0:
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

                R1 = np.linalg.norm(initial_meas + ecef_ref[:, 0, np.newaxis, :], axis=2, keepdims=True)
                R1 = np.mean(R1, axis=1)
                R1 = np.where(np.less(R1, np.ones_like(R1) * self.RE), np.ones_like(R1) * self.RE, R1)
                rad_temp = np.power(R1, 3)
                GMt1 = np.divide(self.GM, rad_temp)
                acc = get_legendre_np(GMt1, pos + ecef_ref[:, 0, :], R1)

                # acc = (initial_meas[:, 2, :] - 2*initial_meas[:, 1, :] + initial_meas[:, 0, :]) / (dtn**2)
                initial_state = np.expand_dims(np.concatenate([pos, vel, acc, np.random.normal(loc=np.zeros_like(acc), scale=0.001)], axis=1), 1)
                initial_state = initial_state[:, :, self.idxi]

                dt0 = initial_time[:, -1, np.newaxis, :] - initial_time[:, -2, np.newaxis, :]
                dt1 = current_time[:, 0, np.newaxis, :] - prev_time[:, 0, np.newaxis, :]
                current_state, covariance_out, converted_meas, pred_state, pred_covariance, prev_Q, prev_R, prev_cov = \
                    unscented_kalman_np(self.batch_size_np, prev_x.shape[1], initial_state[:, -1, :], prev_x, prev_time, dt0, dt1, lla_datar, sensor_vector2)

                initial_Q = prev_Q
                initial_R = prev_R

                current_state_estimate = current_state[:, :, self.idxo]
                current_cov_estimate = covariance_out[-1]
                prev_state_estimate = initial_state[:, :, self.idxo]
                prev_covariance_estimate = prev_cov

            update = False

            prev_state_estimate = prev_state_estimate[:, :, self.idxi]
            current_y = current_y[:, :, self.idxi]
            prev_y = prev_y[:, :, self.idxi]
            current_state_estimate = current_state_estimate[:, :, self.idxi]

            if tstep == 0:
                # current_state_estimate = prev_state_estimate
                current_cov_estimate = prev_covariance_estimate[:, -1, :, :]

                std = 0.0
                fd.update({self.cell_state1: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                fd.update({self.cell_state2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                fd.update({self.cell_state3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

            fd.update({self.measurement[t]: current_x[:, t, :].reshape(-1, self.num_meas) for t in range(self.max_seq)})
            fd.update({self.prev_measurement: prev_x.reshape(-1, self.num_meas)})
            fd.update({self.prev_covariance_estimate: prev_covariance_estimate[:, -1, :, :]})
            fd.update({self.truth_state[t]: current_y[:, t, :].reshape(-1, self.num_state) for t in range(self.max_seq)})
            fd.update({self.prev_state_truth: prev_y.reshape(-1, self.num_state)})
            fd.update({self.prev_state_estimate: prev_state_estimate.reshape(-1, self.num_state)})
            fd.update({self.sensor_ecef: ecef_ref[:, 0, :]})
            fd.update({self.sensor_lla: lla_datar})
            fd.update({self.sensor_vector: sensor_vector})
            fd.update({self.seqlen: seqlen})
            fd.update({self.int_time: int_time})
            fd.update({self.update_condition: update})
            fd.update({self.is_training: True})
            fd.update({self.seqweightin: seqweight})
            fd.update({self.P_inp: current_cov_estimate})
            fd.update({self.Q_inp: initial_Q})
            fd.update({self.R_inp: initial_R})
            fd.update({self.R_static: sensor_vector2})
            fd.update({self.state_input: current_state_estimate.reshape(-1, self.num_state)})
            fd.update({self.prev_time: prev_time[:, :, 0]})
            fd.update({self.current_timei[t]: current_time[:, t, :].reshape(-1, 1) for t in range(self.max_seq)})
            fd.update({self.drop_rate: 1.0})

            pred_output0, pred_output00, pred_output1, q_out_t, q_out, _, rmsp, rmsv, rmsa, rmsj, LR, \
            cov_pos_loss, cov_vel_loss, kalman_cov_loss, maha_loss, MD, trace_loss, rl, \
            entropy, qt_out, rt_out, at_out, q_loss, state_fwf, state_fwf2, state_fwf3, new_meas, \
            y_t_resh, Cz_t, MDP, mvn_inv, state_error, num_el = \
                self.sess.run([self.final_state_filter,
                               self.final_state_filter,
                               self.final_state_filter,
                               self.final_cov_filter,
                               self.final_cov_filter,
                               self.train_q,
                               self.rmse_pos,
                               self.rmse_vel,
                               self.rmse_acc,
                               self.rmse_jer,
                               self.learning_rate_inp,
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
                               self.state_fwf3,
                               self.new_meas,
                               self.y_t_resh,
                               self.Cz_t,
                               self.MDP,
                               self.mvn_inv,
                               self.state_error,
                               self.num_el],
                              fd)

            print("Epoch: {0:2d} MB: {1:1d} Time: {2:3d} "
                  "RMSP: {3:2.2e} RMSV: {4:2.2e} RMSA: {5:2.2e} RMSJ: {6:2.2e} "
                  "LR: {7:1.2e} ST: {8:1.2f} CPL: {9:1.2f} "
                  "CVL: {10:1.2f} EN: {11:1.2f} QL: {12:1.2f} "
                  "MD: {13:1.2f} RL: {14:1.2f} COV {15:1.2f} ".format(epoch, minibatch_index, tstep,
                                                                      rmsp, rmsv, rmsa, rmsj,
                                                                      LR, max_t, cov_pos_loss,
                                                                      cov_vel_loss, entropy, q_loss,
                                                                      MD, rl, kalman_cov_loss))

            current_y = current_y[:, :, self.idxo]
            prev_y = prev_y[:, :, self.idxo]
            filter_output = filter_output[:, :, self.idxo]
            smooth_output = smooth_output[:, :, self.idxo]

            if stateful is True:
                fd.update({self.cell_state1: state_fwf1[:, -1, :]})
                fd.update({self.cell_state2: state_fwf2[:, -1, :]})
                fd.update({self.cell_state3: state_fwf3[:, -1, :]})

            else:
                fd.update({self.cell_state1: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                fd.update({self.cell_state2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                fd.update({self.cell_state3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

            idx = -1
            prev_x = np.concatenate([prev_x, current_x], axis=1)
            prev_x = prev_x[:, idx, np.newaxis, :]

            prev_state_estimate = filter_output[:, -3, np.newaxis, :]
            current_state_estimate = filter_output[:, -2, np.newaxis, :]

            initial_Q = qt_out[:, -1, :, :]
            initial_R = rt_out[:, -1, :, :]

            prev_y = np.concatenate([prev_y, current_y], axis=1)
            prev_y = prev_y[:, idx, np.newaxis, :]

            prev_time = np.concatenate([prev_time, current_time], axis=1)
            prev_time = prev_time[:, idx, np.newaxis, :]

            prev_cov = np.concatenate([prev_cov[:, -1, np.newaxis, :, :], q_out_t], axis=1)
            prev_cov = prev_cov[:, idx, np.newaxis, :, :]

            prev_covariance_estimate = q_out_t[:, -3, np.newaxis, :, :]
            current_cov_estimate = q_out_t[:, -2, :, :]

            if tstep == 0:
                # out_plot_F = full_final_output[0, np.newaxis, :, :]
                out_plot_smooth = smooth_output[plt_idx, np.newaxis, :, :]
                out_plot_filter = filter_output[plt_idx, np.newaxis, :, :]

                q_plott = q_out_t[plt_idx, np.newaxis, :, :, :]
                q_plot = q_out[plt_idx, np.newaxis, :, :, :]
                qt_plot = qt_out[plt_idx, np.newaxis, :, :]
                rt_plot = rt_out[plt_idx, np.newaxis, :, :]
                at_plot = at_out[plt_idx, np.newaxis, :, :]
                # trans_plot = trans_out[0, np.newaxis, :, :]

                time_vals = current_time[plt_idx, np.newaxis, :, :]
                meas_plot = new_meas[plt_idx, np.newaxis, :, :]
                truth_plot = current_y[plt_idx, np.newaxis, :, :]

            else:
                # new_vals_F = full_final_output[0, :, :]  # current step
                new_vals_smooth = smooth_output[plt_idx, :, :]
                new_vals_filter = filter_output[plt_idx, :, :]
                new_q = q_out[plt_idx, :, :, :]
                new_qt = q_out_t[plt_idx, :, :, :]
                new_qtt = qt_out[plt_idx, :, :, :]
                new_rtt = rt_out[plt_idx, :, :, :]
                new_att = at_out[plt_idx, :, :, :]
                # new_trans = trans_out[0, :, :]

                new_time = current_time[plt_idx, :, 0]
                new_meas = new_meas[plt_idx, :, :]
                new_truth = current_y[plt_idx, :, :]

            if tstep > 0:
                # out_plot_F = np.concatenate([out_plot_F, new_vals_F[np.newaxis, :, :]], axis=1)
                out_plot_filter = np.concatenate([out_plot_filter, new_vals_filter[np.newaxis, :, :]], axis=1)
                out_plot_smooth = np.concatenate([out_plot_smooth, new_vals_smooth[np.newaxis, :, :]], axis=1)
                meas_plot = np.concatenate([meas_plot, new_meas[np.newaxis, :, :]], axis=1)
                truth_plot = np.concatenate([truth_plot, new_truth[np.newaxis, :, :]], axis=1)
                time_vals = np.concatenate([time_vals, new_time[np.newaxis, :, np.newaxis]], axis=1)
                q_plot = np.concatenate([q_plot, new_q[np.newaxis, :, :, :]], axis=1)
                q_plott = np.concatenate([q_plott, new_qt[np.newaxis, :, :, :]], axis=1)
                qt_plot = np.concatenate([qt_plot, new_qtt[np.newaxis, :, :, :]], axis=1)
                rt_plot = np.concatenate([rt_plot, new_rtt[np.newaxis, :, :, :]], axis=1)
                at_plot = np.concatenate([at_plot, new_att[np.newaxis, :, :, :]], axis=1)
                # trans_plot = np.concatenate([trans_plot, new_trans[np.newaxis, :, :]], axis=1)

        plotpath = self.plot_eval_dir + '/epoch_' + str(epoch) + '_eval_B_' + str(minibatch_index) + '_step_' + str(step)
        if not os.path.isdir(plotpath):
            os.mkdir(plotpath)
        output_plots(out_plot_filter, out_plot_smooth, meas_plot, truth_plot, q_plott, q_plot, time_plotter, plotpath, qt_plot, rt_plot, meas_list[plt_idx], self.max_seq)

        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        print("Saving filter Weights for epoch" + str(epoch))
        save_path = self.saver.save(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(epoch) + '_' + str(step) + ".ckpt", global_step=step)
        print("Checkpoint saved at: ", save_path)

    def test(self, data_rate, max_exp_seq):

        self.data_rate = data_rate
        self.max_exp_seq = max_exp_seq
        tf.global_variables_initializer().run()

        save_files = os.listdir(self.checkpoint_dir + '/')
        save_files = natsorted(save_files, reverse=True)
        recent = str.split(save_files[1], '_')
        start_epoch = recent[2]
        step = str.split(recent[3], '.')[0]
        print("Resuming run from epoch " + str(start_epoch) + ' and step ' + str(step))
        step = int(step)
        print('Loading filter...')
        self.saver = tf.train.import_meta_graph(self.checkpoint_dir + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step) + '.meta')
        self.saver.restore(self.sess, self.checkpoint_dir + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step))
        print("filter restored.")

        if self.preprocessed is True:
            ds = DataServerPrePro(self.train_dir, self.test_dir)
        else:
            ds = DataServerLive(self.meas_dir, self.state_dir, decimate_data=self.decimate_data)

        n_train_batches = int(ds.num_examples_train)

        for minibatch_index in range(n_train_batches):

            testing = True
            print('Testing filter for batch ' + str(minibatch_index))

            # Data is unnormalized at this point
            if self.preprocessed is False:
                x_data, y_data, batch_number, total_batches, ecef_ref, lla_data, sensor_vector, sensor_vector2, meas_list = ds.load(batch_size=self.batch_size_np, constant=self.constant, test=testing,
                                                                                                                                    max_seq_len=self.max_exp_seq, HZ=self.data_rate)
                lla_datar = copy.copy(lla_data)
                ecef_ref = np.ones([self.batch_size_np, y_data.shape[1], 3]) * ecef_ref[:, np.newaxis, :]

            else:
                x_data, y_data, ecef_ref, lla_data = ds.load(batch_size=self.batch_size_np, constant=self.constant, test=testing)
                lla_datar = copy.copy(lla_data[:, 0, :])

            lla_datar[:, 0] = lla_datar[:, 0] * np.pi / 180
            lla_datar[:, 1] = lla_datar[:, 1] * np.pi / 180

            x_data = np.concatenate([x_data[:, :, 0, np.newaxis], x_data[:, :, 4:7]], axis=2)  # rae measurements

            y_uvw = y_data[:, :, :3] - ecef_ref
            # y_enu = np.zeros_like(y_uvw)
            # y_rae = np.zeros_like(y_uvw)
            zero_rows = (y_data[:, :, :3] == 0).all(2)
            for i in range(y_data.shape[0]):
                zz = zero_rows[i, :, np.newaxis]
                y_uvw[i, :, :] = np.where(zz, np.zeros_like(y_uvw[i, :, :]), y_uvw[i, :, :])

                y_data = np.concatenate([y_uvw, y_data[:, :, 3:]], axis=2)

            s_data = x_data

            x, y, meta, prev_y, prev_x, prev_time, prev_meta, initial_meas, initial_state_truth, initial_time, max_length = prepare_batch(0, x_data, y_data, s_train,
                                                                                                                                          seq_len=self.max_seq, batch_size=self.batch_size_np,
                                                                                                                                          new_batch=True)

            count, _, _, _, _, _, q_plot, q_plott, k_plot, out_plot_filter, out_plot_F, out_plot_smooth, time_vals, \
            meas_plot, truth_plot, Q_plot, R_plot, maha_plot = initialize_run_variables(self.batch_size_np, self.max_seq, self.num_state)

            fd = {}

            # time_plotter = np.zeros([self.batch_size_np, int(x.shape[1]), 1])

            tstep = 0
            r1 = tstep * self.max_seq
            r2 = r1 + self.max_seq

            # x_data, y_data, time_data, meta_data = \
            #     get_feed_time_asynch(x, y, meta, prev_x, prev_y, prev_time, prev_meta, self.max_seq, tstep, self.num_state)

            x_data = x[:, :, 1:]
            time_data = x[:, :, 0, np.newaxis]
            y_data = y
            meta_data = meta

            seqlen = np.ones(shape=[self.batch_size_np, ])
            int_time = np.zeros(shape=[self.batch_size_np, x.shape[1]])
            seqweight = np.zeros(shape=[self.batch_size_np, x.shape[1]])

            for i in range(self.batch_size_np):
                current_yt = y_data[i, :, :3]
                m = ~(current_yt == 0).all(1)
                yf = current_yt[m]
                seq = yf.shape[0]
                seqlen[i] = seq
                int_time[i, :] = range(r1, r2)
                seqweight[i, :] = m.astype(int)

            time_plotter = x_data[:, :, 0, np.newaxis]
            max_t = np.max(time_plotter[0, :, 0])

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

            R1 = np.linalg.norm(initial_meas + ecef_ref[:, 0, np.newaxis, :], axis=2, keepdims=True)
            R1 = np.mean(R1, axis=1)
            R1 = np.where(np.less(R1, np.ones_like(R1) * self.RE), np.ones_like(R1) * self.RE, R1)
            rad_temp = np.power(R1, 3)
            GMt1 = np.divide(self.GM, rad_temp)
            acc = get_legendre_np(GMt1, pos + ecef_ref[:, 0, :], R1)

            # acc = (initial_meas[:, 2, :] - 2*initial_meas[:, 1, :] + initial_meas[:, 0, :]) / (dtn**2)
            initial_state = np.expand_dims(np.concatenate([pos, vel, acc, np.random.normal(loc=np.zeros_like(acc), scale=0.001)], axis=1), 1)
            initial_state = initial_state[:, :, self.idxi]

            dt0 = initial_time[:, -1, np.newaxis, :] - initial_time[:, -2, np.newaxis, :]
            dt1 = time_data[:, 0, np.newaxis, :] - prev_time[:, 0, np.newaxis, :]
            current_state, covariance_out, converted_meas, pred_state, pred_covariance, prev_Q, prev_R, prev_cov = \
                unscented_kalman_np(self.batch_size_np, prev_x.shape[1], initial_state[:, -1, :], prev_x, prev_time, dt0, dt1, lla_datar, sensor_vector2)

            initial_Q = prev_Q
            initial_R = prev_R

            current_state_estimate = current_state[:, :, self.idxo]
            current_cov_estimate = covariance_out[-1]
            prev_state_estimate = initial_state[:, :, self.idxo]
            prev_covariance_estimate = prev_cov

            fd.update({self.is_training: False})
            stateful = True
            update = False

            for step in range(0, time_data.shape[1]):

                if step == 0:
                    std = 0.0
                    fd.update({self.state_fw_in_state: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                    fd.update({self.state_fw_in_state2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                    fd.update({self.state_fw_in_state3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

                prev_state = copy.copy(prev_y)
                prev_meas = copy.copy(prev_x)

                current_y = y_data[:, step, self.idxi]
                prev_state = prev_state[:, :, self.idxi]
                prev_state_estimate = prev_state_estimate[:, :, self.idxi]
                current_state_estimate = current_state_estimate[:, :, self.idxi]
                current_x = x_data[:, step, np.newaxis, :]

                if np.all(current_x == 0):
                    continue

                current_time = time_data[:, step, np.newaxis, :]
                current_int = int_time[:, step, tfna]
                current_weight = seqweight[:, step, tfna]

                fd.update({self.measurement[0]: current_x.reshape(-1, self.num_meas)})
                fd.update({self.prev_measurement: prev_x.reshape(-1, self.num_meas)})
                fd.update({self.prev_covariance_estimate: prev_covariance_estimate[:, -1, :, :]})
                fd.update({self.truth_state[0]: current_y.reshape(-1, self.num_state)})
                fd.update({self.prev_state_truth: prev_y.reshape(-1, self.num_state)})
                fd.update({self.prev_state_estimate: prev_state_estimate.reshape(-1, self.num_state)})
                fd.update({self.sensor_ecef: ecef_ref[:, 0, :]})
                fd.update({self.sensor_lla: lla_datar})
                fd.update({self.sensor_vector: sensor_vector})
                fd.update({self.seqlen: seqlen})
                fd.update({self.int_time: current_int})
                fd.update({self.update_condition: update})
                fd.update({self.is_training: True})
                fd.update({self.seqweightin: current_weight})
                fd.update({self.P_inp: current_cov_estimate})
                fd.update({self.Q_inp: initial_Q})
                fd.update({self.R_inp: initial_R})
                fd.update({self.R_static: sensor_vector2})
                fd.update({self.state_input: current_state_estimate.reshape(-1, self.num_state)})
                fd.update({self.prev_time: prev_time[:, :, 0]})
                fd.update({self.current_timei[0]: current_time.reshape(-1, 1)})
                fd.update({self.drop_rate: 1.0})

                filter_output, smooth_output, refined_output, q_out_t, q_outs, q_out_refine, rmsp, rmsv, rmsa, rmsj, LR, \
                cov_pos_loss, cov_vel_loss, kalman_cov_loss, maha_loss, MD, trace_loss, rl, \
                entropy, qt_out, rt_out, at_out, q_loss, state_fwf1, state_fwf2, state_fwf3, new_meas, \
                y_t_resh, Cz_t, MDP, mvn_inv, state_error = \
                    self.sess.run([self.final_state_filter,
                                   self.final_state_smooth,
                                   self.final_state_prediction,
                                   self.final_cov_filter,
                                   self.final_cov_smooth,
                                   self.final_cov_prediction,
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
                                   self.state_fwf1,
                                   self.state_fwf2,
                                   self.state_fwf3,
                                   self.new_meas,
                                   self.y_t_resh,
                                   self.Cz_t,
                                   self.MDP,
                                   self.mvn_inv,
                                   self.state_error],
                                  fd)

                print("Test: {0:2d} MB: {1:1d} Time: {2:3d} "
                      "RMSP: {3:2.2e} RMSV: {4:2.2e} RMSA: {5:2.2e} RMSJ: {6:2.2e} "
                      "LR: {7:1.2e} ST: {8:1.2f} CPL: {9:1.2f} "
                      "CVL: {10:1.2f} EN: {11:1.2f} QL: {12:1.2f} "
                      "MD: {13:1.2f} RL: {14:1.2f} COV {15:1.2f} ".format(0, minibatch_index, step,
                                                                          rmsp, rmsv, rmsa, rmsj,
                                                                          LR, max_t, cov_pos_loss,
                                                                          cov_vel_loss, entropy, q_loss,
                                                                          MD, rl, kalman_cov_loss))


                current_y = current_y[:, np.newaxis, self.idxo]
                prev_y = prev_y[:, :, self.idxo]
                filter_output = filter_output[:, :, self.idxo]
                smooth_output = smooth_output[:, :, self.idxo]
                refined_output = refined_output[:, :, self.idxo]

                if stateful is True:
                    fd.update({self.cell_state1: state_fwf1[:, -1, :]})
                    fd.update({self.cell_state2: state_fwf2[:, -1, :]})
                    fd.update({self.cell_state3: state_fwf3[:, :]})

                else:
                    fd.update({self.cell_state1: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                    fd.update({self.cell_state2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                    fd.update({self.cell_state3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

                idx = -1
                prev_x = np.concatenate([prev_x, current_x], axis=1)
                prev_x = prev_x[:, idx, np.newaxis, :]

                prev_state_estimate = current_state_estimate
                current_state_estimate = filter_output[:, -1, np.newaxis, :]

                initial_Q = qt_out[:, -1, :, :]
                initial_R = rt_out[:, -1, :, :]

                prev_y = np.concatenate([prev_y, current_y], axis=1)
                prev_y = prev_y[:, idx, np.newaxis, :]

                prev_time = np.concatenate([prev_time, current_time], axis=1)
                prev_time = prev_time[:, idx, np.newaxis, :]

                prev_covariance_estimate = prev_cov[:, -1, np.newaxis, :, :]

                prev_cov = np.concatenate([prev_cov[:, -1, np.newaxis, :, :], q_out_t], axis=1)
                prev_cov = prev_cov[:, idx, np.newaxis, :, :]

                current_cov_estimate = q_out_t[:, -1, :, :]

                if step == 0:
                    # out_plot_F = full_final_output[0, np.newaxis, :, :]
                    out_plot_smooth = smooth_output[plt_idx, np.newaxis, :, :]
                    out_plot_filter = filter_output[plt_idx, np.newaxis, :, :]
                    out_plot_refined = refined_output[plt_idx, np.newaxis, :, :]

                    q_plott = q_out_t[plt_idx, np.newaxis, :, :, :]
                    q_plots = q_outs[plt_idx, np.newaxis, :, :, :]
                    q_plotr = q_out_refine[plt_idx, np.newaxis, :, :, :]
                    qt_plot = qt_out[plt_idx, np.newaxis, :, :]
                    rt_plot = rt_out[plt_idx, np.newaxis, :, :]
                    at_plot = at_out[plt_idx, np.newaxis, :, :]
                    # trans_plot = trans_out[0, np.newaxis, :, :]

                    time_vals = current_time[plt_idx, np.newaxis, :, :]
                    meas_plot = new_meas[plt_idx, np.newaxis, :, :]
                    truth_plot = current_y[plt_idx, np.newaxis, :, :]

                else:
                    # new_vals_F = full_final_output[0, :, :]  # current step
                    new_vals_smooth = smooth_output[plt_idx, :, :]
                    new_vals_filter = filter_output[plt_idx, :, :]
                    new_vals_refined = refined_output[plt_idx, :, :]

                    new_qs = q_outs[plt_idx, :, :, :]
                    new_qt = q_out_t[plt_idx, :, :, :]
                    new_qr = q_out_refine[plt_idx, :, :, :]
                    new_qtt = qt_out[plt_idx, :, :, :]
                    new_rtt = rt_out[plt_idx, :, :, :]
                    new_att = at_out[plt_idx, :, :, :]
                    # new_trans = trans_out[0, :, :]

                    new_time = current_time[plt_idx, :, 0]
                    new_meas = new_meas[plt_idx, :, :]
                    new_truth = current_y[plt_idx, :, :]

                if step > 0:
                    # out_plot_F = np.concatenate([out_plot_F, new_vals_F[np.newaxis, :, :]], axis=1)
                    out_plot_filter = np.concatenate([out_plot_filter, new_vals_filter[np.newaxis, :, :]], axis=1)
                    out_plot_smooth = np.concatenate([out_plot_smooth, new_vals_smooth[np.newaxis, :, :]], axis=1)
                    out_plot_refined = np.concatenate([out_plot_refined, new_vals_refined[np.newaxis, :, :]], axis=1)
                    meas_plot = np.concatenate([meas_plot, new_meas[np.newaxis, :, :]], axis=1)
                    truth_plot = np.concatenate([truth_plot, new_truth[np.newaxis, :, :]], axis=1)
                    time_vals = np.concatenate([time_vals, new_time[np.newaxis, :, np.newaxis]], axis=1)
                    q_plots = np.concatenate([q_plots, new_qs[np.newaxis, :, :, :]], axis=1)
                    q_plott = np.concatenate([q_plott, new_qt[np.newaxis, :, :, :]], axis=1)
                    q_plotr = np.concatenate([q_plotr, new_qr[np.newaxis, :, :, :]], axis=1)
                    qt_plot = np.concatenate([qt_plot, new_qtt[np.newaxis, :, :, :]], axis=1)
                    rt_plot = np.concatenate([rt_plot, new_rtt[np.newaxis, :, :, :]], axis=1)
                    at_plot = np.concatenate([at_plot, new_att[np.newaxis, :, :, :]], axis=1)
                    # trans_plot = np.concatenate([trans_plot, new_trans[np.newaxis, :, :]], axis=1)


            plotpath = self.plot_test_dir + '/epoch_' + str(9999) + '_TEST_B_' + str(minibatch_index)
            if ~os.path.isdir(plotpath):
                    os.mkdir(plotpath)

            output_plots(out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, q_plott, q_plots, q_plotr, time_plotter, plotpath, qt_plot, rt_plot, meas_list[plt_idx],
                         self.max_seq)
