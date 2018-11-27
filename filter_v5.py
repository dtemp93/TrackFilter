import math
import time
import random

from tensorflow.contrib.layers import fully_connected as FCL

from helper import *
from data_loaders import DataServerPrePro, DataServerLive
from plotting import *
from propagation_utils import *

from natsort import natsorted

tfd = tfp.distributions
tfna = tf.newaxis

setattr(tfc.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)


class Filter(object):
    def __init__(self, sess, state_type='GRU', mode='training',
                 data_dir='', filter_name='', save_dir='',
                 F_hidden=12, num_state=12, num_meas=3, max_seq=2, num_mixtures=4,
                 max_epoch=10000, batch_size=10, learning_rate=1e-3,
                 plot_interval=1, checkpoint_interval=50, dropout_rate=0.5,
                 constant=False, decimate_data=False):

        self.sess = sess
        self.mode = mode
        self.max_seq = max_seq
        self.num_mixtures = num_mixtures
        self.F_hidden = F_hidden
        self.num_state = num_state
        self.num_meas = num_meas
        self.plot_dir = save_dir + '/plots/'
        self.plot_eval_dir = save_dir + '/plots_eval/'
        self.plot_test_dir = save_dir + '/plots_test/'
        self.checkpoint_dir = save_dir + '/checkpoints/'
        self.log_dir = save_dir + '/logs/'
        self.max_epoch = max_epoch
        self.state_type = state_type
        self.filter_name = filter_name
        self.constant = constant
        self.learning_rate_main = learning_rate
        self.dropout_rate_main = dropout_rate

        self.batch_size_np = batch_size
        self.meas_dir = 'NoiseRAE/'
        self.state_dir = 'Translate/'

        self.root = 'D:/TrackFilterData'
        self.data_dir = data_dir

        self.train_dir = self.root + '/OOPBroad/5k25hz_oop_broad_data'
        self.test_dir = self.root + '/OOPBroad/5k25hz_oop_broad_data'

        self.preprocessed = False
        self.plot_interval = plot_interval
        self.checkpoint_interval = checkpoint_interval
        self.convert_to_uvw = True

        self.decimate_data = decimate_data

        self.global_step = tf.Variable(initial_value=0, name="global_step", trainable=False, collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES], dtype=tf.int32)

        self.idxi = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        self.idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

        # Meta Variables
        self.plen = int(self.max_seq)
        self.pi_val = tf.constant(math.pi, dtype=tf.float64)

        total_samples = 45000
        avg_seq_len = 3000
        self.decay_steps = (total_samples / self.batch_size_np) * (avg_seq_len / self.max_seq)
        self.e_sq = 0.00669437999014132
        self.RE = 6378137
        self.GM = 398600441890000

        self.vdtype = tf.float64
        self.vdp_np = np.float64

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

    def alpha(self, int_time, dt, pstate, meas_rae, cov_est, Q_m1, R_m1, LLA, sensor_onehot, sensor_noise, states):

        with tf.variable_scope('alpha'):

            state1 = states[0]
            state2 = states[1]
            state3 = states[2]

            lat = LLA[:, 0, tfna]
            lon = LLA[:, 1, tfna]
            alt = LLA[:, 2, tfna]

            chi = self.RE / tf.sqrt(1 - self.e_sq * (tf.sin(lat)) ** 2)

            xs = (chi + alt) * tf.cos(lat) * tf.cos(lon)
            ys = (chi + alt) * tf.cos(lat) * tf.sin(lon)
            zs = (alt + chi * (1 - self.e_sq)) * tf.sin(lat)

            ecef_ref = tf.concat([xs, ys, zs], axis=1)  # This is correct

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
            meas_ecef = meas_uvw + ecef_ref

            R1 = tf.linalg.norm(meas_ecef, axis=1, keepdims=True)
            R1 = tf.where(tf.less(R1, tf.ones_like(R1) * self.RE), tf.ones_like(R1) * self.RE, R1)
            rad_temp = tf.pow(R1, 3)
            GMt1 = tf.divide(self.GM, rad_temp)
            gravity = get_legendre(GMt1, meas_ecef, R1, self.vdtype)

            altitude = (R1 - self.RE)

            rho0 = 1.22  # kg / m**3
            k0 = 0.14141e-3
            area = 0.25  # / FLAGS.RE  # meter squared
            cd = 0.03  # unitless

            _, At, _, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                                 dimension=int(self.num_state / 3),
                                 sjix=self.om[:, :, 0] * 0.01 ** 2,
                                 sjiy=self.om[:, :, 0] * 0.01 ** 2,
                                 sjiz=self.om[:, :, 0] * 0.01 ** 2,
                                 aji=self.om[:, :, 0] * 1.0)

            prop_state = tf.matmul(At, pstate[:, :, tfna])
            # prop_cov = tf.matmul(tf.matmul(At, cov_est), At, transpose_b=True)

            pos_part = tf.matmul(self.meas_mat, prop_state)
            vel_part = tf.concat([prop_state[:, 1, tfna], prop_state[:, 5, tfna], prop_state[:, 9, tfna]], axis=1)
            acc_part = tf.concat([prop_state[:, 2, tfna], prop_state[:, 6, tfna], prop_state[:, 10, tfna]], axis=1)
            jer_part = tf.concat([prop_state[:, 3, tfna], prop_state[:, 7, tfna], prop_state[:, 11, tfna]], axis=1)
            pre_residual = meas_uvw[:, :, tfna] - pos_part

            gs_input = acc_part[:, :, 0] / gravity
            gs_input = tf.where(tf.is_nan(gs_input), tf.ones_like(gs_input), gs_input)

            rho = rho0 * tf.exp(-k0 * altitude)
            acc_drag = tf.negative(0.5 * rho * vel_part[:, :, 0] * tf.norm(vel_part[:, :, 0], axis=1, keepdims=True) * (cd * area))

            cov_diag = tf.matrix_diag_part(cov_est)
            # Q_diag = tf.matrix_diag_part(Q_est)
            # R_diag = tf.matrix_diag_part(R_est)

            cov_diag_n = cov_diag / tf.ones_like(cov_diag)
            # prop_state_n = prop_state / (tf.ones_like(prop_state) * 10000)
            pos_part_n = pos_part[:, :, 0] / (tf.ones_like(pos_part[:, :, 0]) * 10000)
            vel_part_n = vel_part[:, :, 0] / (tf.ones_like(vel_part[:, :, 0]) * 1000)
            acc_part_n = acc_part[:, :, 0] / (tf.ones_like(acc_part[:, :, 0]) * 100)
            jer_part_n = jer_part[:, :, 0] / (tf.ones_like(jer_part[:, :, 0]) * 50)

            # eet = tf.matmul(pre_residual, pre_residual, transpose_b=True)
            # HPH = tf.matmul(tf.matmul(self.meas_mat, prop_cov), self.meas_mat, transpose_b=True)

            # Rt_temp_uvw = eet + HPH

            # lat = LLA[:, 0, tfna, tfna]
            # lon = LLA[:, 1, tfna, tfna]

            # uvw_to_enu = uvw2enu_tf(lat, lon)

            # y_enu = tf.squeeze(tf.matmul(uvw_to_enu, pos_part), -1)

            # rae_to_enu = rae2enu_tf(y_enu, self.pi_val)
            # enu_to_rae = tf.matrix_transpose(rae_to_enu)

            # enu_cov = tf.matmul(tf.matmul(uvw_to_enu, Rt_temp_uvw), uvw_to_enu, transpose_b=True)

            # Rt_temp_rae = tf.matmul(tf.matmul(enu_to_rae, enu_cov), enu_to_rae, transpose_b=True)

            # Q_diag_n = Q_diag / tf.ones_like(Q_diag)
            # R_diag_n = R_diag / tf.ones_like(R_diag)
            pre_res_n = pre_residual[:, :, 0] / tf.ones_like(pre_residual[:, :, 0])
            # meas_uvw_n = meas_uvw / (tf.ones_like(meas_uvw) * 10000)
            altitude_n = altitude / (tf.ones_like(altitude) * 1000)
            gravity_n = gravity / (tf.ones_like(gravity) * 100)

            rnn_inpa = tf.concat([dt, pre_res_n, cov_diag_n, pos_part_n, vel_part_n, acc_part_n, jer_part_n, altitude_n, gravity_n, gs_input, acc_drag, A, E], axis=1)
            # rnn_inpb = tf.concat([dt, pre_res_n, cov_diag_n, pos_part_n, vel_part_n, acc_part_n, jer_part_n], axis=1)

            inp_width = rnn_inpa.shape[1].value

            rnn_inpa0 = FCL(rnn_inpa, inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpa0/q_cov', reuse=tf.AUTO_REUSE)
            rnn_inpa1 = FCL(tf.concat([rnn_inpa0], axis=1), inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpa1/q_cov', reuse=tf.AUTO_REUSE)
            rnn_inpa2 = FCL(tf.concat([rnn_inpa0, rnn_inpa1], axis=1), self.F_hidden, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpa2/q_cov', reuse=tf.AUTO_REUSE)

            rnn_inpb0 = FCL(rnn_inpa, inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpb0/r_cov', reuse=tf.AUTO_REUSE)
            rnn_inpb1 = FCL(tf.concat([rnn_inpb0], axis=1), inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpb1/r_cov', reuse=tf.AUTO_REUSE)
            rnn_inpb2 = FCL(tf.concat([rnn_inpb0, rnn_inpb1], axis=1), self.F_hidden, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpb2/r_cov', reuse=tf.AUTO_REUSE)

            rnn_inpc0 = FCL(rnn_inpa, inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpc0/input_u', reuse=tf.AUTO_REUSE)
            rnn_inpc1 = FCL(tf.concat([rnn_inpc0], axis=1), inp_width, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpc1/input_u', reuse=tf.AUTO_REUSE)
            rnn_inpc2 = FCL(tf.concat([rnn_inpc0, rnn_inpc1], axis=1), self.F_hidden, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inpc2/input_u', reuse=tf.AUTO_REUSE)

            # rnn_inpa = tfc.layers.dropout(rnn_inpa2, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_inputsa/q_cov')
            # rnn_inpb = tfc.layers.dropout(rnn_inpb2, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_inputsa/r_cov')
            # rnn_inpc = tfc.layers.dropout(rnn_inpc2, keep_prob=self.drop_rate, is_training=self.is_training, scope='dropout_inputsa/input_u')

            with tf.variable_scope('Cell_1/q_cov', reuse=tf.AUTO_REUSE):
                (outa, state1) = self.source_fwf(rnn_inpa2, state=state1)

            with tf.variable_scope('Cell_2/r_cov', reuse=tf.AUTO_REUSE):
                (outb, state2) = self.source_fwf2(rnn_inpb2, state=state2)

            with tf.variable_scope('Cell_3/input_u', reuse=tf.AUTO_REUSE):
                (outc, state3) = self.source_fwf3(rnn_inpc2, state=state3)

            nv = outb.shape[1].value // 3

            rm0 = FCL(tf.concat([outb], axis=1), outb.shape[1].value, activation_fn=tf.nn.elu, scope='r_cov/1', reuse=tf.AUTO_REUSE)
            # rm1 = FCL(rm0, outb.shape[1].value, activation_fn=tf.nn.elu, scope='r_cov/2', reuse=tf.AUTO_REUSE)
            # d_mult = (tf.nn.sigmoid(rm1[:, -1:], 'r_cov/d_mult') * 50) + tf.ones_like(rm1[:, -1:]) * 1
            # rd = tril_with_diag_softplus_and_shift(rm1[:, :6], diag_shift=0.01, diag_mult=None, name='r_cov/tril')
            asr = FCL(rm0[:, :nv], self.num_mixtures, activation_fn=tf.nn.softmax, scope='r_cov/alphar', reuse=tf.AUTO_REUSE)
            asa = FCL(rm0[:, nv:2 * nv], self.num_mixtures, activation_fn=tf.nn.softmax, scope='r_cov/alphaa', reuse=tf.AUTO_REUSE)
            ase = FCL(rm0[:, 2 * nv:], self.num_mixtures, activation_fn=tf.nn.softmax, scope='r_cov/alphae', reuse=tf.AUTO_REUSE)

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
            rd_mult = tf.concat([ones * sr, ones * tf.sqrt(R) * sa, ones * tf.sqrt(R) * se], axis=1)
            rd = (sensor_noise * rd_mult)

            # rdist = tfd.MultivariateNormalTriL(loc=None, scale_tril=rd)
            rdist = tfd.MultivariateNormalDiag(loc=None, scale_diag=rd)
            Rt = rdist.covariance()

            # u = tf.zeros([self.batch_size, 3], self.vdtype)

            nv = outa.shape[1].value // 3
            qm0 = FCL(outa, outa.shape[1].value, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='q_cov/1', reuse=tf.AUTO_REUSE)
            # qm1 = FCL(qm0, outa.shape[1].value, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='q_cov/2', reuse=tf.AUTO_REUSE)

            qmx = FCL(qm0[:, :nv], self.num_mixtures, activation_fn=tf.nn.softmax, weights_initializer=tf.initializers.variance_scaling, scope='q_cov/x', reuse=tf.AUTO_REUSE)
            qmy = FCL(qm0[:, nv:2 * nv], self.num_mixtures, activation_fn=tf.nn.softmax, weights_initializer=tf.initializers.variance_scaling, scope='q_cov/y', reuse=tf.AUTO_REUSE)
            qmz = FCL(qm0[:, 2 * nv:], self.num_mixtures, activation_fn=tf.nn.softmax, weights_initializer=tf.initializers.variance_scaling, scope='q_cov/z', reuse=tf.AUTO_REUSE)

            sjx_l = list()
            sjy_l = list()
            sjz_l = list()
            sjxl = [0.0001, 0.5, 25, 250]
            sjyl = [0.0001, 0.5, 25, 250]
            sjzl = [0.0001, 0.5, 25, 250]

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

            am0 = FCL(tf.concat([outc], axis=1), outc.shape[1].value, activation_fn=tf.nn.elu, scope='input_u/1', reuse=tf.AUTO_REUSE)
            # rm1 = FCL(rm0, 3, activation_fn=None, scope='r_cov/2', reuse=tf.AUTO_REUSE)
            # d_mult = (tf.nn.sigmoid(rm1[:, -1:], 'r_cov/d_mult') * 50) + tf.ones_like(rm1[:, -1:]) * 1
            # rd = tril_with_diag_softplus_and_shift(rm1[:, :6], diag_shift=0.01, diag_mult=None, name='r_cov/tril')
            aax = FCL(am0[:, :nv], self.num_mixtures, activation_fn=tf.nn.softmax, scope='input_u/2x', reuse=tf.AUTO_REUSE)
            aay = FCL(am0[:, nv:2*nv], self.num_mixtures, activation_fn=tf.nn.softmax, scope='input_u/2y', reuse=tf.AUTO_REUSE)
            aaz = FCL(am0[:, 2*nv:], self.num_mixtures, activation_fn=tf.nn.softmax, scope='input_u/2z', reuse=tf.AUTO_REUSE)

            jlx_l = list()
            jly_l = list()
            jlz_l = list()
            srl = [-5, -0.1, 0.1, 5]
            for ppp in range(self.num_mixtures):
                jlx_l.append(tf.ones([1, 1], dtype=self.vdtype) * srl[ppp])
                jly_l.append(tf.ones([1, 1], dtype=self.vdtype) * srl[ppp])
                jlz_l.append(tf.ones([1, 1], dtype=self.vdtype) * srl[ppp])

            jx_vals = tf.squeeze(tf.stack(jlx_l, axis=1), 0)
            jy_vals = tf.squeeze(tf.stack(jly_l, axis=1), 0)
            jz_vals = tf.squeeze(tf.stack(jlz_l, axis=1), 0)

            jx_input = tf.matmul(aax, jx_vals)
            jy_input = tf.matmul(aay, jy_vals)
            jz_input = tf.matmul(aaz, jz_vals)

            zu = tf.zeros_like(jx_input)

            u = tf.concat([zu, zu, zu, jx_input,
                           zu, zu, zu, jy_input,
                           zu, zu, zu, jz_input], axis=1)

            # u = tf.concat([jx_input, jy_input, jz_input], axis=1)

            Qt, At, _, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                                   dimension=int(self.num_state / 3),
                                   sjix=self.om[:, :, 0] * sjx ** 2,
                                   sjiy=self.om[:, :, 0] * sjy ** 2,
                                   sjiz=self.om[:, :, 0] * sjz ** 2,
                                   aji=self.om[:, :, 0] * 1.0)

            Qt_diag = tf.where(tf.matrix_diag_part(Qt) <= tf.ones_like(tf.matrix_diag_part(Qt)) * 1e-16, tf.ones_like(tf.matrix_diag_part(Qt)), tf.matrix_diag_part(Qt))
            Qt = tf.matrix_set_diag(Qt, Qt_diag)
            qdist = tfd.MultivariateNormalTriL(loc=None, scale_tril=tf.cholesky(Qt))
            Qt = qdist.covariance()

            states = (state1, state2, state3)

            Bt = At

            return meas_uvw, Qt, At, Rt, Bt, u, gravity, altitude, states

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

            meas_uvw, Qt, At, Rt, Bt, u, gravity, altitude, states = self.alpha(int_time, dt, mu_t0, meas_rae,
                                                                                Sigma_t0, Qt0, Rt0, LLA,
                                                                                sensor_onehot, sensor_noise, states)

            lat = LLA[:, 0, tfna, tfna]
            lon = LLA[:, 1, tfna, tfna]

            # mu_t0 = tf.concat([mu_t0[:, 0, tf.newaxis], mu_t0[:, 1, tf.newaxis], gravity[:, 0, tf.newaxis], mu_t0[:, 3, tf.newaxis],
            #                    mu_t0[:, 4, tf.newaxis], mu_t0[:, 5, tf.newaxis], gravity[:, 1, tf.newaxis], mu_t0[:, 7, tf.newaxis],
            #                    mu_t0[:, 8, tf.newaxis], mu_t0[:, 9, tf.newaxis], gravity[:, 2, tf.newaxis], mu_t0[:, 11, tf.newaxis]], axis=1)

            mu_pred = tf.squeeze(tf.matmul(At, tf.expand_dims(mu_t0, 2)), -1) + tf.squeeze(tf.matmul(Bt, u[:, :, tfna]), -1)

            # jerk_part = tf.concat([mu_pred[:, 3, tfna], mu_pred[:, 7, tfna], mu_pred[:, 11, tfna]], axis=1)
            # jerk_part = tf.where(jerk_part > 500, tf.ones_like(jerk_part)*500, jerk_part)
            # jerk_part = tf.where(jerk_part < -500, tf.ones_like(jerk_part)*-500, jerk_part)

            # mu_pred = tf.concat([mu_pred[:, 0, tfna], mu_pred[:, 1, tfna], mu_pred[:, 3, tfna], jerk_part[:, 0, tfna],
            #                      mu_pred[:, 4, tfna], mu_pred[:, 5, tfna], mu_pred[:, 6, tfna], jerk_part[:, 1, tfna],
            #                      mu_pred[:, 8, tfna], mu_pred[:, 9, tfna], mu_pred[:, 10, tfna], jerk_part[:, 2, tfna]], axis=1)

            # mu_pred = propagatef2(mu_t0, u, dt, altitude)
            Sigma_pred = tf.matmul(tf.matmul(At, Sigma_t0), At, transpose_b=True) + Qt

            ####
            # qcholr = tf.cholesky(Sigma_t0)
            # Am = tf.expand_dims(self.c, axis=2) * tf.cast(qcholr, tf.float64)
            # Y = tf.tile(tf.expand_dims(mu_t0, axis=2), [1, 1, self.num_state])
            # X = tf.concat([tf.expand_dims(mu_t0, axis=2), Y + Am, Y - Am], axis=2)
            # X = tf.transpose(X, [0, 2, 1])
            #
            # x1, X1, P1, X2 = ut_state_batch(X, u, gravity, altitude, self.Wm, self.Wc, Qt, self.num_state, self.batch_size, dt, At, Bt)
            #
            # uvw_to_enu = uvw2enu_tf(lat, lon)
            # enu_to_uvw = tf.transpose(uvw_to_enu, [0, 2, 1])
            # y_enu = tf.squeeze(tf.matmul(uvw_to_enu, tf.matmul(self.meas_mat, x1)), -1)
            # rae_to_enu = rae2enu_tf(y_enu, self.pi_val)
            # Rt = Rt * (tf.ones_like(Rt) * weight)
            # enu_cov = tf.matmul(tf.matmul(rae_to_enu, Rt), rae_to_enu, transpose_b=True)
            # Rt = tf.matmul(tf.matmul(enu_to_uvw, enu_cov), enu_to_uvw, transpose_b=True)
            #
            # z1, Z1, P2, Z2 = ut_meas(X1, self.Wm, self.Wc, Rt, self.meas_mat, self.batch_size, lat, lon, self.pi_val)
            #
            # P12 = tf.matmul(tf.matmul(X2, tf.matrix_diag(self.Wc)), Z2, transpose_b=True)
            #
            # S_inv = P2
            #
            # gain = tf.matmul(P12, tf.matrix_inverse(P2)) * weight
            # pos_res2 = meas_uvw[:, :, tf.newaxis] - tf.matmul(self.meas_mat, x1)
            # mu_t = x1 + tf.matmul(gain, pos_res2)
            #
            # cov_est_t0 = P1 - tf.matmul(gain, P12, transpose_b=True)
            # Sigma_t = (cov_est_t0 + tf.transpose(cov_est_t0, [0, 2, 1])) / 2
            #
            # mu_t = mu_t[:, :, 0]
            #
            # ####

            # alpha = 0.3
            mu_pred_uvw = tf.matmul(self.meas_mat, mu_pred[:, :, tfna])
            pos_res_uvw = meas_uvw[:, :, tfna] - mu_pred_uvw

            # eet = tf.matmul(pos_res_uvw, pos_res_uvw, transpose_b=True)
            # HPH = tf.matmul(tf.matmul(self.meas_mat, Sigma_pred), self.meas_mat, transpose_b=True)

            uvw_to_enu = uvw2enu_tf(lat, lon)
            enu_to_uvw = tf.transpose(uvw_to_enu, [0, 2, 1])

            y_enu = tf.squeeze(tf.matmul(uvw_to_enu, mu_pred_uvw), -1)

            rae_to_enu = rae2enu_tf(y_enu, self.pi_val)

            HPH = tf.matmul(tf.matmul(self.meas_mat, Sigma_pred), self.meas_mat, transpose_b=True)

            # Rt = Rt * (tf.ones_like(Rt) * weight)
            Rdiag = tf.matrix_diag_part(Rt)
            Rdiag = tf.where(Rdiag <= 1e-12, tf.ones_like(Rdiag)*1000, Rdiag)
            Rt = tf.matrix_set_diag(Rt, Rdiag)

            # Rt = tf.scalar_mul(alpha, (Rt0 + Rt) / 2) + tf.scalar_mul(1-alpha, eet + HPH)

            enu_cov = tf.matmul(tf.matmul(rae_to_enu, Rt), rae_to_enu, transpose_b=True)

            Rt = tf.matmul(tf.matmul(enu_to_uvw, enu_cov), enu_to_uvw, transpose_b=True)

            S = HPH + Rt

            S_inv = tf.matrix_inverse(S)

            gain = tf.matmul(tf.matmul(Sigma_pred, self.meas_mat, transpose_b=True), S_inv)
            gain = tf.where(tf.equal(weight * gain, tf.zeros_like(gain)), tf.zeros_like(gain), gain)

            mu_t = mu_pred[:, :, tfna] + tf.matmul(gain, pos_res_uvw)
            mu_t = mu_t[:, :, 0]

            I_KC = self.I_12 - tf.matmul(gain, self.meas_mat)  # (bs, dim_z, dim_z)
            Sigma_t = tf.matmul(tf.matmul(I_KC, Sigma_pred), I_KC, transpose_b=True) + tf.matmul(tf.matmul(gain, Rt), gain, transpose_b=True)
            Sigma_t = (Sigma_t + tf.transpose(Sigma_t, [0, 2, 1])) / 2

            mu_pred = tf.squeeze(tf.matmul(At, tf.expand_dims(mu_t, 2)), -1)  # + tf.squeeze(tf.matmul(Bt, u[:, :, tfna]), -1)
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
            init_B = tf.ones([self.batch_size, 12, 12], self.vdtype)
            meas_uvw = tf.zeros([self.batch_size, 3], self.vdtype)
            init_weight = tf.ones([self.batch_size, 1, 1], self.vdtype)
            init_u = tf.zeros([self.batch_size, 12], self.vdtype)

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

    def refine(self, filtered_state, filtered_covariance, meas_uvw):

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
            rnn_inp1 = FCL(rnn_inp0, inp_width * 4, activation_fn=tf.nn.elu, weights_initializer=tf.initializers.variance_scaling, scope='rnn_inp1', reuse=tf.AUTO_REUSE)
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

            # if self.mode == 'training':
            #     Az_tm1 = tf.matmul(self.ao_list[:, :-1], tf.expand_dims(z_smooth[:, :-1], 3))
            #     Bz_tm1 = tf.matmul(self.bo_list[:, :-1], tf.expand_dims(self.uo_list[:, 1:], 3))
            #     mu_transition = Az_tm1[:, :, :, 0] + Bz_tm1[:, :, :, 0]
            #     z_t_transition = z_smooth[:, 1:, :]
            #     trans_centered = z_t_transition - mu_transition
            #     Qdiag = tf.matrix_diag_part(self.qo_list[:, 1:, :, :])
            #     # log_prob_transition = mvn_transition.log_prob(trans_centered) * self.seqweightin[:, :-1]
            #     # trans_centered_j = tf.concat([trans_centered[:, :, 3, tfna], trans_centered[:, :, 7, tfna], trans_centered[:, :, 11, tfna]], axis=2)
            #     # Qdiag_j = tf.concat([Qdiag[:, :, 3, tfna], Qdiag[:, :, 7, tfna], Qdiag[:, :, 11, tfna]], axis=2)
            #     # trans_centered_a = tf.concat([trans_centered[:, :, 2, tfna], trans_centered[:, :, 6, tfna], trans_centered[:, :, 10, tfna]], axis=2)
            #     # Qdiag_a = tf.concat([Qdiag[:, :, 2, tfna], Qdiag[:, :, 6, tfna], Qdiag[:, :, 10, tfna]], axis=2)
            #     trans_centered_a = tf.concat([trans_centered[:, :, 0, tfna], trans_centered[:, :, 4, tfna], trans_centered[:, :, 8, tfna]], axis=2)
            #     Qdiag_a = tf.concat([Qdiag[:, :, 0, tfna], Qdiag[:, :, 4, tfna], Qdiag[:, :, 8, tfna]], axis=2)
            #     # mvn_transition = tfd.MultivariateNormalTriL(None, tf.cholesky(self.qo_list[:, :-1, :, :]))
            #     mvn_transition = tfd.MultivariateNormalDiag(None, tf.sqrt(Qdiag_a))
            #     log_prob_transition = mvn_transition.log_prob(trans_centered_a) * self.seqweightin[:, :-1]

            self.y_t_resh = tf.concat([all_truth[:, :, 0, tfna], all_truth[:, :, 4, tfna], all_truth[:, :, 8, tfna]], axis=2)
            self.Cz_t = self.new_meas
            emiss_centered = (self.Cz_t - self.y_t_resh)
            emiss_centered = emiss_centered + tf.ones_like(emiss_centered) * 1e-5
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
            self.MDPi = tf.sqrt((tf.ones_like(self.MDP, self.vdtype) / self.MDP))
            self.maha_loss = tf.truediv(tf.reduce_sum((self.MDP * self.seqweightin[:, :, tfna] + self.MDPi * self.seqweightin[:, :, tfna])), num_el2)
            self.maha_out = tf.truediv(tf.reduce_sum(self.MDP * self.seqweightin[:, :, tfna]), num_el2)

            jtemp = tf.squeeze(tf.matmul(tf.matmul(pos_error[:, :, tf.newaxis], self.si_list), pos_error[:, :, tf.newaxis], transpose_b=True), -1) + tf.log(tf.matrix_determinant(self.si_list))[:, :, tfna]
            self.J_loss = tf.truediv(tf.reduce_sum(jtemp * self.seqweightin[:, :, tfna]), num_el2)

            train_cov_full = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=Sigma_smooth)
            train_cov_pos = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=cov_pos)
            train_cov_vel = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=cov_vel)
            train_cov_acc = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=cov_acc)

            self.trace_loss = tf.log(tf.truediv(tf.reduce_sum(tf.sqrt(tf.pow(tf.matrix_diag_part(Sigma_smooth), 2))), num_el2))
            self.trace_loss_Q = tf.log(tf.truediv(tf.reduce_sum(tf.sqrt(tf.pow(tf.matrix_diag_part(self.qo_list), 2))), num_el2))

            self.error_loss_pos = tf.truediv(tf.reduce_sum(tf.negative(train_cov_pos.log_prob(pos_error)) * self.seqweightin), num_el2 * 9) + 1e-5
            self.error_loss_vel = tf.truediv(tf.reduce_sum(tf.negative(train_cov_vel.log_prob(vel_error)) * self.seqweightin), num_el2 * 9) + 1e-5
            self.error_loss_acc = tf.truediv(tf.reduce_sum(tf.negative(train_cov_acc.log_prob(acc_error)) * self.seqweightin), num_el2 * 9) + 1e-5

            self.error_loss_full = tf.truediv(tf.reduce_sum(tf.negative(train_cov_full.log_prob(self.state_error[:, :, :, 0])) * self.seqweightin), (num_el2 * 144)) + 1e-5

            self.entropy = tf.truediv(tf.reduce_sum(mvn_smooth.log_prob(z_smooth) * self.seqweightin), num_el2 * 12)

            self.rl = tf.truediv(tf.reduce_sum(tf.negative(mvn_emission.log_prob(emiss_centered)) * self.seqweightin), num_el2 * 9)

            self.num_el = num_el
            self.num_el2 = num_el2
            # if self.mode == 'training':
            #     self.error_loss_Q = tf.truediv(tf.reduce_sum(tf.negative(log_prob_transition)), (num_el2 * 12))
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
                self.source_fwf = tfc.rnn.DropoutWrapper(cell_type(num_units=self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
                                                         input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)

            with tf.variable_scope('Cell_2/r_cov'):
                self.source_fwf2 = tfc.rnn.DropoutWrapper(cell_type(num_units=self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
                                                          input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)

            with tf.variable_scope('Cell_3/input_u'):
                self.source_fwf3 = tfc.rnn.DropoutWrapper(cell_type(num_units=self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
                                                          input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)

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

        if self.mode == 'training':
            smooth, filter_out, prediction, A, Q, R, B, S_inv, u, meas_uvw, state1_out, state2_out, state3_out = self.smooth()  # for plotting smoothed posterior
            # filter_out, A, Q, R, B, S_inv, u, meas_uvw, prediction, state1_out, state2_out, state3_out = self.filter()  # for plotting smoothed posterior
            # filter_out, _, Q, R, _, _, _, meas_uvw, prediction, _, _, _ = self.filter()  # for plotting smoothed posterior

        else:
            filter_out, A, Q, R, B, S_inv, u, meas_uvw, prediction, state1_out, state2_out, state3_out = self.filter()

        use_refinement = False

        if use_refinement is True:
            refined_state, refined_covariance, state3_out = self.refine(filter_out[0], filter_out[1], meas_uvw)
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
            self.get_elbo(smooth)
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
            self.learning_rate = tf.train.exponential_decay(self.learning_rate_inp, global_step=self.global_step, decay_steps=self.decay_steps, decay_rate=0.99, staircase=True)

        with tf.variable_scope("TrainOps"):
            print('Updating Gradients')
            all_vars = tf.trainable_variables()
            other_vars = [var for var in all_vars if 'r_cov' and 'q_cov' not in var.name]
            q_vars = [var for var in all_vars if 'q_cov' in var.name]
            r_vars = [var for var in all_vars if 'r_cov' in var.name]

            # self.lower_bound = (tf.log(self.meas_err_use + 1) + 0.4) * (tf.sqrt(self.rmse_pos) + tf.sqrt(self.rmse_vel) + tf.sqrt(self.rmse_acc) + tf.sqrt(self.rmse_jer))
            self.lower_bound = self.J_loss

            if use_refinement is True:
                refinement_error = self.meas_error_refined  # * (self.error_loss_pos_refined + self.error_loss_vel_refined + self.entropy_refined)

            opt1 = tfc.opt.MultitaskOptimizerWrapper(tfc.opt.AdamWOptimizer(weight_decay=1e-5, learning_rate=self.learning_rate, name='opt1'))
            gradvars1 = opt1.compute_gradients(self.J_loss + self.rmse_pos + tf.sqrt(self.rmse_vel), other_vars, colocate_gradients_with_ops=False)
            gradvars1, _ = tfc.opt.clip_gradients_by_global_norm(gradvars1, 1.0)
            self.train_1 = opt1.apply_gradients(gradvars1, global_step=self.global_step)

            # if use_refinement is True:
            opt2 = tfc.opt.MultitaskOptimizerWrapper(tfc.opt.AdamWOptimizer(weight_decay=1e-5, learning_rate=self.learning_rate, name='opt2'))
            gradvars2 = opt2.compute_gradients(self.J_loss + self.rl, r_vars, colocate_gradients_with_ops=False)
            gradvars2, _ = tfc.opt.clip_gradients_by_global_norm(gradvars2, 1.0)
            self.train_2 = opt1.apply_gradients(gradvars2, global_step=self.global_step)

            opt3 = tfc.opt.MultitaskOptimizerWrapper(tfc.opt.AdamWOptimizer(weight_decay=1e-5, learning_rate=self.learning_rate, name='opt3'))
            gradvars3 = opt3.compute_gradients(self.J_loss + self.error_loss_full, q_vars, colocate_gradients_with_ops=False)
            gradvars3, _ = tfc.opt.clip_gradients_by_global_norm(gradvars3, 1.0)
            self.train_3 = opt1.apply_gradients(gradvars3, global_step=self.global_step)

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
            tf.summary.scalar("Meas_Err_Norm", tf.log(self.meas_err_use + 1) + 0.4)
            tf.summary.scalar("Trace", self.trace_loss)
            tf.summary.scalar("Cov_Meas", self.rl)
            tf.summary.scalar("Cov_Q", self.error_loss_Q)
            tf.summary.scalar("Trace_Q", self.trace_loss_Q)
            tf.summary.scalar("Entropy", self.entropy)
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
            tf.summary.scalar("J_loss", self.J_loss)

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

                if self.convert_to_uvw is True:
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

                count, out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot = initialize_run_variables()

                fd = {}

                self.fader = 0.9 * np.power(0.95, (step / 250)) + 0.00001

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
                        seqweight[i, :] = m.astype(int)

                    cur_time = x[:, r1:r2, 0]
                    time_plotter[:, r1:r2, :] = cur_time[:, :, np.newaxis]
                    max_t = np.max(time_plotter[0, :, 0])
                    count += 1

                    if tstep == 0:
                        current_state_estimate, current_cov_estimate, prev_state_estimate, prev_covariance_estimate, initial_Q, initial_R = \
                            initialize_filter(self.batch_size_np, initial_time, initial_meas, prev_time, prev_x, current_time, lla_datar, sensor_vector2, ecef_ref)

                    prev_state_estimate = prev_state_estimate[:, :, self.idxi]
                    current_y = current_y[:, :, self.idxi]
                    prev_y = prev_y[:, :, self.idxi]
                    current_state_estimate = current_state_estimate[:, :, self.idxi]

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
                    fd.update({self.is_training: True})
                    fd.update({self.seqweightin: seqweight})
                    fd.update({self.P_inp: current_cov_estimate})
                    fd.update({self.Q_inp: initial_Q})
                    fd.update({self.R_inp: initial_R})
                    fd.update({self.R_static: sensor_vector2})
                    fd.update({self.state_input: current_state_estimate.reshape(-1, self.num_state)})
                    fd.update({self.prev_time: prev_time[:, :, 0]})
                    fd.update({self.current_timei[t]: current_time[:, t, :].reshape(-1, 1) for t in range(self.max_seq)})
                    fd.update({self.drop_rate: self.dropout_rate_main})

                    randn = random.random()
                    if randn > self.fader:
                        stateful = True
                    else:
                        stateful = False

                    fd.update({self.learning_rate_inp: lr})

                    if epoch < 1 and minibatch_index < int(n_train_batches // 4):
                        ittt = 1
                    else:
                        ittt = 1

                    for _ in range(ittt):
                        filter_output, smooth_output, refined_output, q_out_t, q_outs, q_out_refine, _, _, _, rmsp, rmsv, rmsa, rmsj, LR, \
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
                                           self.train_2,
                                           self.train_3,
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

                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    step += 1

                    print("Epoch: {0:2d} MB: {1:1d} Time: {2:3d} "
                          "RMSP: {3:2.3f} RMSV: {4:2.3f} RMSA: {5:2.3f} RMSJ: {6:2.2f} "
                          "LR: {7:1.2f} ST: {8:1.2f} CPL: {9:1.2f} "
                          "CVL: {10:1.2f} EN: {11:1.2f} QL: {12:1.2f} "
                          "MD: {13:1.2f} RL: {14:1.2f} COV {15:1.2f} FV {16:1.2f} ".format(epoch, minibatch_index, tstep,
                                                                                            rmsp, rmsv, rmsa, rmsj,
                                                                                            LR, max_t, cov_pos_loss,
                                                                                            cov_vel_loss, entropy, q_loss,
                                                                                            MD, rl, kalman_cov_loss, self.fader))

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

                    prev_covariance_estimate = q_out_t[:, -2, np.newaxis, :, :]
                    current_cov_estimate = q_out_t[:, -1, :, :]

                    out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot = \
                        append_output_vaulues(tstep, current_time, current_y, new_meas, smooth_output, filter_output, refined_output, q_out_t, q_outs, q_out_refine, qt_out, rt_out, at_out,
                                              out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot,
                                              testing=False, plt_idx=plt_idx)

                if minibatch_index % self.plot_interval == 0:
                    plotpath = self.plot_dir + 'epoch_' + str(epoch) + '_B_' + str(minibatch_index) + '_step_' + str(step)
                    check_folder(plotpath)

                    out_plot_refined = out_plot_filter
                    q_plotr = q_plott
                    output_plots(out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, q_plott, q_plots, q_plotr, time_plotter, plotpath, qt_plot, rt_plot, meas_list[plt_idx], self.max_seq)

                if minibatch_index % self.checkpoint_interval == 0 and minibatch_index != 0:
                    print("Saving Weights for Epoch " + str(epoch))
                    save_path = self.saver.save(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(epoch) + '_' + str(step) + ".ckpt", global_step=step)
                    print("Checkpoint saved at :: ", save_path)

    def evaluate(self, x_data, y_data, ecef_ref, lla_datar, epoch, minibatch_index, step):

        x_data = np.concatenate([x_data[:, :, 0, np.newaxis], x_data[:, :, 4:7]], axis=2)  # rae measurements

        if self.convert_to_uvw is True:
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
        meas_plot, truth_plot, Q_plot, R_plot, maha_plot = initialize_run_variables()

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
                current_state_estimate, current_cov_estimate, prev_state_estimate, prev_covariance_estimate, initial_Q, initial_R = \
                    initialize_filter(self.batch_size_np, initial_time, initial_meas, prev_time, prev_x, current_time, lla_datar, sensor_vector2, ecef_ref)

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

            prev_state_estimate = filter_output[:, idx-1, np.newaxis, :]
            current_state_estimate = filter_output[:, idx, np.newaxis, :]

            initial_Q = qt_out[:, idx, :, :]
            initial_R = rt_out[:, idx, :, :]

            prev_y = np.concatenate([prev_y, current_y], axis=1)
            prev_y = prev_y[:, idx, np.newaxis, :]

            prev_time = np.concatenate([prev_time, current_time], axis=1)
            prev_time = prev_time[:, idx, np.newaxis, :]

            prev_cov = np.concatenate([prev_cov[:, idx, np.newaxis, :, :], q_out_t], axis=1)
            prev_cov = prev_cov[:, idx, np.newaxis, :, :]

            prev_covariance_estimate = q_out_t[:, idx-1, np.newaxis, :, :]
            current_cov_estimate = q_out_t[:, idx, :, :]

            out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot = \
                append_output_vaulues(step, current_time, current_y, new_meas, smooth_output, filter_output, refined_output, q_out_t, q_outs, q_out_refine,
                                      qt_out, rt_out, at_out, testing=False, plt_idx=0)

        plotpath = self.plot_eval_dir + '/epoch_' + str(epoch) + '_eval_B_' + str(minibatch_index) + '_step_' + str(step)
        check_folder(plotpath)
        output_plots(out_plot_filter, out_plot_smooth, meas_plot, truth_plot, q_plott, q_plot, time_plotter, plotpath, qt_plot, rt_plot, meas_list[plt_idx], self.max_seq)

        # check_folder(self.checkpoint_dir)
        # print("Saving filter Weights for epoch" + str(epoch))
        # save_path = self.saver.save(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(epoch) + '_' + str(step) + ".ckpt", global_step=step)
        # print("Checkpoint saved at: ", save_path)

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
            ds = DataServerLive(self.data_dir, self.meas_dir, self.state_dir, decimate_data=self.decimate_data)

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

            if self.convert_to_uvw is True:
                y_uvw = y_data[:, :, :3] - ecef_ref
                zero_rows = (y_data[:, :, :3] == 0).all(2)
                for i in range(y_data.shape[0]):
                    zz = zero_rows[i, :, np.newaxis]
                    y_uvw[i, :, :] = np.where(zz, np.zeros_like(y_uvw[i, :, :]), y_uvw[i, :, :])
                    y_data = np.concatenate([y_uvw, y_data[:, :, 3:]], axis=2)

            s_data = x_data

            x, y, meta, prev_y, prev_x, prev_time, prev_meta, initial_meas, initial_state_truth, initial_time, max_length = prepare_batch(0, x_data, y_data, s_data,
                                                                                                                                          seq_len=self.max_seq, batch_size=self.batch_size_np,
                                                                                                                                          new_batch=True)

            count, out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot = initialize_run_variables()

            fd = {}

            r1 = self.max_seq
            r2 = r1 + self.max_seq

            x_data = x[:, :, 1:]
            time_data = x[:, :, 0, np.newaxis]
            y_data = y

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

            time_plotter = time_data

            current_state_estimate, current_cov_estimate, prev_state_estimate, prev_covariance_estimate, initial_Q, initial_R = \
                initialize_filter(self.batch_size_np, initial_time, initial_meas, prev_time, prev_x, time_data[:, 0, np.newaxis, :], lla_datar, sensor_vector2, ecef_ref)

            fd.update({self.is_training: False})
            stateful = True
            update = False

            std = 0.0
            fd.update({self.cell_state1: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            fd.update({self.cell_state2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            fd.update({self.cell_state3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

            for tstep in range(0, time_data.shape[1]):

                current_y = y_data[:, tstep, self.idxi]
                prev_state_estimate = prev_state_estimate[:, :, self.idxi]
                current_state_estimate = current_state_estimate[:, :, self.idxi]
                current_x = x_data[:, tstep, np.newaxis, :]

                if np.all(current_x == 0):
                    continue

                current_time = time_data[:, tstep, np.newaxis, :]
                current_int = int_time[:, tstep, tfna]
                current_weight = seqweight[:, tstep, tfna]

                max_t = np.max(current_time)

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
                      "MD: {13:1.2f} RL: {14:1.2f} COV {15:1.2f} ".format(0, minibatch_index, tstep,
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
                    fd.update({self.cell_state3: state_fwf3[:, -1, :]})

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

                prev_covariance_estimate = current_cov_estimate[:, np.newaxis, :, :]

                current_cov_estimate = q_out_t[:, -1, :, :]

                out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot = \
                    append_output_vaulues(tstep, current_time, current_y, new_meas, smooth_output, filter_output, refined_output, q_out_t, q_outs, q_out_refine, qt_out, rt_out, at_out,
                                          out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot, testing=True)

            for pidx in range(self.batch_size_np):
                plotpath = self.plot_test_dir + '/epoch_' + str(9999) + '_MB_' + str(minibatch_index) + '_BATCH_IDX_' + str(pidx)
                check_folder(plotpath)

                output_plots(out_plot_filter[pidx, np.newaxis, :, :],
                             out_plot_smooth[pidx, np.newaxis, :, :],
                             out_plot_refined[pidx, np.newaxis, :, :],
                             meas_plot[pidx, np.newaxis, :, :],
                             truth_plot[pidx, np.newaxis, :, :],
                             q_plott[pidx, np.newaxis, :, :, :],
                             q_plots[pidx, np.newaxis, :, :, :],
                             q_plotr[pidx, np.newaxis, :, :, :],
                             time_plotter[pidx, np.newaxis, :, :],
                             plotpath,
                             qt_plot[pidx, np.newaxis, :, :, :],
                             rt_plot[pidx, np.newaxis, :, :, :],
                             meas_list[pidx],
                             self.max_seq)
