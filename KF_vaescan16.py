import math
import time
import random

from tensorflow.contrib.layers import fully_connected as FCL

from helper2 import *
from helper3 import *
from load_all_data_4 import DataServerPrePro
from plotting import *
from propagation_utils import *

from natsort import natsorted

tfd = tfp.distributions
tfb = tfp.bijectors

setattr(tfc.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)


class Filter(object):
    def __init__(self, sess, trainable_state=False, state_type='GRU', mode='training',
                 data_dir='', filter_name='', save_dir='',
                 F_hidden=12, num_state=12, num_meas=3, max_seq=2, num_mixtures=5,
                 max_epoch=10000, RE=6378137, GM=398600441890000, batch_size=10, constant=False):

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

        self.batch_size_np = batch_size
        self.meas_dir = data_dir + '/NoiseRAE/'
        self.state_dir = data_dir + '/Translate/'

        self.root = 'D:/TrackFilterData'
        self.train_dir = self.root + '/OOPBroad_preprocessed/Train/'
        self.test_dir = self.root + '/OOPBroad_preprocessed/Test/'
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

        alpha = 1e-3 * tf.ones([self.batch_size, 1], dtype=self.vdtype)
        beta = 2. * tf.ones([self.batch_size, 1], dtype=self.vdtype)
        k = 0. * tf.ones([self.batch_size, 1], dtype=self.vdtype)

        L = tf.cast(self.num_state, dtype=self.vdtype)
        lam = alpha * (L + k) - L
        c1 = L + lam
        tmat = tf.ones([1, 2 * self.num_state], dtype=self.vdtype)
        self.Wm = tf.concat([(lam / c1), (0.5 / c1) * tmat], axis=1)
        Wc1 = tf.expand_dims(copy.copy(self.Wm[:, 0]), axis=1) + (tf.ones_like(alpha, dtype=self.vdtype) - alpha + beta)
        self.Wc = tf.concat([Wc1, copy.copy(self.Wm[:, 1:])], axis=1)
        self.c = tf.sqrt(c1)

        if 'OOP' in self.train_dir:
            # OOP
            self.aeg_loc = [None] * 3
            self.tpy_loc = [None] * 3
            self.pat_loc = [None] * 3
            self.aeg_loc[0], self.aeg_loc[1], self.aeg_loc[2] = [0.17, -0.085], [0.00001, -0.089], [-0.089, -0.084]
            self.pat_loc[0], self.pat_loc[1], self.pat_loc[2] = [-0.0823, 0.0359], [0.0269, 0.0359], [0.0898, -0.1556]
            self.tpy_loc[0], self.tpy_loc[1], self.tpy_loc[2] = [0.15, 0.0449], [0.00001, 0.09], [-0.09, 0.06]
        elif 'Advanced' in self.train_dir:
            # ADVANCED
            self.aeg_loc = [None] * 3
            self.tpy_loc = [None] * 3
            self.pat_loc = [None] * 3
            self.aeg_loc[0], self.aeg_loc[1], self.aeg_loc[2] = [0.17, -0.085], [0.00001, -0.089], [-0.089, -0.084]
            self.pat_loc[0], self.pat_loc[1], self.pat_loc[2] = [1e-5, -0.0449], [0.09, 0.0449], [0.0898, -0.1556]
            self.tpy_loc[0], self.tpy_loc[1], self.tpy_loc[2] = [-0.45, 0.0449], [0.00001, 0.0449], [0.18, 0.0449]

    def alpha(self, int_time, dt, pstate, meas_rae, cov_est, Q_est, R_est, LLA, sensor_onehot, state1=None, state2=None, state3=None):

        with tf.variable_scope('alpha'):
            lat = LLA[:, 0, tf.newaxis]
            lon = LLA[:, 1, tf.newaxis]
            # alt = LLA[:, 2, tf.newaxis]

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

            _, At, _, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                                 dimension=int(self.num_state / 3),
                                 sjix=self.om[:, :, 0] * 1 ** 2,
                                 sjiy=self.om[:, :, 0] * 1 ** 2,
                                 sjiz=self.om[:, :, 0] * 1 ** 2,
                                 aji=self.om[:, :, 0] * 1.0)

            prop_state = tf.matmul(At, pstate[:, :, tf.newaxis])
            pre_residual = meas_uvw1[:, :, tf.newaxis] - tf.matmul(self.meas_mat, prop_state)

            cov_diag = tf.matrix_diag_part(cov_est)
            Q_diag = tf.matrix_diag_part(Q_est)
            R_diag = tf.matrix_diag_part(R_est)

            cov_diag_n = cov_diag / tf.ones_like(cov_diag) * 100
            Q_diag_n = Q_diag / tf.ones_like(Q_diag) * 100
            R_diag_n = R_diag / tf.ones_like(R_diag) * 100
            pre_res_n = pre_residual[:, :, 0] / tf.ones_like(pre_residual[:, :, 0]) * 100

            rnn_inp = tf.concat([pre_res_n, cov_diag_n, Q_diag_n, R_diag_n, sensor_onehot], axis=1)

            rnn_inpa1 = FCL(rnn_inp, rnn_inp.shape[1].value, activation_fn=tf.nn.elu, scope='r_cov/rnn_inp1', reuse=tf.AUTO_REUSE)
            rnn_inpa = FCL(rnn_inpa1, self.F_hidden, activation_fn=tf.nn.elu, scope='r_cov/rnn_inp2', reuse=tf.AUTO_REUSE)

            rnn_inpb1 = FCL(rnn_inp, rnn_inp.shape[1].value, activation_fn=tf.nn.elu, scope='q_cov/rnn_inp1', reuse=tf.AUTO_REUSE)
            rnn_inpb = FCL(rnn_inpb1, self.F_hidden, activation_fn=tf.nn.elu, scope='q_cov/rnn_inp2', reuse=tf.AUTO_REUSE)

            rnn_inpa = tfc.layers.dropout(rnn_inpa, keep_prob=self.drop_rate, is_training=self.is_training, scope='r_cov/dropout_inputs')
            rnn_inpb = tfc.layers.dropout(rnn_inpb, keep_prob=self.drop_rate, is_training=self.is_training, scope='q_cov/dropout_inputs')

            with tf.variable_scope('Source_Track_Forward/r_cov', reuse=tf.AUTO_REUSE):
                (outa, state1) = self.source_fwf((int_time, rnn_inpa), state=state1)

            with tf.variable_scope('Source_Track_Forward2/q_cov', reuse=tf.AUTO_REUSE):
                (outb, state2) = self.source_fwf2((int_time, rnn_inpb), state=state2)

            rm0 = FCL(tf.concat([outa], axis=1), self.F_hidden, activation_fn=None, scope='r_cov/1', reuse=tf.AUTO_REUSE)
            # rm1 = FCL(rm0, 6, activation_fn=None, scope='r_cov/2', reuse=tf.AUTO_REUSE)
            # d_mult = (tf.nn.sigmoid(rm1[:, -1:], 'r_cov/d_mult') * 50) + tf.ones_like(rm1[:, -1:]) * 1
            # rd = tril_with_diag_softplus_and_shift(rm1[:, :6], diag_shift=0.01, diag_mult=None, name='r_cov/tril')
            asr = FCL(rm0, self.num_mixtures, activation_fn=tf.nn.softmax, scope='r_cov/alphar', reuse=tf.AUTO_REUSE)
            # asa = FCL(rm0[:, 12:24], self.num_mixtures, activation_fn=tf.nn.softmax, scope='r_cov/alphaa', reuse=tf.AUTO_REUSE)
            # ase = FCL(rm0[:, 24:], self.num_mixtures, activation_fn=tf.nn.softmax, scope='r_cov/alphae', reuse=tf.AUTO_REUSE)

            sr_list = list()
            sa_list = list()
            se_list = list()
            srl = [0.0001, 0.01, 10, 250, 500]
            sal = [1e-7, 1e-6, 5e-6, 1e-5, 2.5e-5]
            sel = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]

            srl = [0.01, 1, 10, 100, 500]
            sal = [1e-7, 1e-6, 5e-6, 1e-5, 2.5e-5]
            sel = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001]

            for ppp in range(self.num_mixtures):
                sr_list.append(tf.ones([1, 1], dtype=self.vdtype) * srl[ppp])
                sa_list.append(tf.ones([1, 1], dtype=self.vdtype) * sal[ppp])
                se_list.append(tf.ones([1, 1], dtype=self.vdtype) * sel[ppp])

            sr_vals = tf.squeeze(tf.stack(sr_list, axis=1), 0)
            sa_vals = tf.squeeze(tf.stack(sa_list, axis=1), 0)
            se_vals = tf.squeeze(tf.stack(se_list, axis=1), 0)

            # sr = FCL(rm1[:, 0, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, scope='sr', reuse=tf.AUTO_REUSE) * 200 + tf.ones_like(rm1[:, 0, tf.newaxis]) * 0.1
            # sa = FCL(rm1[:, 1, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, scope='sa', reuse=tf.AUTO_REUSE) * 1e-1 + tf.ones_like(rm1[:, 0, tf.newaxis]) * 5e-4
            # se = FCL(rm1[:, 2, tf.newaxis], 1, activation_fn=tf.nn.sigmoid, scope='sz', reuse=tf.AUTO_REUSE) * 0.5 + tf.ones_like(rm1[:, 0, tf.newaxis]) * 1e-3

            sr = tf.matmul(asr, tf.sqrt(sr_vals))
            sa = tf.matmul(asr, tf.sqrt(sa_vals))
            se = tf.matmul(asr, tf.sqrt(se_vals))

            rd = tf.concat([sr, sa, se], axis=1)

            # rd = tf.matrix_set_diag(self.I_3z, r_diag)

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

            # rdist = tfd.MultivariateNormalTriL(loc=None, scale_tril=rd)
            rdist = tfd.MultivariateNormalDiag(loc=None, scale_diag=rd)
            Rt = rdist.covariance()

            meas_uvw = meas_uvw1

            # alpha0 = FCL(out[:, hp:], out[:, hp:].shape[1].value, activation_fn=None, scope='alpha0', reuse=tf.AUTO_REUSE)
            # alpha1 = FCL(alpha0, alpha0.shape[1].value, activation_fn=None, scope='alpha1', reuse=tf.AUTO_REUSE)
            # alpha = FCL(alpha1, self.num_mixtures, activation_fn=tf.nn.softmax, scope='alpha', reuse=tf.AUTO_REUSE)

            # u0 = FCL(out[:, hp:hp+6], 3, activation_fn=None, scope='u/1', reuse=tf.AUTO_REUSE)
            # ul = FCL(u0, 3, activation_fn=None, scope='u/2', reuse=tf.AUTO_REUSE)
            # u = FCL(ul, 3, activation_fn=None, scope='u/3', reuse=tf.AUTO_REUSE)

            u = tf.zeros([self.batch_size, 3], self.vdtype)

            qm0 = FCL(tf.concat([outb], axis=1), 36, activation_fn=None, scope='q_cov/1', reuse=tf.AUTO_REUSE)
            # qm1 = FCL(qm0, 3, activation_fn=None, scope='q_cov/2', reuse=tf.AUTO_REUSE)
            qmx = FCL(qm0[:, :12], self.num_mixtures, activation_fn=tf.nn.softmax, scope='q_cov/x', reuse=tf.AUTO_REUSE)
            qmy = FCL(qm0[:, 12:24], self.num_mixtures, activation_fn=tf.nn.softmax, scope='q_cov/y', reuse=tf.AUTO_REUSE)
            qmz = FCL(qm0[:, 24:], self.num_mixtures, activation_fn=tf.nn.softmax, scope='q_cov/z', reuse=tf.AUTO_REUSE)

            # alphas = tf.stop_gradient(tf.concat([asr, asa, ase, qmx, qmy, qmz], axis=1))

            sjx_l = list()
            sjy_l = list()
            sjz_l = list()
            sjxl = [0.1, 1, 25, 50, 100]
            sjyl = [0.1, 1, 25, 50, 100]
            sjzl = [0.1, 1, 25, 50, 100]

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

            Qt, At, Bt, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                                   dimension=int(self.num_state / 3),
                                   sjix=self.om[:, :, 0] * sjx ** 2,
                                   sjiy=self.om[:, :, 0] * sjy ** 2,
                                   sjiz=self.om[:, :, 0] * sjz ** 2,
                                   aji=self.om[:, :, 0] * 1.0)

            qdist = tfd.MultivariateNormalTriL(loc=None, scale_tril=tf.cholesky(Qt))
            Qt = qdist.covariance()

            return meas_uvw, Qt, At, Rt, Bt, u, state1, state2, state3

    def beta(self, int_time, pos_res, dt, mu_pred, Sigma_pred, sensor_onehot, cur_weight, state3=None):
        with tf.variable_scope('beta'):
            weight = cur_weight[:, :, tf.newaxis]

            layer_input = tf.concat([dt, pos_res, tf.matrix_diag_part(Sigma_pred)], axis=1)

            # rnn_inpr = self.filter_layer(layer_input, name='rnn_inp_beta')
            rnn_inp = tf.concat([layer_input], axis=1)
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

        with tf.variable_scope('forward_step_fn'):
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

            meas_uvw, Qt, At, Rt, Bt, u, state1, state2, state3 = self.alpha(int_time, dt, mu_t0, meas_rae,
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

            mu_pred_uvw = tf.matmul(self.meas_mat, mu_pred[:, :, tf.newaxis])
            pos_res_uvw = meas_uvw[:, :, tf.newaxis] - mu_pred_uvw

            # r = sqrt(enu(1) * enu(1) + enu(2) * enu(2) + enu(3) * enu(3));
            # az = atan2(enu(1), enu(2));
            # el = asin(enu(3) / r);
            lat = LLA[:, 0, tf.newaxis, tf.newaxis]
            lon = LLA[:, 1, tf.newaxis, tf.newaxis]
            tz = tf.zeros_like(lon)

            t00 = -tf.sin(lon)
            t01 = tf.cos(lon)
            t10 = -tf.sin(lat) * tf.cos(lon)
            t11 = -tf.sin(lat) * tf.sin(lon)
            t12 = tf.cos(lat)
            t20 = tf.cos(lat) * tf.cos(lon)
            t21 = tf.cos(lat) * tf.sin(lon)
            t22 = tf.sin(lat)

            # t00 = -tf.sin(lat) * tf.cos(lon)
            # t01 = -tf.sin(lon)
            # t02 = -tf.cos(lat)*tf.cos(lon)
            # t10 = -tf.sin(lat) * tf.sin(lon)
            # t11 = tf.cos(lon)
            # t12 = -tf.cos(lat)*tf.sin(lon)
            # t20 = tf.cos(lat)
            # # t21 = tf.cos(lat) * tf.sin(lon)
            # t22 = -tf.sin(lat)
            #
            Ti2e = tf.concat([tf.concat([t00, t01, tz], axis=2), tf.concat([t10, t11, t12], axis=2), tf.concat([t20, t21, t22], axis=2)], axis=1)

            y_enu = tf.squeeze(tf.matmul(Ti2e, mu_pred_uvw), -1)

            rng = tf.sqrt(y_enu[:, 0] * y_enu[:, 0] + y_enu[:, 1] * y_enu[:, 1] + y_enu[:, 2] * y_enu[:, 2])
            az = tf.atan2(y_enu[:, 0], y_enu[:, 1])
            az = tf.where(az < 0, az + tf.ones_like(az) * (2 * self.pi_val), az)
            el = tf.asin(y_enu[:, 2] / rng)

            # y_ned = tf.squeeze(tf.matmul(Ti2e, mu_pred_uvw), -1)

            # east = y_enu[:, 1, tf.newaxis]
            # north = y_enu[:, 0, tf.newaxis]
            # up = tf.negative(y_enu[:, 2, tf.newaxis])

            # y_enu = tf.concat([east, north, up], axis=1)

            # rae_pred0 = tf.concat([rng[:, tf.newaxis], az[:, tf.newaxis], el[:, tf.newaxis]], axis=1)

            # rng = tf.sqrt(y_ned[:, 0] * y_ned[:, 0] + y_ned[:, 1] * y_ned[:, 1] + y_ned[:, 2] * y_ned[:, 2])
            # az = tf.atan2(y_ned[:, 1], y_ned[:, 0])
            # az = tf.where(az < 0, az + tf.ones_like(az) * (2 * self.pi_val), az)
            # el = tf.acos(y_ned[:, 2] / rng)
            #
            # # rae_pred = tf.concat([rng[:, tf.newaxis], az[:, tf.newaxis], el[:, tf.newaxis]], axis=1)
            #
            az = az[:, tf.newaxis, tf.newaxis]
            el = el[:, tf.newaxis, tf.newaxis]

            # m01 = tf.sin(az)*tf.cos(el) + tf.cos(az)*tf.cos(el)
            # m10 = tf.cos(az)*tf.cos(el)
            # m11 = -tf.sin(az)*tf.cos(el)
            # m12 = -tf.cos(az)*tf.sin(el)
            # m22 = tf.sin(el) + tf.cos(el)

            # rae2enu = [sin(am) * cos(em), sin(am) * cos(em) + cos(am) * cos(em), - sin(am) * sin(em)
            #            cos(am) * cos(em), -sin(am) * cos(em) - cos(am) * sin(em),
            #            sin(em), 0                                                                            sin(em) + cos(em)]

            m00 = tf.sin(az) * tf.cos(el)
            m01 = tf.cos(az) * tf.cos(el)
            m02 = -tf.sin(az) * tf.sin(el)
            m10 = tf.cos(az) * tf.cos(el)
            m11 = tf.sin(az) * tf.cos(el)
            m12 = -tf.cos(az) * tf.sin(el)
            m20 = tf.sin(el)
            m22 = tf.cos(el)

            # m00 = tf.cos(az) * tf.sin(el)
            # m01 = -tf.sin(az) * tf.sin(el)
            # m02 = -tf.cos(az) * tf.cos(el)
            # m10 = tf.sin(az) * tf.sin(el)
            # m11 = tf.cos(az) * tf.sin(el)
            # m12 = tf.sin(az) * tf.cos(el)
            # m20 = tf.cos(el)
            # m22 = -tf.sin(el)

            cov_xform = tf.concat([tf.concat([m00, m01, m02], axis=2), tf.concat([m10, m11, m12], axis=2), tf.concat([m20, tz, m22], axis=2)], axis=1)

            Rt = Rt * (tf.ones_like(Rt) * weight)

            enu_cov = tf.matmul(tf.matmul(cov_xform, Rt), cov_xform, transpose_b=True)

            # n00 = neu_cov[:, 1, 1, tf.newaxis, tf.newaxis]
            # n01 = neu_cov[:, 0, 1, tf.newaxis, tf.newaxis]
            # n02 = neu_cov[:, 1, 2, tf.newaxis, tf.newaxis]
            # n10 = neu_cov[:, 1, 0, tf.newaxis, tf.newaxis]
            # n11 = neu_cov[:, 0, 0, tf.newaxis, tf.newaxis]
            # n12 = neu_cov[:, 0, 2, tf.newaxis, tf.newaxis]
            # n20 = neu_cov[:, 2, 1, tf.newaxis, tf.newaxis]
            # n21 = neu_cov[:, 2, 0, tf.newaxis, tf.newaxis]
            # n22 = neu_cov[:, 2, 2, tf.newaxis, tf.newaxis]
            #
            # enu_cov = tf.concat([tf.concat([n00, n01, n02], axis=2), tf.concat([n10, n11, n12], axis=2), tf.concat([n20, n21, n22], axis=2)], axis=1)

            # c = tf.cholesky(enu_cov)

            Ti2et = tf.transpose(Ti2e, [0, 2, 1])

            Rt = tf.matmul(tf.matmul(Ti2et, enu_cov), Ti2et, transpose_b=True)

            Rt = Rt * (tf.ones_like(Rt) * weight)

            Rt_d = tf.matrix_diag_part(Rt)
            Rt0 = tf.zeros_like(Rt)
            Rt = tf.matrix_set_diag(Rt0, Rt_d)

            # c = tf.cholesky(Rt)

            # Rt = tf.where(tf.is_nan(Rt), tf.ones_like(Rt), Rt)
            # Rt = tf.where(tf.equal(Rt, 0.), tf.ones_like(Rt), Rt)

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

            mu_t = mu_pred[:, :, tf.newaxis] + tf.matmul(gain, pos_res_uvw)
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

            mu_pred = tf.squeeze(tf.matmul(At, tf.expand_dims(mu_t, 2)), -1) + tf.squeeze(tf.matmul(Bt, u[:, :, tf.newaxis]), -1)
            # mu_pred = mu_pred + tf.squeeze(tf.matmul(Bt, u[:, :, tf.newaxis]), -1)
            Sigma_pred = tf.matmul(tf.matmul(At, Sigma_t), At, transpose_b=True) + Qt
            # Sigma_pred = (Sigma_pred + tf.matrix_transpose(Sigma_pred))/2
            # # Sigma_pred1 = tf.matmul(tf.matmul(At, Sigma_t), At, transpose_b=True)

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

        with tf.variable_scope('compute_forwards'):
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

            sensor_onehot = tf.expand_dims(self.sensor_vector, axis=1)
            sensor_onehot = tf.tile(sensor_onehot, [1, meas_rae.shape[1], 1])

            inputs = tf.concat([meas_time, prev_time, int_time[:, :, tf.newaxis], meas_rae[:, 1:, :], self.seqweightin[:, :, tf.newaxis], sensor_lla[:, 1:, :], sensor_onehot[:, 1:, :]], axis=2)

            cov_input = self.prev_covariance_estimate
            init_Q = self.Q_inp
            init_R = self.R_inp

            meas_uvw, Qt, At, Rt, Bt, u, state1, state2, state3 = self.alpha(int_time[:, 0, tf.newaxis], dt0, self.prev_state_estimate, meas_rae[:, 0, :],
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
            # init_alphas = alphas

            state3 = self.state_fw_in_state3

            forward_states = tf.scan(self.forward_step_fn, tf.transpose(inputs, [1, 0, 2]),
                                     initializer=(self.mu, self.Sigma, self.mu, self.Sigma, meas_uvw,
                                                  state1, state2, state3,
                                                  init_Q, init_R, init_A, init_B, init_Si,
                                                  init_weight, u),
                                     parallel_iterations=1, name='forward')
            return forward_states

    def compute_backwards(self, forward_states):
        with tf.variable_scope('compute_backwards'):

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
        with tf.variable_scope('filter'):
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

            forward_states_filter = [mu_filt, Sigma_filt]
            forward_states_pred = [mu_pred, Sigma_pred]

            # alphas = tf.transpose(alphas, [1, 0, 2])

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

        with tf.variable_scope('smooth'):
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
                   tf.transpose(u, [1, 0, 2]), tf.transpose(meas_uvw, [1, 0, 2]), \
                   state1_out, state2_out, state3_out

    @staticmethod
    def _sast(a, s):
        _, dim_1, dim_2 = s.get_shape().as_list()
        sastt = tf.matmul(tf.reshape(s, [-1, dim_2]), a, transpose_b=True)
        sastt = tf.transpose(tf.reshape(sastt, [-1, dim_1, dim_2]), [0, 2, 1])
        sastt = tf.matmul(s, sastt)
        return sastt

    def get_elbo(self, backward_states):
        with tf.variable_scope('Loss_Calculation'):

            mu_smooth = backward_states[0]
            Sigma_smooth = backward_states[1]

            ssdiag = tf.matrix_diag_part(Sigma_smooth)
            ssdiag = tf.where(tf.less_equal(ssdiag, tf.zeros_like(ssdiag)), tf.ones_like(ssdiag) * 1, ssdiag)

            rodiag = tf.matrix_diag_part(self.ro_list)
            rodiag = tf.where(tf.less_equal(rodiag, tf.zeros_like(rodiag)), tf.ones_like(rodiag) * 1, rodiag)

            Sigma_smooth = tf.matrix_set_diag(Sigma_smooth, ssdiag)
            self.ro_list = tf.matrix_set_diag(self.ro_list, rodiag)

            all_truth = tf.stack(self.truth_state, axis=1)
            # mvn_smooth = tfd.MultivariateNormalTriL(mu_smooth, tf.cholesky(Sigma_smooth))
            mvn_smooth = tfd.MultivariateNormalDiag(mu_smooth, tf.sqrt(tf.matrix_diag_part(Sigma_smooth)))
            self.mvn_inv = tf.matrix_inverse(mvn_smooth.covariance())

            z_smooth = mvn_smooth.sample()
            self.state_error = all_truth - z_smooth
            self.state_error = tf.where(self.state_error < 1e-6, tf.ones_like(self.state_error) * 1e-6, self.state_error)
            self.state_error = self.state_error[:, :, :, tf.newaxis]

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
            # self.to_list = trans_centered
            # self.trans1 = mu_transition
            # self.trans2 = z_t_transition

            Qdiag = tf.matrix_diag_part(self.qo_list[:, :-1])
            # mvn_transition = tfd.MultivariateNormalTriL(tf.zeros(self.num_state, dtype=self.vdtype), tf.cholesky(self.qo_list[:, :-1]))
            # mvn_transition = tfd.MultivariateNormalDiag(None, tf.sqrt(Qdiag))
            # log_prob_transition = mvn_transition.log_prob(trans_centered) * self.seqweightin[:, :-1]

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
            emiss_centered = (self.Cz_t - self.y_t_resh)
            # emiss_centered = tf.where(emiss_centered < 1., tf.sqrt(emiss_centered), emiss_centered)
            # mvn_emission = tfd.MultivariateNormalTriL(None, tf.cholesky(self.ro_list))
            mvn_emission = tfd.MultivariateNormalDiag(None, tf.sqrt(tf.matrix_diag_part(self.ro_list)))

            # Distribution of the initial state p(z_1|z_0)
            z_0 = z_smooth[:, 0, :]
            mvn_0 = tfd.MultivariateNormalTriL(self.mu, tf.cholesky(self.Sigma))

            num_el = tf.reduce_sum(self.seqweightin) / tf.cast(self.batch_size, self.vdtype)
            num_el = tf.cast(tf.reduce_sum(self.batch_size * self.max_seq), self.vdtype)
            num_el2 = tf.reduce_sum(tf.cast(self.batch_size, self.vdtype))

            state_error_pos = truth_pos - smooth_pos
            meas_error_pos = tf.sqrt(tf.square(truth_pos - self.new_meas))
            # meas_error_pos = meas_error_pos + tf.ones_like(meas_error_pos) * 1e-20

            # meas_dist = tfd.MultivariateNormalTriL(meas_error_pos, tf.cholesky(meas_error_pos_cov))
            meas_dist = tfd.MultivariateNormalDiag(None, tf.sqrt(meas_error_pos))
            self.meas_error = tf.truediv(tf.reduce_sum(tf.negative(meas_dist.log_prob(state_error_pos)) * self.seqweightin), num_el * 3.)

            cov_pos = tf.concat([tf.concat([Sigma_smooth[:, :, 0, 0, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 0, 4, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 0, 8, tf.newaxis, tf.newaxis]], axis=3),
                                 tf.concat([Sigma_smooth[:, :, 4, 0, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 4, 4, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 4, 8, tf.newaxis, tf.newaxis]], axis=3),
                                 tf.concat([Sigma_smooth[:, :, 8, 0, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 8, 4, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 8, 8, tf.newaxis, tf.newaxis]], axis=3)],
                                axis=2)

            cov_vel = tf.concat([tf.concat([Sigma_smooth[:, :, 1, 1, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 1, 5, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 1, 9, tf.newaxis, tf.newaxis]], axis=3),
                                 tf.concat([Sigma_smooth[:, :, 5, 1, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 5, 5, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 5, 9, tf.newaxis, tf.newaxis]], axis=3),
                                 tf.concat([Sigma_smooth[:, :, 9, 1, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 9, 5, tf.newaxis, tf.newaxis], Sigma_smooth[:, :, 9, 9, tf.newaxis, tf.newaxis]], axis=3)],
                                axis=2)

            pos_error = pos_error[:, :, :, tf.newaxis]
            M1P = tf.matmul(pos_error, tf.matrix_inverse(cov_pos), transpose_a=True)
            M2P = tf.matmul(M1P, pos_error)

            self.MDP = tf.sqrt(tf.squeeze(M2P / 3, -1))
            self.MDPi = tf.sqrt((tf.ones_like(self.MDP, self.vdtype) / tf.squeeze(M2P, -1)))
            self.maha_loss = tf.truediv(tf.reduce_sum((self.MDP * self.seqweightin[:, :, tf.newaxis] + self.MDPi * self.seqweightin[:, :, tf.newaxis])), num_el)

            # train_cov00 = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=Sigma_smooth)
            train_cov00 = tfd.MultivariateNormalDiag(loc=None, scale_diag=tf.sqrt(tf.matrix_diag_part(Sigma_smooth)))
            # train_cov_pos = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=cov_pos)
            train_cov_pos = tfd.MultivariateNormalDiag(loc=None, scale_diag=tf.sqrt(tf.matrix_diag_part(cov_pos)))
            # train_cov_vel = tfd.MultivariateNormalFullCovariance(loc=None, covariance_matrix=cov_vel)
            train_cov_vel = tfd.MultivariateNormalDiag(loc=None, scale_diag=tf.sqrt(tf.matrix_diag_part(cov_vel)))

            self.trace_loss = tf.truediv(tf.reduce_sum(tf.sqrt(tf.pow(tf.matrix_diag_part(Sigma_smooth), 2))), num_el)

            self.error_loss_pos = tf.truediv(tf.reduce_sum(tf.negative(train_cov_pos.log_prob(pos_error[:, :, :, 0])) * self.seqweightin), num_el * 3.)
            self.error_loss_vel = tf.truediv(tf.reduce_sum(tf.negative(train_cov_vel.log_prob(vel_error)) * self.seqweightin), num_el * 3.)

            self.error_loss_full = tf.truediv(tf.reduce_sum(tf.negative(train_cov00.log_prob(self.state_error[:, :, :, 0])) * self.seqweightin), (num_el * 12.))
            self.error_loss_initial = tf.truediv(tf.reduce_sum(tf.negative(mvn_0.log_prob(z_0) * self.seqweightin[:, 0])), (num_el * 12.))

            self.entropy = tf.truediv(tf.reduce_sum(mvn_smooth.log_prob(z_smooth) * self.seqweightin), num_el * 12.)
            self.rl = tf.truediv(tf.reduce_sum(tf.negative(mvn_emission.log_prob(emiss_centered)) * self.seqweightin), num_el * 3.)

            self.z_smooth = z_smooth
            self.num_el = num_el
            self.num_el2 = num_el2
            self.error_loss_Q = tf.truediv(tf.reduce_sum(tf.negative(log_prob_transition)), (num_el * 3.))
            # self.error_loss_Q = tf.cast(tf.reduce_sum(0.0), self.vdtype)

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
            self.state_input = tf.placeholder(self.vdtype, shape=(None, self.num_state), name="state_input")
            self.truth_state = [tf.placeholder(self.vdtype, shape=(None, self.num_state), name="y_truth_{}".format(t)) for t in range(self.max_seq)]
            self.seqweightin = tf.placeholder(self.vdtype, [None, self.max_seq], name='seqweight')

            self.init_c_fwf = tf.placeholder(name='init_c_fwf', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.init_h_fwf = tf.placeholder(name='init_h_fwf', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_fw_in_state = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwf, self.init_h_fwf)

            self.init_c_fwf2 = tf.placeholder(name='init_c_fwf2', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.init_h_fwf2 = tf.placeholder(name='init_h_fwf2', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.state_fw_in_state2 = tf.contrib.rnn.LSTMStateTuple(self.init_c_fwf2, self.init_h_fwf2)

            self.init_c_fwf3 = tf.placeholder(name='init_c_fwf3', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.init_h_fwf3 = tf.placeholder(name='init_h_fwf3', shape=[None, self.F_hidden], dtype=self.vdtype)
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
            with tf.variable_scope('Source_Track_Forward/r_cov'):
                if use_dropout:
                    self.source_fwf = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
                                                             input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
                else:
                    if is_training:
                        self.source_fwf = cell_type(self.F_hidden, ratio_on=0.1, period_init_min=1, period_init_max=self.max_seq, leak=0.01)
                    else:
                        self.source_fwf = cell_type(self.F_hidden, ratio_on=0.1, period_init_min=1, period_init_max=self.max_seq, leak=0.0)

            with tf.variable_scope('Source_Track_Forward2/q_cov'):
                if use_dropout:
                    self.source_fwf2 = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
                                                              input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
                else:
                    if is_training:
                        self.source_fwf2 = cell_type(self.F_hidden, ratio_on=0.1, period_init_min=1, period_init_max=self.max_seq, leak=0.01)
                    else:
                        self.source_fwf2 = cell_type(self.F_hidden, ratio_on=0.1, period_init_min=1, period_init_max=self.max_seq, leak=0.0)

            with tf.variable_scope('Source_Track_Forward3/state'):
                if use_dropout:
                    self.source_fwf3 = tfc.rnn.DropoutWrapper(cell_type(self.F_hidden), input_keep_prob=self.drop_rate, variational_recurrent=True,
                                                              input_size=tf.TensorShape([self.F_hidden]), dtype=self.vdtype)
                else:
                    if is_training:
                        self.source_fwf3 = cell_type(self.F_hidden, ratio_on=0.1, period_init_min=1, period_init_max=self.max_seq, leak=0.01)
                    else:
                        self.source_fwf3 = cell_type(self.F_hidden, ratio_on=0.1, period_init_min=1, period_init_max=self.max_seq, leak=0.0)

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
            smooth, A, Q, R, B, S_inv, u, meas_uvw, state1_out, state2_out, state3_out = self.smooth()  # for plotting smoothed posterior
            # filter_out, A, Q, R, B, S_inv, u, meas_uvw, prediction, state1_out, state2_out, state3_out = self.filter()  # for plotting smoothed posterior
            filter_out, _, _, _, _, _, _, _, prediction, _, _, _ = self.filter()  # for plotting smoothed posterior

        else:
            filter_out, A, Q, R, B, S_inv, u, meas_uvw, prediction, state1_out, state2_out, state3_out = self.filter()

        self.ao_list = A
        self.qo_list = Q
        self.ro_list = R
        self.uo_list = u
        self.bo_list = B
        self.si_list = S_inv

        self.new_meas = meas_uvw
        # self.alphas = alphas
        self.final_state_filter = filter_out[0]
        self.final_state_prediction = prediction[0]
        if self.mode == 'training':
            self.final_state_smooth = smooth[0]
        else:
            self.final_state_smooth = filter_out[0]

        self.final_cov_filter = filter_out[1]
        self.final_cov_prediction = prediction[1]
        if self.mode == 'training':
            self.final_cov_smooth = smooth[1]
        else:
            self.final_cov_smooth = filter_out[1]

        self.state_fwf = state1_out
        self.state_fwf2 = state2_out
        self.state_fwf3 = state3_out

        _y = tf.stack(self.truth_state, axis=1)

        self.get_elbo(filter_out)

        with tf.variable_scope('regression_loss'):
            total_weight = tf.cast(self.seqweightin, self.vdtype)
            tot = tf.cast(self.max_seq, self.vdtype)
            loss_func = weighted_mape_tf

            # Measurement Error
            pos1m_err = loss_func(_y[:, :, 0], self.new_meas[:, :, 0], total_weight, tot, name='merr1')
            pos2m_err = loss_func(_y[:, :, 4], self.new_meas[:, :, 1], total_weight, tot, name='merr1')
            pos3m_err = loss_func(_y[:, :, 8], self.new_meas[:, :, 2], total_weight, tot, name='merr1')

            # State Error
            pos1e_err = loss_func(_y[:, :, 0], self.z_smooth[:, :, 0], total_weight, tot, name='serr1')
            pos2e_err = loss_func(_y[:, :, 4], self.z_smooth[:, :, 4], total_weight, tot, name='serr1')
            pos3e_err = loss_func(_y[:, :, 8], self.z_smooth[:, :, 8], total_weight, tot, name='serr1')

            total_err_state = pos1e_err + pos2e_err + pos3e_err
            total_err_meas = pos1m_err + pos2m_err + pos3m_err

            e1 = (pos1e_err / pos1m_err)
            e2 = (pos2e_err / pos2m_err)
            e3 = (pos3e_err / pos3m_err)
            self.meas_error = total_err_state / total_err_meas

            # state_loss_pos100 = tf.where(pos1e_err > pos1m_err, 10000*tf.ones_like(pos1e_err), pos1e_err)
            # state_loss_pos200 = tf.where(pos2e_err > pos2m_err, 10000*tf.ones_like(pos1e_err), pos2e_err)
            # state_loss_pos300 = tf.where(pos3e_err > pos3m_err, 10000*tf.ones_like(pos1e_err), pos3e_err)

            state_loss_pos100 = loss_func(_y[:, :, 0], self.z_smooth[:, :, 0], total_weight, tot, name='pos1_err')
            state_loss_pos200 = loss_func(_y[:, :, 4], self.z_smooth[:, :, 4], total_weight, tot, name='pos2_err')
            state_loss_pos300 = loss_func(_y[:, :, 8], self.z_smooth[:, :, 8], total_weight, tot, name='pos3_err')
            state_loss_vel100 = loss_func(_y[:, :, 1], self.z_smooth[:, :, 1], total_weight, tot, name='vel1_err')
            state_loss_vel200 = loss_func(_y[:, :, 5], self.z_smooth[:, :, 5], total_weight, tot, name='vel2_err')
            state_loss_vel300 = loss_func(_y[:, :, 9], self.z_smooth[:, :, 9], total_weight, tot, name='vel3_err')
            state_loss_acc100 = loss_func(_y[:, :, 2], self.z_smooth[:, :, 2], total_weight, tot, name='acc1_err')
            state_loss_acc200 = loss_func(_y[:, :, 6], self.z_smooth[:, :, 6], total_weight, tot, name='acc2_err')
            state_loss_acc300 = loss_func(_y[:, :, 10], self.z_smooth[:, :, 10], total_weight, tot, name='acc3_err')
            state_loss_jer100 = loss_func(_y[:, :, 3], self.z_smooth[:, :, 3], total_weight, tot, name='jer1_err')
            state_loss_jer200 = loss_func(_y[:, :, 7], self.z_smooth[:, :, 7], total_weight, tot, name='jer2_err')
            state_loss_jer300 = loss_func(_y[:, :, 11], self.z_smooth[:, :, 11], total_weight, tot, name='jer3_err')

            self.SLPf1 = state_loss_pos100 + state_loss_pos200 + state_loss_pos300
            self.SLVf1 = state_loss_vel100 + state_loss_vel200 + state_loss_vel300
            self.SLAf1 = state_loss_acc100 + state_loss_acc200 + state_loss_acc300
            self.SLJf1 = state_loss_jer100 + state_loss_jer200 + state_loss_jer300

            # self.SLPf1 = tf.truediv(self.SLPf1, tf.cast(tf.reduce_sum(self.batch_size), self.vdtype))
            # self.SLVf1 = tf.truediv(self.SLVf1, tf.cast(tf.reduce_sum(self.batch_size), self.vdtype))
            # self.SLAf1 = tf.truediv(self.SLAf1, tf.cast(tf.reduce_sum(self.batch_size), self.vdtype))
            # self.SLJf1 = tf.truediv(self.SLJf1, tf.cast(tf.reduce_sum(self.batch_size), self.vdtype))

            self.SLPf1 = tf.truediv(self.SLPf1, self.num_el)
            self.SLVf1 = tf.truediv(self.SLVf1, self.num_el)
            self.SLAf1 = tf.truediv(self.SLAf1, self.num_el)
            self.SLJf1 = tf.truediv(self.SLJf1, self.num_el)

            self.rmse_pos = self.SLPf1
            self.rmse_vel = self.SLVf1
            self.rmse_acc = self.SLAf1
            self.rmse_jer = self.SLJf1

            self.maha_out = tf.truediv(tf.reduce_sum(self.MDP * self.seqweightin[:, :, tf.newaxis]), self.num_el)

        with tf.variable_scope('learning_rate'):
            self.learning_rate = tf.train.exponential_decay(self.learning_rate_inp, global_step=self.global_step, decay_steps=self.decay_steps, decay_rate=0.9, staircase=True)


        with tf.variable_scope("TrainOps"):
            print('Updating Gradients')
            all_vars = tf.trainable_variables()
            # tfc.opt.MomentumWOptimizer(learning_rate=self.learning_rate, momentum=0.9, weight_decay=1e-10, name='r3')
            # tfc.opt.AdamWOptimizer(weight_decay=1e-7, learning_rate=self.learning_rate, name='opt')

            # optq = tfc.opt.MultitaskOptimizerWrapper(tfc.opt.AdamWOptimizer(weight_decay=1e-10, learning_rate=self.learning_rate, name='opt'))
            optq = tfc.opt.MomentumWOptimizer(learning_rate=self.learning_rate, momentum=0.95, weight_decay=1e-10, name='opt')
            lower_bound = (self.rl * 1 + self.entropy * 1 + self.maha_loss * 0 + self.error_loss_full + self.error_loss_Q * 0 + self.rmse_pos * 1 + self.error_loss_pos * 0 + self.meas_error)
            gradvarsq = optq.compute_gradients(lower_bound, all_vars, colocate_gradients_with_ops=False)
            gradvarsq, _ = tfc.opt.clip_gradients_by_global_norm(gradvarsq, 5.)
            self.train_q = optq.apply_gradients(gradvarsq, global_step=self.global_step)

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
        # tf.summary.scalar("Acceleration", self.acc_error)
        tf.summary.scalar("MeasurementCovariance", self.rl)
        tf.summary.scalar("TransitionCovariance", self.error_loss_Q)
        tf.summary.scalar("Entropy", self.entropy)
        tf.summary.scalar("PositionCovariance", self.error_loss_pos)
        tf.summary.scalar("VelocityCovariance", self.error_loss_vel)
        tf.summary.scalar("TotalCovariance", self.error_loss_full)
        tf.summary.scalar("InitialCovariance", self.error_loss_initial)
        tf.summary.scalar("MahalanobisLoss", self.maha_loss)
        tf.summary.scalar("MahalanobisDistance", tf.truediv(tf.reduce_sum(self.MDP * self.seqweightin[:, :, tf.newaxis]), self.num_el))
        tf.summary.scalar("MahalanobisInverse", tf.truediv(tf.reduce_sum(self.MDPi * self.seqweightin[:, :, tf.newaxis]), self.num_el))
        tf.summary.scalar("RMSE_pos", self.rmse_pos)
        tf.summary.scalar("RMSE_vel", self.rmse_vel)
        tf.summary.scalar("RMSE_acc", self.rmse_acc)
        tf.summary.scalar("RMSE_jer", self.rmse_jer)
        tf.summary.scalar("Learning_Rate", self.learning_rate)
        tf.summary.scalar("Num_ele", self.num_el)

    def train(self, data_rate, max_exp_seq):

        # rho0 = 1.22  # kg / m**3
        # k0 = 0.14141e-3
        # area = 0.25  # / self.RE  # meter squared
        # cd = 0.03  # unitless
        # gmn = self.GM / (self.RE ** 3)

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
            self.saver = tf.train.import_meta_graph(self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step) + '.meta')
            self.saver.restore(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step))
            print("filter restored.")
        except:
            self.saver = tf.train.Saver(save_relative_paths=True)
            print("Could not restore filter")
            start_epoch = 0
            step = 0

        ds = DataServerPrePro(self.train_dir, self.test_dir)

        for epoch in range(int(start_epoch), self.max_epoch):

            n_train_batches = int(ds.num_examples_train)

            for minibatch_index in range(n_train_batches):

                testing = False

                # if minibatch_index % 100 == 0 and minibatch_index != 0:
                #     testing = True
                #     print('Testing filter for epoch ' + str(epoch))
                # else:
                #     testing = False
                #     print('Training filter for epoch ' + str(epoch))

                # Data is unnormalized at this point
                # x_data, y_data, batch_number, total_batches, ecef_ref, lla_data = ds.load(batch_size=self.batch_size_np, constant=self.constant, test=testing,
                #                                                                             max_seq_len=self.max_exp_seq, HZ=self.data_rate)

                x_data, y_data, ecef_ref, lla_data = ds.load(batch_size=self.batch_size_np, constant=self.constant, test=testing)

                if shuffle_data:
                    shuf = np.arange(x_data.shape[0])
                    np.random.shuffle(shuf)
                    x_data = x_data[shuf]
                    y_data = y_data[shuf]

                lla_datar = copy.copy(lla_data[:, 0, :])

                # GET SENSOR VECTOR
                sensor_vector = np.zeros([self.batch_size_np, 3])
                for i in range(lla_datar.shape[0]):
                    cur_lla = list(lla_datar[i, :2])
                    if cur_lla in self.aeg_loc:
                        vec = [1, 0, 0]
                    elif cur_lla in self.pat_loc:
                        vec = [0, 1, 0]
                    elif cur_lla in self.tpy_loc:
                        vec = [0, 0, 1]
                    else:
                        pdb.set_trace()
                        print("unknown sensor location")
                        pass
                    sensor_vector[i, :] = vec

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

                count, _, _, _, _, _, prev_cov, prev_Q, prev_R, q_plot, q_plott, k_plot, out_plot_filter, out_plot_F, out_plot_smooth, time_vals, \
                meas_plot, truth_plot, Q_plot, R_plot, maha_plot = initialize_run_variables(self.batch_size_np, self.max_seq, self.num_state)

                fd = {}

                windows = int((x.shape[1]) / self.max_seq)
                time_plotter = np.zeros([self.batch_size_np, int(x.shape[1]), 1])
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
                        initial_Q = prev_Q[:, -1, :, :]
                        initial_R = prev_R[:, -1, :, :]
                        initial_state = initial_state[:, :, self.idxi]

                        dt0 = prev_time - initial_time[:, -1:, :]
                        dt1 = current_time[:, 0, :] - prev_time[:, 0, :]
                        current_state, covariance_out, converted_meas, pred_state, pred_covariance = \
                            unscented_kalman_np(self.batch_size_np, prev_x.shape[1], initial_state[:, -1, :], prev_cov[:, -1, :, :], prev_x, prev_time, dt0, dt1, lla_datar)

                        current_state_estimate = current_state[:, :, self.idxo]
                        current_cov_estimate = covariance_out[-1]
                        prev_state_estimate = initial_state[:, :, self.idxo]
                        prev_covariance_estimate = prev_cov

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
                        std = 0.3
                        fd.update({self.init_c_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.init_h_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.init_c_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.init_h_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.init_c_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.init_h_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

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
                    fd.update({self.state_input: current_state_estimate.reshape(-1, self.num_state)})
                    fd.update({self.prev_time: prev_time[:, :, 0]})
                    fd.update({self.current_timei[t]: current_time[:, t, :].reshape(-1, 1) for t in range(self.max_seq)})
                    fd.update({self.drop_rate: 0.8})

                    lr = 1e-3
                    randn = random.random()
                    # randn = 0.
                    if epoch < 1 and minibatch_index < 25:
                        if randn > 0.1:
                            stateful = False
                        else:
                            stateful = True
                    elif epoch >= 1 and epoch < 5:
                        if randn > 0.5:
                            stateful = False
                        else:
                            stateful = True
                    else:
                        stateful = True

                    fd.update({self.learning_rate_inp: lr})

                    t00 = time.time()

                    if tstep == 0:
                        ittt = 1
                    else:
                        ittt = 1

                    for _ in range(ittt):
                        filter_output, smooth_output, q_out_t, q_out, _, rmsp, rmsv, rmsa, rmsj, LR, \
                        cov_pos_loss, cov_vel_loss, kalman_cov_loss, maha_loss, MD, trace_loss, rl, \
                        entropy, qt_out, rt_out, at_out, q_loss, state_fwf, state_fwf2, state_fwf3, new_meas, \
                        y_t_resh, Cz_t, MDP, mvn_inv, state_error, summary_str = \
                            self.sess.run([self.final_state_filter,
                                           self.final_state_smooth,
                                           self.final_cov_filter,
                                           self.final_cov_filter,
                                           self.train_q,
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

                    if stateful is True:
                        fd.update({self.init_c_fwf: state_fwf[0]})
                        fd.update({self.init_h_fwf: state_fwf[1]})
                        fd.update({self.init_c_fwf2: state_fwf2[0]})
                        fd.update({self.init_h_fwf2: state_fwf2[1]})
                        fd.update({self.init_c_fwf3: state_fwf3[0]})
                        fd.update({self.init_h_fwf3: state_fwf3[1]})

                    else:
                        fd.update({self.init_c_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.init_h_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.init_c_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.init_h_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.init_c_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                        fd.update({self.init_h_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

                    idx = -1
                    prev_x = np.concatenate([prev_x, current_x], axis=1)
                    prev_x = prev_x[:, idx, np.newaxis, :]

                    # err0 = np.sum(np.abs(temp_prev_y - temp_pred0))
                    # err1 = np.sum(np.abs(temp_prev_y - temp_pred1))
                    # if err1 < err0 and e > 5:
                    #     new_prev = temp_pred0
                    # else:
                    #     new_prev = temp_pred1

                    # plt.plot(current_time[0, :, :], new_meas[0, :, 1], 'b')
                    # plt.plot(current_time[0, :, :], current_y[0, :, 1], 'r')
                    # plt.plot(current_time[0, :, :], prop_output[0, :, 1], 'm')
                    # plt.plot(current_time[0, :, :], pred_output[0, :, 1], 'g')

                    prev_state_estimate = filter_output[:, -2, np.newaxis, :]
                    current_state_estimate = filter_output[:, -1, np.newaxis, :]

                    initial_Q = qt_out[:, -1, :, :]
                    initial_R = rt_out[:, -1, :, :]

                    prev_y = np.concatenate([prev_y, current_y], axis=1)
                    prev_y = prev_y[:, idx, np.newaxis, :]

                    prev_time = np.concatenate([prev_time, current_time], axis=1)
                    prev_time = prev_time[:, idx, np.newaxis, :]

                    prev_cov = np.concatenate([prev_cov[:, -1:, :, :], q_out_t], axis=1)
                    prev_cov = prev_cov[:, idx, np.newaxis, :, :]
                    current_cov_estimate = q_out_t[:, -1, :, :]
                    prev_covariance_estimate = q_out_t[:, -2, np.newaxis, :, :]

                    if tstep == 0:
                        # out_plot_F = full_final_output[0, np.newaxis, :, :]
                        out_plot_smooth = smooth_output[0, np.newaxis, :, :]
                        out_plot_filter = filter_output[0, np.newaxis, :, :]

                        q_plott = q_out_t[0, np.newaxis, :, :, :]
                        q_plot = q_out[0, np.newaxis, :, :, :]
                        qt_plot = qt_out[0, np.newaxis, :, :]
                        rt_plot = rt_out[0, np.newaxis, :, :]
                        at_plot = at_out[0, np.newaxis, :, :]
                        # trans_plot = trans_out[0, np.newaxis, :, :]

                        time_vals = current_time[0, np.newaxis, :, :]
                        meas_plot = new_meas[0, np.newaxis, :, :]
                        truth_plot = current_y[0, np.newaxis, :, :]

                    else:
                        # new_vals_F = full_final_output[0, :, :]  # current step
                        new_vals_smooth = smooth_output[0, :, :]
                        new_vals_filter = filter_output[0, :, :]
                        new_q = q_out[0, :, :, :]
                        new_qt = q_out_t[0, :, :, :]
                        new_qtt = qt_out[0, :, :, :]
                        new_rtt = rt_out[0, :, :, :]
                        new_att = at_out[0, :, :, :]
                        # new_trans = trans_out[0, :, :]

                        new_time = current_time[0, :, 0]
                        new_meas = new_meas[0, :, :]
                        new_truth = current_y[0, :, :]

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

                if minibatch_index % 5 == 0:
                    plotpath = self.plot_dir + '/epoch_' + str(epoch) + '_B_' + str(minibatch_index) + '_step_' + str(step)
                    if ~os.path.isdir(plotpath):
                        os.mkdir(plotpath)
                    # plot_all2(out_plot_X, out_plot_F, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, mean_y)
                    trans_plot = at_plot
                    comparison_plot(out_plot_filter, out_plot_smooth, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, qt_plot, rt_plot, trans_plot)

                if (n_train_batches - minibatch_index) == 1:
                    print("Saving Weights for Epoch " + str(epoch))
                    save_path = self.saver.save(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(epoch) + '_' + str(step) + ".ckpt", global_step=step)
                    print("Checkpoint saved at :: ", save_path)

    def evaluate(self, x_data, y_data, ecef_ref, lla_datar, epoch, minibatch_index, step):

        # GET SENSOR ONE HOTS
        sensor_vector = np.zeros([self.batch_size_np, 3])
        for i in range(lla_datar.shape[0]):
            cur_lla = list(lla_datar[i, :2])
            if cur_lla in self.aeg_loc:
                vec = [1, 0, 0]
            elif cur_lla in self.pat_loc:
                vec = [0, 1, 0]
            elif cur_lla in self.tpy_loc:
                vec = [0, 0, 1]
            else:
                pdb.set_trace()
                print("unknown sensor location")
                pass
            sensor_vector[i, :] = vec

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

                # acc = (initial_meas[:, 2, :] - 2*initial_meas[:, 1, :] + initial_meas[:, 0, :]) / dtn**2

                initial_state = np.expand_dims(np.concatenate([pos, vel, acc, np.random.normal(loc=np.zeros_like(acc), scale=1)], axis=1), 1)
                # initial_cov = prev_cov[:, -1, :, :]
                # initial_Q = prev_Q[:, -1, :, :]
                # initial_R = prev_R[:, -1, :, :]
                initial_state = initial_state[:, :, self.idxi]

                dt0 = prev_time - initial_time[:, -1:, :]
                dt1 = current_time[:, 0, :] - prev_time[:, 0, :]
                current_state, covariance_out, converted_meas, pred_state, pred_covariance = \
                    unscented_kalman_np(self.batch_size_np, prev_x.shape[1], initial_state[:, -1, :], prev_cov[:, -1, :, :], prev_x, prev_time, dt0, dt1, lla_datar)

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
                current_state_estimate = prev_state_estimate

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
            # fd.update({self.Q_inp: initial_Q})
            # fd.update({self.R_inp: initial_R})
            fd.update({self.state_input: current_state_estimate.reshape(-1, self.num_state)})
            fd.update({self.prev_time: prev_time[:, :, 0]})
            fd.update({self.current_timei[t]: current_time[:, t, :].reshape(-1, 1) for t in range(self.max_seq)})
            fd.update({self.drop_rate: 1.0})

            std = 0.0

            fd.update({self.init_c_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            fd.update({self.init_h_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            fd.update({self.init_c_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            fd.update({self.init_h_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            fd.update({self.init_c_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            fd.update({self.init_h_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

            ee = minibatch_index

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
            pred_output0 = pred_output0[:, :, self.idxo]
            pred_output00 = pred_output00[:, :, self.idxo]
            pred_output1 = pred_output1[:, :, self.idxo]

            stateful = True
            if stateful is True:
                fd.update({self.init_c_fwf: state_fwf[0]})
                fd.update({self.init_h_fwf: state_fwf[1]})
                fd.update({self.init_c_fwf2: state_fwf2[0]})
                fd.update({self.init_h_fwf2: state_fwf2[1]})
                fd.update({self.init_c_fwf3: state_fwf3[0]})
                fd.update({self.init_h_fwf3: state_fwf3[1]})

            else:
                std = 0.3
                fd.update({self.init_c_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                fd.update({self.init_h_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                fd.update({self.init_c_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                fd.update({self.init_h_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                fd.update({self.init_c_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
                fd.update({self.init_h_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

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
            prev_x = np.concatenate([prev_x, current_x], axis=1)
            prev_x = prev_x[:, idx, np.newaxis, :]

            # err0 = np.sum(np.abs(temp_prev_y - temp_pred0))
            # err1 = np.sum(np.abs(temp_prev_y - temp_pred1))
            # if err1 < err0 and e > 5:
            #     new_prev = temp_pred0
            # else:
            #     new_prev = temp_pred1

            # plt.plot(current_time[0, :, :], new_meas[0, :, 1], 'b')
            # plt.plot(current_time[0, :, :], current_y[0, :, 1], 'r')
            # plt.plot(current_time[0, :, :], prop_output[0, :, 1], 'm')
            # plt.plot(current_time[0, :, :], pred_output[0, :, 1], 'g')

            prev_state_estimate = prop_output[:, -2, np.newaxis, :]
            current_state_estimate = prop_output[:, -1, np.newaxis, :]

            prev_y = np.concatenate([prev_y, current_y], axis=1)
            prev_y = prev_y[:, idx, np.newaxis, :]

            prev_time = np.concatenate([prev_time, current_time], axis=1)
            prev_time = prev_time[:, idx, np.newaxis, :]

            prev_cov = np.concatenate([prev_cov[:, -1:, :, :], q_out_t], axis=1)
            prev_cov = prev_cov[:, idx, np.newaxis, :, :]
            current_cov_estimate = q_out_t[:, -1, :, :]
            prev_covariance_estimate = q_out_t[:, -2, np.newaxis, :, :]

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
                # trans_plot = trans_out[0, np.newaxis, :, :]

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
                # new_trans = trans_out[0, :, :]

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
                # trans_plot = np.concatenate([trans_plot, new_trans[np.newaxis, :, :]], axis=1)

        plotpath = self.plot_eval_dir + '/epoch_' + str(epoch) + '_eval_B_' + str(minibatch_index) + '_step_' + str(step)
        if not os.path.isdir(plotpath):
            os.mkdir(plotpath)

        # plot_all2(out_plot_X, out_plot_F, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, mean_y)
        trans_plot = at_plot
        comparison_plot(out_plot_X, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, qt_plot, rt_plot, trans_plot)

        # if minibatch_index % 50 == 0 and minibatch_index != 0:
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        print("Saving filter Weights for epoch" + str(epoch))
        save_path = self.saver.save(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(epoch) + '_' + str(step) + ".ckpt", global_step=step)
        print("Checkpoint saved at: ", save_path)

    def test(self, data_rate, max_exp_seq):

        shuffle_data = False
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
        self.saver = tf.train.import_meta_graph(self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step) + '.meta')
        self.saver.restore(self.sess, self.checkpoint_dir + '/' + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step))
        print("filter restored.")

        ds = DataServerPrePro(self.train_dir, self.test_dir)

        n_train_batches = int(ds._num_examples_train)

        for minibatch_index in range(n_train_batches):

            testing = True
            print('Testing filter for batch ' + str(minibatch_index))

            # Data is unnormalized at this point
            x_train, y_train, ecef_ref, lla_data = ds.load(batch_size=self.batch_size_np, constant=self.constant, test=testing)

            lla_datar = copy.copy(lla_data[:, 0, :])

            # GET SENSOR ONE HOTS
            sensor_vector = np.zeros([self.batch_size_np, 3])
            for i in range(lla_datar.shape[0]):
                cur_lla = list(lla_datar[i, :2])
                if cur_lla in self.aeg_loc:
                    vec = [1, 0, 0]
                elif cur_lla in self.pat_loc:
                    vec = [0, 1, 0]
                elif cur_lla in self.tpy_loc:
                    vec = [0, 0, 1]
                else:
                    pdb.set_trace()
                    print("unknown sensor location")
                    pass
                sensor_vector[i, :] = vec

            lla_datar[:, 0] = lla_datar[:, 0] * np.pi / 180
            lla_datar[:, 1] = lla_datar[:, 1] * np.pi / 180

            x_train = np.concatenate([x_train[:, :, 0, np.newaxis], x_train[:, :, 4:7]], axis=2)  # rae measurements

            y_uvw = y_train[:, :, :3] - ecef_ref
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

            x, y, meta, prev_y, prev_x, prev_time, prev_meta, initial_meas, initial_state_truth, initial_time, max_length = prepare_batch(0, x_train, y_train, s_train,
                                                                                                                                             seq_len=self.max_seq, batch_size=self.batch_size_np,
                                                                                                                                             new_batch=True)

            count, _, _, _, _, _, prev_cov, prev_Q, prev_R, q_plot, q_plott, k_plot, out_plot_X, out_plot_F, out_plot_P, time_vals, \
            meas_plot, truth_plot, Q_plot, R_plot, maha_plot = initialize_run_variables(self.batch_size_np, self.max_seq, self.num_state)

            prev_cov = prev_cov[:, -1:, :, :]
            fd = {}

            time_plotter = np.zeros([self.batch_size_np, int(x.shape[1]), 1])

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

            cur_time = x[:, 0, 0]

            time_plotter[:, 0, :] = cur_time[:, np.newaxis]
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

            R1 = np.linalg.norm(initial_meas_uvw + ecef_ref[:, 0, np.newaxis, :], axis=2, keepdims=True)
            R1 = np.mean(R1, axis=1)
            R1 = np.where(np.less(R1, np.ones_like(R1) * self.RE), np.ones_like(R1) * self.RE, R1)
            rad_temp = np.power(R1, 3)
            GMt1 = np.divide(self.GM, rad_temp)
            acc = get_legendre_np(GMt1, pos + ecef_ref[:, 0, :], R1)

            # acc = (initial_meas[:, 2, :] - 2*initial_meas[:, 1, :] + initial_meas[:, 0, :]) / dtn**2

            std = 0.0

            fd.update({self.init_c_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            fd.update({self.init_h_fwf: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            fd.update({self.init_c_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            fd.update({self.init_h_fwf2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            fd.update({self.init_c_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
            fd.update({self.init_h_fwf3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

            fd.update({self.is_training: False})
            fd.update({self.deterministic: True})
            stateful = True

            initial_state = np.expand_dims(np.concatenate([pos, vel, acc, np.random.normal(loc=np.zeros_like(acc), scale=1)], axis=1), 1)
            initial_cov = prev_cov[:, -1, :, :]
            initial_Q = prev_Q[:, -1, :, :]
            initial_R = prev_R[:, -1, :, :]
            initial_state = initial_state[:, :, self.idxi]

            prev_state = copy.copy(prev_y)
            prev_meas = copy.copy(prev_x)

            dt0 = prev_time - initial_time[:, -1:, :]
            dt1 = time_data[:, 0, :] - prev_time[:, 0, :]
            current_state_estimate, covariance_out, converted_meas, pred_state, pred_covariance = \
                unscented_kalman_np(self.batch_size_np, prev_meas.shape[1], initial_state[:, -1, :], prev_cov[:, -1, :, :], prev_meas, prev_time, dt0, dt1, lla_datar)
            steps = x_data.shape[1]

            current_cov_estimate = covariance_out[-1]
            prev_state_estimate = initial_state[:, -1, np.newaxis, :]
            prev_covariance_estimate = prev_cov[:, -1, :, :]
            current_state_estimate = current_state_estimate[:, :, self.idxo]
            prev_state_estimate = prev_state_estimate[:, :, self.idxo]

            update = False
            for step in range(0, steps):

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
                current_int = int_time[:, step, tf.newaxis]
                current_weight = seqweight[:, step, tf.newaxis]

                fd.update({self.measurement[0]: current_x.reshape(-1, self.num_meas)})
                fd.update({self.prev_measurement: prev_meas.reshape(-1, self.num_meas)})
                fd.update({self.prev_covariance_estimate: prev_covariance_estimate})
                fd.update({self.truth_state[0]: current_y.reshape(-1, self.num_state)})
                fd.update({self.prev_state_truth: prev_state.reshape(-1, self.num_state)})
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
                fd.update({self.state_input: current_state_estimate.reshape(-1, self.num_state)})
                fd.update({self.prev_time: prev_time[:, :, 0]})
                fd.update({self.current_timei[0]: current_time.reshape(-1, 1)})
                fd.update({self.drop_rate: 1.0})

                pred_output0, pred_output00, pred_output1, q_out_t, q_out, rmsp, rmsv, rmsa, rmsj, LR, \
                cov_pos_loss, cov_vel_loss, kalman_cov_loss, maha_loss, MD, trace_loss, rl, \
                entropy, qt_out, rt_out, at_out, q_loss, \
                state_fwf, state_fwf2, new_meas, y_t_resh, Cz_t = \
                    self.sess.run([self.final_state_filter,
                                   self.final_state_filter,
                                   self.final_state_smooth,
                                   self.final_cov_filter,
                                   self.final_cov_filter,
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
                                   self.y_t_resh,
                                   self.Cz_t],
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

                if stateful is True:
                    fd.update({self.init_c_fwf: state_fwf[0]})
                    fd.update({self.init_h_fwf: state_fwf[1]})
                    fd.update({self.init_c_fwf2: state_fwf2[0]})
                    fd.update({self.init_h_fwf2: state_fwf2[1]})

                # prev_state = prev_state[:, np.newaxis, :]
                current_y = current_y[:, np.newaxis, :]

                current_y = current_y[:, :, self.idxo]
                prev_state = prev_state[:, :, self.idxo]

                pred_output0 = pred_output0[:, :, self.idxo]
                pred_output00 = pred_output00[:, :, self.idxo]
                pred_output1 = pred_output1[:, :, self.idxo]

                prop_output = np.array(pred_output0)
                pred_output = np.array(pred_output1)
                full_final_output = np.array(pred_output00)

                idx = -1
                prev_meas = np.concatenate([prev_meas, current_x], axis=1)
                prev_meas = prev_meas[:, idx, np.newaxis, :]

                # err0 = np.sum(np.abs(temp_prev_y - temp_pred0))
                # err1 = np.sum(np.abs(temp_prev_y - temp_pred1))
                # if err1 < err0 and e > 5:
                #     new_prev = temp_pred0
                # else:
                #     new_prev = temp_pred1

                # plt.plot(current_time[0, :, :], new_meas[0, :, 1], 'b')
                # plt.plot(current_time[0, :, :], current_y[0, :, 1], 'r')
                # plt.plot(current_time[0, :, :], prop_output[0, :, 1], 'm')
                # plt.plot(current_time[0, :, :], pred_output[0, :, 1], 'g')

                prev_state_estimate = copy.copy(current_state_estimate)
                current_state_estimate = prop_output[:, -1, np.newaxis, :]

                initial_Q = qt_out[:, -1, :, :]
                initial_R = rt_out[:, -1, :, :]

                prev_state = np.concatenate([prev_state, current_y], axis=1)
                prev_state = prev_state[:, idx, np.newaxis, :]

                prev_time = np.concatenate([prev_time, current_time], axis=1)
                prev_time = prev_time[:, idx, np.newaxis, :]

                prev_cov = np.concatenate([prev_cov[:, -1:, :, :], q_out_t], axis=1)
                prev_cov = prev_cov[:, idx, np.newaxis, :, :]

                prev_covariance_estimate = copy.copy(current_cov_estimate)
                current_cov_estimate = q_out_t[:, -1, :, :]

                prev_y = copy.copy(prev_state)
                prev_x = copy.copy(prev_meas)

                if step == 0:
                    out_plot_F = full_final_output[0, np.newaxis, :, :]
                    out_plot_X = pred_output[0, np.newaxis, :, :]
                    out_plot_P = prop_output[0, np.newaxis, :, :]

                    q_plott = q_out_t[0, np.newaxis, :, :, :]
                    q_plot = q_out[0, np.newaxis, :, :, :]
                    qt_plot = qt_out[0, np.newaxis, :, :]
                    rt_plot = rt_out[0, np.newaxis, :, :]
                    at_plot = at_out[0, np.newaxis, :, :]
                    # trans_plot = trans_out[0, np.newaxis, :, :]

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
                    # new_trans = trans_out[0, :, :]

                    new_time = current_time[0, :, 0]
                    new_meas = new_meas[0, :, :]
                    new_truth = current_y[0, :, :]

                if step > 0:
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
                    # trans_plot = np.concatenate([trans_plot, new_trans[np.newaxis, :, :]], axis=1)

            trans_plot = np.zeros_like(out_plot_X)

            plotpath = self.plot_test_dir + '/epoch_' + str(9999) + '_TEST_B_' + str(minibatch_index)
            if ~os.path.isdir(plotpath): \
                    os.mkdir(plotpath)

            comparison_plot(out_plot_X, out_plot_P, meas_plot, meas_plot, truth_plot, q_plot, q_plott, time_vals, tstep, plotpath, ecef_ref, qt_plot, rt_plot, trans_plot)
