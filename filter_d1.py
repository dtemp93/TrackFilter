import math
from tensorflow.contrib.layers import fully_connected as FCL

from .helper import *
from .plotting import *
from .propagation_utils import *

from tqdm import tqdm

from .rnn_cell import IndyGRUCell
tfna = tf.newaxis
varsci = tf.initializers.variance_scaling
tfar = tf.AUTO_REUSE
ELU = tf.nn.elu


class Filter(object):
    def __init__(self, sess, state_type='INDYGRU', mode='training',
                 data_dir='', filter_name='', save_dir='', outpath='',
                 F_hidden=18, num_state=12, num_meas=3, max_seq=1, num_mixtures=4,
                 max_epoch=100, batch_size=1, learning_rate=1e-3,
                 plot_interval=1, checkpoint_interval=50, dropout_rate=1.0,
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
        self.plot_test_dir = outpath + '/plots_test/'
        self.outpath = outpath
        self.checkpoint_dir = save_dir + '/checkpoints/'
        self.log_dir = save_dir + '/logs/'
        self.max_epoch = max_epoch
        self.state_type = state_type
        self.filter_name = filter_name
        self.constant = constant
        self.learning_rate_main = learning_rate
        self.dropout_rate_main = dropout_rate

        self.batch_size_np = batch_size

        self.plot_interval = plot_interval
        self.checkpoint_interval = checkpoint_interval
        self.convert_to_uvw = True

        self.decimate_data = decimate_data

        with tf.variable_scope('Class_Init'):
            self.global_step = tf.Variable(initial_value=0, name="global_step", trainable=False, collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES], dtype=tf.int32)

            self.idxi = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
            self.idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

            # Meta Variables
            self.plen = int(self.max_seq)
            self.pi_val = tf.constant(math.pi, dtype=tf.float64)

            total_samples = 15000
            avg_seq_len = 2500
            self.decay_steps = (total_samples / self.batch_size_np) * (avg_seq_len / self.max_seq)
            self.e_sq = 0.00669437999014132
            self.RE = 6378137
            self.GM = 398600441890000

            self.vdtype = tf.float64
            self.vdp_np = np.float64

            self.seqlen = tf.placeholder(tf.int32, [None])
            self.int_time = tf.placeholder(tf.float64, [None, self.max_seq])
            self.batch_size = tf.shape(self.seqlen)[0]

            alpha = 1e-1 * tf.ones([self.batch_size, 1], dtype=self.vdtype)
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

    def alpha(self, dt, pstate, meas_rae, prev_meas_uvw, cov_est, LLA, sensor_noise, states):

        with tf.variable_scope('alpha'):
            state1 = states[0]
            state2 = states[1]
            state3 = states[2]
            state4 = states[3]

            lat = LLA[:, 0, tfna]
            lon = LLA[:, 1, tfna]
            alt = LLA[:, 2, tfna]

            chi = self.RE / tf.sqrt(1 - self.e_sq * (tf.sin(lat)) ** 2)

            xs = (chi + alt) * tf.cos(lat) * tf.cos(lon)
            ys = (chi + alt) * tf.cos(lat) * tf.sin(lon)
            zs = (alt + chi * (1 - self.e_sq)) * tf.sin(lat)

            ecef_ref = tf.concat([xs, ys, zs], axis=1)

            R = meas_rae[:, 0, tfna]
            A = meas_rae[:, 1, tfna]
            E = meas_rae[:, 2, tfna]

            east = (R * tf.sin(A) * tf.cos(E))
            north = (R * tf.cos(E) * tf.cos(A))
            up = (R * tf.sin(E))

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

            _, At, _, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                                 dimension=int(self.num_state / 3),
                                 sjix=self.om[:, :, 0] * 1 ** 2,
                                 sjiy=self.om[:, :, 0] * 1 ** 2,
                                 sjiz=self.om[:, :, 0] * 1 ** 2,
                                 aji=self.om[:, :, 0] * 1.0)

            prop_state = tf.matmul(At, pstate[:, :, tfna])
            prop_cov = tf.matmul(tf.matmul(At, cov_est), At, transpose_b=True)

            delta_state = prop_state[:, :, 0] - pstate
            # delta_cov = prop_cov - cov_est
            delta_meas = meas_uvw - prev_meas_uvw

            pos_part = tf.matmul(self.meas_mat, prop_state)
            cov_diag = tf.matrix_diag_part(prop_cov)
            # vel_part = tf.concat([prop_state[:, 1, tfna], prop_state[:, 5, tfna], prop_state[:, 9, tfna]], axis=1)
            acc_part = tf.concat([prop_state[:, 2, tfna], prop_state[:, 6, tfna], prop_state[:, 10, tfna]], axis=1)
            # jer_part = tf.concat([prop_state[:, 3, tfna], prop_state[:, 7, tfna], prop_state[:, 11, tfna]], axis=1)
            pre_residual = meas_uvw[:, :, tfna] - pos_part

            gs_input = acc_part[:, :, 0] / tf.ones_like(acc_part[:, :, 0]) * 9.81
            gs_input = tf.where(tf.is_nan(gs_input), tf.ones_like(gs_input), gs_input)

            cov_diag_n = cov_diag / tf.ones_like(cov_diag)
            pre_res_n = pre_residual[:, :, 0] / tf.ones_like(pre_residual[:, :, 0])
            altitude_n = altitude / (tf.ones_like(altitude) * 1000)
            gravity_n = gravity / (tf.ones_like(gravity) * 9.81)

            rnn_inp = tf.concat([dt, pre_res_n, cov_diag_n, delta_state, delta_meas, altitude_n, gravity_n, gs_input], axis=1)

            inp_width = rnn_inp.shape[1].value

            rnn_inpa0 = FCL(rnn_inp, inp_width, activation_fn=ELU, weights_initializer=varsci, scope='rnn_inpa0/q_cov', reuse=tfar)
            rnn_inpa1 = FCL(rnn_inpa0, inp_width, activation_fn=ELU, weights_initializer=varsci, scope='rnn_inpa1/q_cov', reuse=tfar)
            rnn_inpa2 = FCL(tf.concat([rnn_inpa0, rnn_inpa1], axis=1), self.F_hidden, activation_fn=ELU, weights_initializer=varsci, scope='rnn_inpa2/q_cov',
                            reuse=tfar)

            with tf.variable_scope('Cell_1/q_cov', reuse=tfar):
                (outa1, state1) = self.source_fwf(rnn_inpa2, state=state1)

            with tf.variable_scope('Cell_2/q_cov', reuse=tfar):
                (outa2, state2) = self.source_fwf2(outa1, state=state2)

            outa = tf.concat([outa1, outa2], axis=1)

            with tf.variable_scope('Cell_3/q_cov', reuse=tfar):
                (outb1, state3) = self.source_fwf3(outa, state=state3)

            with tf.variable_scope('Cell_4/q_cov', reuse=tfar):
                (outb2, state4) = self.source_fwf4(outb1, state=state4)

            outb = tf.concat([outb1, outb2], axis=1)

            out = outa + outb

            Rtemp = tf.where(R <= sensor_noise[:, 0, tfna], self.om[:, :, 0] * 1000, R)
            rd_mult = tf.concat([self.om[:, :, 0], self.om[:, :, 0] * Rtemp, self.om[:, :, 0] * Rtemp], axis=1)
            rd = (rd_mult * sensor_noise)

            Rt = tf.zeros([self.batch_size, self.num_meas, self.num_meas], dtype=self.vdtype)
            Rt = tf.matrix_set_diag(Rt, tf.pow(rd, 2))

            split = 3
            nv = out.shape[1].value // split
            qm0 = FCL(out, self.num_mixtures * split, activation_fn=ELU, weights_initializer=varsci, normalizer_fn=tfc.layers.layer_norm, scope='q_cov/1', reuse=tfar)
            qm1 = FCL(qm0, qm0.shape[1].value,        activation_fn=ELU, weights_initializer=varsci, scope='q_cov/2', reuse=tfar)
            qm2 = FCL(qm1, qm1.shape[1].value,        activation_fn=ELU, weights_initializer=varsci, scope='q_cov/3', reuse=tfar)

            qmx = FCL(qm2[:, :nv],     self.num_mixtures, activation_fn=tf.nn.softmax, weights_initializer=varsci, scope='q_cov/x', reuse=tfar)
            qmy = FCL(qm2[:, nv:2*nv], self.num_mixtures, activation_fn=tf.nn.softmax, weights_initializer=varsci, scope='q_cov/y', reuse=tfar)
            qmz = FCL(qm2[:, 2*nv:],   self.num_mixtures, activation_fn=tf.nn.softmax, weights_initializer=varsci, scope='q_cov/z', reuse=tfar)
            # atx = FCL(qm0[:, 3*nv:], self.num_mixtures, activation_fn=tf.nn.softmax, weights_initializer=varsci, scope='q_cov/t', reuse=tfar)

            sjx_l = list()
            sjy_l = list()
            sjz_l = list()
            atc_l = list()
            atcl = [0.1, 0.25, 0.75, 1.0]
            sjxl = [0.001, 0.1, 1, 50]
            sjyl = [0.001, 0.1, 1, 50]
            sjzl = [0.001, 0.1, 1, 50]

            for ppp in range(self.num_mixtures):
                # atc_l.append(tf.ones([1, 1], dtype=self.vdtype) * atcl[ppp])
                sjx_l.append(tf.ones([1, 1], dtype=self.vdtype) * sjxl[ppp])
                sjy_l.append(tf.ones([1, 1], dtype=self.vdtype) * sjyl[ppp])
                sjz_l.append(tf.ones([1, 1], dtype=self.vdtype) * sjzl[ppp])

            sjx_vals = tf.squeeze(tf.stack(sjx_l, axis=1), 0)
            # atc_vals = tf.squeeze(tf.stack(atc_l, axis=1), 0)
            sjy_vals = tf.squeeze(tf.stack(sjy_l, axis=1), 0)
            sjz_vals = tf.squeeze(tf.stack(sjz_l, axis=1), 0)

            sjx = tf.matmul(qmx, sjx_vals)
            sjy = tf.matmul(qmy, sjy_vals)
            sjz = tf.matmul(qmz, sjz_vals)
            # atc = tf.matmul(atx, atc_vals)

            zu = tf.zeros([self.batch_size, 1], self.vdtype)

            u = tf.concat([zu, zu, zu, zu,
                           zu, zu, zu, zu,
                           zu, zu, zu, zu], axis=1)

            Qt, At, _, _ = get_QP(dt, self.om, self.zm, self.I_3z, self.I_4z, self.zb,
                                  dimension=int(self.num_state / 3),
                                  sjix=self.om[:, :, 0] * sjx ** 2,
                                  sjiy=self.om[:, :, 0] * sjy ** 2,
                                  sjiz=self.om[:, :, 0] * sjz ** 2,
                                  aji=self.om[:, :, 0] * 1.0)

            states = (state1, state2, state3, state4)

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
            sensor_noise = inputs[:, -3:]

            weight = cur_weight[:, :, tfna]

            _, _, mu_t0, Sigma_t0, prev_meas_uvw, prev_meas_rae, state1, state2, state3, state4, _, _, _, _, _, _, _ = params

            states = (state1, state2, state3, state4)

            dt = current_time - prev_time
            dt = tf.where(dt <= 1 / 100, tf.ones_like(dt) * 1 / 25, dt)

            meas_uvw, Qt, At, Rt, Bt, u, gravity, altitude, states = self.alpha(dt, mu_t0, meas_rae, prev_meas_uvw, Sigma_t0, LLA, sensor_noise, states)

            lat = LLA[:, 0, tfna, tfna]
            lon = LLA[:, 1, tfna, tfna]

            ### UNSCENTED
            qcholr = tf.cholesky(Sigma_t0)
            Am = tf.expand_dims(self.c, axis=2) * tf.cast(qcholr, tf.float64)
            Y = tf.tile(tf.expand_dims(mu_t0, axis=2), [1, 1, self.num_state])
            X = tf.concat([tf.expand_dims(mu_t0, axis=2), Y + Am, Y - Am], axis=2)
            X = tf.transpose(X, [0, 2, 1])

            x1, X1, P1, X2 = ut_state_batch(X, u, gravity, altitude, self.Wm, self.Wc, Qt, self.num_state, self.batch_size, dt, At, Bt)

            uvw_to_enu = uvw2enu_tf(lat, lon)
            enu_to_uvw = tf.transpose(uvw_to_enu, [0, 2, 1])
            y_enu = tf.squeeze(tf.matmul(uvw_to_enu, tf.matmul(self.meas_mat, x1)), -1)
            rae_to_enu = rae2enu_tf(y_enu, self.pi_val)
            Rt = Rt * (tf.ones_like(Rt) * weight)
            enu_cov = tf.matmul(tf.matmul(rae_to_enu, Rt), rae_to_enu, transpose_b=True)
            Rt = tf.matmul(tf.matmul(enu_to_uvw, enu_cov), enu_to_uvw, transpose_b=True)

            z1, Z1, P2, Z2 = ut_meas(X1, self.Wm, self.Wc, Rt, self.meas_mat, self.batch_size, lat, lon, self.pi_val)

            P12 = tf.matmul(tf.matmul(X2, tf.matrix_diag(self.Wc)), Z2, transpose_b=True)

            S_inv = tf.matrix_inverse(P2)

            gain = tf.matmul(P12, tf.matrix_inverse(P2))
            gain = tf.where(tf.less_equal(weight, tf.zeros_like(gain)), tf.zeros_like(gain), gain)
            pos_res2 = meas_uvw[:, :, tf.newaxis] - tf.matmul(self.meas_mat, x1)
            mu_t = x1 + tf.matmul(gain, pos_res2)

            cov_est_t0 = P1 - tf.matmul(gain, P12, transpose_b=True)
            Sigma_t = (cov_est_t0 + tf.transpose(cov_est_t0, [0, 2, 1])) / 2

            mu_t = mu_t[:, :, 0]

            ###
            # STANDARD
            # mu_pred = tf.squeeze(tf.matmul(At, tf.expand_dims(mu_t0, 2)), -1) + tf.squeeze(tf.matmul(Bt, u[:, :, tfna]), -1)
            # Sigma_pred = tf.matmul(tf.matmul(At, Sigma_t0), At, transpose_b=True) + Qt
            # mu_pred_uvw = tf.matmul(self.meas_mat, mu_pred[:, :, tfna])
            # pos_res_uvw = meas_uvw[:, :, tfna] - mu_pred_uvw
            #
            # uvw_to_enu = uvw2enu_tf(lat, lon)
            # enu_to_uvw = tf.transpose(uvw_to_enu, [0, 2, 1])
            #
            # y_enu = tf.squeeze(tf.matmul(uvw_to_enu, mu_pred_uvw), -1)
            #
            # rae_to_enu = rae2enu_tf(y_enu, self.pi_val)
            #
            # HPH = tf.matmul(tf.matmul(self.meas_mat, Sigma_pred), self.meas_mat, transpose_b=True)
            #
            # enu_cov = tf.matmul(tf.matmul(rae_to_enu, Rt), rae_to_enu, transpose_b=True)
            #
            # Rt = tf.matmul(tf.matmul(enu_to_uvw, enu_cov), enu_to_uvw, transpose_b=True)
            #
            # S = HPH + Rt
            #
            # S_inv = tf.matrix_inverse(S)
            #
            # gain = tf.matmul(tf.matmul(Sigma_pred, self.meas_mat, transpose_b=True), S_inv)
            # gain = tf.where(tf.less_equal(weight, tf.zeros_like(gain)), tf.zeros_like(gain), gain)
            #
            # mu_t = mu_pred[:, :, tfna] + tf.matmul(gain, pos_res_uvw)
            # mu_t = mu_t[:, :, 0]
            #
            # I_KC = self.I_12 - tf.matmul(gain, self.meas_mat)  # (bs, dim_z, dim_z)
            # Sigma_t = tf.matmul(tf.matmul(I_KC, Sigma_pred), I_KC, transpose_b=True) + tf.matmul(tf.matmul(gain, Rt), gain, transpose_b=True)
            # Sigma_t = (Sigma_t + tf.transpose(Sigma_t, [0, 2, 1])) / 2

            # mu_t = tfd.MultivariateNormalTriL(loc=mu_t, scale_tril=tf.cholesky(Sigma_t)).sample(1)
            # mu_t = tf.transpose(mu_t, [1, 0, 2])
            # mu_t = mu_t[:, 0, :]

            # state_measurement_pos = tf.concat([mu_t[:, :, 0, tf.newaxis], mu_t[:, :, 4, tf.newaxis], mu_t[:, :, 8, tf.newaxis]], axis=2)
            # meas_uvwt = tf.tile(meas_uvw[:, tfna, :], [1, 10, 1])
            # # mu_tile = tf.tile(mu_t[:, tfna, :], [1, 100, 1])
            #
            # pos_res_uvwt = state_measurement_pos - meas_uvwt
            # Rt_tile = tf.tile(Rt[:, tfna, :, :], [1, 10, 1, 1])
            #
            # lpmeas = tfd.MultivariateNormalTriL(loc=None, scale_tril=tf.cholesky(Rt_tile)).log_prob(pos_res_uvwt)
            # lpmeas = lpmeas[:, :, tfna]

            # num_units = 1
            # Q = tf.layers.dense(mu_t, num_units, activation=None, reuse=tfar, name='Q')
            # K = tf.layers.dense(lpmeas, num_units, activation=None, reuse=tfar, name='K')
            # V = tf.layers.dense(lpmeas, num_units, activation=None, reuse=tfar, name='V')

            # outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (h*N, T_q, T_k)
            # outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)
            # outputs = tf.nn.softmax(tf.matmul(outputs, V))
            # outputs = tf.nn.softmax(lpmeas)

            # mu_t = tf.reduce_mean(mu_t * outputs, 1)

            mu_pred = tf.squeeze(tf.matmul(At, tf.expand_dims(mu_t, 2)), -1)  # + tf.squeeze(tf.matmul(Bt, u[:, :, tfna]), -1)
            Sigma_pred = tf.matmul(tf.matmul(At, Sigma_t), At, transpose_b=True) + Qt

            return mu_pred, Sigma_pred, mu_t, Sigma_t, meas_uvw, meas_rae, state1, state2, state3, state4, Qt, Rt, At, Bt, S_inv, weight, u

    def compute_forwards(self):
        with tf.variable_scope('Forward'):
            self.mu = self.state_input
            self.Sigma = tf.reshape(self.P_inp, [self.batch_size, self.num_state, self.num_state])

            all_time = tf.concat([self.prev_time[:, tfna], tf.stack(self.current_timei, axis=1)], axis=1)
            # meas_rae = tf.concat([self.prev_measurement[:, tfna], tf.stack(self.measurement, axis=1)], axis=1)

            meas_rae = tf.stack(self.measurement_rae, axis=1)

            current_time = all_time[:, 1:, :]
            prev_time = all_time[:, :-1, :]

            int_time = self.int_time

            sensor_lla = tf.expand_dims(self.sensor_lla, axis=1)
            sensor_lla = tf.tile(sensor_lla, [1, meas_rae.shape[1], 1])

            sensor_noise = tf.stack(self.meas_variance, axis=1)

            inputs = tf.concat([current_time, prev_time, int_time[:, :, tfna], meas_rae,
                                self.seqweightin[:, :, tfna], sensor_lla, sensor_noise], axis=2)

            init_Q = self.Q_inp
            init_R = self.R_inp

            state1 = self.cell_state1
            state2 = self.cell_state2
            state3 = self.cell_state3
            state4 = self.cell_state4

            init_Si = tf.ones([self.batch_size, 3, 3], self.vdtype)
            init_A = tf.ones([self.batch_size, 12, 12], self.vdtype)
            init_B = tf.ones([self.batch_size, 12, 12], self.vdtype)
            meas_uvw = tf.zeros([self.batch_size, 3], self.vdtype)
            meas_rae = self.prev_measurement_rae
            init_weight = tf.zeros([self.batch_size, 1, 1], self.vdtype)
            init_u = tf.zeros([self.batch_size, 12], self.vdtype)

            forward_states = tf.scan(self.forward_step_fn, tf.transpose(inputs, [1, 0, 2]),
                                     initializer=(self.mu, self.Sigma, self.mu, self.Sigma,
                                                  meas_uvw, meas_rae,
                                                  state1, state2, state3, state4,
                                                  init_Q, init_R, init_A, init_B, init_Si,
                                                  init_weight, init_u),
                                     parallel_iterations=1, name='forward')
            return forward_states

    def filter(self):
        mu_pred, Sigma_pred, mu_filt, Sigma_filt, meas_uvw, _, state1, state2, state3, state4, Q, R, A, B, S_inv, weights, u = forward_states = \
            self.compute_forwards()

        state1_out = tf.transpose(state1, [1, 0, 2])
        state2_out = tf.transpose(state2, [1, 0, 2])
        state3_out = tf.transpose(state3, [1, 0, 2])
        state4_out = tf.transpose(state4, [1, 0, 2])

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
               state1_out, state2_out, state3_out, state4_out

    def build_model(self):

        with tf.variable_scope('Input_Placeholders'):

            self.drop_rate = tf.Variable(0.5, trainable=False, dtype=tf.float64, name='dropout_rate')
            self.learning_rate_inp = tf.Variable(0.0, trainable=False, dtype=tf.float64, name='learning_rate_input')
            self.update_condition = tf.placeholder(tf.bool, name='update_condition')

            self.grad_clip = tf.placeholder(self.vdtype, name='grad_clip')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.measurement_rae = [tf.placeholder(self.vdtype, shape=(None, self.num_meas), name="meas_rae_{}".format(t)) for t in range(self.max_seq)]
            self.meas_variance = [tf.placeholder(self.vdtype, shape=(None, self.num_meas), name="meas_var_{}".format(t)) for t in range(self.max_seq)]

            self.sensor_ecef = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name='sensor_ecef')
            self.sensor_lla = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name='sensor_lla')

            self.prev_measurement_rae = tf.placeholder(self.vdtype, shape=(None, self.num_meas), name="px")
            self.prev_covariance_estimate = tf.placeholder(self.vdtype, shape=(None, self.num_state, self.num_state), name="pcov")
            self.prev_time = tf.placeholder(self.vdtype, shape=(None, 1), name="ptime")
            self.prev_state_estimate = tf.placeholder(self.vdtype, shape=(None, self.num_state), name="prev_state_estimate")

            self.current_timei = [tf.placeholder(self.vdtype, shape=(None, 1), name="current_time_{}".format(t)) for t in range(self.max_seq)]
            self.P_inp = tf.placeholder(self.vdtype, shape=(None, self.num_state, self.num_state), name="p_inp")
            self.Q_inp = tf.placeholder(self.vdtype, shape=(None, self.num_state, self.num_state), name="q_inp")
            self.R_inp = tf.placeholder(self.vdtype, shape=(None, self.num_meas, self.num_meas), name="r_inp")
            self.state_input = tf.placeholder(self.vdtype, shape=(None, self.num_state), name="state_input")
            self.seqweightin = tf.placeholder(self.vdtype, [None, self.max_seq], name='seqweight')

            self.cell_state1 = tf.placeholder(name='cell_state1', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.cell_state2 = tf.placeholder(name='cell_state2', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.cell_state3 = tf.placeholder(name='cell_state3', shape=[None, self.F_hidden], dtype=self.vdtype)
            self.cell_state4 = tf.placeholder(name='cell_state4', shape=[None, self.F_hidden], dtype=self.vdtype)

            # self.input_cell_states = [tf.placeholder(name="GRU_state_{}".format(t), shape=(None, self.F_hidden), dtype=self.vdtype) for t in self.num_cells

            cell_type = IndyGRUCell

            with tf.variable_scope('Cell_1/q_cov'):
                self.source_fwf = cell_type(num_units=self.F_hidden)

            with tf.variable_scope('Cell_2/r_cov'):
                self.source_fwf2 = cell_type(num_units=self.F_hidden)

            with tf.variable_scope('Cell_3/q_cov'):
                self.source_fwf3 = cell_type(num_units=self.F_hidden)

            with tf.variable_scope('Cell_4/r_cov'):
                self.source_fwf4 = cell_type(num_units=self.F_hidden)

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

        filter_out, A, Q, R, B, S_inv, u, meas_uvw, prediction, state1_out, state2_out, state3_out, state4_out = self.filter()

        self.ao_list = A
        self.qo_list = Q
        self.ro_list = R
        self.uo_list = u
        self.bo_list = B
        self.si_list = S_inv
        self.new_meas = meas_uvw

        refined_state = filter_out[0]
        refined_covariance = filter_out[1]

        self.z_smooth = filter_out[0]
        self.final_state_filter = filter_out[0]
        self.final_state_prediction = refined_state
        self.final_state_smooth = filter_out[0]

        self.final_cov_filter = filter_out[1]
        self.final_cov_prediction = refined_covariance
        self.final_cov_smooth = filter_out[1]

        self.state_fwf1 = state1_out
        self.state_fwf2 = state2_out
        self.state_fwf3 = state3_out
        self.state_fwf4 = state4_out

    def test(self, x_data, y_data, ecef_ref, lla_data, filename, traj_id=0, sensor_name=''):

        tf.global_variables_initializer().run()

        save_files = os.listdir(self.checkpoint_dir + '/')
        recent = str.split(save_files[0], '_')
        start_epoch = recent[2]
        step = str.split(recent[3], '.')[0]
        print("Resuming run from epoch " + str(start_epoch) + ' and step ' + str(step))
        step = int(step)
        print('Loading filter...')
        self.saver = tf.train.import_meta_graph(self.checkpoint_dir + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step) + '.meta')
        self.saver.restore(self.sess, self.checkpoint_dir + self.filter_name + '_' + str(start_epoch) + '_' + str(step) + '.ckpt-' + str(step))
        print("filter restored.")

        lla_datar = copy.copy(lla_data)
        lla_datar[:, 0] = lla_datar[:, 0] * np.pi / 180
        lla_datar[:, 1] = lla_datar[:, 1] * np.pi / 180

        x_data = np.concatenate([x_data[:, :, 0, np.newaxis], x_data[:, :, 4:10]], axis=2)  # rae measurements

        # Get the initialized batch of data
        x, y, prev_x, prev_time, initial_meas, initial_time = prepare_batch_testing(x_data, y_data)

        default_unc = np.ones_like(x[:, :, -3:]) * np.max(x[:, :, -3:], axis=1, keepdims=True)
        x[:, :, -3:] = np.where(x[:, :, -3:] <= 1e-12, default_unc, x[:, :, -3:])

        ecef_ref = np.ones([self.batch_size_np, x.shape[1], 3]) * ecef_ref[:, np.newaxis, :]

        # Initialize placeholder lists for plotting
        out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot = initialize_run_variables()

        fd = {}

        x_data = x[:, :, 1:]
        time_data = x[:, :, 0, np.newaxis]
        y_data = y

        seqlen = np.ones(shape=[self.batch_size_np, ])
        int_time = np.zeros(shape=[self.batch_size_np, x.shape[1]])
        seqweight = np.ones(shape=[self.batch_size_np, x.shape[1]])

        time_plotter = time_data

        initial_meas = initial_meas[:, :, :3]

        current_state_estimate, current_cov_estimate, prev_state_estimate, prev_covariance_estimate, initial_Q, initial_R, converted_meas_init = \
            initialize_filter(self.batch_size_np, initial_time, initial_meas, prev_time, prev_x, time_data[:, 0, np.newaxis, :], lla_datar)

        fd.update({self.is_training: False})

        std = 0.0
        fd.update({self.cell_state1: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
        fd.update({self.cell_state2: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
        fd.update({self.cell_state3: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})
        fd.update({self.cell_state4: get_zero_state(1, self.F_hidden, self.batch_size_np, 4, std)})

        print('Beginning trajectory filtering. ')
        for tstep in tqdm(range(0, time_data.shape[1])):

            prev_state_estimate = prev_state_estimate[:, :, self.idxi]
            current_state_estimate = current_state_estimate[:, :, self.idxi]
            current_x = x_data[:, tstep, np.newaxis, :]
            if len(y_data) > 0:
                current_y = y_data[:, tstep, np.newaxis, :]

            if np.all(current_x == 0):
                continue

            current_time = time_data[:, tstep, np.newaxis, :]
            current_int = int_time[:, tstep, np.newaxis]
            current_weight = seqweight[:, tstep, np.newaxis]

            fd.update({self.measurement_rae[0]: current_x[:, 0, :3].reshape(-1, self.num_meas)})
            fd.update({self.meas_variance[0]: current_x[:, 0, 3:].reshape(-1, self.num_meas)})
            fd.update({self.prev_measurement_rae: prev_x.reshape(-1, self.num_meas)})
            fd.update({self.prev_covariance_estimate: prev_covariance_estimate[:, -1, :, :]})
            fd.update({self.prev_state_estimate: prev_state_estimate.reshape(-1, self.num_state)})
            fd.update({self.sensor_ecef: ecef_ref[:, 0, :]})
            fd.update({self.sensor_lla: lla_datar})
            fd.update({self.seqlen: seqlen})
            fd.update({self.int_time: current_int})
            fd.update({self.is_training: True})
            fd.update({self.seqweightin: current_weight})
            fd.update({self.P_inp: current_cov_estimate})
            fd.update({self.Q_inp: initial_Q})
            fd.update({self.R_inp: initial_R})
            fd.update({self.state_input: current_state_estimate.reshape(-1, self.num_state)})
            fd.update({self.prev_time: prev_time[:, :, 0]})
            fd.update({self.current_timei[0]: current_time.reshape(-1, 1)})
            fd.update({self.drop_rate: 1.0})

            filter_output, smooth_output, refined_output, q_out_t, q_outs, q_out_refine, \
            qt_out, rt_out, at_out, state_fwf1, state_fwf2, state_fwf3, state_fwf4, new_meas = \
                self.sess.run([self.final_state_filter,
                               self.final_state_smooth,
                               self.final_state_prediction,
                               self.final_cov_filter,
                               self.final_cov_smooth,
                               self.final_cov_prediction,
                               self.qo_list,
                               self.ro_list,
                               self.ao_list,
                               self.state_fwf1,
                               self.state_fwf2,
                               self.state_fwf3,
                               self.state_fwf4,
                               self.new_meas],
                              fd)

            # print("Test: {0:2d} MB: {1:1d} Time: {2:3d} Total: {3:3d}".format(0, run_id, tstep, time_data.shape[1]))

            filter_output = filter_output[:, :, self.idxo]
            smooth_output = smooth_output[:, :, self.idxo]
            refined_output = refined_output[:, :, self.idxo]

            fd.update({self.cell_state1: state_fwf1[:, -1, :]})
            fd.update({self.cell_state2: state_fwf2[:, -1, :]})
            fd.update({self.cell_state3: state_fwf3[:, -1, :]})
            fd.update({self.cell_state4: state_fwf4[:, -1, :]})

            idx = -1
            prev_x = np.concatenate([prev_x, current_x], axis=1)
            prev_x = prev_x[:, idx, np.newaxis, :]

            prev_state_estimate = current_state_estimate
            current_state_estimate = filter_output[:, idx, np.newaxis, :]

            initial_Q = qt_out[:, idx, :, :]
            initial_R = rt_out[:, idx, :, :]

            prev_time = np.concatenate([prev_time, current_time], axis=1)
            prev_time = prev_time[:, idx, np.newaxis, :]

            prev_covariance_estimate = current_cov_estimate[:, np.newaxis, :, :]

            current_cov_estimate = q_out_t[:, idx, :, :]

            if len(y_data) == 0:
                current_y = []

            out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot = \
                append_output_vaulues(tstep, current_time, current_y, new_meas, smooth_output, filter_output, refined_output, q_out_t, q_outs, q_out_refine, qt_out, rt_out, at_out,
                                      out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot, testing=True)

        print('Completed Trajectory Filtering ')
        print(' ')
        print('Outputting data to specified output directory ')

        for pidx in range(self.batch_size_np):

            lla = lla_datar[pidx, :]

            plotpath = self.plot_test_dir + '/BATCH_IDX_' + str(pidx) + '_Trajectory_' + str(traj_id) + '_LLA_' + '[' + str(lla[0]) + ',' + str(lla[1]) + ',' + str(lla[2]) + ']_' + 'SEN_' + sensor_name

            check_folder(plotpath)

            out_plot_filter[pidx, :, :3] = out_plot_filter[pidx, :, :3] + ecef_ref
            out_plot_smooth[pidx, :, :3] = out_plot_smooth[pidx, :, :3] + ecef_ref
            out_plot_refined[pidx, :, :3] = out_plot_refined[pidx, :, :3] + ecef_ref
            meas_plot[pidx, :, :3] = meas_plot[pidx, :, :3] + ecef_ref

            output_plots_testing(out_plot_filter[pidx, np.newaxis, :, :],
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
                                 self.max_seq,
                                 self.outpath,
                                 traj_id=traj_id,
                                 sensor_lla=lla,
                                 sensor_name=sensor_name)
