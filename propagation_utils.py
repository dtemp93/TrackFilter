import copy
import tensorflow as tf
import numpy as np
import pdb
# import scipy


def get_legendre(GMR3, position, R, dtype):
    q = tf.expand_dims(position[:, 2], axis=1)

    SMA = R / 1.0
    SMA2 = SMA * SMA
    SMA3 = SMA2 * SMA
    SMA4 = SMA3 * SMA
    SMA5 = SMA4 * SMA
    SMA6 = SMA5 * SMA

    c2 = tf.constant(-0.484165371736e-3, dtype) * -tf.sqrt(tf.constant(5, dtype=dtype)) * (tf.constant(3, dtype=dtype) / tf.constant(2, dtype)) / SMA2
    c3 = tf.constant(0.957254173792e-6, dtype) * -tf.sqrt(tf.constant(7, dtype)) * (tf.constant(5. / 2., dtype)) / SMA3
    c4 = tf.constant(0.539873863789e-6, dtype) * -tf.sqrt(tf.constant(9., dtype)) * (tf.constant(-5., dtype) / tf.constant(8., dtype)) / SMA4
    c5 = tf.constant(0.685323475630e-7, dtype) * -tf.sqrt(tf.constant(11., dtype)) * (tf.constant(-3., dtype) / tf.constant(8., dtype)) / SMA5
    c6 = tf.constant(-0.149957994714e-6, dtype) * -tf.sqrt(tf.constant(13., dtype)) * (tf.constant(1., dtype) / tf.constant(16., dtype)) / SMA6

    q2 = q * q
    q4 = q2 * q2
    q6 = q2 * q4

    X = tf.constant(1., dtype) + c2 * (tf.constant(1., dtype) - tf.constant(5., dtype) * q2) + \
        c3 * (tf.constant(3., dtype) - tf.constant(7., dtype) * q2) * q + \
        c4 * (tf.constant(3., dtype) - tf.constant(42., dtype) * q2 + tf.constant(63., dtype) * q4) + \
        c5 * (tf.constant(35., dtype) - tf.constant(210., dtype) * q2 + tf.constant(231., dtype) * q4) * q + \
        c6 * (tf.constant(35., dtype) - tf.constant(945., dtype) * q2 + tf.constant(3465., dtype) * q4 - tf.constant(3003., dtype) * q6)

    Z = tf.constant(1., dtype) + c2 * (tf.constant(3., dtype) - tf.constant(5., dtype) * q2) + \
        c3 * (tf.constant(6., dtype) - tf.constant(7., dtype) * q2) * q + \
        c4 * (tf.constant(15., dtype) - tf.constant(70., dtype) * q2 + tf.constant(63., dtype) * q4) + \
        c5 * (tf.constant(105., dtype) - tf.constant(315., dtype) * q2 + tf.constant(231., dtype) * q4) * q + \
        c6 * (tf.constant(245., dtype) - tf.constant(2205., dtype) * q2 + tf.constant(4851., dtype) * q4 - tf.constant(3003., dtype) * q6)

    Zc = (tf.constant(3., dtype) / tf.constant(2., dtype)) * (tf.constant(2., dtype) / tf.constant(5., dtype)) * c3 - \
         (tf.constant(15., dtype) / tf.constant(8., dtype)) * (tf.constant(-8., dtype) / tf.constant(3., dtype)) * c5

    acc1 = -GMR3 * tf.expand_dims(position[:, 0], axis=1) * X
    acc2 = -GMR3 * tf.expand_dims(position[:, 1], axis=1) * X
    acc3 = -GMR3 * (tf.expand_dims(position[:, 2], axis=1) * Z - R * Zc)

    acc = tf.concat([acc1, acc2, acc3], axis=1)

    return acc


def get_legendreb(GMR3, position, R, dtype):
    q = tf.expand_dims(position[:, :, 2], axis=2)

    SMA = R / 1.0
    SMA2 = SMA * SMA
    SMA3 = SMA2 * SMA
    SMA4 = SMA3 * SMA
    SMA5 = SMA4 * SMA
    SMA6 = SMA5 * SMA

    c2 = tf.constant(-0.484165371736e-3, dtype) * -tf.sqrt(tf.constant(5, dtype=dtype)) * (tf.constant(3, dtype=dtype) / tf.constant(2, dtype)) / SMA2
    c3 = tf.constant(0.957254173792e-6, dtype) * -tf.sqrt(tf.constant(7, dtype)) * (tf.constant(5. / 2., dtype)) / SMA3
    c4 = tf.constant(0.539873863789e-6, dtype) * -tf.sqrt(tf.constant(9., dtype)) * (tf.constant(-5., dtype) / tf.constant(8., dtype)) / SMA4
    c5 = tf.constant(0.685323475630e-7, dtype) * -tf.sqrt(tf.constant(11., dtype)) * (tf.constant(-3., dtype) / tf.constant(8., dtype)) / SMA5
    c6 = tf.constant(-0.149957994714e-6, dtype) * -tf.sqrt(tf.constant(13., dtype)) * (tf.constant(1., dtype) / tf.constant(16., dtype)) / SMA6

    q2 = q * q
    q4 = q2 * q2
    q6 = q2 * q4

    X = tf.constant(1., dtype) + c2 * (tf.constant(1., dtype) - tf.constant(5., dtype) * q2) + \
        c3 * (tf.constant(3., dtype) - tf.constant(7., dtype) * q2) * q + \
        c4 * (tf.constant(3., dtype) - tf.constant(42., dtype) * q2 + tf.constant(63., dtype) * q4) + \
        c5 * (tf.constant(35., dtype) - tf.constant(210., dtype) * q2 + tf.constant(231., dtype) * q4) * q + \
        c6 * (tf.constant(35., dtype) - tf.constant(945., dtype) * q2 + tf.constant(3465., dtype) * q4 - tf.constant(3003., dtype) * q6)

    Z = tf.constant(1., dtype) + c2 * (tf.constant(3., dtype) - tf.constant(5., dtype) * q2) + \
        c3 * (tf.constant(6., dtype) - tf.constant(7., dtype) * q2) * q + \
        c4 * (tf.constant(15., dtype) - tf.constant(70., dtype) * q2 + tf.constant(63., dtype) * q4) + \
        c5 * (tf.constant(105., dtype) - tf.constant(315., dtype) * q2 + tf.constant(231., dtype) * q4) * q + \
        c6 * (tf.constant(245., dtype) - tf.constant(2205., dtype) * q2 + tf.constant(4851., dtype) * q4 - tf.constant(3003., dtype) * q6)

    Zc = (tf.constant(3., dtype) / tf.constant(2., dtype)) * (tf.constant(2., dtype) / tf.constant(5., dtype)) * c3 - \
         (tf.constant(15., dtype) / tf.constant(8., dtype)) * (tf.constant(-8., dtype) / tf.constant(3., dtype)) * c5

    acc1 = -GMR3 * tf.expand_dims(position[:, :, 0], axis=2) * X
    acc2 = -GMR3 * tf.expand_dims(position[:, :, 1], axis=2) * X
    acc3 = -GMR3 * (tf.expand_dims(position[:, :, 2], axis=2) * Z - R * Zc)

    acc = tf.concat([acc1, acc2, acc3], axis=2)

    return acc


def get_legendre_np(GMR3, position, R):
    q = np.expand_dims(position[:, 2], axis=1)
    SMA = R / 1.0

    c2 = -0.484165371736e-3 * -np.sqrt(5.) * (3. / 2.) / (SMA ** 2)
    c3 = 0.957254173792e-6 * -np.sqrt(7.) * (5. / 2.) / (SMA ** 3)
    c4 = 0.539873863789e-6 * -np.sqrt(9.) * (-5. / 8.) / (SMA ** 4)
    c5 = 0.685323475630e-7 * -np.sqrt(11.) * (-3. / 8.) / (SMA ** 5)
    c6 = -0.149957994714e-6 * -np.sqrt(13.) * (1. / 16.) / (SMA ** 6)

    X = 1. + c2 * (1. - 5. * q ** 2) + \
        c3 * (3. - 7. * q ** 2) * q + \
        c4 * (3. - 42. * q ** 2 + 63. * q ** 4) + \
        c5 * (35. - 210. * q ** 2 + 231. * q ** 4) * q + \
        c6 * (35. - 945. * q ** 2 + 3465. * q ** 4 - 3003. * q ** 6)

    Z = 1. + c2 * (3. - 5. * q ** 2) + \
        c3 * (6. - 7. * q ** 2) * q + \
        c4 * (15. - 70. * q ** 2 + 63. * q ** 4) + \
        c5 * (105. - 315. * q ** 2 + 231. * q ** 4) * q + \
        c6 * (245. - 2205. * q ** 2 + 4851. * q ** 4 - 3003. * q ** 6)

    Zc = (3. / 2.) * (2. / 5.) * c3 - (15. / 8.) * (-8. / 3.) * c5

    acc1 = -GMR3 * np.expand_dims(position[:, 0], axis=1) * X
    acc2 = -GMR3 * np.expand_dims(position[:, 1], axis=1) * X
    acc3 = -GMR3 * (np.expand_dims(position[:, 2], axis=1) * Z - R * Zc)

    acc = np.concatenate([acc1, acc2, acc3], axis=1)

    return acc


def propagatefb2(X, u, G, B, dt, alt):

    # idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    # xt = X[:, :, :3]
    # sensor_ecef = tf.tile(sensor_ecef[:, tf.newaxis, :], [1, X.shape[1], 1])
    G = tf.tile(G[:, tf.newaxis, :], [1, X.shape[1], 1])

    # u = tf.tile(u[:, tf.newaxis, :], [1, X.shape[1], 1])
    X = tf.expand_dims(X, 3)
    xt = tf.concat([X[:, :, 0], X[:, :, 4], X[:, :, 8]], 2)
    vt = tf.concat([X[:, :, 1], X[:, :, 5], X[:, :, 9]], 2)
    at = tf.concat([X[:, :, 2], X[:, :, 6], X[:, :, 10]], 2)
    jt = tf.concat([X[:, :, 3], X[:, :, 7], X[:, :, 11]], 2)
    # jt = tf.zeros_like(at)

    dt = tf.ones_like(xt) * dt[:, :, tf.newaxis]

    inp_jerk = tf.matrix_transpose(tf.matmul(B, u[:, :, tf.newaxis]))
    inp_jerk = tf.tile(inp_jerk, [1, X.shape[1], 1])

    # rho0 = 1.22  # kg / m**3
    # k0 = 0.14141e-3
    # area = 0.25  # / FLAGS.RE  # meter squared
    # cd = 0.03  # unitless

    # R1 = tf.norm(xt + sensor_ecef, axis=2, keepdims=True)
    # R1 = tf.where(tf.less(R1, tf.ones_like(R1)*6378137), tf.ones_like(R1)*6378137, R1)
    # rad_temp = tf.pow(R1, 3)
    # alt = rad_temp - 1.
    # alt = tf.where(tf.less(alt, tf.zeros_like(alt)), tf.zeros_like(alt), alt)

    # rho = rho0 * tf.exp(-k0 * alt)
    # rho = rho[:, :, tf.newaxis]
    # Ka = tf.negative(0.5 * rho * vt * tf.norm(vt, axis=2, keepdims=True) * (cd * area))

    at1 = at  # + Ka

    xt0 = xt + vt * dt + 0.5 * tf.pow(at1, 2) + (1 / 6) * tf.pow(jt, 3)
    vt0 = vt + at1 * dt + 0.5 * tf.pow(jt, 2)
    at0 = at1 + jt * dt
    jt0 = jt

    xt0 = tf.expand_dims(xt0, 3)
    vt0 = tf.expand_dims(vt0, 3)
    at0 = tf.expand_dims(at0, 3)
    jt0 = tf.expand_dims(jt0, 3)

    state_est = tf.concat([xt0[:, :, 0], vt0[:, :, 0], at0[:, :, 0], jt0[:, :, 0],
                           xt0[:, :, 1], vt0[:, :, 1], at0[:, :, 1], jt0[:, :, 1],
                           xt0[:, :, 2], vt0[:, :, 2], at0[:, :, 2], jt0[:, :, 2]], axis=2)

    state_est_out = state_est + inp_jerk

    return state_est_out


def propagatef2(X, u, dt, alt):

    # idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    # xt = X[:, :, :3]
    # sensor_ecef = tf.tile(sensor_ecef[:, tf.newaxis, :], [1, X.shape[1], 1])
    X = tf.expand_dims(X, 2)
    xt = tf.concat([X[:, 0], X[:, 4], X[:, 8]], 1)
    vt = tf.concat([X[:, 1], X[:, 5], X[:, 9]], 1)
    at = tf.concat([X[:, 2], X[:, 6], X[:, 10]], 1)
    jt = tf.concat([X[:, 3], X[:, 7], X[:, 11]], 1)
    jt = jt + u

    dt = tf.ones_like(xt) * dt

    rho0 = 1.22  # kg / m**3
    k0 = 0.14141e-3
    area = 0.25  # / FLAGS.RE  # meter squared
    cd = 0.03  # unitless

    # R1 = tf.norm(xt + sensor_ecef, axis=2, keepdims=True)
    # R1 = tf.where(tf.less(R1, tf.ones_like(R1)*6378137), tf.ones_like(R1)*6378137, R1)
    # rad_temp = tf.pow(R1, 3)
    # alt = rad_temp - 1.
    # alt = tf.where(tf.less(alt, tf.zeros_like(alt)), tf.zeros_like(alt), alt)
    rho = rho0 * tf.exp(-k0 * alt * 6378137)
    Ka = tf.negative(0.5 * rho * vt * tf.norm(vt, axis=1, keepdims=True) * (cd * area))

    at1 = at

    xt0 = xt + vt * dt + 0.5 * tf.pow(at1, 2) + (1 / 6) * tf.pow(jt, 3)
    vt0 = vt + at1 * dt + 0.5 * tf.pow(jt, 2)
    at0 = at1 + jt * dt
    jt0 = jt

    xt0 = tf.expand_dims(xt0, 2)
    vt0 = tf.expand_dims(vt0, 2)
    at0 = tf.expand_dims(at0, 2)
    jt0 = tf.expand_dims(jt0, 2)

    state_est = tf.concat([xt0[:, 0], vt0[:, 0], at0[:, 0], jt0[:, 0],
                           xt0[:, 1], vt0[:, 1], at0[:, 1], jt0[:, 1],
                           xt0[:, 2], vt0[:, 2], at0[:, 2], jt0[:, 2]], axis=1)
    return state_est


def ut_state_batch(X, u, gravity, altitude, Wm, Wc, R, num_state, batch_size, dt, prop, B):
    # acc_est = tf.tile(tf.expand_dims(acc_est0, axis=1), [1, X.shape[1], 1])

    # Y = tf.transpose(propagatefb2(X, u, gravity, B, dt, altitude), [0, 2, 1])

    input = tf.matrix_transpose(tf.matmul(B, u[:, :, tf.newaxis]))
    input = tf.tile(input[:, :], [1, 25, 1])
    Y = tf.matmul(prop, X, transpose_b=True) + tf.matrix_transpose(input)

    y = tf.zeros([batch_size, num_state, 1], dtype=tf.float64)

    for q in range(Y.shape[2]):
        y += tf.expand_dims(tf.expand_dims(Wm[:, q], 1) * Y[:, :, q], axis=2)
    Y1 = Y - tf.tile(y, [1, 1, Y.shape[2]])
    P = tf.matmul(tf.matmul(Y1, tf.matrix_diag(Wc)), tf.transpose(Y1, [0, 2, 1])) + R

    return y, Y, P, Y1


def ut_meas(X, Wm, Wc, R, meas_mat, batch_size, lat, lon, pi_val):

    Y = tf.matmul(X, meas_mat, transpose_a=True, transpose_b=True)

    # uvw_to_enu = uvw2enu_tf(lat, lon)
    # enu_to_uvw = tf.transpose(uvw_to_enu, [0, 2, 1])

    # y_enu = tf.matrix_transpose(tf.matmul(uvw_to_enu, Y, transpose_b=True))

    # rae_to_enu = rae2enu_tfb(y_enu, pi_val)

    # R = tf.tile(R[:, tf.newaxis, :, :], [1, Y.shape[1].value, 1, 1])
    # enu_to_uvw = tf.tile(enu_to_uvw[:, tf.newaxis, :, :], [1, Y.shape[1].value, 1, 1])

    # enu_cov = tf.matmul(tf.matmul(rae_to_enu, R), rae_to_enu, transpose_b=True)

    # R = tf.matmul(tf.matmul(enu_to_uvw, enu_cov), enu_to_uvw, transpose_b=True)
    # Rd = tf.matrix_diag_part(R)
    # R = tf.matrix_set_diag(R, tf.ones_like(Rd)*1000)

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


def initialize_covariance(Rt, dt):

    sigx = np.maximum(Rt[:, 0, 0], np.ones_like(Rt[:, 0, 0])) ** 1
    sigy = np.maximum(Rt[:, 1, 1], np.ones_like(Rt[:, 0, 0])) ** 1
    sigz = np.maximum(Rt[:, 2, 2], np.ones_like(Rt[:, 0, 0])) ** 1

    dt = dt[:, 0, 0]

    batch_size = Rt.shape[0]

    small_covx = np.zeros(shape=[batch_size, 4, 4])
    small_covy = np.zeros(shape=[batch_size, 4, 4])
    small_covz = np.zeros(shape=[batch_size, 4, 4])
    zero_cov = np.zeros_like(small_covx)

    sig_jer = 10

    small_covx[:, 0, 0] = sigx**2
    # small_covx[:, 0, 1] = (sigx**2)/dt
    # small_covx[:, 0, 2] = (sigx**2)/dt**2
    # small_covx[:, 0, 3] = 0

    # small_covx[:, 1, 0] = small_covx[:, 0, 1]
    small_covx[:, 1, 1] = (2*sigx**2)/(dt**2)
    # small_covx[:, 1, 2] = (3*sigx**2)/(dt**3)
    # small_covx[:, 1, 3] = (5*sig_jer**2*dt**2)/6

    # small_covx[:, 2, 0] = small_covx[:, 0, 2]
    # small_covx[:, 2, 1] = small_covx[:, 1, 2]
    small_covx[:, 2, 2] = (6*sigx**2)/(dt**3)
    # small_covx[:, 2, 3] = sig_jer**2 * dt

    # small_covx[:, 3, 0] = 0
    # small_covx[:, 3, 1] = small_covx[:, 1, 3]
    # small_covx[:, 3, 2] = small_covx[:, 2, 3]
    small_covx[:, 3, 3] = sig_jer**2

    small_covy[:, 0, 0] = sigy ** 2
    # small_covy[:, 0, 1] = (sigy ** 2) / dt
    # small_covy[:, 0, 2] = (sigy ** 2) / dt ** 2
    # small_covy[:, 0, 3] = 0

    # small_covy[:, 1, 0] = small_covy[:, 0, 1]
    small_covy[:, 1, 1] = (2 * sigy ** 2) / (dt ** 2)
    # small_covy[:, 1, 2] = (3 * sigy ** 2) / (dt ** 3)
    # small_covy[:, 1, 3] = (5 * sig_jer ** 2 * dt ** 2) / 6

    # small_covy[:, 2, 0] = small_covy[:, 0, 2]
    # small_covy[:, 2, 1] = small_covy[:, 1, 2]
    small_covy[:, 2, 2] = (6 * sigy ** 2) / (dt ** 3)
    # small_covy[:, 2, 3] = sig_jer ** 2 * dt

    # small_covy[:, 3, 0] = 0
    # small_covy[:, 3, 1] = small_covy[:, 1, 3]
    # small_covy[:, 3, 2] = small_covy[:, 2, 3]
    small_covy[:, 3, 3] = sig_jer ** 2

    small_covz[:, 0, 0] = sigz ** 2
    # small_covz[:, 0, 1] = (sigz ** 2) / dt
    # small_covz[:, 0, 2] = (sigz ** 2) / dt ** 2
    # small_covz[:, 0, 3] = 0

    # small_covz[:, 1, 0] = small_covz[:, 0, 1]
    small_covz[:, 1, 1] = (2 * sigz ** 2) / (dt ** 2)
    # small_covz[:, 1, 2] = (3 * sigz ** 2) / (dt ** 3)
    # small_covz[:, 1, 3] = (5 * sig_jer ** 2 * dt ** 2) / 6

    # small_covz[:, 2, 0] = small_covz[:, 0, 2]
    # small_covz[:, 2, 1] = small_covz[:, 1, 2]
    small_covz[:, 2, 2] = (6 * sigz ** 2) / (dt ** 3)
    # small_covz[:, 2, 3] = sig_jer ** 2 * dt

    # small_covz[:, 3, 0] = 0
    # small_covz[:, 3, 1] = small_covz[:, 1, 3]
    # small_covz[:, 3, 2] = small_covz[:, 2, 3]
    small_covz[:, 3, 3] = sig_jer ** 2

    small_covariance = np.concatenate([np.concatenate([small_covx, zero_cov, zero_cov], axis=2),
                                         np.concatenate([zero_cov, small_covy, zero_cov], axis=2),
                                         np.concatenate([zero_cov, zero_cov, small_covz], axis=2)], axis=1)

    prev_cov = np.zeros(shape=[batch_size, 12, 12])
    sj2 = 10
    RE = 1

    prev_cov[:, 0, 0] = (sigx**2 / RE) ** 1
    prev_cov[:, 4, 4] = (sigy**2 / RE) ** 1
    prev_cov[:, 8, 8] = (sigz**2 / RE) ** 1

    prev_cov[:, 1, 1] = ((2*sigx**2)/(dt**2) / RE) ** 1
    prev_cov[:, 5, 5] = ((2*sigy**2)/(dt**2) / RE) ** 1
    prev_cov[:, 9, 9] = ((2*sigz**2)/(dt**2) / RE) ** 1

    prev_cov[:, 2, 2] = (1000 / RE) ** 1
    prev_cov[:, 6, 6] = (1000 / RE) ** 1
    prev_cov[:, 10, 10] = (1000 / RE) ** 1

    prev_cov[:, 3, 3] = (sj2 / RE) ** 1
    prev_cov[:, 7, 7] = (sj2 / RE) ** 1
    prev_cov[:, 11, 11] = (sj2 / RE) ** 1

    dt_mat = np.ones_like(small_covariance) * dt[:, np.newaxis, np.newaxis]
    initial_covariance = np.where(dt_mat > 1, prev_cov, small_covariance)
    # initial_covariance = prev_cov

    return initial_covariance


def eci_to_rae_np(eci_data, lla_datar, reference_ecef_position):
    # input is in [batch, time, [x,y,z]]
    # lla_datar is [lat lon alt] in radians and meters, respectively
    # reference_ecef_position is ecef position [x, y, z] in meters

    y_uvw = eci_data - np.ones_like(eci_data) * reference_ecef_position[:, np.newaxis, :]
    y_enu = np.zeros_like(y_uvw)
    y_rae = np.zeros_like(y_uvw)
    zero_rows = (eci_data == 0).all(2)
    for i in range(eci_data.shape[0]):  # batch size
        zz = zero_rows[i, :, np.newaxis]
        y_uvw[i, :, :] = np.where(zz, np.zeros_like(y_uvw[i, :, :]), y_uvw[i, :, :])

        Ti2e = np.zeros(shape=[3, 3])
        Ti2e[0, 0] = -np.sin(lla_datar[i, 1])
        Ti2e[0, 1] = np.cos(lla_datar[i, 1])
        Ti2e[1, 0] = -np.sin(lla_datar[i, 0]) * np.cos(lla_datar[i, 1])
        Ti2e[1, 1] = -np.sin(lla_datar[i, 0]) * np.sin(lla_datar[i, 1])
        Ti2e[1, 2] = np.cos(lla_datar[i, 0])
        Ti2e[2, 0] = np.cos(lla_datar[i, 0]) * np.cos(lla_datar[i, 1])
        Ti2e[2, 1] = np.cos(lla_datar[i, 0]) * np.sin(lla_datar[i, 1])
        Ti2e[2, 2] = np.sin(lla_datar[i, 0])

        for ii in range(eci_data.shape[1]):  # time dimension
            y_enu[i, ii, :] = np.squeeze(np.matmul(Ti2e, y_uvw[i, ii, np.newaxis, :].T), -1)
            y_rae[i, ii, 0] = np.sqrt(y_enu[i, ii, 0] * y_enu[i, ii, 0] + y_enu[i, ii, 1] * y_enu[i, ii, 1] + y_enu[i, ii, 2] * y_enu[i, ii, 2])
            y_rae[i, ii, 1] = np.arctan2(y_enu[i, ii, 0], y_enu[i, ii, 1])
            if y_rae[i, ii, 1] < 0:
                y_rae[i, ii, 1] = (2 * np.pi) + y_rae[i, ii, 1]
            y_rae[i, ii, 2] = np.arcsin(y_enu[i, ii, 2] / y_rae[i, ii, 0])

        y_enu[i, :, :] = np.where(zz, np.zeros_like(y_enu[i, :, :]), y_enu[i, :, :])
        y_rae[i, :, :] = np.where(zz, np.zeros_like(y_rae[i, :, :]), y_rae[i, :, :])

    return y_rae


def eci_to_rae_tf(eci_data, lla_datar, reference_ecef_position, pi_val):
    # itfut is in [batch, time, [x,y,z]]
    # lla_datar is [lat lon alt] in radians and meters, respectively
    # reference_ecef_position is ecef position [x, y, z] in meters

    y_uvw = eci_data - tf.ones_like(eci_data) * reference_ecef_position

    t00 = -tf.sin(lla_datar[:, 1, tf.newaxis, tf.newaxis])
    t01 = tf.cos(lla_datar[:, 1, tf.newaxis, tf.newaxis])
    t02 = tf.zeros_like(t01)
    t10 = -tf.sin(lla_datar[:, 0, tf.newaxis, tf.newaxis]) * tf.cos(lla_datar[:, 1, tf.newaxis, tf.newaxis])
    t11 = -tf.sin(lla_datar[:, 0, tf.newaxis, tf.newaxis]) * tf.sin(lla_datar[:, 1, tf.newaxis, tf.newaxis])
    t12 = tf.cos(lla_datar[:, 0, tf.newaxis, tf.newaxis])
    t20 = tf.cos(lla_datar[:, 0, tf.newaxis, tf.newaxis]) * tf.cos(lla_datar[:, 1, tf.newaxis, tf.newaxis])
    t21 = tf.cos(lla_datar[:, 0, tf.newaxis, tf.newaxis]) * tf.sin(lla_datar[:, 1, tf.newaxis, tf.newaxis])
    t22 = tf.sin(lla_datar[:, 0, tf.newaxis, tf.newaxis])

    Ti2e = tf.concat([tf.concat([t00, t01, t02], axis=2), tf.concat([t10, t11, t12], axis=2), tf.concat([t20, t21, t22], axis=2)], axis=1)

    y_enu = tf.matmul(Ti2e, y_uvw[:, :, tf.newaxis])
    r = tf.norm(y_enu, axis=1)
    a = tf.math.atan2(y_enu[:, 0], y_enu[:, 1])
    e = tf.math.asin(y_enu[:, 2] / r)

    a = tf.where(a < 0, a + tf.ones_like(a) * pi_val, a)

    y_rae = tf.concat([r, a, e], axis=1)

    return y_rae


def trans(A):
    return np.transpose(A, [0, 2, 1])


def unscented_kalman_np(batch_size, initial_state, rae_data, prev_time, dt0, dt1, lla_datar, units):
    seqlen = rae_data.shape[1]

    rae_meas = rae_data[:, 0, :3]
    rae_variance = rae_data[:, 0, 3:]

    RE = 6378137
    e_sq = 0.00669437999014132

    Ql = [None] * seqlen
    Sl = [None] * seqlen
    al = [None] * seqlen
    pl = [None] * seqlen
    meas_list = [None] * seqlen

    I_4z = np.eye(4, dtype=np.float64) * 0
    I_3z = np.eye(3, dtype=np.float64) * 0

    I_4z = np.tile(I_4z[np.newaxis, :, :], [batch_size, 1, 1])
    I_3z = np.tile(I_3z[np.newaxis, :, :], [batch_size, 1, 1])

    zb = np.zeros([batch_size, 4, 2], dtype=np.float64)
    om = np.ones([batch_size, 1, 1], dtype=np.float64)
    zm = np.zeros([batch_size, 1, 1], dtype=np.float64)
    omp = np.ones([1, 1], np.float64)
    zmp = np.zeros([1, 1], np.float64)

    num_state = 12

    m1 = np.concatenate([omp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp], axis=1).astype(np.float64)
    m2 = np.concatenate([zmp, zmp, zmp, zmp, omp, zmp, zmp, zmp, zmp, zmp, zmp, zmp], axis=1).astype(np.float64)
    m3 = np.concatenate([zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, omp, zmp, zmp, zmp], axis=1).astype(np.float64)
    meas_mat = np.tile(np.expand_dims(np.concatenate([m1, m2, m3], axis=0), axis=0), [batch_size, 1, 1])

    # alpha = 1e-1 * np.ones([batch_size, 1], dtype=np.float64)
    # beta = 2. * np.ones([batch_size, 1], dtype=np.float64)
    # k = 0. * np.ones([batch_size, 1], dtype=np.float64)
    #
    # L = num_state
    # lam = alpha * (L + k) - L
    # c1 = L + lam
    # tmat = np.ones([1, 2 * num_state], dtype=np.float64)
    # Wm = np.concatenate([(lam / c1), (0.5 / c1) * tmat], axis=1)
    # Wc1 = np.expand_dims(copy.copy(Wm[:, 0]), axis=1) + (np.ones_like(alpha, dtype=np.float64) - (alpha) + beta)
    # Wc = np.concatenate([Wc1, copy.copy(Wm[:, 1:])], axis=1)
    # c = np.sqrt(c1)

    for q in range(seqlen):
        if q == 0:
            pstate_est = initial_state
        else:
            pstate_est = Sl[q - 1]

        R = rae_meas[:, 0, np.newaxis]
        A = rae_meas[:, 1, np.newaxis]
        E = rae_meas[:, 2, np.newaxis]

        east = (R * np.sin(A) * np.cos(E))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        north = (R * np.cos(E) * np.cos(A))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        up = (R * np.sin(E))  # * ((tf.exp(tf.negative(tf.pow(se, 2) / 2))))

        lat = lla_datar[:, 0, np.newaxis]
        lon = lla_datar[:, 1, np.newaxis]
        alt = lla_datar[:, 2, np.newaxis]

        uvw2enu = np.zeros([lat.shape[0], 3, 3])
        uvw2enu[:, 0, 0] = np.squeeze(-np.sin(lon), -1)
        uvw2enu[:, 0, 1] = np.squeeze(np.cos(lon), -1)
        uvw2enu[:, 0, 2] = 0
        uvw2enu[:, 1, 0] = np.squeeze(-np.sin(lat) * np.cos(lon), -1)
        uvw2enu[:, 1, 1] = np.squeeze(-np.sin(lat) * np.sin(lon), -1)
        uvw2enu[:, 1, 2] = np.squeeze(np.cos(lat), -1)
        uvw2enu[:, 2, 0] = np.squeeze(np.cos(lat) * np.cos(lon), -1)
        uvw2enu[:, 2, 1] = np.squeeze(np.cos(lat) * np.sin(lon), -1)
        uvw2enu[:, 2, 2] = np.squeeze(np.sin(lat), -1)

        # enu2uvw = np.transpose(uvw2enu, [0, 2, 1])

        rae2enu = np.zeros([lat.shape[0], 3, 3])
        rae2enu[:, 0, 0] = np.squeeze(np.sin(A) * np.cos(E), -1)
        rae2enu[:, 0, 1] = np.squeeze(np.cos(A) * np.cos(E), -1)
        rae2enu[:, 0, 2] = np.squeeze(-np.sin(A) * np.sin(E), -1)
        rae2enu[:, 1, 0] = np.squeeze(np.cos(A) * np.cos(E), -1)
        rae2enu[:, 1, 1] = np.squeeze(-np.sin(A) * np.cos(E), -1)
        rae2enu[:, 1, 2] = np.squeeze(-np.cos(A) * np.sin(E), -1)
        rae2enu[:, 2, 0] = np.squeeze(np.sin(E), -1)
        rae2enu[:, 2, 1] = 0
        rae2enu[:, 2, 2] = np.squeeze(np.cos(E), -1)

        # enu2rae = np.matmul(enu2uvw, rae2enu)
        # rae2enu = trans(enu2rae)

        cosPhi = np.cos(lat)
        sinPhi = np.sin(lat)
        cosLambda = np.cos(lon)
        sinLambda = np.sin(lon)

        tv = cosPhi * up - sinPhi * north
        wv = sinPhi * up + cosPhi * north
        uv = cosLambda * tv - sinLambda * east
        vv = sinLambda * tv + cosLambda * east

        chi = RE / np.sqrt(1 - e_sq * (np.sin(lat)) ** 2)

        xs = (chi + alt) * np.cos(lat) * np.cos(lon)
        ys = (chi + alt) * np.cos(lat) * np.sin(lon)
        zs = (alt + chi * (1 - e_sq)) * np.sin(lat)

        ecef_ref = np.concatenate([xs, ys, zs], axis=1)

        cur_meas_uvw = np.concatenate([uv, vv, wv], axis=1)
        cur_meas_ecef = cur_meas_uvw + ecef_ref

        if q > 0:
            dt = prev_time[:, q, :] - prev_time[:, q - 1, :]
            dt = dt[:, :, np.newaxis]
        else:
            dt = dt0

        Qt, _, Bt, At = get_QP_np(dt, om, zm, I_3z, I_4z, zb,
                                    dimension=int(num_state / 3),
                                    sjix=om * 1 ** 2,
                                    sjiy=om * 1 ** 2,
                                    sjiz=om * 1 ** 2,
                                    aji=om * 1.0)

        # acc_part = np.concatenate([pstate_est[:, 2, np.newaxis], pstate_est[:, 6, np.newaxis], pstate_est[:, 10, np.newaxis]], axis=1)
        # jer_part = np.concatenate([pstate_est[:, 3, np.newaxis], pstate_est[:, 7, np.newaxis], pstate_est[:, 11, np.newaxis]], axis=1)
        # acc_norm = np.linalg.norm(acc_part, axis=1)
        #
        # omx = (acc_part[:, 1] * jer_part[:, 2] - acc_part[:, 2] * jer_part[:, 1]) / acc_norm
        # omy = (acc_part[:, 2] * jer_part[:, 0] - acc_part[:, 0] * jer_part[:, 2]) / acc_norm
        # omz = (acc_part[:, 0] * jer_part[:, 1] - acc_part[:, 1] * jer_part[:, 0]) / acc_norm
        #
        # omx = omx[:, np.newaxis, np.newaxis]
        # omy = omy[:, np.newaxis, np.newaxis]
        # omz = omz[:, np.newaxis, np.newaxis]
        #
        # Qt, _, Bt, At = get_QP_np_snap(dt, om, zm, I_4z, zb,
        #                                omx=omx,
        #                                omy=omy,
        #                                omz=omz,
        #                                sjix=om * 1 ** 2,
        #                                sjiy=om * 1 ** 2,
        #                                sjiz=om * 1 ** 2,
        #                                aji=om * 1.0)

        al[q] = At
        pl[q] = Qt

        mu_pred = np.matmul(At, pstate_est[:, :, np.newaxis])

        if units == 'ecef':
            mu_pred_pos_ecef = np.matmul(meas_mat, mu_pred)
            mu_pred_pos_uvw = mu_pred_pos_ecef - ecef_ref[:, :, np.newaxis]

            y_enu = np.squeeze(np.matmul(uvw2enu, mu_pred_pos_uvw), -1)
        elif units == 'uvw':

            mu_pred_pos_uvw = np.matmul(meas_mat, mu_pred)
            y_enu = np.squeeze(np.matmul(uvw2enu, mu_pred_pos_uvw), -1)

        y_rae = np.zeros_like(y_enu)
        y_rae[:, 0] = np.sqrt(y_enu[:, 0] * y_enu[:, 0] + y_enu[:, 1] * y_enu[:, 1] + y_enu[:, 2] * y_enu[:, 2])
        y_rae[:, 1] = np.arctan2(y_enu[:, 0], y_enu[:, 1])
        y_rae[:, 1] = np.where(y_rae[:, 1] < 0, (2 * np.pi) + y_rae[:, 1], y_rae[:, 1])
        y_rae[:, 2] = np.arcsin(y_enu[:, 2] / y_rae[:, 0])

        rng = np.sqrt(y_enu[:, 0] * y_enu[:, 0] + y_enu[:, 1] * y_enu[:, 1] + y_enu[:, 2] * y_enu[:, 2])
        az = np.arctan2(y_enu[:, 0], y_enu[:, 1])
        az = np.where(az < 0, az + np.ones_like(az) * (2 * np.pi), az)
        el = np.arcsin(y_enu[:, 2] / rng)

        Rt = np.eye(3, dtype=np.float64)
        Rt = np.tile(Rt[np.newaxis, :, :], [batch_size, 1, 1])
        Rt[:, 0, 0] = np.ones_like(el) * rae_variance[:, 0] ** 1
        Rt[:, 1, 1] = rng * np.ones_like(el) * rae_variance[:, 1] ** 1
        Rt[:, 2, 2] = rng * np.ones_like(el) * rae_variance[:, 2] ** 1

        enu_cov = np.matmul(np.matmul(rae2enu, Rt), trans(rae2enu))

        Rt = np.matmul(np.matmul(trans(uvw2enu), enu_cov), uvw2enu)

        if q == 0:
            cov_est0 = initialize_covariance(Rt, dt)
            initial_cov = cov_est0[:, np.newaxis, :, :]
        else:
            cov_est0 = Ql[q - 1]

        cov_est0 = np.matmul(np.matmul(At, cov_est0), np.transpose(At, [0, 2, 1])) + Qt

        # Am = np.expand_dims(c, axis=2) * qcholr
        # Y = np.tile(np.expand_dims(pstate_est, axis=2), [1, 1, 12])
        # X = np.concatenate([np.expand_dims(pstate_est, axis=2), Y + Am, Y - Am], axis=2)
        # X = np.transpose(X, [0, 2, 1])

        # x1, X1, P1, X2 = ut_state_batch_np(X, Wm, Wc, Qt, num_state, batch_size, At)
        # z1, Z1, P2, Z2 = ut_meas_np(X1, Wm, Wc, Rt, meas_mat, batch_size)

        diag_eye12 = np.tile(np.eye(12)[np.newaxis, :, :], [batch_size, 1, 1])

        # P12 = np.matmul(np.matmul(X2, diag_eye), np.transpose(Z2, [0, 2, 1]))

        # gain = np.matmul(P12, np.linalg.inv(P2))
        if units == 'uvw':
            pos_res = cur_meas_uvw[:, :, np.newaxis] - np.matmul(meas_mat, mu_pred)
        elif units == 'ecef':
            pos_res = cur_meas_ecef[:, :, np.newaxis] - np.matmul(meas_mat, mu_pred)
        # x = x1 + np.matmul(gain, pos_res2)

        # cov_est_t0 = P1 - np.matmul(gain, np.transpose(P12, [0, 2, 1]))
        # cov_est_t = (cov_est_t0 + np.transpose(cov_est_t0, [0, 2, 1])) / 2

        HPH = np.matmul(np.matmul(meas_mat, cov_est0), np.transpose(meas_mat, [0, 2, 1]))
        S = HPH + Rt

        try:
            S_inv = np.linalg.inv(S)
        except:
            pdb.set_trace()
            pass
        gain = np.matmul(np.matmul(cov_est0, np.transpose(meas_mat, [0, 2, 1])), S_inv)

        # gain = tf.matmul(P12, tf.matrix_inverse(P2)) * cur_weight[:, tf.newaxis, :]

        mu_t = mu_pred + np.matmul(gain, pos_res)

        # pos_res2 = meas_uvw[:, :, tf.newaxis] - tf.matmul(self.meas_mat, mu_t)
        # gain2 = gain2 * cur_weight[:, tf.newaxis, :]
        # mu_t = mu_t + tf.matmul(gain2, pos_res2)

        # x = mu_t[:, :, 0]

        I_KC = diag_eye12 - np.matmul(gain, meas_mat)  # (bs, dim_z, dim_z)
        cov_est_t = np.matmul(np.matmul(I_KC, cov_est0), np.transpose(I_KC, [0, 2, 1])) + np.matmul(np.matmul(gain, Rt), np.transpose(gain, [0, 2, 1]))

        try:
            np.linalg.inv(cov_est_t)
        except:
            pdb.set_trace()
            pass
        cov_est_t = (cov_est_t + np.transpose(cov_est_t, [0, 2, 1])) / 2

        Ql[q] = cov_est_t
        if units == 'uvw':
            meas_list[q] = cur_meas_uvw
        elif units == 'ecef':
            meas_list[q] = cur_meas_ecef
        Sl[q] = mu_t[:, :, 0]

    # final_state = np.stack(Sl, 1)  # final smoothed estimate

    # Smoothing
    # j = [None] * (seqlen + 1)
    # xtemp = copy.copy(np.split(np.expand_dims(final_state, 3), seqlen, axis=1))
    xtemp = copy.copy(Sl)
    # for q in range(seqlen):
    #     xtemp[q] = np.expand_dims(xtemp[q], axis=2)
    # Ptemp = copy.copy(Ql)
    #
    # for q in range(seqlen - 2, -1, -1):
    #     if q >= 0:
    #         P_pred = np.matmul(np.matmul(al[q], Ptemp[q]), np.transpose(al[q], [0, 2, 1]))  # + Pl[q]
    #         j[q] = np.matmul(np.matmul(Ptemp[q], np.transpose(al[q], [0, 2, 1])), np.linalg.inv(P_pred))
    #         xtemp[q] += np.matmul(j[q], xtemp[q + 1] - np.matmul(al[q], xtemp[q]))
    #         Ptemp[q] += np.matmul(np.matmul(j[q], Ptemp[q + 1] - P_pred), np.transpose(j[q], [0, 2, 1]))

    final_state = np.stack(xtemp, axis=1)
    converted_meas = np.stack(meas_list, axis=1)

    Qt, At, Bt, At2 = get_QP_np(dt1, om, zm, I_3z, I_4z, zb,
                                dimension=int(num_state / 3),
                                sjix=om * 1 ** 2,
                                sjiy=om * 1 ** 2,
                                sjiz=om * 1 ** 2,
                                aji=om * 1.)

    pred_state = np.matmul(At, Sl[-1][:, :, np.newaxis])
    pred_covariance = np.matmul(np.matmul(At, Ql[-1]), np.transpose(At, [0, 2, 1])) + Qt

    return final_state, Ql, converted_meas, pred_state, pred_covariance, Qt, Rt, initial_cov


def get_QP_np(dt, om, zm, I_3z, I_4z, zb, dimension=3, sjix=50e-6, sjiy=50e-6, sjiz=50e-6, aji=0.1):
    # dt = dt[:, np.newaxis, :]

    dt7 = dt ** 7
    dt6 = dt ** 6
    dt5 = dt ** 5
    dt4 = dt ** 4
    dt3 = dt ** 3
    dt2 = dt ** 2

    # aji = np.ones_like(sji[:, np.newaxis]) * aji
    aj = aji

    aj7 = np.power(aj, 7)
    aj6 = np.power(aj, 6)
    aj5 = np.power(aj, 5)
    aj4 = np.power(aj, 4)
    aj3 = np.power(aj, 3)
    aj2 = np.power(aj, 2)

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

    emadt = np.exp(-aj * dt)

    q11j = (1 / (2 * aj7)) * (((aj5 * dt5) / 10) - ((aj4 * dt4) / 2) + ((4 * aj3 * dt3) / 3) + (2 * aj * dt) - (2 * aj2 * dt2) - 3 + (4 * emadt) + (2 * aj2 * dt2 * emadt) - np.exp(-2 * aj * dt))
    q22j = ((1 / (2 * aj5)) * (1 - np.exp(-2 * aj * dt) + ((2 * aj3 * dt3) / 2) + (2 * aj * dt) - (2 * aj2 * dt2) - (4 * aj * dt * emadt)))
    q33j = ((1 / (2 * aj3)) * (4 * emadt + (2 * aj * dt) - (np.exp(-2 * aj * dt)) - 3))
    q44j = ((1 / (2 * aj)) * (1 - np.exp(-2 * aj * dt)))

    q12j = (1 / (2 * aj6)) * (1 - (2 * aj * dt) + (2 * aj2 * dt2) - (aj3 * dt3) + ((aj4 * dt4) / 4)
                              + np.exp(-2 * aj * dt) + (2 * aj * dt * emadt) - (2 * emadt) - (aj2 * dt2 * emadt))
    q13j = (1 / (2 * aj5)) * (((aj3 * dt3) / 3) + (2 * aj * dt) - (aj2 * dt2) - 3
                              + (4 * emadt) + (aj2 * dt2 * emadt) - np.exp(-2 * aj * dt))

    # q14j = ((1 / (2 * aj4)) * (1 - (2 * np.exp(-2 * aj * dt)) - (aj2 * dt2 * emadt) + np.exp(-2 * aj * dt)))
    q14j = ((1 / (2 * aj4)) * (1 + np.exp(-2 * aj * dt) - (2 * emadt) - (aj2 * dt2 * emadt)))

    q23j = ((1 / (2 * aj4)) * (1 - (2 * aj * dt) + (aj2 * dt2) + (2 * aj * dt * emadt) + np.exp(-2 * aj * dt) - 2 * emadt))
    q24j = ((1 / (2 * aj3)) * (1 - 2 * aj * dt * emadt - np.exp(-2 * aj * dt)))
    q34j = ((1 / (2 * aj2)) * (1 - 2 * emadt + np.exp(-2 * aj * dt)))

    pj = ((2 - (2 * aj * dt) + (aj2 * dt2) - 2 * emadt) / (2 * aj3))
    qj = ((emadt - 1 + (aj * dt)) / aj2)
    rj = ((1 - emadt) / aj)
    sj = emadt

    # pj = q24
    # qj = q34
    # rj = om * dt
    # sj = om

    sj1 = 2 * sjix * aj
    sj2 = 2 * sjiy * aj
    sj3 = 2 * sjiz * aj

    mat_part = np.concatenate(
        [np.concatenate([q11j, q12j, q13j, q14j], axis=2),
         np.concatenate([q12j, q22j, q23j, q24j], axis=2),
         np.concatenate([q13j, q23j, q33j, q34j], axis=2),
         np.concatenate([q14j, q24j, q34j, q44j], axis=2)],
        axis=1)

    zeta1j = copy.copy(mat_part) * sj1

    zeta2j = copy.copy(mat_part) * sj2

    zeta3j = copy.copy(mat_part) * sj3

    Q = np.concatenate([np.concatenate([zeta1j, I_4z, I_4z], axis=2),
                        np.concatenate([I_4z, zeta2j, I_4z], axis=2),
                        np.concatenate([I_4z, I_4z, zeta3j], axis=2)], axis=1)

    phi = np.concatenate([np.concatenate([om, dt, q34, pj], axis=2),
                          np.concatenate([zm, om, dt, qj], axis=2),
                          np.concatenate([zm, zm, om, rj], axis=2),
                          np.concatenate([zm, zm, zm, sj], axis=2)],
                         axis=1)

    A = np.concatenate([np.concatenate([phi, I_4z, I_4z], axis=2), np.concatenate([I_4z, phi, I_4z], axis=2), np.concatenate([I_4z, I_4z, phi], axis=2)], axis=1)

    # zeta1 = np.concatenate([np.concatenate([q11, q12, q13, q14], axis=2), np.concatenate([q12, q22, q23, q24], axis=2), np.concatenate([q13, q23, q33, q34], axis=2),
    #                         np.concatenate([q14, q24, q34, q44], axis=2)], axis=1) * sj1
    # zeta2 = np.concatenate([np.concatenate([q11, q12, q13, q14], axis=2), np.concatenate([q12, q22, q23, q24], axis=2), np.concatenate([q13, q23, q33, q34], axis=2),
    #                         np.concatenate([q14, q24, q34, q44], axis=2)], axis=1) * sj2
    # zeta3 = np.concatenate([np.concatenate([q11, q12, q13, q14], axis=2), np.concatenate([q12, q22, q23, q24], axis=2), np.concatenate([q13, q23, q33, q34], axis=2),
    #                         np.concatenate([q14, q24, q34, q44], axis=2)], axis=1) * sj3
    #
    # Q = np.concatenate([np.concatenate([zeta1, I_4z, I_4z], axis=2),
    #                     np.concatenate([I_4z, zeta2, I_4z], axis=2),
    #                     np.concatenate([I_4z, I_4z, zeta3], axis=2)], axis=1)

    phi2 = np.concatenate([np.concatenate([om, dt, q34, q24], axis=2),
                           np.concatenate([zm, om, dt, q34], axis=2),
                           np.concatenate([zm, zm, om, dt], axis=2),
                           np.concatenate([zm, zm, zm, om], axis=2)], axis=1)

    A2 = np.concatenate([np.concatenate([phi2, I_4z, I_4z], axis=2),
                         np.concatenate([I_4z, phi2, I_4z], axis=2),
                         np.concatenate([I_4z, I_4z, phi2], axis=2)], axis=1)

    tb = np.concatenate([np.concatenate([q34, q24], axis=2),
                         np.concatenate([q44, q34], axis=2),
                         np.concatenate([om, q44], axis=2),
                         np.concatenate([zm, om], axis=2)], axis=1)

    B = np.concatenate([np.concatenate([tb, zb, zb], axis=2),
                        np.concatenate([zb, tb, zb], axis=2),
                        np.concatenate([zb, zb, tb], axis=2)], axis=1)

    return Q, A, B, A2


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

    return Q, A, B, A


def get_QP_snap(dt, om, zm, I_4z, omx, omy, omz, sjix=50e-6, sjiy=50e-6, sjiz=50e-6, aji=0.1):

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

    At = tf.concat([tf.concat([zm, om, zm, zm, zm, zm, zm, zm, zm, zm, zm, zm], axis=2),
                    tf.concat([zm, zm, om, zm, zm, zm, zm, zm, zm, zm, zm, zm], axis=2),
                    tf.concat([zm, zm, zm, om, zm, zm, zm, zm, zm, zm, zm, zm], axis=2),
                    tf.concat([zm, zm, omx ** 2 * omy ** 2, zm, zm, zm, -omx * omy, -2 * omz, zm, zm, -omx * omz, 2 * omy], axis=2),

                    tf.concat([zm, zm, zm, zm, zm, om, zm, zm, zm, zm, zm, zm], axis=2),
                    tf.concat([zm, zm, zm, zm, zm, zm, om, zm, zm, zm, zm, zm], axis=2),
                    tf.concat([zm, zm, zm, zm, zm, zm, zm, om, zm, zm, zm, zm], axis=2),
                    tf.concat([zm, zm, -omx * omy, 2 * omx, zm, zm, omx ** 2 + omz ** 2, zm, zm, zm, -omz * omy, -2 * omx], axis=2),

                    tf.concat([zm, zm, zm, zm, zm, zm, zm, zm, zm, om, zm, zm], axis=2),
                    tf.concat([zm, zm, zm, zm, zm, zm, zm, zm, zm, zm, om, zm], axis=2),
                    tf.concat([zm, zm, zm, zm, zm, zm, zm, zm, zm, zm, zm, om], axis=2),
                    tf.concat([zm, zm, -omx * omz, -2 * omy, zm, zm, -omy * omz, 2 * omx, zm, zm, omx ** 2 + omy ** 2, zm], axis=2)],
                    axis=1)
    
    A = tf.stop_gradient(tf.linalg.expm(At * dt))
    
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

    B = tf.concat([tf.concat([zm, zm, zm], axis=2),
                    tf.concat([zm, zm, zm], axis=2),
                    tf.concat([zm, zm, zm], axis=2),
                    tf.concat([om, zm, zm], axis=2),
                    tf.concat([zm, zm, zm], axis=2),
                    tf.concat([zm, zm, zm], axis=2),
                    tf.concat([zm, zm, zm], axis=2),
                    tf.concat([zm, om, zm], axis=2),
                    tf.concat([zm, zm, zm], axis=2),
                    tf.concat([zm, zm, zm], axis=2),
                    tf.concat([zm, zm, zm], axis=2),
                    tf.concat([zm, zm, om], axis=2)], axis=1)

    return Q, A, B, A


def get_QP_np_snap(dt, om, zm, I_4z, omx, omy, omz, sjix=50e-6, sjiy=50e-6, sjiz=50e-6, aji=0.1):
    # dt = dt[:, np.newaxis, :]

    dt7 = dt ** 7
    dt6 = dt ** 6
    dt5 = dt ** 5
    dt4 = dt ** 4
    dt3 = dt ** 3
    dt2 = dt ** 2

    # aji = np.ones_like(sji[:, np.newaxis]) * aji
    aj = aji

    aj7 = np.power(aj, 7)
    aj6 = np.power(aj, 6)
    aj5 = np.power(aj, 5)
    aj4 = np.power(aj, 4)
    aj3 = np.power(aj, 3)
    aj2 = np.power(aj, 2)

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

    emadt = np.exp(-aj * dt)

    q11j = (1 / (2 * aj7)) * (((aj5 * dt5) / 10) - ((aj4 * dt4) / 2) + ((4 * aj3 * dt3) / 3) + (2 * aj * dt) - (2 * aj2 * dt2) - 3 + (4 * emadt) + (2 * aj2 * dt2 * emadt) - np.exp(-2 * aj * dt))
    q22j = ((1 / (2 * aj5)) * (1 - np.exp(-2 * aj * dt) + ((2 * aj3 * dt3) / 2) + (2 * aj * dt) - (2 * aj2 * dt2) - (4 * aj * dt * emadt)))
    q33j = ((1 / (2 * aj3)) * (4 * emadt + (2 * aj * dt) - (np.exp(-2 * aj * dt)) - 3))
    q44j = ((1 / (2 * aj)) * (1 - np.exp(-2 * aj * dt)))

    q12j = (1 / (2 * aj6)) * (1 - (2 * aj * dt) + (2 * aj2 * dt2) - (aj3 * dt3) + ((aj4 * dt4) / 4)
                              + np.exp(-2 * aj * dt) + (2 * aj * dt * emadt) - (2 * emadt) - (aj2 * dt2 * emadt))
    q13j = (1 / (2 * aj5)) * (((aj3 * dt3) / 3) + (2 * aj * dt) - (aj2 * dt2) - 3
                              + (4 * emadt) + (aj2 * dt2 * emadt) - np.exp(-2 * aj * dt))

    # q14j = ((1 / (2 * aj4)) * (1 - (2 * np.exp(-2 * aj * dt)) - (aj2 * dt2 * emadt) + np.exp(-2 * aj * dt)))
    q14j = ((1 / (2 * aj4)) * (1 + np.exp(-2 * aj * dt) - (2 * emadt) - (aj2 * dt2 * emadt)))

    q23j = ((1 / (2 * aj4)) * (1 - (2 * aj * dt) + (aj2 * dt2) + (2 * aj * dt * emadt) + np.exp(-2 * aj * dt) - 2 * emadt))
    q24j = ((1 / (2 * aj3)) * (1 - 2 * aj * dt * emadt - np.exp(-2 * aj * dt)))
    q34j = ((1 / (2 * aj2)) * (1 - 2 * emadt + np.exp(-2 * aj * dt)))

    pj = ((2 - (2 * aj * dt) + (aj2 * dt2) - 2 * emadt) / (2 * aj3))
    qj = ((emadt - 1 + (aj * dt)) / aj2)
    rj = ((1 - emadt) / aj)
    sj = emadt

    # pj = q24
    # qj = q34
    # rj = om * dt
    # sj = om

    sj1 = 2 * sjix * aj
    sj2 = 2 * sjiy * aj
    sj3 = 2 * sjiz * aj

    mat_part = np.concatenate(
        [np.concatenate([q11j, q12j, q13j, q14j], axis=2),
         np.concatenate([q12j, q22j, q23j, q24j], axis=2),
         np.concatenate([q13j, q23j, q33j, q34j], axis=2),
         np.concatenate([q14j, q24j, q34j, q44j], axis=2)],
        axis=1)

    zeta1j = copy.copy(mat_part) * sj1

    zeta2j = copy.copy(mat_part) * sj2

    zeta3j = copy.copy(mat_part) * sj3

    Q = np.concatenate([np.concatenate([zeta1j, I_4z, I_4z], axis=2),
                        np.concatenate([I_4z, zeta2j, I_4z], axis=2),
                        np.concatenate([I_4z, I_4z, zeta3j], axis=2)], axis=1)

    At = np.concatenate([np.concatenate([zm, om, zm, zm, zm, zm, zm, zm, zm, zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, om, zm, zm, zm, zm, zm, zm, zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, zm, om, zm, zm, zm, zm, zm, zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, omx**2 * omy**2, zm, zm, zm, -omx*omy, -2*omz, zm, zm, -omx*omz, 2*omy], axis=2),

                        np.concatenate([zm, zm, zm, zm, zm, om, zm, zm, zm, zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, zm, zm, zm, zm, om, zm, zm, zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, zm, zm, zm, zm, zm, om, zm, zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, -omx*omy, 2*omx, zm, zm, omx ** 2 + omz ** 2, zm, zm, zm, -omz*omy, -2*omx], axis=2),

                        np.concatenate([zm, zm, zm, zm, zm, zm, zm, zm, zm, om, zm, zm], axis=2),
                        np.concatenate([zm, zm, zm, zm, zm, zm, zm, zm, zm, zm, om, zm], axis=2),
                        np.concatenate([zm, zm, zm, zm, zm, zm, zm, zm, zm, zm, zm, om], axis=2),
                        np.concatenate([zm, zm, -omx*omz, -2*omy, zm, zm, -omy*omz, 2*omx, zm, zm, omx**2 + omy**2, zm], axis=2)],
                        axis=1)

    al = list()
    for bb in range(At.shape[0]):
        al.append(scipy.linalg.expm(At[bb, :, :]*dt[bb]))

    A = np.stack(al, axis=0)

    B = np.concatenate([np.concatenate([zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, zm], axis=2),
                        np.concatenate([om, zm, zm], axis=2),
                        np.concatenate([zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, zm], axis=2),
                        np.concatenate([zm, om, zm], axis=2),
                        np.concatenate([zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, zm], axis=2),
                        np.concatenate([zm, zm, om], axis=2)], axis=1)

    # p1 = A[0] - np.eye(12)
    # p2 = np.linalg.solve(A[0], p1)
    # bb = np.matmul(p2, B[0])
    
    return Q, A, B, A


def uvw2enu_tf(lat, lon):
    tz = tf.zeros_like(lon)

    t00 = -tf.sin(lon)
    t01 = tf.cos(lon)
    t10 = -tf.sin(lat) * tf.cos(lon)
    t11 = -tf.sin(lat) * tf.sin(lon)
    t12 = tf.cos(lat)
    t20 = tf.cos(lat) * tf.cos(lon)
    t21 = tf.cos(lat) * tf.sin(lon)
    t22 = tf.sin(lat)

    uvw_to_enu = tf.concat([tf.concat([t00, t01, tz], axis=2), tf.concat([t10, t11, t12], axis=2), tf.concat([t20, t21, t22], axis=2)], axis=1)

    return uvw_to_enu


def rae2enu_tf(y_enu, pi_val):
    rng = tf.sqrt(y_enu[:, 0] * y_enu[:, 0] + y_enu[:, 1] * y_enu[:, 1] + y_enu[:, 2] * y_enu[:, 2])
    az = tf.atan2(y_enu[:, 0], y_enu[:, 1])
    az = tf.where(az < 0, az + tf.ones_like(az) * (2 * pi_val), az)
    el = tf.asin(y_enu[:, 2] / rng)

    az = az[:, tf.newaxis, tf.newaxis]
    el = el[:, tf.newaxis, tf.newaxis]

    m00 = tf.sin(az) * tf.cos(el)
    m01 = tf.cos(az) * tf.cos(el)
    m02 = -tf.sin(az) * tf.sin(el)
    m10 = tf.cos(az) * tf.cos(el)
    m11 = -tf.sin(az) * tf.cos(el)
    m12 = -tf.cos(az) * tf.sin(el)
    m20 = tf.sin(el)
    m22 = tf.cos(el)

    tz = tf.zeros_like(el)

    rae_to_enu = tf.concat([tf.concat([m00, m01, m02], axis=2), tf.concat([m10, m11, m12], axis=2), tf.concat([m20, tz, m22], axis=2)], axis=1)

    return rae_to_enu


def rae2enu_tfb(y_enu, pi_val):
    rng = tf.sqrt(y_enu[:, :, 0] * y_enu[:, :, 0] + y_enu[:, :, 1] * y_enu[:, :, 1] + y_enu[:, :, 2] * y_enu[:, :, 2])
    az = tf.atan2(y_enu[:, :, 0], y_enu[:, :, 1])
    az = tf.where(az < 0, az + tf.ones_like(az) * (2 * pi_val), az)
    el = tf.asin(y_enu[:, :, 2] / rng)

    az = az[:, :, tf.newaxis, tf.newaxis]
    el = el[:, :,  tf.newaxis, tf.newaxis]

    m00 = tf.sin(az) * tf.cos(el)
    m01 = tf.cos(az) * tf.cos(el)
    m02 = -tf.sin(az) * tf.sin(el)
    m10 = tf.cos(az) * tf.cos(el)
    m11 = -tf.sin(az) * tf.cos(el)
    m12 = -tf.cos(az) * tf.sin(el)
    m20 = tf.sin(el)
    m22 = tf.cos(el)

    tz = tf.zeros_like(el)

    rae_to_enu = tf.concat([tf.concat([m00, m01, m02], axis=3), tf.concat([m10, m11, m12], axis=3), tf.concat([m20, tz, m22], axis=3)], axis=2)

    return rae_to_enu


def initialize_filter(batch_size, initial_time, initial_meas, prev_time, prev_x, current_time, lla_datar,
                      GM=398600441890000, RE=6378137, e_sq=0.00669437999014132, units='uvw'):

    idxi = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

    R = initial_meas[:, :, 0, np.newaxis]
    A = initial_meas[:, :, 1, np.newaxis]
    E = initial_meas[:, :, 2, np.newaxis]

    east = (R * np.sin(A) * np.cos(E))
    north = (R * np.cos(E) * np.cos(A))
    up = (R * np.sin(E))

    lat = lla_datar[:, 0, np.newaxis, np.newaxis]
    lon = lla_datar[:, 1, np.newaxis, np.newaxis]
    alt = lla_datar[:, 2, np.newaxis, np.newaxis]

    cosPhi = np.cos(lat)
    sinPhi = np.sin(lat)
    cosLambda = np.cos(lon)
    sinLambda = np.sin(lon)

    tv = cosPhi * up - sinPhi * north
    wv = sinPhi * up + cosPhi * north
    uv = cosLambda * tv - sinLambda * east
    vv = sinLambda * tv + cosLambda * east

    chi = RE / np.sqrt(1 - e_sq * (np.sin(lat)) ** 2)

    xs = (chi + alt) * np.cos(lat) * np.cos(lon)
    ys = (chi + alt) * np.cos(lat) * np.sin(lon)
    zs = (alt + chi * (1 - e_sq)) * np.sin(lat)

    ecef_ref = np.concatenate([xs, ys, zs], axis=2)

    initial_meas_uvw = np.concatenate([uv, vv, wv], axis=2)  # + ecef_ref[:, 0, np.newaxis, :]
    initial_meas_ecef = initial_meas_uvw + ecef_ref[:, 0, np.newaxis, :]

    dtn = np.sum(np.diff(initial_time, axis=1), axis=1)
    if units == 'uvw':
        pos = initial_meas_uvw[:, 2, :]
        vel = (initial_meas_uvw[:, 2, :] - initial_meas_uvw[:, 0, :]) / (2 * dtn)

        R1 = np.linalg.norm(initial_meas_uvw + ecef_ref[:, 0, np.newaxis, :], axis=2, keepdims=True)
        R1 = np.mean(R1, axis=1)
        R1 = np.where(np.less(R1, np.ones_like(R1) * RE), np.ones_like(R1) * RE, R1)
        rad_temp = np.power(R1, 3)
        GMt1 = np.divide(GM, rad_temp)
        accg = get_legendre_np(GMt1, pos + ecef_ref[:, 0, :], R1)
        acc = (initial_meas_uvw[:, 2, :] - 2*initial_meas_uvw[:, 1, :] + initial_meas_uvw[:, 0, :]) / (dtn**2)
        acc = accg

    elif units == 'ecef':

        pos = initial_meas_ecef[:, 2, :]
        vel = (initial_meas_ecef[:, 2, :] - initial_meas_ecef[:, 0, :]) / (2 * dtn)

        R1 = np.linalg.norm(initial_meas_ecef, axis=2, keepdims=True)
        R1 = np.mean(R1, axis=1)
        R1 = np.where(np.less(R1, np.ones_like(R1) * RE), np.ones_like(R1) * RE, R1)
        rad_temp = np.power(R1, 3)
        GMt1 = np.divide(GM, rad_temp)
        accg = get_legendre_np(GMt1, pos, R1)
        acc = (initial_meas_ecef[:, 2, :] - 2 * initial_meas_ecef[:, 1, :] + initial_meas_ecef[:, 0, :]) / (dtn ** 2)
        acc = (acc + accg) / 2

    initial_state = np.expand_dims(np.concatenate([pos, vel, acc, np.random.normal(loc=np.zeros_like(acc), scale=1.0)], axis=1), 1)
    initial_state = initial_state[:, :, idxi]

    dt0 = initial_time[:, -1, np.newaxis, :] - initial_time[:, -2, np.newaxis, :]
    dt1 = current_time[:, 0, np.newaxis, :] - prev_time[:, 0, np.newaxis, :]
    current_state, covariance_out, converted_meas, pred_state, pred_covariance, prev_Q, prev_R, prev_cov = \
        unscented_kalman_np(batch_size, initial_state[:, -1, :], prev_x, prev_time, dt0, dt1, lla_datar, units)

    current_state_estimate = current_state[:, :, idxo]
    current_cov_estimate = covariance_out[-1]
    prev_state_estimate = initial_state[:, :, idxo]
    prev_covariance_estimate = prev_cov

    return current_state_estimate, current_cov_estimate, prev_state_estimate, prev_covariance_estimate, prev_Q, prev_R, converted_meas

# def test_ecef_2_eci(ecef_data, eci_data, time):
