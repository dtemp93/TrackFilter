import copy
import tensorflow as tf
from scipy.signal._savitzky_golay import *


def get_legendre(GMR3, position, R, dtype):
    q = tf.expand_dims(position[:, 2], axis=1)

    SMA = R / 1.0
    SMA2 = SMA * SMA
    SMA3 = SMA2 * SMA
    SMA4 = SMA3 * SMA
    SMA5 = SMA4 * SMA
    SMA6 = SMA5 * SMA

    c2 = tf.constant(-0.484165371736e-3, dtype) * -tf.sqrt(tf.constant(5, dtype=dtype)) * (tf.constant(3, dtype=dtype) / tf.constant(2, dtype)) / SMA2
    c3 = tf.constant(0.957254173792e-6, dtype) * -tf.sqrt(tf.constant(7, dtype)) * (tf.constant(5./2., dtype)) / SMA3
    c4 = tf.constant(0.539873863789e-6, dtype) * -tf.sqrt(tf.constant(9., dtype)) * (tf.constant(-5., dtype) / tf.constant(8., dtype)) / SMA4
    c5 = tf.constant(0.685323475630e-7, dtype) * -tf.sqrt(tf.constant(11., dtype)) * (tf.constant(-3., dtype) / tf.constant(8., dtype)) / SMA5
    c6 = tf.constant(-0.149957994714e-6, dtype) * -tf.sqrt(tf.constant(13., dtype)) * (tf.constant(1., dtype) / tf.constant(16., dtype)) / SMA6

    q2 = q*q
    q4 = q2*q2
    q6 = q2*q4

    X = tf.constant(1., dtype) + c2 * (tf.constant(1., dtype) - tf.constant(5.,dtype) * q2) + \
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
    c3 = tf.constant(0.957254173792e-6, dtype) * -tf.sqrt(tf.constant(7, dtype)) * (tf.constant(5./2., dtype)) / SMA3
    c4 = tf.constant(0.539873863789e-6, dtype) * -tf.sqrt(tf.constant(9., dtype)) * (tf.constant(-5., dtype) / tf.constant(8., dtype)) / SMA4
    c5 = tf.constant(0.685323475630e-7, dtype) * -tf.sqrt(tf.constant(11., dtype)) * (tf.constant(-3., dtype) / tf.constant(8., dtype)) / SMA5
    c6 = tf.constant(-0.149957994714e-6, dtype) * -tf.sqrt(tf.constant(13., dtype)) * (tf.constant(1., dtype) / tf.constant(16., dtype)) / SMA6

    q2 = q*q
    q4 = q2*q2
    q6 = q2*q4

    X = tf.constant(1., dtype) + c2 * (tf.constant(1., dtype) - tf.constant(5.,dtype) * q2) + \
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


def get_legendre_np2(GMR3, position, R):
    q = position[2]
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

    acc1 = GMR3 * np.expand_dims(position[0], axis=1) * X
    acc2 = GMR3 * np.expand_dims(position[1], axis=1) * X
    acc3 = GMR3 * (np.expand_dims(position[2], axis=1) * Z - R * Zc)

    acc = np.concatenate([acc1, acc2, acc3])

    return acc


def propagate(state_estimate, dt, gmn):
    xt = state_estimate[:, :3]
    vt = state_estimate[:, 3:6]

    R1 = tf.norm(xt, axis=1, keepdims=True)
    R1 = tf.where(tf.less(R1, tf.ones_like(R1)), tf.ones_like(R1), R1)
    rad_temp = tf.pow(R1, 3)
    # alt = rad_temp - 1.
    # alt = tf.where(tf.less(alt, tf.zeros_like(alt)), tf.zeros_like(alt), alt)
    # rho = rho0 * tf.exp(-k0 * alt * FLAGS.RE)
    # K = tf.negative(0.5 * rho * vt * tf.norm(vt, axis=1, keepdims=True) * (cd * area))

    GMt1 = tf.negative(tf.divide(gmn, rad_temp))
    acc = get_legendre(GMt1, xt, R1, tf.float32)
    A = tf.multiply(dt, acc)

    x2 = xt + tf.divide(dt, 2) * vt + tf.divide(dt, 8) * A
    v2 = vt + A / 2

    R2 = tf.norm(x2, axis=1, keepdims=True)
    R2 = tf.where(tf.less(R2, tf.ones_like(R2)), tf.ones_like(R2), R2)
    rad_temp2 = tf.pow(R2, 3)
    # alt2 = rad_temp2 - 1.
    # alt2 = tf.where(tf.less(alt2, tf.zeros_like(alt2)), tf.zeros_like(alt2), alt2)
    # rho = rho0 * tf.exp(-k0 * alt2 * FLAGS.RE)
    # K2 = tf.negative(0.5 * rho * v2 * tf.norm(v2, axis=1, keepdims=True) * (cd * area))
    GMt2 = tf.negative(tf.divide(gmn, rad_temp2))
    acc2 = get_legendre(GMt2, x2, R2, tf.float32)
    B = dt * acc2

    x3 = x2
    v3 = vt + B / 2

    R3 = tf.norm(x3, axis=1, keepdims=True)
    R3 = tf.where(tf.less(R3, tf.ones_like(R3)), tf.ones_like(R3), R3)
    rad_temp3 = tf.pow(R3, 3)
    # alt3 = rad_temp3 - 1.
    # alt3 = tf.where(tf.less(alt3, tf.zeros_like(alt3)), tf.zeros_like(alt3), alt3)
    # rho = rho0 * tf.exp(-k0 * alt3 * FLAGS.RE)
    # K3 = tf.negative(0.5 * rho * v3 * tf.norm(v3, axis=1, keepdims=True) * (cd * area))
    GMt3 = tf.negative(tf.divide(gmn, rad_temp3))
    acc3 = get_legendre(GMt3, x3, R3, tf.float32)
    C = dt * acc3

    x4 = xt + (dt * vt) + tf.divide(dt, 2) * C
    v4 = vt + C
    R4 = tf.norm(x4, axis=1, keepdims=True)
    R4 = tf.where(tf.less(R4, tf.ones_like(R4)), tf.ones_like(R4), R4)
    rad_temp4 = tf.pow(R4, 3)
    # alt4 = rad_temp4 - 1.
    # alt4 = tf.where(tf.less(alt4, tf.zeros_like(alt4)), tf.zeros_like(alt4), alt4)
    # rho = rho0 * tf.exp(-k0 * alt4 * FLAGS.RE)
    # K4 = tf.negative(0.5 * rho * v4 * tf.norm(v4, axis=1, keepdims=True) * (cd * area))
    GMt4 = tf.negative(tf.divide(gmn, rad_temp4))
    acc4 = get_legendre(GMt4, x4, R4, tf.float32)
    D = dt * acc4

    xt0 = xt + (dt * (vt + (A + B + C) / 6))
    vt0 = vt + ((A + (2 * B) + (2 * C) + D) / 6)
    at0 = (((A + (2 * B) + (2 * C) + D) / 6) / dt)

    state_est = tf.concat([xt0, vt0], axis=1)
    coefs = [(dt * (vt + (A + B + C) / 6)),
             ((A + (2 * B) + (2 * C) + D) / 6),
             (((A + (2 * B) + (2 * C) + D) / 6) / dt)]

    return state_est


def propagatef(state_estimate, dt, gmn, input_acceleration, alpha):
    xt = state_estimate[:, :3]
    vt = state_estimate[:, 3:6]

    rho0 = 1.22  # kg / m**3
    k0 = 0.14141e-3
    area = 0.25  # / FLAGS.RE  # meter squared
    cd = 0.03  # unitless

    R1 = tf.norm(xt, axis=1, keepdims=True)
    R1 = tf.where(tf.less(R1, tf.ones_like(R1)), tf.ones_like(R1), R1)
    rad_temp = tf.pow(R1, 3)
    alt = rad_temp - 1.
    alt = tf.where(tf.less(alt, tf.zeros_like(alt)), tf.zeros_like(alt), alt)
    rho = rho0 * tf.exp(-k0 * alt * 6378100)
    K = tf.negative(0.5 * rho * vt * tf.norm(vt, axis=1, keepdims=True) * (cd * area))

    GMt1 = tf.negative(tf.divide(gmn, rad_temp))
    gravity = get_legendre(GMt1, xt, R1, tf.float32)
    acc = gravity + K * alpha + input_acceleration
    A = acc * dt

    # x2 = xt + tf.divide(dt, 2) * vt + tf.divide(dt, 8) * A
    # v2 = vt + A / 2

    # R2 = tf.norm(x2, axis=1, keepdims=True)
    # R2 = tf.where(tf.less(R2, tf.ones_like(R2)), tf.ones_like(R2), R2)
    # rad_temp2 = tf.pow(R2, 3)
    # alt2 = rad_temp2 - 1.
    # alt2 = tf.where(tf.less(alt2, tf.zeros_like(alt2)), tf.zeros_like(alt2), alt2)
    # rho = rho0 * tf.exp(-k0 * alt2 * FLAGS.RE)
    # K2 = tf.negative(0.5 * rho * v2 * tf.norm(v2, axis=1, keepdims=True) * (cd * area))
    # GMt2 = tf.negative(tf.divide(gmn, rad_temp2))
    # acc2 = get_legendre(GMt2, x2, R2, tf.float32)
    B = dt * acc

    # x3 = x2
    # v3 = vt + B / 2

    # R3 = tf.norm(x3, axis=1, keepdims=True)
    # R3 = tf.where(tf.less(R3, tf.ones_like(R3)), tf.ones_like(R3), R3)
    # rad_temp3 = tf.pow(R3, 3)
    # alt3 = rad_temp3 - 1.
    # alt3 = tf.where(tf.less(alt3, tf.zeros_like(alt3)), tf.zeros_like(alt3), alt3)
    # rho = rho0 * tf.exp(-k0 * alt3 * FLAGS.RE)
    # K3 = tf.negative(0.5 * rho * v3 * tf.norm(v3, axis=1, keepdims=True) * (cd * area))
    # GMt3 = tf.negative(tf.divide(gmn, rad_temp3))
    # acc3 = get_legendre(GMt3, x3, R3, tf.float32)
    C = dt * acc

    # x4 = xt + (dt * vt) + tf.divide(dt, 2) * C
    # v4 = vt + C
    # R4 = tf.norm(x4, axis=1, keepdims=True)
    # R4 = tf.where(tf.less(R4, tf.ones_like(R4)), tf.ones_like(R4), R4)
    # rad_temp4 = tf.pow(R4, 3)
    # alt4 = rad_temp4 - 1.
    # alt4 = tf.where(tf.less(alt4, tf.zeros_like(alt4)), tf.zeros_like(alt4), alt4)
    # rho = rho0 * tf.exp(-k0 * alt4 * FLAGS.RE)
    # K4 = tf.negative(0.5 * rho * v4 * tf.norm(v4, axis=1, keepdims=True) * (cd * area))
    # GMt4 = tf.negative(tf.divide(gmn, rad_temp4))
    # acc4 = get_legendre(GMt4, x4, R4, tf.float32)
    D = dt * acc

    xt0 = xt + (dt * (vt + (A + B + C) / 6))
    vt0 = vt + ((A + (2 * B) + (2 * C) + D) / 6)
    at0 = (((A + (2 * B) + (2 * C) + D) / 6) / dt)

    state_est = tf.concat([xt0, vt0, at0], axis=1)
    # coefs = [(dt * (vt + (A + B + C) / 6)),
    #          ((A + (2 * B) + (2 * C) + D) / 6),
    #          (((A + (2 * B) + (2 * C) + D) / 6) / dt)]

    return state_est, gravity


def propagate_input(dt, acc, jerk):
    xt = tf.zeros_like(acc)
    vt = tf.zeros_like(acc)
    at = acc
    jt = jerk

    xt0 = xt + vt * dt + 0.5 * tf.pow(at, 2) + (1 / 6) * tf.pow(jt, 3)
    vt0 = vt + at * dt + 0.5 * tf.pow(jt, 2)
    at0 = at + jt * dt

    state_est = tf.concat([xt0, vt0, at0, jt], axis=1)

    return state_est


def propagatef_jerk(state_estimate, dt, acc):
    xt = state_estimate[:, :3]
    vt = state_estimate[:, 3:6]
    # at = state_estimate[:, 6:9]
    # jt = state_estimate[:, 9:12]

    at = acc
    jt = tf.zeros_like(acc)

    xt0 = xt + vt * dt + 0.5 * tf.pow(at, 2) + (1 / 6) * tf.pow(jt, 3)
    vt0 = vt + at * dt + 0.5 * tf.pow(jt, 2)
    at0 = at + jt * dt

    state_est = tf.concat([xt0, vt0, at0, jt], axis=1)

    return state_est


def propagatefb(X, dt, sensor_ecef):

    # idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    # xt = X[:, :, :3]
    sensor_ecef = tf.tile(sensor_ecef[:, tf.newaxis, :], [1, X.shape[1], 1])
    X = tf.expand_dims(X, 3)
    xt = tf.concat([X[:, :, 0], X[:, :, 4], X[:, :, 8]], 2)
    vt = tf.concat([X[:, :, 1], X[:, :, 5], X[:, :, 9]], 2)
    at = tf.concat([X[:, :, 2], X[:, :, 6], X[:, :, 10]], 2)
    # at = tf.zeros_like(at) * at

    jt = tf.concat([X[:, :, 3], X[:, :, 7], X[:, :, 11]], 2)
    jt = tf.zeros_like(jt) * jt

    dt = tf.ones_like(xt) * dt[:, tf.newaxis, :]

    rho0 = 1.22  # kg / m**3
    k0 = 0.14141e-3
    area = 0.25  # / FLAGS.RE  # meter squared
    cd = 0.03  # unitless

    R1 = tf.norm(xt + sensor_ecef, axis=2, keepdims=True)
    R1 = tf.where(tf.less(R1, tf.ones_like(R1)*6378137), tf.ones_like(R1)*6378137, R1)
    rad_temp = tf.pow(R1, 3)
    alt = rad_temp - 1.
    alt = tf.where(tf.less(alt, tf.zeros_like(alt)), tf.zeros_like(alt), alt)
    rho = rho0 * tf.exp(-k0 * alt * 6378137)
    Ka = tf.negative(0.5 * rho * vt * tf.norm(vt, axis=2, keepdims=True) * (cd * area))

    GMt1 = tf.divide(398600441890000, rad_temp)
    n_acc = get_legendreb(GMt1, xt + sensor_ecef, R1, tf.float64)
    b_acc = Ka  # * alpha_atmos[:, :, tf.newaxis]
    real_acc = n_acc + b_acc
    noise_acc = at - real_acc

    total_acc = real_acc + noise_acc
    # inp_acc = copy.copy(pstate_est[:, 6:9]) - n_acc - b_acc
    at1 = total_acc

    xt0 = xt + vt * dt + 0.5 * tf.pow(at1, 2) + (1 / 6) * tf.pow(jt, 3)
    vt0 = vt + at1 * dt + 0.5 * tf.pow(jt, 2)
    at0 = at1 + jt * dt
    jt0 = jt

    # idxi = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    # state_est = tf.expand_dims(tf.concat([xt0, vt0, at0, jt], axis=2), 3)
    # state_est = state_est[:, :, idxi]
    xt0 = tf.expand_dims(xt0, 3)
    vt0 = tf.expand_dims(vt0, 3)
    at0 = tf.expand_dims(at0, 3)
    jt0 = tf.expand_dims(jt0, 3)

    state_est = tf.concat([xt0[:, :, 0], vt0[:, :, 0], at0[:, :, 0], jt0[:, :, 0],
                           xt0[:, :, 1], vt0[:, :, 1], at0[:, :, 1], jt0[:, :, 1],
                           xt0[:, :, 2], vt0[:, :, 2], at0[:, :, 2], jt0[:, :, 2]], axis=2)
    return state_est


def propagatef2(X, dt):

    # idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    # xt = X[:, :, :3]
    X = tf.expand_dims(X, 2)
    xt = tf.concat([X[:, 0], X[:, 4], X[:, 8]], 1)
    vt = tf.concat([X[:, 1], X[:, 5], X[:, 9]], 1)
    at = tf.concat([X[:, 2], X[:, 6], X[:, 10]], 1)
    jt = tf.concat([X[:, 3], X[:, 7], X[:, 11]], 1)

    dt = tf.ones_like(xt) * dt

    rho0 = 1.22  # kg / m**3
    k0 = 0.14141e-3
    area = 0.25  # / FLAGS.RE  # meter squared
    cd = 0.03  # unitless

    R1 = tf.norm(xt, axis=1, keepdims=True)
    R1 = tf.where(tf.less(R1, tf.ones_like(R1)*6378137), tf.ones_like(R1)*6378137, R1)
    rad_temp = tf.pow(R1, 3)
    alt = rad_temp - 1.
    alt = tf.where(tf.less(alt, tf.zeros_like(alt)), tf.zeros_like(alt), alt)
    rho = rho0 * tf.exp(-k0 * alt * 6378137)
    Ka = tf.negative(0.5 * rho * vt * tf.norm(vt, axis=1, keepdims=True) * (cd * area))

    GMt1 = tf.divide(398600441890000, rad_temp)
    n_acc = get_legendre(GMt1, xt, R1, tf.float64)
    b_acc = Ka  # * alpha_atmos[:, :, tf.newaxis]
    real_acc = n_acc + b_acc
    noise_acc = at - real_acc

    total_acc = real_acc + noise_acc
    # inp_acc = copy.copy(pstate_est[:, 6:9]) - n_acc - b_acc
    at1 = total_acc

    xt0 = xt + vt * dt + 0.5 * tf.pow(at1, 2) + (1 / 6) * tf.pow(jt, 3)
    vt0 = vt + at1 * dt + 0.5 * tf.pow(jt, 2)
    at0 = at1 + jt * dt
    jt0 = jt

    # idxi = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    # state_est = tf.expand_dims(tf.concat([xt0, vt0, at0, jt], axis=2), 3)
    # state_est = state_est[:, :, idxi]
    xt0 = tf.expand_dims(xt0, 2)
    vt0 = tf.expand_dims(vt0, 2)
    at0 = tf.expand_dims(at0, 2)
    jt0 = tf.expand_dims(jt0, 2)

    state_est = tf.concat([xt0[:, 0], vt0[:, 0], at0[:, 0], jt0[:, 0],
                           xt0[:, 1], vt0[:, 1], at0[:, 1], jt0[:, 1],
                           xt0[:, 2], vt0[:, 2], at0[:, 2], jt0[:, 2]], axis=1)
    return state_est


def kalman_tf(state, covariance, gmn, dt, measurement, batch_size):
    state_est1 = propagate(state, dt, gmn)

    I_3 = tf.eye(3, batch_shape=[batch_size], dtype=tf.float32)
    position = tf.expand_dims(copy.copy(state_est1[:, :3]), axis=1)
    velocity = tf.expand_dims(copy.copy(state_est1[:, 3:]), axis=1)
    R = tf.norm(position, axis=2, keepdims=True)

    C = (gmn / (R ** 3)) * (3 * (position * tf.matrix_transpose(position)) / R ** 2 - I_3)
    F = (3 * gmn / (R ** 5)) * (
        (tf.matrix_transpose(position) * velocity) * (I_3 - (5 * (position * tf.matrix_transpose(position))) / (R ** 2)) + (position * tf.matrix_transpose(velocity)) +
        (velocity * tf.matrix_transpose(position)))
    # dt2 = tf.expand_dims(dt, axis=1)
    dt2 = dt
    p1l = I_3 + C / 2 * dt2 ** 2 + F / 6 * dt2 ** 3
    p1r = I_3 * dt2 + C / 6 * dt2 ** 3
    p1 = tf.concat([p1l, p1r], axis=2)
    p2l = C * dt2 + F / 2 * dt2 ** 2
    p2r = I_3 + C / 2 * dt2 ** 2 + F / 3 * dt2 ** 3
    p2 = tf.concat([p2l, p2r], axis=2)

    Pr = tf.concat([p1, p2], axis=1)

    pcov = tf.matmul(tf.matmul(Pr, covariance), tf.matrix_transpose(Pr))

    mmr = tf.zeros(shape=[batch_size, 3, 3])
    mml = tf.eye(num_rows=3, batch_shape=[batch_size])
    meas_mat = tf.concat([mml, mmr], axis=2)
    meas_unc = meas_mat[:, :3, :3] * 1e2

    num = tf.matmul(pcov, meas_mat, transpose_b=True)
    den = tf.matmul(meas_mat, num) + meas_unc

    den = tf.where(tf.is_nan(den), tf.ones_like(den), den)
    # num = tf.where(tf.is_nan(num), tf.ones_like(num), num)

    si = tf.py_func(np.linalg.pinv, [den], tf.float32)
    gain = tf.matmul(num, si)

    pre_res = measurement - tf.squeeze(tf.matmul(meas_mat, tf.expand_dims(state_est1, axis=2)), axis=2)

    state_update1 = state_est1 + tf.squeeze(tf.matmul(gain, tf.expand_dims(pre_res, axis=2)), axis=2)

    I_KH = tf.eye(6, batch_shape=[batch_size]) - tf.matmul(gain, meas_mat)
    pupdate = tf.matmul(tf.matmul(I_KH, pcov), I_KH, transpose_b=True) + tf.matmul(tf.matmul(gain, meas_unc), gain, transpose_b=True)

    return state_update1, pupdate, gain


def kalman_np(state, covariance, gmn, dt, measurement, batch_size):
    state_est1 = propagate(state, dt, gmn)

    I_3 = np.repeat(np.expand_dims(np.eye(3), axis=0), batch_size, axis=0)
    position = np.expand_dims(copy.copy(state_est1[:, :3]), axis=1)
    velocity = np.expand_dims(copy.copy(state_est1[:, 3:]), axis=1)
    R = np.linalg.norm(position, axis=2, keepdims=True)

    C = (gmn / (R ** 3)) * (3 * (position * np.transpose(position)) / R ** 2 - I_3)
    F = (3 * gmn / (R ** 5)) * (
        (np.transpose(position) * velocity) * (I_3 - (5 * (position * np.transpose(position))) / (R ** 2)) + (position * np.transpose(velocity)) +
        (velocity * np.transpose(position)))
    # dt2 = np.expand_dims(dt, axis=1)
    dt2 = dt
    p1l = I_3 + C / 2 * dt2 ** 2 + F / 6 * dt2 ** 3
    p1r = I_3 * dt2 + C / 6 * dt2 ** 3
    p1 = np.concatenate([p1l, p1r], axis=2)
    p2l = C * dt2 + F / 2 * dt2 ** 2
    p2r = I_3 + C / 2 * dt2 ** 2 + F / 3 * dt2 ** 3
    p2 = np.concatenate([p2l, p2r], axis=2)

    Pr = np.concatenate([p1, p2], axis=1)

    pcov = np.matmul(np.matmul(Pr, covariance), np.transpose(Pr))

    mmr = np.zeros(shape=[batch_size, 3, 3])
    mml = np.repeat(np.expand_dims(np.eye(3), axis=0), batch_size, axis=0)
    meas_mat = np.concatenate([mml, mmr], axis=2)
    meas_unc = meas_mat[:, :3, :3] * 1e2

    num = np.matmul(pcov, np.transpose(meas_mat))
    den = np.matmul(meas_mat, num) + meas_unc

    den = np.where(np.isnan(den), np.ones_like(den), den)
    # num = np.where(np.is_nan(num), np.ones_like(num), num)

    si = np.linalg.pinv([den])
    gain = np.matmul(num, si)

    pre_res = measurement - np.squeeze(np.matmul(meas_mat, np.expand_dims(state_est1, axis=2)), axis=2)

    state_update1 = state_est1 + np.squeeze(np.matmul(gain, np.expand_dims(pre_res, axis=2)), axis=2)

    I_KH = np.repeat(np.expand_dims(np.eye(6), axis=0), batch_size, axis=0) - np.matmul(gain, meas_mat)
    pupdate = np.matmul(np.matmul(I_KH, pcov), np.transpose(I_KH)) + np.matmul(np.matmul(gain, meas_unc), np.transpose(gain))

    return state_update1, pupdate, gain


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

    a = tf.where(a < 0, a+tf.ones_like(a)*pi_val, a)

    y_rae = tf.concat([r, a, e], axis=1)

    return y_rae


def ut_state_batch(X, Wm, Wc, R, num_state, batch_size, A, dt, sensor_ecef):
    # acc_est = tf.tile(tf.expand_dims(acc_est0, axis=1), [1, X.shape[1], 1])

    # Y = tf.transpose(propagatefb(X, dt, sensor_ecef), [0, 2, 1])
    Y = tf.matmul(A, X, transpose_b=True)  #  + tf.matmul(B, u[:, :, tf.newaxis])
    # Y = Y[:, :, :num_state]
    # Y = tf.transpose(Y, [0, 2, 1])
    y = tf.zeros([batch_size, num_state, 1], dtype=tf.float64)

    for q in range(Y.shape[2]):
        y += tf.expand_dims(tf.expand_dims(Wm[:, q], 1) * Y[:, :, q], axis=2)
    # y = tf.matmul(Y, Wm[:, :, tf.newaxis])
    # y = tf.reduce_mean(Y, axis=2, keepdims=True)
    # y = tf.expand_dims(Wm, axis=2) * Y
    Y1 = Y - tf.tile(y, [1, 1, Y.shape[2]])
    P = tf.matmul(tf.matmul(Y1, tf.matrix_diag(Wc)), tf.transpose(Y1, [0, 2, 1])) + R
    # P = tf.matmul(Y1, tf.transpose(Y1, [0, 2, 1])) + R

    return y, Y, P, Y1


def ut_state_batch_no_prop(X, Wm, Wc, R, num_state, batch_size, A, dt):
    # acc_est = tf.tile(tf.expand_dims(acc_est0, axis=1), [1, X.shape[1], 1])

    # Y = tf.transpose(propagatefb(X, dt), [0, 2, 1])
    # Y = tf.matmul(A, X, transpose_b=True) #  + tf.matmul(B, u[:, :, tf.newaxis])
    # Y = Y[:, :, :num_state]
    # Y = tf.transpose(Y, [0, 2, 1])

    Y = tf.transpose(X, [0, 2, 1])
    y = tf.zeros([batch_size, num_state, 1], dtype=tf.float64)

    for q in range(Y.shape[2]):
        y += tf.expand_dims(tf.expand_dims(Wm[:, q], 1) * Y[:, :, q], axis=2)
    # y = tf.matmul(Y, Wm[:, :, tf.newaxis])
    # y = tf.reduce_mean(Y, axis=2, keepdims=True)
    # y = tf.expand_dims(Wm, axis=2) * Y
    Y1 = Y - tf.tile(y, [1, 1, Y.shape[2]])
    P = tf.matmul(tf.matmul(Y1, tf.matrix_diag(Wc)), tf.transpose(Y1, [0, 2, 1])) + R
    # P = tf.matmul(Y1, tf.transpose(Y1, [0, 2, 1])) + R

    return y, Y, P, Y1


def ut_state_batch_np(X, Wm, Wc, R, num_state, batch_size, prop):
    # acc_est = tf.tile(tf.expand_dims(acc_est0, axis=1), [1, X.shape[1], 1])

    # Y = propagatefb(X, 0.1)
    Y = np.matmul(prop, np.transpose(X, [0, 2, 1]))
    # Y = Y[:, :, :num_state]
    # Y = tf.transpose(Y, [0, 2, 1])
    y = np.zeros([batch_size, num_state, 1], dtype=np.float64)

    for q in range(Y.shape[2]):
        y += np.expand_dims(np.expand_dims(Wm[:, q], 1) * Y[:, :, q], axis=2)
    # y = tf.matmul(Y, Wm[:, :, tf.newaxis])
    # y = tf.reduce_mean(Y, axis=2, keepdims=True)
    # y = tf.expand_dims(Wm, axis=2) * Y
    Y1 = Y - np.tile(y, [1, 1, Y.shape[2]])

    diag_eye = np.tile(np.eye(25)[np.newaxis, :, :], [batch_size, 1, 1]) * Wc[0, :]

    P = np.matmul(np.matmul(Y1, diag_eye), np.transpose(Y1, [0, 2, 1])) + R
    # P = tf.matmul(Y1, tf.transpose(Y1, [0, 2, 1])) + R

    return y, Y, P, Y1


def ut_state_batch_cw(X, weights, R, num_state, batch_size, prop):
    # acc_est = tf.tile(tf.expand_dims(acc_est0, axis=1), [1, X.shape[1], 1])

    # Y = propagatefb(X, 0.1)
    Y = tf.matmul(prop, X, transpose_b=True)
    # Y = Y[:, :, :num_state]
    # Y = tf.transpose(Y, [0, 2, 1])
    y = tf.zeros([batch_size, num_state, 1], dtype=tf.float64)

    weights = tf.tile(tf.cast(weights[:, tf.newaxis, :], tf.float64), [1, Y.shape[1], 1])
    # weights = tf.cast(weights, tf.float64)
    y = tf.reduce_sum((Y * weights), axis=2, keepdims=True) # / tf.reduce_sum(weights, keepdims=True)
    # y = tf.reduce_sum(Y * weights, axis=2, keepdims=True) / Y.shape[2].value
    # for q in range(Y.shape[2]):
    #     y += tf.expand_dims(tf.expand_dims(weights[:, q], 1) * Y[:, :, q], axis=2)
    # y = tf.matmul(Y, weights[:, :, tf.newaxis]) # / Y.shape[2].value
    # y = y / Y.shape[2].value
    # y = tf.reduce_mean(Y, axis=2, keepdims=True)
    # y = tf.expand_dims(Wm, axis=2) * Y
    Y1 = Y - tf.tile(y, [1, 1, Y.shape[2]])
    # P = tf.matmul(tf.matmul(Y1, tf.matrix_diag(weights)), tf.transpose(Y1, [0, 2, 1])) + R
    P = tf.matmul(Y1, tf.transpose(Y1, [0, 2, 1])) + R

    return y, Y, P, Y1, weights


def ut_meas(X, Wm, Wc, R, meas_mat, batch_size):
    # Y = X[:, :3, :]
    Y = tf.matmul(X, meas_mat, transpose_a=True, transpose_b=True)
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


def ut_meas_rae2(mu_pred, Wm, Wc, R, LLA, ecef_ref, pi_val, batch_size):
    # Y = X[:, :3, :]

    rae_list = list()
    for _ in range(mu_pred.shape[2].value):
        pos_part = tf.concat([mu_pred[:, 0, _, tf.newaxis], mu_pred[:, 4, _, tf.newaxis], mu_pred[:, 8, _, tf.newaxis]], axis=1)
        rae_state = eci_to_rae_tf(pos_part, LLA, ecef_ref, pi_val)
        rae_list.append(rae_state)

    Y = tf.stack(rae_list, axis=1)
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


def ut_meas_np(X, Wm, Wc, R, meas_mat, batch_size):
    # Y = X[:, :3, :]
    Y = np.matmul(np.transpose(X, [0, 2, 1]), np.transpose(meas_mat, [0, 2, 1]))
    Y = np.transpose(Y, [0, 2, 1])
    y = np.zeros([batch_size, Y.shape[1], 1], dtype=np.float64)
    for q in range(Y.shape[2]):
        y = y + np.expand_dims(np.expand_dims(Wm[:, q], 1) * Y[:, :, q], axis=2)
    # y = np.expand_dims(Wm, axis=2) * Y
    # y = np.reduce_mean(Y, axis=2, keepdims=True)
    Y1 = Y - np.tile(y, [1, 1, Y.shape[2]])

    diag_eye = np.tile(np.eye(25)[np.newaxis, :, :], [batch_size, 1, 1]) * Wc[0, :]

    P = (np.matmul(np.matmul(Y1, diag_eye), np.transpose(Y1, [0, 2, 1])) / (1 ** 2)) + R
    # P = np.matmul(Y1, np.transpose(Y1, [0, 2, 1])) + R

    return y, Y, P, Y1


def trans(A):
    return np.transpose(A, [0, 2, 1])


def unscented_kalman_np(batch_size, seqlen, initial_state, initial_covariance, measurement_list, prev_time, dt0, dt1, lla_datar):
    Ql = [None] * seqlen
    Sl = [None] * seqlen
    al = [None] * seqlen
    pl = [None] * seqlen
    meas_list = [None] * seqlen

    import pdb

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
    num_meas = 3

    m1 = np.concatenate([omp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp], axis=1).astype(np.float64)
    m2 = np.concatenate([zmp, zmp, zmp, zmp, omp, zmp, zmp, zmp, zmp, zmp, zmp, zmp], axis=1).astype(np.float64)
    m3 = np.concatenate([zmp, zmp, zmp, zmp, zmp, zmp, zmp, zmp, omp, zmp, zmp, zmp], axis=1).astype(np.float64)
    meas_mat = np.tile(np.expand_dims(np.concatenate([m1, m2, m3], axis=0), axis=0), [batch_size, 1, 1])

    alpha = 1e-1 * np.ones([batch_size, 1], dtype=np.float64)
    beta = 2. * np.ones([batch_size, 1], dtype=np.float64)
    k = 0. * np.ones([batch_size, 1], dtype=np.float64)

    L = num_state
    lam = alpha * (L + k) - L
    c1 = L + lam
    tmat = np.ones([1, 2 * num_state], dtype=np.float64)
    Wm = np.concatenate([(lam / c1), (0.5 / c1) * tmat], axis=1)
    Wc1 = np.expand_dims(copy.copy(Wm[:, 0]), axis=1) + (np.ones_like(alpha, dtype=np.float64) - (alpha) + beta)
    Wc = np.concatenate([Wc1, copy.copy(Wm[:, 1:])], axis=1)
    c = np.sqrt(c1)
    
    for q in range(seqlen):
        if q == 0:
            pstate_est = initial_state
            cov_est0 = initial_covariance
        else:
            pstate_est = Sl[q - 1]
            cov_est0 = Ql[q - 1]

        rae_meas = measurement_list[:, q, :]

        R = rae_meas[:, 0, np.newaxis]
        A = rae_meas[:, 1, np.newaxis]
        E = rae_meas[:, 2, np.newaxis]

        east = (R * np.sin(A) * np.cos(E))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        north = (R * np.cos(E) * np.cos(A))  # * ((tf.exp(tf.negative(tf.pow(sa, 2) / 2)) * tf.exp(tf.negative(tf.pow(se, 2) / 2))))
        up = (R * np.sin(E))  # * ((tf.exp(tf.negative(tf.pow(se, 2) / 2))))

        enu_meas = np.concatenate([east, north, up], axis=1)

        lat = lla_datar[:, 0, np.newaxis]
        lon = lla_datar[:, 1, np.newaxis]

        # Ti2e = np.zeros
        # t00 = -tf.sin(lon)
        # t01 = tf.cos(lon)
        # t10 = -tf.sin(lat) * tf.cos(lon)
        # t11 = -tf.sin(lat) * tf.sin(lon)
        # t12 = tf.cos(lat)
        # t20 = tf.cos(lat) * tf.cos(lon)
        # t21 = tf.cos(lat) * tf.sin(lon)
        # t22 = tf.sin(lat)

        uvw2enu = np.zeros([lat.shape[0], 3, 3])
        uvw2enu[:, 0, 0] = np.squeeze(-np.sin(lon), -1)
        uvw2enu[:, 0, 1] = np.squeeze(np.cos(lon), -1)
        uvw2enu[:, 0, 2] = 0
        uvw2enu[:, 1, 0] = np.squeeze(-np.sin(lat)*np.cos(lon), -1)
        uvw2enu[:, 1, 1] = np.squeeze(-np.sin(lat)*np.sin(lon), -1)
        uvw2enu[:, 1, 2] = np.squeeze(np.cos(lat), -1)
        uvw2enu[:, 2, 0] = np.squeeze(np.cos(lat) * np.cos(lon), -1)
        uvw2enu[:, 2, 1] = np.squeeze(np.cos(lat)*np.sin(lon), -1)
        uvw2enu[:, 2, 2] = np.squeeze(np.sin(lat), -1)

        enu2uvw = np.transpose(uvw2enu, [0, 2, 1])

        rae2enu = np.zeros([lat.shape[0], 3, 3])
        rae2enu[:, 0, 0] = np.squeeze(np.sin(A)*np.cos(E), -1)
        rae2enu[:, 0, 1] = np.squeeze(np.cos(A) * np.cos(E), -1)
        rae2enu[:, 0, 2] = np.squeeze(-np.sin(A) * np.sin(E), -1)
        rae2enu[:, 1, 0] = np.squeeze(np.cos(A)*np.cos(E), -1)
        rae2enu[:, 1, 1] = np.squeeze(-np.sin(A) * np.cos(E), -1)
        rae2enu[:, 1, 2] = np.squeeze(-np.cos(A)*np.sin(E), -1)
        rae2enu[:, 2, 0] = np.squeeze(np.sin(E), -1)
        rae2enu[:, 2, 1] = 0
        rae2enu[:, 2, 2] = np.squeeze(np.cos(E), -1)

        enu2rae = np.matmul(enu2uvw, rae2enu)
        
        cosPhi = np.cos(lat)
        sinPhi = np.sin(lat)
        cosLambda = np.cos(lon)
        sinLambda = np.sin(lon)

        tv = cosPhi * up - sinPhi * north
        wv = sinPhi * up + cosPhi * north
        uv = cosLambda * tv - sinLambda * east
        vv = sinLambda * tv + cosLambda * east

        cur_meas_temp = np.concatenate([uv, vv, wv], axis=1)

        if q > 0:
            dt = prev_time[:, q, :] - prev_time[:, q-1, :]
            dt = dt[:, :, np.newaxis]
        else:
            dt = dt0

        Qt, At, Bt, At2 = get_QP_np(dt, om, zm, I_3z, I_4z, zb,
                                 dimension=int(num_state / 3),
                                 sjix=om * 1 ** 2,
                                 sjiy=om * 1 ** 2,
                                 sjiz=om * 1 ** 2,
                                 aji=om * 1.)

        al[q] = At
        pl[q] = Qt
        # Rt = np.eye(3, dtype=np.float64) * 50
        # Rt = np.tile(Rt[np.newaxis, :, :], [batch_size, 1, 1])

        mu_pred = np.matmul(At, pstate_est[:, :, np.newaxis])
        cov_est0 = np.matmul(np.matmul(At, cov_est0), np.transpose(At, [0, 2, 1])) + Qt

        mu_pred_pos_uvw = np.matmul(meas_mat, mu_pred)

        # Ti2e = np.zeros(shape=[lla_datar.shape[0], 3, 3])
        # Ti2e[:, 0, 0] = np.squeeze(-np.sin(lon), -1)
        # Ti2e[:, 0, 1] = np.squeeze(np.cos(lon), -1)
        # Ti2e[:, 1, 0] = np.squeeze(-np.sin(lat) * np.cos(lon), -1)
        # Ti2e[:, 1, 1] = np.squeeze(-np.sin(lat) * np.sin(lon), -1)
        # Ti2e[:, 1, 2] = np.squeeze(np.cos(lat), -1)
        # Ti2e[:, 2, 0] = np.squeeze(np.cos(lat) * np.cos(lon), -1)
        # Ti2e[:, 2, 1] = np.squeeze(np.cos(lat) * np.sin(lon), -1)
        # Ti2e[:, 2, 2] = np.squeeze(np.sin(lat), -1)

        y_enu = np.squeeze(np.matmul(uvw2enu, mu_pred_pos_uvw), -1)

        y_rae = np.zeros_like(y_enu)
        y_rae[:, 0] = np.sqrt(y_enu[:, 0] * y_enu[:, 0] + y_enu[:, 1] * y_enu[:, 1] + y_enu[:, 2] * y_enu[:, 2])
        y_rae[:, 1] = np.arctan2(y_enu[:, 0], y_enu[:, 1])
        y_rae[:, 1] = np.where(y_rae[:, 1] < 0, (2 * np.pi) + y_rae[:, 1], y_rae[:, 1])
        y_rae[:, 2] = np.arcsin(y_enu[:, 2] / y_rae[:, 0])

        # east = y_enu[:, 0, np.newaxis]
        # north = y_enu[:, 1, np.newaxis]
        # up = y_enu[:, 2, np.newaxis]

        # y_ned = np.concatenate([north, east, np.negative(up)], axis=1)

        rng = np.sqrt(y_enu[:, 0] * y_enu[:, 0] + y_enu[:, 1] * y_enu[:, 1] + y_enu[:, 2] * y_enu[:, 2])
        az = np.arctan2(y_enu[:, 0], y_enu[:, 1])
        az = np.where(az < 0, az + np.ones_like(az) * (2 * np.pi), az)
        el = np.arcsin(y_enu[:, 2] / rng)

        # rae_pred0 = np.concatenate([rng[:, np.newaxis], az[:, np.newaxis], el[:, np.newaxis]], axis=1)
        #
        # rng = np.sqrt(y_ned[:, 0] * y_ned[:, 0] + y_ned[:, 1] * y_ned[:, 1] + y_ned[:, 2] * y_ned[:, 2])
        # az = np.arctan2(y_ned[:, 1], y_ned[:, 0])
        # az = np.where(az < 0, az + np.ones_like(az) * (2 * np.pi), az)
        # el = np.arccos(y_ned[:, 2] / rng)
        #
        # rae_pred = np.concatenate([rng[:, np.newaxis], az[:, np.newaxis], el[:, np.newaxis]], axis=1)
        #
        # rae_pred = rae_pred-np.pi/2

        # mmat = np.zeros(shape=[lla_datar.shape[0], 3, 3])
        # mmat[:, 0, 0] = np.cos(az)*np.sin(el)
        # mmat[:, 0, 1] = -np.sin(az)*np.sin(el)
        # mmat[:, 0, 2] = np.cos(az)*np.cos(el)
        # mmat[:, 1, 0] = np.sin(az)*np.sin(el)
        # mmat[:, 1, 1] = np.cos(az)*np.sin(el)
        # mmat[:, 1, 2] = np.sin(az)*np.cos(el)
        # mmat[:, 2, 0] = np.cos(el)
        # # mmat[:, 2, 1] = np.squeeze(np.zeros_like, -1)
        # mmat[:, 2, 2] = -np.sin(el)

        Rt = np.eye(3, dtype=np.float64)
        # r_diag = np.concatenate([np.ones_like(el[:, np.newaxis])*50, np.ones_like(el[:, np.newaxis])*5e-3, np.ones_like(el[:, np.newaxis])*0.01], axis=1)
        # Rt = np.eye(3, dtype=np.float64) * 50
        Rt = np.tile(Rt[np.newaxis, :, :], [batch_size, 1, 1])
        Rt[:, 0, 0] = np.ones_like(el)*50
        Rt[:, 1, 1] = np.ones_like(el)*5e-3 * R[:, 0]
        Rt[:, 2, 2] = np.ones_like(el)*0.001 * R[:, 0]
        # Rt = np.fill_diagonal(Rt, r_diag)

        enu_cov = np.matmul(np.matmul(trans(enu2rae), Rt), enu2rae)

        Rt = np.matmul(np.matmul(trans(uvw2enu), enu_cov), uvw2enu)

        # Am = np.expand_dims(c, axis=2) * qcholr
        # Y = np.tile(np.expand_dims(pstate_est, axis=2), [1, 1, 12])
        # X = np.concatenate([np.expand_dims(pstate_est, axis=2), Y + Am, Y - Am], axis=2)
        # X = np.transpose(X, [0, 2, 1])
        #
        # x1, X1, P1, X2 = ut_state_batch_np(X, Wm, Wc, Qt, num_state, batch_size, At)
        # z1, Z1, P2, Z2 = ut_meas_np(X1, Wm, Wc, Rt, meas_mat, batch_size)
        #
        diag_eye12 = np.tile(np.eye(12)[np.newaxis, :, :], [batch_size, 1, 1])

        # P12 = np.matmul(np.matmul(X2, diag_eye), np.transpose(Z2, [0, 2, 1]))
        #
        # gain = np.matmul(P12, np.linalg.inv(P2))
        pos_res1 = cur_meas_temp[:, :, np.newaxis] - np.matmul(meas_mat, mu_pred)
        # x = x1 + np.matmul(gain, pos_res2)
        #
        # cov_est_t0 = P1 - np.matmul(gain, np.transpose(P12, [0, 2, 1]))
        # cov_est_t = (cov_est_t0 + np.transpose(cov_est_t0, [0, 2, 1])) / 2

        sp1 = np.matmul(np.matmul(meas_mat, cov_est0), np.transpose(meas_mat, [0, 2, 1]))
        S = sp1 + Rt

        try:
            S_inv = np.linalg.inv(S)
        except:
            pdb.set_trace()
            pass
        # S_inv = pinv(S)
        gain = np.matmul(np.matmul(cov_est0, np.transpose(meas_mat, [0, 2, 1])), S_inv)

        # gain = tf.matmul(P12, tf.matrix_inverse(P2)) * cur_weight[:, tf.newaxis, :]

        mu_t = mu_pred + np.matmul(gain, pos_res1)

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
        meas_list[q] = cur_meas_temp
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

    return final_state, Ql, converted_meas, pred_state, pred_covariance, Qt, Rt


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
    
    q11j = (1 / (2 * aj7)) * ( ((aj5 * dt5) / 10) - ((aj4 * dt4) / 2) + ((4 * aj3 * dt3) / 3) + (2 * aj * dt) - (2 * aj2 * dt2) - 3 + (4 * emadt) + (2 * aj2 * dt2 * emadt) - np.exp(-2 * aj * dt) )
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
    sj = (emadt)

    # pj = q24
    # qj = q34
    # rj = om * dt
    # sj = om

    sj1 = 2 * sjix * aj
    sj2 = 2 * sjiy * aj
    sj3 = 2 * sjiz * aj

    # # C = GMt1 * ((3 * (np.matmul(pstate_est[:, :3, np.newaxis], pstate_est[:, :3, np.newaxis], transpose_b=True)) / rad_temp2) - I_3)#

    if dimension == 4:

        zeta1j = np.concatenate(
            [np.concatenate([q11j, q12j, q13j, q14j], axis=2), np.concatenate([q12j, q22j, q23j, q24j], axis=2), np.concatenate([q13j, q23j, q33j, q34j], axis=2), np.concatenate([q14j, q24j, q34j, q44j], axis=2)],
            axis=1) * sj1

        zeta2j = np.concatenate(
            [np.concatenate([q11j, q12j, q13j, q14j], axis=2), np.concatenate([q12j, q22j, q23j, q24j], axis=2), np.concatenate([q13j, q23j, q33j, q34j], axis=2), np.concatenate([q14j, q24j, q34j, q44j], axis=2)],
            axis=1) * sj2

        zeta3j = np.concatenate(
            [np.concatenate([q11j, q12j, q13j, q14j], axis=2), np.concatenate([q12j, q22j, q23j, q24j], axis=2), np.concatenate([q13j, q23j, q33j, q34j], axis=2), np.concatenate([q14j, q24j, q34j, q44j], axis=2)],
            axis=1) * sj3

        Q = np.concatenate([np.concatenate([zeta1j, I_4z, I_4z], axis=2), np.concatenate([I_4z, zeta2j, I_4z], axis=2), np.concatenate([I_4z, I_4z, zeta3j], axis=2)], axis=1)

        phi = np.concatenate([np.concatenate([om, dt, q34, pj], axis=2), np.concatenate([zm, om, dt, qj], axis=2), np.concatenate([zm, zm, om, rj], axis=2), np.concatenate([zm, zm, zm, sj], axis=2)], axis=1)

        A = np.concatenate([np.concatenate([phi, I_4z, I_4z], axis=2), np.concatenate([I_4z, phi, I_4z], axis=2), np.concatenate([I_4z, I_4z, phi], axis=2)], axis=1)

        # zeta1 = np.concatenate([np.concatenate([q11, q12, q13, q14], axis=2), np.concatenate([q12, q22, q23, q24], axis=2), np.concatenate([q13, q23, q33, q34], axis=2), np.concatenate([q14, q24, q34, q44], axis=2)], axis=1)
        # zeta2 = np.concatenate([np.concatenate([q11, q12, q13, q14], axis=2), np.concatenate([q12, q22, q23, q24], axis=2), np.concatenate([q13, q23, q33, q34], axis=2), np.concatenate([q14, q24, q34, q44], axis=2)], axis=1)
        # zeta3 = np.concatenate([np.concatenate([q11, q12, q13, q14], axis=2), np.concatenate([q12, q22, q23, q24], axis=2), np.concatenate([q13, q23, q33, q34], axis=2), np.concatenate([q14, q24, q34, q44], axis=2)], axis=1)
        # Q = np.scalar_mul(2, np.concatenate([np.concatenate([zeta1, I_4z, I_4z], axis=2), np.concatenate([I_4z, zeta2, I_4z], axis=2), np.concatenate([I_4z, I_4z, zeta3], axis=2)], axis=1)) \
        #     * np.cast(sji[:, :, np.newaxis], dtype=np.float64) * aj

        phi2 = np.concatenate([np.concatenate([om, dt, q34, q24], axis=2), np.concatenate([zm, om, dt, q34], axis=2), np.concatenate([zm, zm, om, dt], axis=2), np.concatenate([zm, zm, zm, om], axis=2)], axis=1)
        A2 = np.concatenate([np.concatenate([phi2, I_4z, I_4z], axis=2), np.concatenate([I_4z, phi2, I_4z], axis=2), np.concatenate([I_4z, I_4z, phi2], axis=2)], axis=1)

        tb = np.concatenate([np.concatenate([q34, q24], axis=2), np.concatenate([q44, q34], axis=2), np.concatenate([om, q44], axis=2), np.concatenate([zm, om], axis=2)], axis=1)

        B = np.concatenate([np.concatenate([tb, zb, zb], axis=2), np.concatenate([zb, tb, zb], axis=2), np.concatenate([zb, zb, tb], axis=2)], axis=1)

    elif dimension == 3:
        zeta1 = np.concatenate([np.concatenate([q22, q23, q24], axis=2), np.concatenate([q23, q33, q34], axis=2), np.concatenate([q24, q34, q44], axis=2)], axis=1)
        zeta2 = np.concatenate([np.concatenate([q22, q23, q24], axis=2), np.concatenate([q23, q33, q34], axis=2), np.concatenate([q24, q34, q44], axis=2)], axis=1)
        zeta3 = np.concatenate([np.concatenate([q22, q23, q24], axis=2), np.concatenate([q23, q33, q34], axis=2), np.concatenate([q24, q34, q44], axis=2)], axis=1)
        Q = np.concatenate([np.concatenate([zeta1, I_3z, I_3z], axis=2), np.concatenate([I_3z, zeta2, I_3z], axis=2), np.concatenate([I_3z, I_3z, zeta3], axis=2)], axis=1) * 2

        phi = np.concatenate([np.concatenate([om, dt, q34], axis=2), np.concatenate([zm, om, dt], axis=2), np.concatenate([zm, zm, dt], axis=2)], axis=1)
        A = np.concatenate([np.concatenate([phi, I_3z, I_3z], axis=2), np.concatenate([I_3z, phi, I_3z], axis=2), np.concatenate([I_3z, I_3z, phi], axis=2)], axis=1)
        B = A

    q = Q[0, :, :]
    q
    return Q, A, B, A2


def debias_measurement_tf():
    sr = tf.ones_like(rm[:, 0, tf.newaxis]) * 1
    sa = tf.ones_like(sr) * 1e-6
    se = tf.ones_like(sa) * 1e-6

    lb = tf.exp(-tf.pow(sa, 2))
    lb2 = tf.exp(-2*tf.pow(sa, 2))
    le = tf.exp(-tf.pow(se, 2))
    le2 = tf.exp(-2*tf.pow(se, 2))

    rxx = (-tf.pow(lb, 2)*tf.pow(le, 2)*tf.pow(R, 2)*tf.pow(tf.cos(A), 2)*tf.pow(tf.cos(E), 2)) + 0.25*(tf.pow(R, 2) + tf.pow(sr, 2))
    rxx = ((tf.pow(sr, 2)*tf.pow(tf.cos(tf.pow(E, 2)), 2)) + tf.pow(R, 2)) * (tf.pow(se, 2)*tf.sin(tf.pow(E, 2))*tf.cos(tf.pow(A, 2)) + tf.pow(sa, 2)*tf.cos(tf.pow(E, 2))*tf.sin(tf.pow(A, 2)))

    ryy = ((tf.pow(sr, 2)*tf.cos(tf.pow(E, 2))*tf.sin(tf.pow(A, 2))) + tf.pow(R, 2)) * (tf.pow(se, 2)*tf.sin(tf.pow(E, 2))*tf.sin(tf.pow(A, 2)) + tf.pow(sa, 2)*tf.cos(tf.pow(E, 2))*tf.cos(tf.pow(A, 2)))

    rzz = (tf.pow(sr, 2)*tf.sin(tf.pow(E, 2))) + tf.pow(R, 2)*tf.pow(se, 2)*tf.cos(tf.pow(E, 2))

    rxy = (1/2) * (tf.pow(sr, 2)*tf.cos(tf.pow(E, 2))*tf.sin(2*A) + tf.pow(R, 2)*(tf.pow(se, 2)*tf.sin(tf.pow(E, 2))*tf.sin(2*A) - tf.pow(sa, 2)*tf.cos(tf.pow(E, 2))*tf.sin(2*A)))

    rxz = (1/2) * (tf.pow(sr, 2)*tf.sin(2*E)*tf.cos(A) - tf.pow(R, 2)*tf.pow(se, 2)*tf.sin(2*E)*tf.cos(A))

    ryz = (1/2) * (tf.pow(sr, 2)*tf.sin(2*E)*tf.sin(A) - tf.pow(R, 2)*tf.pow(se, 2)*tf.sin(2*E)*tf.sin(A))

    rzer = tf.zeros_like(rxx[:, tf.newaxis, tf.newaxis])

    self.rd1 = tf.concat([tf.concat([rxx[:, tf.newaxis, tf.newaxis], rxy[:, tf.newaxis, tf.newaxis], rxz[:, tf.newaxis, tf.newaxis]], axis=2),
                          tf.concat([rxy[:, tf.newaxis, tf.newaxis], ryy[:, tf.newaxis, tf.newaxis], ryz[:, tf.newaxis, tf.newaxis]], axis=2),
                          tf.concat([rxz[:, tf.newaxis, tf.newaxis], ryz[:, tf.newaxis, tf.newaxis], rzz[:, tf.newaxis, tf.newaxis]], axis=2)], axis=1)

    self.rd1 = tf.squeeze(self.rd1, -1)
    

def debias_measurement_np(R, A, E):
    
    sr = np.ones_like(rm[:, 0, np.newaxis]) * 1
    sa = np.ones_like(sr) * 1e-6
    se = np.ones_like(sa) * 1e-6

    lb = np.exp(-np.power(sa, 2))
    lb2 = np.exp(-2*np.power(sa, 2))
    le = np.exp(-np.power(se, 2))
    le2 = np.exp(-2*np.power(se, 2))

    # rxx = (-np.power(lb, 2)*np.power(le, 2)*np.power(R, 2)*np.power(np.cos(A), 2)*np.power(np.cos(E), 2)) + 0.25*(np.power(R, 2) + np.power(sr, 2))
    rxx = ((np.power(sr, 2)*np.power(np.cos(np.power(E, 2)), 2)) + np.power(R, 2)) * (np.power(se, 2)*np.sin(np.power(E, 2))*np.cos(np.power(A, 2)) + np.power(sa, 2)*np.cos(np.power(E, 2))*np.sin(np.power(A, 2)))

    ryy = ((np.power(sr, 2)*np.cos(np.power(E, 2))*np.sin(np.power(A, 2))) + np.power(R, 2)) * (np.power(se, 2)*np.sin(np.power(E, 2))*np.sin(np.power(A, 2)) + np.power(sa, 2)*np.cos(np.power(E, 2))*np.cos(np.power(A, 2)))

    rzz = (np.power(sr, 2)*np.sin(np.power(E, 2))) + np.power(R, 2)*np.power(se, 2)*np.cos(np.power(E, 2))

    rxy = (1/2) * (np.power(sr, 2)*np.cos(np.power(E, 2))*np.sin(2*A) + np.power(R, 2)*(np.power(se, 2)*np.sin(np.power(E, 2))*np.sin(2*A) - np.power(sa, 2)*np.cos(np.power(E, 2))*np.sin(2*A)))

    rxz = (1/2) * (np.power(sr, 2)*np.sin(2*E)*np.cos(A) - np.power(R, 2)*np.power(se, 2)*np.sin(2*E)*np.cos(A))

    ryz = (1/2) * (np.power(sr, 2)*np.sin(2*E)*np.sin(A) - np.power(R, 2)*np.power(se, 2)*np.sin(2*E)*np.sin(A))

    rzer = np.zeros_like(rxx[:, np.newaxis, np.newaxis])

    rd1 = np.concatenate([np.concatenate([rxx[:, np.newaxis, np.newaxis], rxy[:, np.newaxis, np.newaxis], rxz[:, np.newaxis, np.newaxis]], axis=2),
                                  np.concatenate([rxy[:, np.newaxis, np.newaxis], ryy[:, np.newaxis, np.newaxis], ryz[:, np.newaxis, np.newaxis]], axis=2),
                                  np.concatenate([rxz[:, np.newaxis, np.newaxis], ryz[:, np.newaxis, np.newaxis], rzz[:, np.newaxis, np.newaxis]], axis=2)], axis=1)

    rd1 = np.squeeze(rd1, -1)


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
    m11 = tf.sin(az) * tf.cos(el)
    m12 = -tf.cos(az) * tf.sin(el)
    m20 = tf.sin(el)
    m22 = tf.cos(el)

    tz = tf.zeros_like(el)

    rae_to_enu = tf.concat([tf.concat([m00, m01, m02], axis=2), tf.concat([m10, m11, m12], axis=2), tf.concat([m20, tz, m22], axis=2)], axis=1)

    return rae_to_enu

