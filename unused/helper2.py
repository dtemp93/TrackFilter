import copy
import pdb

import matplotlib.pyplot as plt
import tensorflow as tf
# from scipy.signal._savitzky_golay import *
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
import numpy as np


def safe_div(numerator, denominator, name):
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.truediv(numerator, denominator),
      tf.zeros_like(numerator),
      name=name)


def swish(x, name='swish'):
    with tf.name_scope(name) as scope:
        x = tf.nn.sigmoid(x) * x
    return x


def cosx(x, name=None):
  with ops.name_scope(name, "Cos", [x]) as name:
    x = gen_math_ops.cos(x, name=name) - x
    return x


def sinc(x):
    atzero = tf.divide(tf.sin(x), 1)
    atother = tf.divide(tf.sin(x), x)
    value = tf.where(tf.equal(x, 0), atzero, atother)
    return value


def sinca(x, name=None):
    with ops.name_scope(name, 'sinc', [x]) as name:
        x = sinc(x) + x
    return x


def prepare_batch(minibatch_index, X, Y, S, seq_len, batch_size, new_batch):
    if new_batch is True:
        xf0 = X[int(minibatch_index * batch_size): int((minibatch_index + 1) * batch_size)]
        yf0 = Y[int(minibatch_index * batch_size): int((minibatch_index + 1) * batch_size)]
        sf0 = S[int(minibatch_index * batch_size): int((minibatch_index + 1) * batch_size)]

        ll = copy.copy(xf0.shape[0])

        XL = list()
        SL = list()
        YL = list()
        PS = list()
        PM = list()
        PT = list()
        PMeta = list()
        IT = list()
        IM = list()
        IS = list()
        length_list = np.zeros(shape=[batch_size, 1])
        for i in range(ll):
            yf = yf0[i, :, :2]
            m = ~(yf == 0).all(1)
            yf = yf[m]
            length_list[i, 0] = yf.shape[0]

        max_length = np.max(length_list)

        for i in range(ll):
            yf = yf0[i, :, :]
            sf = sf0[i, :, :]
            xf = xf0[i, :, :]

            m = ~(yf[:, :2] == 0).all(1)
            yf = yf[m]
            sf = sf[m]
            xf = xf[m]

            time_temp = copy.copy(xf[:, 0])
            # max_tt = np.max(time_temp)
            min_tt = np.min(time_temp)

            # xf[:, 0] = copy.copy((time_temp - min_tt) / (max_tt - min_tt))
            xf[:, 0] = copy.copy(time_temp - min_tt)

            delta = int(max_length - yf.shape[0])

            z1 = np.zeros(shape=[delta, xf.shape[1]])
            z2 = np.zeros(shape=[delta, sf.shape[1]])
            z3 = np.zeros(shape=[delta, yf.shape[1]])

            z1n = np.concatenate([xf, z1], axis=0)
            z2n = np.concatenate([sf, z2], axis=0)
            z3n = np.concatenate([yf, z3], axis=0)

            yf = np.expand_dims(z3n, axis=0)
            sf = np.expand_dims(z2n, axis=0)
            xf = np.expand_dims(z1n, axis=0)
            
            imeas = 3
            meas00 = copy.copy(xf[:, :imeas, 1:])
            state00 = copy.copy(yf[:, :imeas, :])
            time00 = copy.copy(xf[:, :imeas, 0, np.newaxis])

            prev_state = copy.copy(state00[:, -1, np.newaxis, :])
            prev_meas = copy.copy(meas00[:, -1, np.newaxis, :])
            prev_time = copy.copy(time00[:, -1, np.newaxis, 0, np.newaxis])
            prev_meta = copy.copy(sf[:, -1, np.newaxis, 0, np.newaxis])

            x = copy.copy(xf[:, 1 + imeas:, :])
            y = copy.copy(yf[:, 1 + imeas:, :])
            s = copy.copy(sf[:, 1 + imeas:, :])

            XL.append(x)
            YL.append(y)
            SL.append(s)
            PS.append(prev_state)
            PM.append(prev_meas)
            PT.append(prev_time)
            PMeta.append(prev_meta)
            IM.append(meas00)
            IS.append(state00)
            IT.append(time00)

            del x, y, s, xf, yf, sf, meas00, time00

        xout = np.concatenate(XL, axis=0)
        yout = np.concatenate(YL, axis=0)
        sout = np.concatenate(SL, axis=0)
        psout = np.concatenate(PS, axis=0)
        pmout = np.concatenate(PM, axis=0)
        ptout = np.concatenate(PT, axis=0)
        pmetaout = np.concatenate(PMeta, axis=0)
        im0out = np.concatenate(IM, axis=0)
        is0out = np.concatenate(IS, axis=0)
        it0out = np.concatenate(IT, axis=0)

        # current_time = xout[:, :seq_len, 0, np.newaxis]
        # current_x = xout[:, :seq_len, 1:]
        # current_y = yout[:, :seq_len, :]
        # current_meta = sout[:, :seq_len, :]

        if seq_len > yout.shape[1]:
            delta = int(seq_len - yout.shape[1])

            # z1 = np.zeros(shape=[delta, xout.shape[2]])
            # z2 = np.zeros(shape=[delta, sout.shape[2]])
            # z3 = np.zeros(shape=[delta, yout.shape[2]])

            init_z_y = np.zeros([yout.shape[0], delta, yout.shape[2]])
            init_z_x = np.zeros([xout.shape[0], delta, xout.shape[2]])
            init_z_s = np.zeros([sout.shape[0], delta, sout.shape[2]])

            xout = np.concatenate([xout, init_z_x], axis=1)
            yout = np.concatenate([yout, init_z_y], axis=1)
            sout = np.concatenate([sout, init_z_s], axis=1)

        tw = np.ceil(xout.shape[1] / seq_len)
        twd = int(tw * seq_len - xout.shape[1])

        init_z_y = np.zeros([yout.shape[0], twd, yout.shape[2]])
        init_z_x = np.zeros([xout.shape[0], twd, xout.shape[2]])
        init_z_s = np.zeros([sout.shape[0], twd, sout.shape[2]])

        xout = np.concatenate([xout, init_z_x], axis=1)
        yout = np.concatenate([yout, init_z_y], axis=1)
        sout = np.concatenate([sout, init_z_s], axis=1)

    else:
        xout = copy.copy(X)
        yout = copy.copy(Y)
        sout = copy.copy(S)
        psout = []
        pmout = []
        ptout = []
        pmetaout = []
        im0out = []
        is0out = []
        it0out = []
        current_time = []
        current_x = []
        current_y = []
        current_meta = []

    return xout, yout, sout, psout, pmout, ptout, pmetaout, im0out, is0out, it0out, max_length


def get_feed_time_asynch(x, y, meta, prev_x, prev_y, prev_time, prev_meta, max_seq, step, num_state, window_mode=True):

    # step = step - 1
    xt = copy.copy(prev_x)
    yt = copy.copy(prev_y)
    mt = copy.copy(prev_meta)
    # sl = copy.copy(prev_sl)
    time = copy.copy(prev_time)

    if window_mode:

        r1 = step * max_seq
        r2 = r1 + max_seq
        slc_length = max_seq

        if step > 0:
            xtemp = x[:, r1:r2, :]
            # xtemp = xtemp[:, np.newaxis, :]

            xt = np.concatenate([xt, xtemp[:, :, 1:]], axis=1)
            xt = copy.copy(xt[:, -slc_length:, :])

            mtemp = meta[:, r1:r2, :1]
            # mtemp = mtemp[:, np.newaxis, :]
            mt = np.concatenate([mt, mtemp], axis=1)

            mt = mt[:, -slc_length:, :]

            time_temp = copy.copy(xtemp[:, :, 0])
            time_temp = time_temp[:, :, np.newaxis]
            time = np.concatenate([time, time_temp], axis=1)
            time = time[:, -slc_length:, :]

            ytemp = y[:, r1:r2, :num_state]
            # ytemp = ytemp[:, np.newaxis, :]

            yt = np.concatenate([yt, ytemp], axis=1)
            yt0 = copy.copy(yt[:, -slc_length:, :])
        else:
            xt = x[:, r1:r2, 1:]
            mt = mt[:, r1:r2, :]
            time = x[:, r1:r2, 0, np. newaxis]
            yt0 = y[:, r1:r2, :]
    else:
        slc_length = 1
        xtemp = x[:, step, :]
        xtemp = xtemp[:, np.newaxis, :]

        xt = np.concatenate([xt, xtemp[:, :, 1:]], axis=1)
        xt = copy.copy(xt[:, -slc_length:, :])

        mtemp = meta[:, step, :1]
        mtemp = mtemp[:, np.newaxis, :]
        mt = np.concatenate([mt, mtemp], axis=1)

        mt = mt[:, -slc_length:, :]

        time_temp = copy.copy(xtemp[:, :, 0])
        time_temp = time_temp[:, :, np.newaxis]
        time = np.concatenate([time, time_temp], axis=1)
        time = time[:, -slc_length:, :]

        ytemp = y[:, step, :num_state]
        ytemp = ytemp[:, np.newaxis, :]

        yt = np.concatenate([yt, ytemp], axis=1)
        yt0 = copy.copy(yt[:, -slc_length:, :])

    # xt = current measurements
    # yt0 = current truth state
    # It = Identity matrix
    # time = current time
    # pst = previous truth state
    # mt = current meta data
    # seqlen = sequence length indicator
    # seqweight = sequence weight indicator
    # prev_measurement = previous measurement
    # y_pred = next truth truth (for prediction)
    return xt, yt0, time, mt


def hwg(values, alpha=0.26, beta=0.19):
    #  Holt-Winters default parameters
    hw_alpha = 0.26  # Based on robust optimization in Gelper 2007,
    hw_beta = 0.19  # for Gaussian, fat tail, and outlier data.

    batch, row, col = values.shape
    alphac = 1 - alpha
    betac = 1 - beta
    tmp = values
    l = np.zeros((batch, row, col))
    l[:, 0, :] = tmp[:, 0, :]
    b = np.zeros((batch, row, col))
    for i in range(1, row):
        l[:, i, :] = (alpha * tmp[:, i, :]) + (alphac * (l[:, i - 1, :] + b[:, i - 1, :]))
        ldelta = l[:, i, :] - l[:, i - 1, :]
        b[:, i, :] = (beta * ldelta) + (betac * b[:, i - 1, :])
    values_out = l + b

    return values_out


def get_zero_state(layers, units, batch_size, n, std=0.3):
    state_list = list()
    if n == 2:
        for i in range(layers):
            s = list()
            s.append(np.zeros(shape=(batch_size, units), dtype=np.float32) + np.random.normal(loc=0.0, scale=std))
            # s.append(np.zeros(shape=(batch_size, units), dtype=np.float64) + np.random.normal(loc=0.0, scale=0.05))
            state_list.append(tuple(s))
    elif n == 3:
        state_list = np.zeros(shape=(layers, batch_size, units), dtype=np.float32) + np.random.normal(loc=0.0, scale=std)
        # s.append(np.zeros(shape=(batch_size, units), dtype=np.float64) + np.random.normal(loc=0.0, scale=0.05))
    elif n == 4:
        state_list = np.zeros(shape=(batch_size, units), dtype=np.float32) + np.random.normal(loc=0.0, scale=std)
        # s.append(np.zeros(shape=(batch_size, units), dtype=np.float64) + np.random.normal(loc=0.0, scale=0.05))
    elif n == 5:
        for i in range(layers):
            s = list()
            s.append(np.zeros(shape=(batch_size, units), dtype=np.float32) + np.random.normal(loc=0.0, scale=std))
            s.append(np.zeros(shape=(batch_size, units), dtype=np.float32) + np.random.normal(loc=0.0, scale=std))
            state_list.append(tuple(s))
        state_list.append(np.zeros(shape=(batch_size, int(units / 2)), dtype=np.float64))
        state_list.append(np.zeros(shape=(), dtype=np.int32))
        state_list.append(np.zeros(shape=(batch_size, 5), dtype=np.float64))
        state_list.append(tuple())
    return state_list


def plot_all2(out_plot_X, out_plot_F, out_plot_P, x, clean_meas, y, q_plot, q_plott, time_plotter, tstep, plot_path, ecef_ref, mean_y):
    # ll2 = np.linspace(0, out_plot_X.shape[1] - 1, out_plot_X.shape[1])

    cov_col = 'g'
    meas_col = 'b'
    meas_col2 = 'y'
    p1_col = 'm'
    pf_col = 'k'
    truth_col = 'r'
    f_col = 'g'

    yt = copy.copy(y[0, :, 3:])
    m = ~(yt == 0).all(1)
    yf = yt[m]
    seq = yf.shape[0]
    p_max = seq

    dim = out_plot_X.shape[2]
    if dim == 12:
        w = 1
    else:
        w = 0
    # p_max = seq

    # coeffs = savgol_coeffs(max_seq, polyorder=3, deriv=0, delta=0.1)
    # coeffs = coeffs[np.newaxis, :, np.newaxis]

    # pred_output0s = convolve(out_plot_X, coeffs, mode="constant")
    # pox1 = fit_edges_polyfit(out_plot_X[:, :, 0], max_seq, 2, 0, 0.1, 1, pred_output0s[:, :, 0])
    # pox2 = fit_edges_polyfit(out_plot_X[:, :, 1], max_seq, 2, 0, 0.1, 1, pred_output0s[:, :, 1])
    # pox3 = fit_edges_polyfit(out_plot_X[:, :, 2], max_seq, 2, 0, 0.1, 1, pred_output0s[:, :, 2])
    # pov1 = fit_edges_polyfit(out_plot_X[:, :, 3], max_seq, 2, 0, 0.1, 1, pred_output0s[:, :, 3])
    # pov2 = fit_edges_polyfit(out_plot_X[:, :, 4], max_seq, 2, 0, 0.1, 1, pred_output0s[:, :, 4])
    # pov3 = fit_edges_polyfit(out_plot_X[:, :, 5], max_seq, 2, 0, 0.1, 1, pred_output0s[:, :, 5])
    # poa1 = fit_edges_polyfit(out_plot_X[:, :, 6], max_seq, 2, 0, 0.1, 1, pred_output0s[:, :, 6])
    # poa2 = fit_edges_polyfit(out_plot_X[:, :, 7], max_seq, 2, 0, 0.1, 1, pred_output0s[:, :, 7])
    # poa3 = fit_edges_polyfit(out_plot_X[:, :, 8], max_seq, 2, 0, 0.1, 1, pred_output0s[:, :, 8])
    #
    # out_plot_X = np.concatenate([pox1[:, :, np.newaxis], pox2[:, :, np.newaxis], pox3[:, :, np.newaxis],
    #                                pov1[:, :, np.newaxis], pov2[:, :, np.newaxis], pov3[:, :, np.newaxis],
    #                                poa1[:, :, np.newaxis], poa2[:, :, np.newaxis], poa3[:, :, np.newaxis],
    #                                out_plot_X[:, :, 9:]], axis=2)

    try:

        # out_plot_X2 = denormalize_statenp(copy.copy(np.nan_to_num(out_plot_X)))
        # out_plot_P2 = denormalize_statenp(copy.copy(np.nan_to_num(out_plot_P)))
        # out_plot_F2 = denormalize_statenp(copy.copy(np.nan_to_num(out_plot_F)))
        out_plot_X2 = copy.copy(np.nan_to_num(out_plot_X)) * 1
        out_plot_P2 = copy.copy(np.nan_to_num(out_plot_P)) * 1
        out_plot_F2 = copy.copy(np.nan_to_num(out_plot_F)) * 1

        # pos = copy.copy(out_plot_X2[0, 1:p_max, :3])
        # altitude_est = np.linalg.norm(pos, axis=1) - np.ones_like(pos[:, 0]) * 6378137

        x2 = copy.copy(x) * 1
        y2 = copy.copy(y) * 1
        clean_meas = copy.copy(clean_meas)
        # y2 = denormalize_statenp(copy.copy(y))

        # p_max = np.min([p_max, pos.shape[0]])

        # altitude_truth = np.linalg.norm(y2[0, :max_step, :3], axis=1) - np.ones_like(pos[1:p_max, 0]) * 6378137
        # mag_vel_truth = np.linalg.norm(y2[0, 1:p_max, 3:6], axis=1) - np.ones_like(pos[:, 0]) * 6378137
        # mag_acc_truth = np.linalg.norm(y2[0, 1:p_max, 6:], axis=1) - np.ones_like(pos[:, 0]) * 6378137

        # a0 = q_plot[0, 900, :, :]
        # a1 = q_plott[0, 900, :, :]

        q_plot2 = np.sqrt(np.power(copy.copy(q_plot), 2)) * 1
        q_plott2 = np.sqrt(np.power(copy.copy(q_plott), 2)) * 1

        # q_plot2 = np.sqrt(np.power(copy.copy(q_plot), 2)) * 6378100**2
        # q_plott2 = np.sqrt(np.power(copy.copy(q_plott), 2)) * 6378100**2

        # a02 = q_plot2[0, 900, :, :]
        # a12 = q_plott2[0, 900, :, :]
        n_cov_truth = int(q_plott.shape[2])
        n_cov = int(q_plot.shape[2])

        if n_cov_truth == 12:
            qt_pos_x = q_plott2[0, 1:p_max, 0, 0]
            qt_pos_y = q_plott2[0, 1:p_max, 4, 4]
            qt_pos_z = q_plott2[0, 1:p_max, 8, 8]
            qt_vel_x = q_plott2[0, 1:p_max, 1, 1]
            qt_vel_y = q_plott2[0, 1:p_max, 5, 5]
            qt_vel_z = q_plott2[0, 1:p_max, 9, 9]
            qt_acc_x = q_plott2[0, 1:p_max, 2, 2]
            qt_acc_y = q_plott2[0, 1:p_max, 6, 6]
            qt_acc_z = q_plott2[0, 1:p_max, 10, 10]
            qt_jer_x = q_plott2[0, 1:p_max, 3, 3]
            qt_jer_y = q_plott2[0, 1:p_max, 7, 7]
            qt_jer_z = q_plott2[0, 1:p_max, 11, 11]
            
        elif n_cov == 6:
            qt_pos_x = q_plott2[0, 1:p_max, 0, 0]
            qt_pos_y = q_plott2[0, 1:p_max, 2, 2]
            qt_pos_z = q_plott2[0, 1:p_max, 4, 4]
            qt_vel_x = q_plott2[0, 1:p_max, 1, 1]
            qt_vel_y = q_plott2[0, 1:p_max, 3, 3]
            qt_vel_z = q_plott2[0, 1:p_max, 5, 5]

        if n_cov == 12:
            q_pos_x = q_plot2[0, 1:p_max, 0, 0]
            q_pos_y = q_plot2[0, 1:p_max, 4, 4]
            q_pos_z = q_plot2[0, 1:p_max, 8, 8]
            q_vel_x = q_plot2[0, 1:p_max, 1, 1]
            q_vel_y = q_plot2[0, 1:p_max, 5, 5]
            q_vel_z = q_plot2[0, 1:p_max, 9, 9]
            q_acc_x = q_plot2[0, 1:p_max, 2, 2]
            q_acc_y = q_plot2[0, 1:p_max, 6, 6]
            q_acc_z = q_plot2[0, 1:p_max, 10, 10]
            q_jer_x = q_plot2[0, 1:p_max, 3, 3]
            q_jer_y = q_plot2[0, 1:p_max, 7, 7]
            q_jer_z = q_plot2[0, 1:p_max, 11, 11]
        else:
            q_pos_x = q_plot2[0, 1:p_max, 0, 0]
            q_pos_y = q_plot2[0, 1:p_max, 1, 1]
            q_pos_z = q_plot2[0, 1:p_max, 2, 2]
            q_vel_x = q_plot2[0, 1:p_max, 3, 3]
            q_vel_y = q_plot2[0, 1:p_max, 4, 4]
            q_vel_z = q_plot2[0, 1:p_max, 5, 5]

        # idxo = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        # idxo2 = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        # q_plott2 = q_plott2[:, :, idxo]
        # q_plot2 = q_plot2[:, :, idxo]

        # q_plot2[:, 1:p_max, 0, 0] = np.sqrt(np.square(q_plot2[:, 1:p_max, 0, 0] - out_plot_X2[0, 1:p_max, 0]))
        # q_plot2[:, 1:p_max, 1, 1] = np.sqrt(np.square(q_plot2[:, 1:p_max, 1, 1] - out_plot_X2[0, 1:p_max, 1]))
        # q_plot2[:, 1:p_max, 2, 2] = np.sqrt(np.square(q_plot2[:, 1:p_max, 2, 2] - out_plot_X2[0, 1:p_max, 2]))
        # q_plot2[:, 1:p_max, 3, 3] = np.sqrt(np.square(q_plot2[:, 1:p_max, 3, 3] - out_plot_X2[0, 1:p_max, 3]))
        # q_plot2[:, 1:p_max, 4, 4] = np.sqrt(np.square(q_plot2[:, 1:p_max, 4, 4] - out_plot_X2[0, 1:p_max, 4]))
        # q_plot2[:, 1:p_max, 5, 5] = np.sqrt(np.square(q_plot2[:, 1:p_max, 5, 5] - out_plot_X2[0, 1:p_max, 5]))
        # q_plot2[:, 1:p_max, 6, 6] = np.sqrt(np.square(q_plot2[:, 1:p_max, 6, 6] - out_plot_X2[0, 1:p_max, 6]))
        # q_plot2[:, 1:p_max, 7, 7] = np.sqrt(np.square(q_plot2[:, 1:p_max, 7, 7] - out_plot_X2[0, 1:p_max, 7]))
        # q_plot2[:, 1:p_max, 8, 8] = np.sqrt(np.square(q_plot2[:, 1:p_max, 8, 8] - out_plot_X2[0, 1:p_max, 8]))

        # q_plott2[:, 1:p_max, 0, 0] = q_plot2[:, 1:p_max, 0, 0] - out_plot_X2[0, 1:p_max, 0]
        # q_plott2[:, 1:p_max, 1, 1] = q_plot2[:, 1:p_max, 0, 1] - out_plot_X2[0, 1:p_max, 1]
        # q_plott2[:, 1:p_max, 2, 2] = q_plot2[:, 1:p_max, 0, 2] - out_plot_X2[0, 1:p_max, 2]
        # q_plott2[:, 1:p_max, 3, 3] = q_plot2[:, 1:p_max, 0, 3] - out_plot_X2[0, 1:p_max, 3]
        # q_plott2[:, 1:p_max, 4, 4] = q_plot2[:, 1:p_max, 0, 4] - out_plot_X2[0, 1:p_max, 4]
        # q_plott2[:, 1:p_max, 5, 5] = q_plot2[:, 1:p_max, 0, 5] - out_plot_X2[0, 1:p_max, 5]
        # q_plott2[:, 1:p_max, 6, 6] = q_plot2[:, 1:p_max, 0, 6] - out_plot_X2[0, 1:p_max, 6]
        # q_plott2[:, 1:p_max, 7, 7] = q_plot2[:, 1:p_max, 0, 7] - out_plot_X2[0, 1:p_max, 7]
        # q_plott2[:, 1:p_max, 8, 8] = q_plot2[:, 1:p_max, 0, 8] - out_plot_X2[0, 1:p_max, 8]

        # tp = time_plotter[0, 01:p_max, :]

        # tp = x[0, time_shift:max_step, 0]
        tp = time_plotter[0, 1:p_max] # * tstep * 0.1
        min_t = tp[0]
        max_t = tp[-1]

        # tp = copy.copy(x[0, :, 0])

        errorK = np.sqrt((out_plot_P2[0, 1:p_max, :] - y2[0, 1:p_max, :]) ** 2)  # one steep in past
        errorX = np.sqrt((out_plot_X2[0, 1:p_max, :] - y2[0, 1:p_max, :]) ** 2)  # current state estimate
        # errorF = np.sqrt((out_plot_F2[0, 1:p_max, :] - y2[0, 1:p_max, :]) ** 2)  # current state estimate
        errorM = np.sqrt((x2[0, 1:p_max, :] - y2[0, 1:p_max, :3])**2)

        # if np.any(errorM >= 10):
        #     # pdb.set_trace()
        #     errorM
        #     if not os.path.isfile('./tmp/shelve.pkl'):
        #         filename = './tmp/shelve.pkl'
        #         with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
        #             pickle.dump([x2, y2, out_plot_P2, out_plot_X2, out_plot_F2, time_plotter, p_max, q_plot, q_plott], f)

        errorM2 = np.sqrt((clean_meas[0, 1:p_max, :] - y2[0, 1:p_max, :3])**2)

        # maxX = np.max(errorX, axis=0)
        # qtemp = np.mean(q_plot2[0, -250:, :, :], axis=1)
        # maxQ = np.zeros([6])
        # maxQ[0] = qtemp[0, 0]
        # maxQ[1] = qtemp[1, 1]
        # maxQ[2] = qtemp[2, 2]
        # maxQ[3] = qtemp[3, 3]
        # maxQ[4] = qtemp[4, 4]
        # maxQ[5] = qtemp[5, 5]
        # scale = maxX[:6] / maxQ
        # q_plot2 = q_plot2 * scale
    except:
        pdb.set_trace()
        pass

    # errorX = out_plot_X2[0, :tstep, :] - y2[0, :tstep, :]
    # errorX = np.expand_dims(errorX, axis=0)
    # errorX = hwg(errorX, alpha=0.26, beta=0.19)
    # errorX = np.abs(np.squeeze(errorX))
    # errorF = out_plot_F[0, :, :] - y[0, :, :]
    # errorP = out_plot_P[0, :, :] - y[0, :, :]

    ############################################################################################################################
    # try:
    #     plt.figure()
    #     plt.interactive(False)
    #     alt1 = plt.plot(tp, altitude_est, pf_col, marker='o', label='Altitude Estimate', lw=1.1, ms=1.5)
    #     alt2 = plt.plot(tp, altitude_truth, truth_col, marker='o', label='Altitude Truth', lw=1.1, ms=1.5)
    #     # truth1 = plt.plot(tp, y2[0, 0:tstep, 0], 'r', label='Truth X', lw=1.1)
    #     # predict1 = plt.plot(tp2, out_plot_F2[0, :, 0], 'g', label='Initial Prediction X', lw=1.1)
    #     # predict1p = plt.plot(tp2, out_plot_P2[0, :, 0], 'm', label='Propagated State X', lw=1.1)
    #     # predict1b = plt.plot(tp3, out_plot_X2[0, max_seq:, 0], 'k', label='Final Prediction X', lw=1.1)
    #
    #     # predict1m = plt.plot(tp2, avg_plot[0, :, 0], 'm', label='Predicted X F', lw=1.1)
    #     plt.legend()
    #     plt.title('Altitude', fontsize=12)
    #     plt.ylabel('Meters', fontsize=12)
    #     plt.xlim(min_t, max_t)
    #     plt.ylim(0, 70000)
    #     # plt.ylim(0, 1)
    #     plt.tight_layout(pad=0)
    #     mng = plt.get_current_fig_manager()
    #     mng.full_screen_toggle()
    #     plt.savefig(plot_path + '/' + str(p_max) + '_Altitude_results.png')
    #     # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    #     plt.close()
    # except:
    #     pdb.set_trace()
    #     pass
    ############################################################################################################################

    # ############################################################################################################################
    # plt.figure()
    # plt.subplot(12,1,1)
    # plt.plot(tp, q_plot2[0, 1:p_max, 0, 0], cov_col, label='Variance Position', lw=1.5)
    # plt.plot(tp, q_plott2[0, 1:p_max, 0, 0], 'r', label='Variance Position Truth', lw=1.1)
    # # plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    # # plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    # plt.plot(tp, errorX[:, 0], pf_col, label='Error Final')
    # plt.legend()
    # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 0]) + 2.0 * np.std(errorX[:, 0]))
    # # plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    # plt.title('X Position Variance', fontsize=12)
    #
    # plt.subplot(12,1,2)
    # plt.plot(tp, q_plot2[0, 1:p_max, 3 + w, 3 + w], cov_col, label='Variance Position', lw=1.5)
    # plt.plot(tp, q_plott2[0, 1:p_max, 3 + w, 3 + w], 'r', label='Variance Position Truth', lw=1.1)
    # # plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    # # plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    # plt.plot(tp, errorX[:, 1], pf_col, label='Error Final')
    # plt.legend()
    # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 0]) + 2.0 * np.std(errorX[:, 0]))
    # # plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    # plt.title('Y Position Variance', fontsize=12)
    #
    # plt.subplot(313)
    # plt.plot(tp, q_plot2[0, 1:p_max, 6 + w * 2, 6 + w * 2], cov_col, label='Variance Position', lw=1.5)
    # plt.plot(tp, q_plott2[0, 1:p_max, 6 + w * 2, 6 + w * 2], 'r', label='Variance Position Truth', lw=1.1)
    # # plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    # # plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    # plt.plot(tp, errorX[:, 2], pf_col, label='Error Final')
    # plt.legend()
    # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 0]) + 2.0 * np.std(errorX[:, 0]))
    # # plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    # plt.title('Z Position Variance', fontsize=12)
    #
    # # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 3]) + 4.0 * np.std(errorX[:, 3]))
    # plt.tight_layout(pad=0)
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.savefig(plot_path + '/' + str(p_max) + '_position_cov_.png')
    # # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    # plt.close()
    # ############################################################################################################################

    # ############################################################################################################################
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(tp, q_plot2[0, 1:p_max, 0, 0], cov_col, label='Variance Position', lw=1.5)
    # plt.plot(tp, q_plott2[0, 1:p_max, 0, 0], 'r', label='Variance Position Truth', lw=1.1)
    # # plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    # # plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    # plt.plot(tp, errorX[:, 0], pf_col, label='Error Final')
    # plt.legend()
    # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 0]) + 2.0 * np.std(errorX[:, 0]))
    # # plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    # plt.title('X Position Variance', fontsize=12)
    #
    # plt.subplot(312)
    # plt.plot(tp, q_plot2[0, 1:p_max, 3 + w, 3 + w], cov_col, label='Variance Position', lw=1.5)
    # plt.plot(tp, q_plott2[0, 1:p_max, 3 + w, 3 + w], 'r', label='Variance Position Truth', lw=1.1)
    # # plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    # # plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    # plt.plot(tp, errorX[:, 1], pf_col, label='Error Final')
    # plt.legend()
    # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 0]) + 2.0 * np.std(errorX[:, 0]))
    # # plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    # plt.title('Y Position Variance', fontsize=12)
    #
    # plt.subplot(313)
    # plt.plot(tp, q_plot2[0, 1:p_max, 6 + w * 2, 6 + w * 2], cov_col, label='Variance Position', lw=1.5)
    # plt.plot(tp, q_plott2[0, 1:p_max, 6 + w * 2, 6 + w * 2], 'r', label='Variance Position Truth', lw=1.1)
    # # plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    # # plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    # plt.plot(tp, errorX[:, 2], pf_col, label='Error Final')
    # plt.legend()
    # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 0]) + 2.0 * np.std(errorX[:, 0]))
    # # plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    # plt.title('Z Position Variance', fontsize=12)
    #
    # # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 3]) + 4.0 * np.std(errorX[:, 3]))
    # plt.tight_layout(pad=0)
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.savefig(plot_path + '/' + str(p_max) + '_position_cov_.png')
    # # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    # plt.close()
    # ############################################################################################################################

    # ############################################################################################################################
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(tp, q_plot2[0, 1:p_max, 1, 1], cov_col, label='Variance Velocity', lw=1.5)
    # plt.plot(tp, q_plott2[0, 1:p_max, 1, 1], 'r', label='Variance Velocity Truth', lw=1.1)
    # # plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    # # plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    # plt.plot(tp, errorX[:, 3], pf_col, label='Error Final')
    # plt.legend()
    # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 0]) + 2.0 * np.std(errorX[:, 0]))
    # # plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    # plt.title('X Velocity Variance', fontsize=12)
    #
    # plt.subplot(312)
    # plt.plot(tp, q_plot2[0, 1:p_max, 4+w, 4+w], cov_col, label='Variance Velocity', lw=1.5)
    # plt.plot(tp, q_plott2[0, 1:p_max, 4+w, 4+w], 'r', label='Variance Velocity Truth', lw=1.1)
    # # plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    # # plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    # plt.plot(tp, errorX[:, 4], pf_col, label='Error Final')
    # plt.legend()
    # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 0]) + 2.0 * np.std(errorX[:, 0]))
    # # plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    # plt.title('Y Velocity Variance', fontsize=12)
    #
    # plt.subplot(313)
    # plt.plot(tp, q_plot2[0, 1:p_max, 7+w*2, 7+w*2], cov_col, label='Variance Velocity', lw=1.5)
    # plt.plot(tp, q_plott2[0, 1:p_max, 7+w*2, 7]+w*2, 'r', label='Variance Velocity Truth', lw=1.1)
    # # plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    # # plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    # plt.plot(tp, errorX[:, 5], pf_col, label='Error Final')
    # plt.legend()
    # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 0]) + 2.0 * np.std(errorX[:, 0]))
    # # plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    # plt.title('Z Velocity Variance', fontsize=12)
    #
    # # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 3]) + 4.0 * np.std(errorX[:, 3]))
    # plt.tight_layout(pad=0)
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.savefig(plot_path + '/' + str(p_max) + '_velocity_cov_.png')
    # # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    # plt.close()
    # ############################################################################################################################

    # ############################################################################################################################
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(tp, q_plot2[0, 1:p_max, 2, 2], cov_col, label='Variance Acceleration', lw=1.5)
    # plt.plot(tp, q_plott2[0, 1:p_max, 2, 2], 'r', label='Variance Acceleration Truth', lw=1.1)
    # # plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    # # plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    # plt.plot(tp, errorX[:, 6], pf_col, label='Error Final')
    # plt.legend()
    # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 0]) + 2.0 * np.std(errorX[:, 0]))
    # # plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    # plt.title('X Acceleration Variance', fontsize=12)
    #
    # plt.subplot(312)
    # plt.plot(tp, q_plot2[0, 1:p_max, 5+w, 5+w], cov_col, label='Variance Acceleration', lw=1.5)
    # plt.plot(tp, q_plott2[0, 1:p_max, 5+w, 5+w], 'r', label='Variance Acceleration Truth', lw=1.1)
    # # plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    # # plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    # plt.plot(tp, errorX[:, 7], pf_col, label='Error Final')
    # plt.legend()
    # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 0]) + 2.0 * np.std(errorX[:, 0]))
    # # plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    # plt.title('Y Acceleration Variance', fontsize=12)
    #
    # plt.subplot(313)
    # plt.plot(tp, q_plot2[0, 1:p_max, 8+w*2, 8+w*2], cov_col, label='Variance Acceleration', lw=1.5)
    # plt.plot(tp, q_plott2[0, 1:p_max, 8+w*2, 8+w*2], 'r', label='Variance Acceleration Truth', lw=1.1)
    # # plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    # # plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    # plt.plot(tp, errorX[:, 8], pf_col, label='Error Final')
    # plt.legend()
    # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 0]) + 2.0 * np.std(errorX[:, 0]))
    # # plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    # plt.title('Z Acceleration Variance', fontsize=12)
    #
    # # plt.xlim(min_t, max_t)
    # # plt.ylim(0, np.mean(errorX[:, 3]) + 4.0 * np.std(errorX[:, 3]))
    # plt.tight_layout(pad=0)
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.savefig(plot_path + '/' + str(p_max) + '_Acceleration_cov_.png')
    # # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    # plt.close()
    # ############################################################################################################################
    
    # plt.figure()
    # plt.interactive(False)
    # plt.subplot(311)
    # truthvel1 = plt.plot(tp, y2[0, 1:p_max, 9], truth_col, label='Truth Jerk', lw=1.1)
    # # meas3 = plt.plot(tp, x2[0, 01:p_max, 3], 'b', marker='o', label='Measurement Xdot', lw=1.1, ms=1.5)
    # predictv1 = plt.plot(tp, out_plot_F2[0, 1:p_max, 9], f_col, label='Predicted Xdot F', lw=1.1)
    # predictv1p = plt.plot(tp, out_plot_P2[0, 1:p_max, 9], p1_col, label='Propagated State Xdot', lw=1.1)
    # predictv1b = plt.plot(tp, out_plot_X2[0, 1:p_max, 9], pf_col, label='Predicted Xdot X', lw=1.1)
    # plt.legend()
    # plt.title('X Jerk', fontsize=12)
    # plt.ylabel('M / S^3', fontsize=12)
    # plt.xlim(min_t, max_t)
    # # plt.ylim(np.min(y2[0, 1:p_max, 3]) - 1, np.max(y2[0, 1:p_max, 9]) + 1)
    #
    # plt.subplot(312)
    # truthacc1 = plt.plot(tp, y2[0, 1:p_max, 10] * 1, 'r', label='Truth Jerk', lw=1.1)
    # predictv1 = plt.plot(tp, out_plot_F2[0, 1:p_max, 10] * 1, 'g', label='Predicted Zdot2 F', lw=1.1)
    # predictv1p = plt.plot(tp, out_plot_P2[0, 1:p_max, 10], 'm', label='Propagated State Zdot2', lw=1.1)
    # predictv1b = plt.plot(tp, out_plot_X2[0, 1:p_max, 10] * 1, 'k', label='Predicted Zdot2 X', lw=1.1)
    # plt.legend()
    # plt.title('Y Jerk', fontsize=12)
    # plt.ylabel('M / S^3', fontsize=12)
    # plt.xlim(min_t, max_t)
    # # plt.ylim(np.min(y2[0, 1:p_max, 6]) - 1, np.max(y2[0, 1:p_max, 6]) + 1)
    #
    # plt.subplot(313)
    # truthacc1 = plt.plot(tp, y2[0, 1:p_max, 11] * 1, 'r', label='Truth Jerk', lw=1.1)
    # predictv1 = plt.plot(tp, out_plot_F2[0, 1:p_max, 11] * 1, 'g', label='Predicted Zdot2 F', lw=1.1)
    # predictv1p = plt.plot(tp, out_plot_P2[0, 1:p_max, 11], 'm', label='Propagated State Jerk', lw=1.1)
    # predictv1b = plt.plot(tp, out_plot_X2[0, 1:p_max, 11] * 1, 'k', label='Predicted Zdot2 X', lw=1.1)
    # plt.legend()
    # plt.title('Z Jerk', fontsize=12)
    # plt.ylabel('M / S^3', fontsize=12)
    # plt.xlim(min_t, max_t)
    # # plt.ylim(np.min(y2[0, 1:p_max, 6]) - 1, np.max(y2[0, 1:p_max, 6]) + 1)
    #
    # plt.tight_layout(pad=0)
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.savefig(plot_path + '/' + str(p_max) + '_Jerk_results.png')
    # # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    # plt.close()
    #
    # ############################################################################################################################
    # plt.figure()
    # plt.interactive(False)
    # meas1 = plt.plot(tp, x2[0, 1:p_max, 0], meas_col, marker='o', label='Measurement X', lw=1.1, ms=1.5)
    # meas2 = plt.plot(tp, clean_meas[0, 1:p_max, 0], meas_col2, marker='o', label='Measurement X', lw=1.1, ms=1.5)
    # truth1 = plt.plot(tp, y2[0, 1:p_max, 0], truth_col, label='Truth X', lw=1.1)
    # predict1 = plt.plot(tp, out_plot_F2[0, 1:p_max, 0], f_col, label='Initial Prediction X', lw=1.1)
    # predict1p = plt.plot(tp, out_plot_P2[0, 1:p_max, 0], p1_col, label='Propagated State X', lw=1.1)
    # predict1b = plt.plot(tp, out_plot_X2[0, 1:p_max, 0], pf_col, label='Final Prediction X', lw=1.1)
    #
    # # predict1m = plt.plot(tp2, avg_plot[0, :, 0], 'm', label='Predicted X F', lw=1.1)
    # plt.legend()
    # plt.title('X Position', fontsize=12)
    # plt.ylabel('Meters', fontsize=12)
    # plt.xlim(min_t, max_t)
    # plt.ylim(np.min(y2[0, 1:p_max, 0]), np.max(y2[0, 1:p_max, 0]))
    # # plt.ylim(0, 1)
    # plt.tight_layout(pad=0)
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.savefig(plot_path + '/' + str(tstep) + '_X_results.png')
    # # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    # plt.close()
    # ############################################################################################################################

    ############################################################################################################################
    plt.figure()
    plt.interactive(False)
    plt.subplot(211)
    plt.plot(tp, y2[0, 1:p_max, 3], truth_col, label='Truth Xdot', lw=1.1)
    plt.plot(tp, out_plot_F2[0, 1:p_max, 3], f_col, label='Predicted Xdot F', lw=1.1)
    plt.plot(tp, out_plot_P2[0, 1:p_max, 3], p1_col, label='Propagated State Xdot', lw=1.1)
    plt.plot(tp, out_plot_X2[0, 1:p_max, 3], pf_col, label='Predicted Xdot X', lw=1.1)
    plt.legend()
    plt.title('X Velocity', fontsize=12)
    plt.ylabel('M / S', fontsize=12)
    plt.xlim(min_t, max_t)
    plt.ylim(np.min(y2[0, 1:p_max, 3])-1, np.max(y2[0, 1:p_max, 3])+1)

    plt.subplot(212)
    plt.plot(tp, y2[0, 1:p_max, 6] * 1, 'r', label='Truth Zdot2', lw=1.1)
    plt.plot(tp, out_plot_F2[0, 1:p_max, 6] * 1, f_col, label='Predicted Zdot2 F', lw=1.1)
    plt.plot(tp, out_plot_P2[0, 1:p_max, 6], 'm', label='Propagated State Zdot2', lw=1.1)
    plt.plot(tp, out_plot_X2[0, 1:p_max, 6] * 1, 'k', label='Predicted Zdot2 X', lw=1.1)
    plt.legend()
    plt.title('X Acceleration', fontsize=12)
    plt.ylabel('M / S^2', fontsize=12)
    plt.xlim(min_t, max_t)
    plt.ylim(np.min(y2[0, 1:p_max, 6])-1, np.max(y2[0, 1:p_max, 6])+1)

    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(plot_path + '/' + str(p_max) + '_Xdot_results.png')
    # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    plt.figure()
    plt.interactive(False)
    plt.subplot(211)
    plt.plot(tp, q_pos_x, cov_col, label='Variance Position', lw=1.5)
    plt.plot(tp, qt_pos_x, 'r', label='Variance Position Truth', lw=1.1)
    plt.plot(tp, errorM[:, 0], meas_col, label='Error Measurement')
    plt.plot(tp, errorM2[:, 0], meas_col2, label='Error Measurement')
    plt.plot(tp, errorK[:, 0], p1_col, label='Error Pre')
    plt.plot(tp, errorX[:, 0], pf_col, label='Error Final')
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorM[:, 0]) + 6.0 * np.std(errorM[:, 0]))
    plt.title('X Position Variance', fontsize=12)

    plt.subplot(212)
    plt.plot(tp, q_vel_x, cov_col, label='Variance Velocity', lw=1.5)
    plt.plot(tp, qt_vel_x, 'r', label='Variance Velocity Truth', lw=1.1)
    plt.plot(tp, errorK[:, 3], p1_col, label='Pre')
    plt.plot(tp, errorX[:, 3], pf_col, label='Final')
    plt.legend()
    plt.title('X Velocity Variance', fontsize=12)

    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorX[:, 3]) + 4.0 * np.std(errorX[:, 3]))
    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(plot_path + '/' + str(p_max) + '_X_cov.png')
    # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    plt.figure()
    plt.interactive(False)
    plt.subplot(211)
    if n_cov == 12:
        plt.plot(tp, q_acc_x, cov_col, label='Variance Acceleration', lw=1.5)
        plt.plot(tp, qt_acc_x, 'r', label='Variance Acceleration Truth', lw=1.1)

    plt.plot(tp, errorK[:, 6], p1_col, label='Error Pre')
    plt.plot(tp, errorX[:, 6], pf_col, label='Error Final')
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorX[:, 2]) + 3.0 * np.std(errorX[:, 2]))
    plt.title('X Acceleration Variance', fontsize=12)

    plt.subplot(212)
    if n_cov == 12:
        plt.plot(tp, q_jer_x, cov_col, label='Variance Jerk', lw=1.5)
        plt.plot(tp, qt_jer_x, 'r', label='Variance Jerk Truth', lw=1.1)
    plt.plot(tp, errorK[:, 9], p1_col, label='Pre')
    plt.plot(tp, errorX[:, 9], pf_col, label='Final')
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorX[:, 3]) + 3.0 * np.std(errorX[:, 3]))
    plt.title('X Jerk Variance', fontsize=12)

    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(plot_path + '/' + str(p_max) + '_X2_cov.png')
    # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    plt.figure()
    plt.interactive(False)
    plt.subplot(211)
    if n_cov == 12:
        plt.plot(tp, q_acc_y, cov_col, label='Variance Acceleration', lw=1.5)
        plt.plot(tp, qt_acc_y, 'r', label='Variance Acceleration Truth', lw=1.1)

    plt.plot(tp, errorK[:, 7], p1_col, label='Error Pre')
    plt.plot(tp, errorX[:, 7], pf_col, label='Error Final')
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorX[:, 6]) + 3.0 * np.std(errorX[:, 6]))
    plt.title('Y Acceleration Variance', fontsize=12)

    plt.subplot(212)
    if n_cov == 12:
        plt.plot(tp, q_jer_y, cov_col, label='Variance Jerk', lw=1.5)
        plt.plot(tp, qt_jer_y, 'r', label='Variance Jerk Truth', lw=1.1)
    plt.plot(tp, errorK[:, 10], p1_col, label='Pre')
    plt.plot(tp, errorX[:, 10], pf_col, label='Final')
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorX[:, 7]) + 3.0 * np.std(errorX[:, 7]))
    plt.title('Y Jerk Variance', fontsize=12)

    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(plot_path + '/' + str(p_max) + '_Y2_cov.png')
    # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    plt.figure()
    plt.interactive(False)
    plt.subplot(211)
    if n_cov == 12:
        plt.plot(tp, q_acc_z, cov_col, label='Variance Acceleration', lw=1.5)
        plt.plot(tp, qt_acc_z, 'r', label='Variance Acceleration Truth', lw=1.1)

    plt.plot(tp, errorK[:, 8], p1_col, label='Error Pre')
    plt.plot(tp, errorX[:, 8], pf_col, label='Error Final')
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorX[:, 10]) + 3.0 * np.std(errorX[:, 10]))
    plt.title('Z Acceleration Variance', fontsize=12)

    plt.subplot(212)
    if n_cov == 12:
        plt.plot(tp, q_jer_z, cov_col, label='Variance Jerk', lw=1.5)
        plt.plot(tp, qt_jer_z, 'r', label='Variance Jerk Truth', lw=1.1)
    plt.plot(tp, errorK[:, 11], p1_col, label='Pre')
    plt.plot(tp, errorX[:, 11], pf_col, label='Final')
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorX[:, 11]) + 3.0 * np.std(errorX[:, 11]))
    plt.title('XZ Jerk Variance', fontsize=12)

    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(plot_path + '/' + str(p_max) + '_Z2_cov.png')
    # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    plt.figure()
    plt.interactive(False)
    plt.plot(tp, x2[0, 1:p_max, 1], meas_col, marker='o', label='Measurement Y', lw=1.1, ms=1.5)
    plt.plot(tp, clean_meas[0, 1:p_max, 1], meas_col2, marker='o', label='Measurement Y', lw=1.1, ms=1.5)
    plt.plot(tp, y2[0, 1:p_max, 1], truth_col, label='Truth Y', lw=1.1)
    plt.plot(tp, out_plot_F2[0, 1:p_max, 1], f_col, label='Initial Prediction Y', lw=1.1)
    plt.plot(tp, out_plot_P2[0, 1:p_max, 1], p1_col, label='Propagated State Y', lw=1.1)
    plt.plot(tp, out_plot_X2[0, 1:p_max, 1], pf_col, label='Final Prediction Y', lw=1.1)
    plt.ylim(np.min(y2[0, 1:p_max, 1]), np.max(y2[0, 1:p_max, 1]))
    plt.legend()
    plt.title('Y Position', fontsize=12)
    plt.ylabel('Meters', fontsize=12)
    plt.xlim(min_t, max_t)

    # plt.ylim(0, 1)
    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(plot_path + '/' + str(p_max) + '_Y_results.png')
    # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    plt.figure()
    plt.interactive(False)
    plt.subplot(211)

    plt.plot(tp, y2[0, 1:p_max, 4], truth_col, label='Truth Ydot', lw=1.1)
    plt.plot(tp, out_plot_F2[0, 1:p_max, 4], f_col, label='Initial Prediction Ydot', lw=1.1)
    plt.plot(tp, out_plot_P2[0, 1:p_max, 4], p1_col, label='Propagated State Ydot', lw=1.1)
    plt.plot(tp, out_plot_X2[0, 1:p_max, 4], pf_col, label='Final Prediction Ydot', lw=1.1)
    plt.legend()
    plt.title('Y Velocity', fontsize=12)
    plt.ylabel('M / S', fontsize=12)
    plt.xlim(min_t, max_t)
    plt.ylim(np.min(y2[0, 1:p_max, 4])-1, np.max(y2[0, 1:p_max, 4])+1)

    plt.subplot(212)
    plt.plot(tp, y2[0, 1:p_max, 7] * 1, 'r', label='Truth Zdot2', lw=1.1)
    plt.plot(tp, out_plot_F2[0, 1:p_max, 7] * 1, 'g', label='Predicted Zdot2 F', lw=1.1)
    plt.plot(tp, out_plot_P2[0, 1:p_max, 7], 'm', label='Propagated State Zdot2', lw=1.1)
    plt.plot(tp, out_plot_X2[0, 1:p_max, 7] * 1, 'k', label='Predicted Zdot2 X', lw=1.1)
    plt.legend()
    plt.title('Y Acceleration', fontsize=12)
    plt.ylabel('M / S^2', fontsize=12)
    plt.xlim(min_t, max_t)
    plt.ylim(np.min(y2[0, 1:p_max, 7])-1, np.max(y2[0, 1:p_max, 7])+1)

    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(plot_path + '/' + str(tstep) + '_Ydot_results.png')
    # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    plt.figure()
    plt.interactive(False)
    plt.subplot(211)
    cov2 = plt.plot(tp, q_pos_y, cov_col, label='Variance Position', lw=1.5)
    cov2t = plt.plot(tp, qt_pos_y, 'r', label='Variance Position Truth', lw=1.1)
    errm = plt.plot(tp, errorM[:, 1], meas_col, label='Measurement')
    errm = plt.plot(tp, errorM2[:, 1], meas_col2, label='Measurement')
    errk = plt.plot(tp, errorK[:, 1], p1_col, label='Pre')
    err2 = plt.plot(tp, errorX[:, 1], pf_col, label='Final')
    # err2 = plt.plot(tp, errorF[:, 1], f_col, label='s')
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, 1]) + 2.0 * np.std(errorX[:, 1]))
    plt.ylim(0, np.mean(errorM[:, 1]) + 6.0 * np.std(errorM[:, 1]))
    plt.title('Y Position Variance', fontsize=12)

    plt.subplot(212)
    cov2v = plt.plot(tp, q_vel_y, cov_col, label='Variance Velocity', lw=1.5)
    cov2vt = plt.plot(tp, qt_vel_y, 'r', label='Variance Velocity Truth', lw=1.1)
    errk = plt.plot(tp, errorK[:, 4], p1_col, label='Pre')
    err2v = plt.plot(tp, errorX[:, 4], pf_col, label='Final')
    # err2v = plt.plot(tp, errorF[:, 4], f_col, label='s')
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorX[:, 4]) + 4.0 * np.std(errorX[:, 4]))
    plt.title('Y Velocity Variance', fontsize=12)
    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(plot_path + '/' + str(p_max) + '_Y_cov.png')
    # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    plt.figure()
    plt.interactive(False)
    # gain3 = plt.plot(ll2, k_plot[0, :, 2, 2], 'm', label='Gain')
    plt.plot(tp, x2[0, 1:p_max, 2], meas_col, marker='o', label='Measurement Z', lw=1.1, ms=1.5)
    plt.plot(tp, clean_meas[0, 1:p_max, 2], meas_col2, marker='o', label='Measurement Z', lw=1.1, ms=1.5)
    plt.plot(tp, y2[0, 1:p_max, 2], truth_col, label='Truth Z', lw=1.1)
    plt.plot(tp, out_plot_F2[0, 1:p_max, 2], 'g', label='Initial Prediction Z', lw=1.1)
    plt.plot(tp, out_plot_P2[0, 1:p_max, 2], p1_col, label='Propagated State Z', lw=1.1)
    plt.plot(tp, out_plot_X2[0, 1:p_max, 2], pf_col, label='Final Prediction Z', lw=1.1)
    # predict3m = plt.plot(tp2, avg_plot[0, :, 2], 'm', label='Predicted Z F', lw=1.1)
    plt.legend()
    plt.ylabel('Meters', fontsize=12)
    plt.title('Z Position', fontsize=12)
    plt.xlim(min_t, max_t)
    plt.ylim(np.min(y2[0, 1:p_max, 2]), np.max(y2[0, 1:p_max, 2]))
    # plt.ylim(0, 1)
    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(plot_path + '/' + str(p_max) + '_Z_results.png')
    # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    plt.figure()
    plt.interactive(False)
    plt.subplot(211)
    # gain1 = plt.plot(ll2, k_plot[0, :, 5, 5], 'm', label='Gain')
    # meas3 = plt.plot(tp, x2[0, 01:p_max, 5], 'b', marker='o', label='Measurement Zdot', lw=1.1, ms=1.5)
    truthvel3 = plt.plot(tp, y2[0, 1:p_max, 5], truth_col, label='Truth Zdot', lw=1.1)
    predictv3 = plt.plot(tp, out_plot_F2[0, 1:p_max, 5], f_col, label='Initial Prediction Zdot', lw=1.1)
    predictv3p = plt.plot(tp, out_plot_P2[0, 1:p_max, 5], p1_col, label='Propagated State Zdot', lw=1.1)
    predictv3b = plt.plot(tp, out_plot_X2[0, 1:p_max, 5], pf_col, label='Final Prediction Zdot', lw=1.1)
    plt.legend()
    plt.ylabel('M / S', fontsize=12)
    plt.title('Z Velocity', fontsize=12)
    plt.xlim(min_t, max_t)
    plt.ylim(np.min(y2[0, 1:p_max, 5])-1, np.max(y2[0, 1:p_max, 5])+1)

    plt.subplot(212)
    truthacc1 = plt.plot(tp, y2[0, 1:p_max, 8] * 1, 'r', label='Truth Zdot2', lw=1.1)
    predictv1 = plt.plot(tp, out_plot_F2[0, 1:p_max, 8] * 1, 'g', label='Predicted Zdot2 F', lw=1.1)
    predictv1p = plt.plot(tp, out_plot_P2[0, 1:p_max, 8], 'm', label='Propagated State Zdot2', lw=1.1)
    predictv1b = plt.plot(tp, out_plot_X2[0, 1:p_max, 8] * 1, 'k', label='Predicted Zdot2 X', lw=1.1)
    plt.legend()
    plt.title('Z Acceleration', fontsize=12)
    plt.ylabel('M / S^2', fontsize=12)
    plt.xlim(min_t, max_t)
    plt.ylim(np.min(y2[0, 1:p_max, 8])-1, np.max(y2[0, 1:p_max, 8])+1)
    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(plot_path + '/' + str(p_max) + '_Zdot_results.png')
    # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    plt.figure()
    plt.interactive(False)
    plt.subplot(211)
    plt.plot(tp, q_pos_z, cov_col, label='Variance Position', lw=1.5)
    plt.plot(tp, qt_pos_z, 'r', label='Variance Position Truth', lw=1.1)
    plt.plot(tp, errorM[:, 2], meas_col, label='Measurement')
    plt.plot(tp, errorM2[:, 2], meas_col2, label='Measurement')
    plt.plot(tp, errorK[:, 2], p1_col, label='Pre')
    plt.plot(tp, errorX[:, 2], pf_col, label='Final')
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorM[:, 2]) + 6.0 * np.std(errorM[:, 2]))
    plt.title('Z Position Variance', fontsize=12)

    plt.subplot(212)
    plt.plot(tp, q_vel_z, cov_col, label='Variance Velocity', lw=1.5)
    plt.plot(tp, qt_vel_z, 'r', label='Variance Velocity Truth', lw=1.1)
    plt.plot(tp, errorK[:, 5], 'm', label='Pre')
    plt.plot(tp, errorX[:, 5], pf_col, label='Final')
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorX[:, 5]) + 4.0*np.std(errorX[:, 5]))
    plt.title('Z Velocity Variance', fontsize=12)
    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(plot_path + '/' + str(p_max) + '_Z_cov.png')
    # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    plt.close()
    ############################################################################################################################
    ############################################################################################################################

    # plt.figure()
    # plt.subplot(311)
    # err1 = plt.plot(tp, errorX[:, 0], 'r', label='Error Position')
    # err1v = plt.plot(tp, errorX[:, 3], 'b', label='Error Velocity')
    # plt.title('X Error', fontsize=12)
    # plt.xlim(0, np.sum(all_dt[0, :, :]))
    # # mv = np.mean(errorX[:, 0])
    # # mstd = np.std(errorX[:, 0])
    # up1 = np.max(errorX[10:, 0])
    # try:
    #     plt.ylim(0, up1)
    # except:
    #     plt.ylim(0, 1)
    #
    # plt.subplot(312)
    # err2 = plt.plot(tp, errorX[:, 1], 'r', label='Error Position')
    # err2v = plt.plot(tp, errorX[:, 4], 'b', label='Error Velocity')
    # plt.title('Y Error', fontsize=12)
    # plt.xlim(0, np.sum(all_dt[0, :, :]))
    # up2 = np.max(errorX[10:, 1])
    # try:
    #     plt.ylim(0, up2)
    # except:
    #     plt.ylim(0, 1)
    # plt.subplot(313)
    # err3 = plt.plot(tp, errorX[:, 2], 'r', label='Error Position')
    # err3v = plt.plot(tp, errorX[:, 5], 'b', label='Error Velocity')
    # plt.title('Z Error', fontsize=12)
    # plt.xlim(0, np.sum(all_dt[0, :, :]))
    # up3 = np.max(errorX[10:, 2])
    # try:
    #     plt.ylim(0, up3)
    # except:
    #     plt.ylim(0, 1)
    #
    # # plt.ylim(-1.1 * np.abs(np.mean(q_plot[0, :, 0])), np.abs(np.mean(q_plot[0, :, 0])) * 1.1)
    # plt.legend()
    # plt.tight_layout(pad=0)
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.savefig(plot_path + '/' + str(tstep) + '_Residual.png')
    # # plt.savefig(plot_dir + 'epoch_' + str(epoch) + '_minibatch_' + str(minibatch_index) + '_' + str(tstep) + '_ECI.png')
    # plt.close()


def initialize_run_variables(batch_size, seq_len, num_state):

    count = 0

    RE = 1
    prev_cov = []
    if num_state == 12:
        prev_cov = np.zeros(shape=[batch_size, seq_len, num_state, num_state])
        sj2 = 10
        prev_cov[:, :, 0, 0] = (1000 / RE) ** 1
        prev_cov[:, :, 4, 4] = (1000 / RE) ** 1
        prev_cov[:, :, 8, 8] = (1000 / RE) ** 1

        prev_cov[:, :, 1, 1] = (5000 / RE) ** 1
        prev_cov[:, :, 5, 5] = (5000 / RE) ** 1
        prev_cov[:, :, 9, 9] = (5000 / RE) ** 1

        prev_cov[:, :, 2, 2] = (1000 / RE) ** 1
        prev_cov[:, :, 6, 6] = (1000 / RE) ** 1
        prev_cov[:, :, 10, 10] = (1000 / RE) ** 1

        prev_cov[:, :, 3, 3] = (sj2 / RE) ** 1
        prev_cov[:, :, 7, 7] = (sj2 / RE) ** 1
        prev_cov[:, :, 11, 11] = (sj2 / RE) ** 1
        
        # prev_cov = np.sqrt(prev_cov)
        # prev_cov[:, :, 0, 0] = (sig / RE) ** 1
        # prev_cov[:, :, 4, 4] = (sig / RE) ** 1
        # prev_cov[:, :, 8, 8] = (sig / RE) ** 1
        #
        # prev_cov[:, :, 1, 1] = ((2*sig2/(0.1**2)) / RE) ** 1
        # prev_cov[:, :, 5, 5] = ((2*sig2/(0.1**2)) / RE) ** 1
        # prev_cov[:, :, 9, 9] = ((2*sig2/(0.1**2)) / RE) ** 1
        #
        # prev_cov[:, :, 2, 2] = ((6*sig2/(0.1**4)) / RE) ** 1
        # prev_cov[:, :, 6, 6] = ((6*sig2/(0.1**4)) / RE) ** 1
        # prev_cov[:, :, 10, 10] = ((6*sig2/(0.1**4)) / RE) ** 1
        #
        # prev_cov[:, :, 3, 3] = (sj2 / RE) ** 1
        # prev_cov[:, :, 7, 7] = (sj2 / RE) ** 1
        # prev_cov[:, :, 11, 11] = (sj2 / RE) ** 1

        # prev_cov[:, :, 0, 1] = (sig2/0.1) / RE
        # prev_cov[:, :, 1, 0] = (sig2/0.1) / RE
        # prev_cov[:, :, 0, 2] = (sig2/(0.1**2)) / RE
        # prev_cov[:, :, 2, 0] = (sig2/(0.1**2)) / RE
        # prev_cov[:, :, 1, 2] = (3*sig2/(0.1**3)) / RE
        # prev_cov[:, :, 2, 1] = (3*sig2/(0.1**3)) / RE
        # prev_cov[:, :, 1, 3] = ((5/6)*sj2/(0.1**2)) / RE
        # prev_cov[:, :, 3, 1] = ((5/6)*sj2/(0.1**2)) / RE
        # prev_cov[:, :, 2, 3] = (sj2*0.1) / RE
        # prev_cov[:, :, 3, 2] = (sj2*0.1) / RE
        #
        # prev_cov[:, :, 4, 5] = (sig2 / 0.1) / RE
        # prev_cov[:, :, 5, 4] = (sig2 / 0.1) / RE
        # prev_cov[:, :, 4, 6] = (sig2 / (0.1 ** 2)) / RE
        # prev_cov[:, :, 6, 4] = (sig2 / (0.1 ** 2)) / RE
        # prev_cov[:, :, 5, 6] = (3 * sig2 / (0.1 ** 3)) / RE
        # prev_cov[:, :, 6, 5] = (3 * sig2 / (0.1 ** 3)) / RE
        # prev_cov[:, :, 5, 7] = ((5 / 6) * sj2 / (0.1 ** 2)) / RE
        # prev_cov[:, :, 7, 5] = ((5 / 6) * sj2 / (0.1 ** 2)) / RE
        # prev_cov[:, :, 6, 7] = (sj2 * 0.1) / RE
        # prev_cov[:, :, 7, 6] = (sj2 * 0.1) / RE
        #
        # prev_cov[:, :, 8, 9] = (sig2 / 0.1) / RE
        # prev_cov[:, :, 9, 8] = (sig2 / 0.1) / RE
        # prev_cov[:, :, 8, 10] = (sig2 / (0.1 ** 2)) / RE
        # prev_cov[:, :, 10, 8] = (sig2 / (0.1 ** 2)) / RE
        # prev_cov[:, :, 9, 10] = (3 * sig2 / (0.1 ** 3)) / RE
        # prev_cov[:, :, 10, 9] = (3 * sig2 / (0.1 ** 3)) / RE
        # prev_cov[:, :, 9, 11] = ((5 / 6) * sj2 / (0.1 ** 2)) / RE
        # prev_cov[:, :, 11, 9] = ((5 / 6) * sj2 / (0.1 ** 2)) / RE
        # prev_cov[:, :, 10, 11] = (sj2 * 0.1) / RE
        # prev_cov[:, :, 11, 10] = (sj2 * 0.1) / RE

    prev_R = np.zeros(shape=[batch_size, seq_len, 3, 3])

    prev_R[:, :, 0, 0] = (25 / RE) ** 1
    prev_R[:, :, 1, 1] = (25 / RE) ** 1
    prev_R[:, :, 2, 2] = (25 / RE) ** 1

    prev_Q = np.zeros(shape=[batch_size, seq_len, num_state, num_state])
    prev_Q[:, :, 0, 0] = (0.1 / RE) ** 1
    prev_Q[:, :, 4, 4] = (0.1 / RE) ** 1
    prev_Q[:, :, 8, 8] = (0.1 / RE) ** 1

    prev_Q[:, :, 1, 1] = (0.5 / RE) ** 1
    prev_Q[:, :, 5, 5] = (0.5 / RE) ** 1
    prev_Q[:, :, 9, 9] = (0.5 / RE) ** 1

    prev_Q[:, :, 2, 2] = (1 / RE) ** 1
    prev_Q[:, :, 6, 6] = (1 / RE) ** 1
    prev_Q[:, :, 10, 10] = (1 / RE) ** 1

    prev_Q[:, :, 3, 3] = (10 / RE) ** 1
    prev_Q[:, :, 7, 7] = (10 / RE) ** 1
    prev_Q[:, :, 11, 11] = (10 / RE) ** 1

    q_plot = list()
    q_plott = list()
    k_plot = list()
    out_plot_X = list()
    out_plot_F = list()
    out_plot_P = list()
    time_vals = list()
    meas_plot = list()
    truth_plot = list()
    Q_plot = list()
    R_plot = list()
    maha_plot = list()

    prev_x = list()
    prev_y = list()
    prev_sl = list()
    prev_meta = list()
    prev_time = list()

    return count, prev_x, prev_y, prev_sl, prev_meta, prev_time, prev_cov, prev_Q, prev_R, q_plot, q_plott, k_plot, out_plot_X, out_plot_F, out_plot_P, time_vals, meas_plot, truth_plot, Q_plot, R_plot, maha_plot


def _polyder(p, m):
    """Differentiate polynomials represented with coefficients.

    p must be a 1D or 2D array.  In the 2D case, each column gives
    the coefficients of a polynomial; the first row holds the coefficients
    associated with the highest power.  m must be a nonnegative integer.
    (numpy.polyder doesn't handle the 2D case.)
    """

    if m == 0:
        result = p
    else:
        n = len(p)
        if n <= m:
            result = np.zeros_like(p[:1, ...])
        else:
            dp = p[:-m].copy()
            for k in range(m):
                rng = np.arange(n - k - 1, m - k - 1, -1)
                dp *= rng.reshape((n - m,) + (1,) * (p.ndim - 1))
            result = dp
    return result


def pinv(a, rcond=None, validate_args=False, name=None):
  with tf.name_scope(name, 'pinv', [a, rcond]):
    a = tf.convert_to_tensor(a, name='a')

    if not a.dtype.is_floating:
      raise TypeError('Input `a` must have `float`-like `dtype` '
                      '(saw {}).'.format(a.dtype.name))
    if a.shape.ndims is not None:
      if a.shape.ndims < 2:
        raise ValueError('Input `a` must have at least 2 dimensions '
                         '(saw: {}).'.format(a.shape.ndims))
    elif validate_args:
      assert_rank_at_least_2 = tf.assert_rank_at_least(
          a, rank=2,
          message='Input `a` must have at least 2 dimensions.')
      with tf.control_dependencies([assert_rank_at_least_2]):
        a = tf.identity(a)

    dtype = a.dtype.as_numpy_dtype

    if rcond is None:
      def get_dim_size(dim):
        if a.shape.ndims is not None and a.shape[dim].value is not None:
          return a.shape[dim].value
        return tf.shape(a)[dim]
      num_rows = get_dim_size(-2)
      num_cols = get_dim_size(-1)
      if isinstance(num_rows, int) and isinstance(num_cols, int):
        max_rows_cols = float(max(num_rows, num_cols))
      else:
        max_rows_cols = tf.cast(tf.maximum(num_rows, num_cols), dtype)
      rcond = 10. * max_rows_cols * np.finfo(dtype).eps

    rcond = tf.convert_to_tensor(rcond, dtype=dtype, name='rcond')

    # Calculate pseudo inverse via SVD.
    # Note: if a is symmetric then u == v. (We might observe additional
    # performance by explicitly setting `v = u` in such cases.)
    [
        singular_values,         # Sigma
        left_singular_vectors,   # U
        right_singular_vectors,  # V
    ] = tf.linalg.svd(a, full_matrices=False, compute_uv=True)

    # Saturate small singular values to inf. This has the effect of make
    # `1. / s = 0.` while not resulting in `NaN` gradients.
    cutoff = rcond * tf.reduce_max(singular_values, axis=-1)
    singular_values = tf.where(
        singular_values > cutoff[..., tf.newaxis],
        singular_values,
        tf.fill(tf.shape(singular_values), np.array(np.inf, dtype)))

    # Although `a == tf.matmul(u, s * v, transpose_b=True)` we swap
    # `u` and `v` here so that `tf.matmul(pinv(A), A) = tf.eye()`, i.e.,
    # a matrix inverse has "transposed" semantics.
    a_pinv = tf.matmul(
        right_singular_vectors / singular_values[..., tf.newaxis, :],
        left_singular_vectors,
        adjoint_b=True)

    if a.shape.ndims is not None:
      a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))

    return a_pinv


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


def permute_xyz_dims(x_data, y_data):
    
    rn = np.random.rand()
    if rn < 0.333:
        perm = [0, 1, 2]
    elif rn >= 0.333 and rn < 0.6667:
        perm = [1, 0, 2]
    else:
        perm = [2, 1, 0]

    m1 = copy.copy(x_data[:, :, 1, np.newaxis])
    m2 = copy.copy(x_data[:, :, 2, np.newaxis])
    m3 = copy.copy(x_data[:, :, 3, np.newaxis])

    y1 = copy.copy(y_data[:, :, 0, np.newaxis])
    y2 = copy.copy(y_data[:, :, 1, np.newaxis])
    y3 = copy.copy(y_data[:, :, 2, np.newaxis])
    y4 = copy.copy(y_data[:, :, 3, np.newaxis])
    y5 = copy.copy(y_data[:, :, 4, np.newaxis])
    y6 = copy.copy(y_data[:, :, 5, np.newaxis])
    y7 = copy.copy(y_data[:, :, 6, np.newaxis])
    y8 = copy.copy(y_data[:, :, 7, np.newaxis])
    y9 = copy.copy(y_data[:, :, 8, np.newaxis])
    y10 = copy.copy(y_data[:, :, 9, np.newaxis])
    y11 = copy.copy(y_data[:, :, 10, np.newaxis])
    y12 = copy.copy(y_data[:, :, 11, np.newaxis])

    x_data[:, :, 1 + perm[0], np.newaxis] = m1
    x_data[:, :, 1 + perm[1], np.newaxis] = m2
    x_data[:, :, 1 + perm[2], np.newaxis] = m3

    y_data[:, :, perm[0], np.newaxis] = y1
    y_data[:, :, perm[1], np.newaxis] = y2
    y_data[:, :, perm[2], np.newaxis] = y3
    y_data[:, :, 3 + perm[0], np.newaxis] = y4
    y_data[:, :, 3 + perm[1], np.newaxis] = y5
    y_data[:, :, 3 + perm[2], np.newaxis] = y6
    y_data[:, :, 6 + perm[0], np.newaxis] = y7
    y_data[:, :, 6 + perm[1], np.newaxis] = y8
    y_data[:, :, 6 + perm[2], np.newaxis] = y9
    y_data[:, :, 9 + perm[0], np.newaxis] = y10
    y_data[:, :, 9 + perm[1], np.newaxis] = y11
    y_data[:, :, 9 + perm[2], np.newaxis] = y12

    return x_data, y_data


def loss_normalize(loss, update_condition, epsilon=1e-10):
    # Variable used for storing the scalar-value of the loss-function.
    loss_value = tf.Variable(1.0)

    # Expression used for either updating the scalar-value or
    # just re-using the old value.
    # Note that when loss_value.assign(loss) is evaluated, it
    # first evaluates the loss-function which is a TensorFlow
    # expression, and then assigns the resulting scalar-value to
    # the loss_value variable.
    loss_value_updated = tf.cond(update_condition,
                                 lambda: loss_value.assign(loss),
                                 lambda: loss_value)

    # Expression for the normalized loss-function.
    loss_normalized = loss / (loss_value_updated + epsilon)

    return loss_normalized