import copy
import numpy as np
import matplotlib
import warnings
import pdb

# matplotlib.use('Qt5Agg')
plt = matplotlib.pyplot


def append_output_vaulues(step, current_time, current_y, current_x, smooth_output, filter_output, refined_output, q_out_t, q_outs, q_out_refine, qt_out, rt_out, at_out,
                          out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot, testing=True, plt_idx=0):

    if testing is True:
        if step == 0:
            out_plot_smooth = smooth_output
            out_plot_filter = filter_output
            out_plot_refined = refined_output

            q_plott = q_out_t
            q_plots = q_outs
            q_plotr = q_out_refine
            qt_plot = qt_out
            rt_plot = rt_out
            at_plot = at_out

            time_vals = current_time
            meas_plot = current_x
            truth_plot = current_y

        else:
            new_vals_smooth = smooth_output
            new_vals_filter = filter_output
            new_vals_refined = refined_output

            new_qs = q_outs
            new_qt = q_out_t
            new_qr = q_out_refine
            new_qtt = qt_out
            new_rtt = rt_out
            new_att = at_out

            new_time = current_time[:, :, 0]
            new_meas = current_x
            new_truth = current_y

        if step > 0:
            out_plot_filter = np.concatenate([out_plot_filter, new_vals_filter], axis=1)
            out_plot_smooth = np.concatenate([out_plot_smooth, new_vals_smooth], axis=1)
            out_plot_refined = np.concatenate([out_plot_refined, new_vals_refined], axis=1)
            meas_plot = np.concatenate([meas_plot, new_meas], axis=1)
            truth_plot = np.concatenate([truth_plot, new_truth], axis=1)
            time_vals = np.concatenate([time_vals, new_time[:, :, np.newaxis]], axis=1)
            q_plots = np.concatenate([q_plots, new_qs], axis=1)
            q_plott = np.concatenate([q_plott, new_qt], axis=1)
            q_plotr = np.concatenate([q_plotr, new_qr], axis=1)
            qt_plot = np.concatenate([qt_plot, new_qtt], axis=1)
            rt_plot = np.concatenate([rt_plot, new_rtt], axis=1)
            at_plot = np.concatenate([at_plot, new_att], axis=1)
    else:
        if step == 0:
            out_plot_smooth = smooth_output[plt_idx, np.newaxis, :, :]
            out_plot_filter = filter_output[plt_idx, np.newaxis, :, :]
            out_plot_refined = refined_output[plt_idx, np.newaxis, :, :]

            q_plott = q_out_t[plt_idx, np.newaxis, :, :, :]
            q_plots = q_outs[plt_idx, np.newaxis, :, :, :]
            qt_plot = qt_out[plt_idx, np.newaxis, :, :]
            rt_plot = rt_out[plt_idx, np.newaxis, :, :]
            at_plot = at_out[plt_idx, np.newaxis, :]

            time_vals = current_time[plt_idx, np.newaxis, :, :]
            meas_plot = current_x[plt_idx, np.newaxis, :, :]
            truth_plot = current_y[plt_idx, np.newaxis, :, :]

        else:
            new_vals_smooth = smooth_output[plt_idx, :, :]
            new_vals_filter = filter_output[plt_idx, :, :]
            new_vals_refined = refined_output[plt_idx, :, :]

            new_qs = q_outs[plt_idx, :, :, :]
            new_qt = q_out_t[plt_idx, :, :, :]
            new_qtt = qt_out[plt_idx, :, :, :]
            new_rtt = rt_out[plt_idx, :, :, :]
            new_att = at_out[plt_idx, :, :]

            new_time = current_time[plt_idx, :, 0]
            new_meas = current_x[plt_idx, :, :]
            new_truth = current_y[plt_idx, :, :]

        if step > 0:
            out_plot_filter = np.concatenate([out_plot_filter, new_vals_filter[np.newaxis, :, :]], axis=1)
            out_plot_smooth = np.concatenate([out_plot_smooth, new_vals_smooth[np.newaxis, :, :]], axis=1)
            out_plot_refined = np.concatenate([out_plot_refined, new_vals_refined[np.newaxis, :, :]], axis=1)
            meas_plot = np.concatenate([meas_plot, new_meas[np.newaxis, :, :]], axis=1)
            truth_plot = np.concatenate([truth_plot, new_truth[np.newaxis, :, :]], axis=1)
            time_vals = np.concatenate([time_vals, new_time[np.newaxis, :, np.newaxis]], axis=1)
            q_plots = np.concatenate([q_plots, new_qs[np.newaxis, :, :, :]], axis=1)
            q_plott = np.concatenate([q_plott, new_qt[np.newaxis, :, :, :]], axis=1)
            qt_plot = np.concatenate([qt_plot, new_qtt[np.newaxis, :, :, :]], axis=1)
            rt_plot = np.concatenate([rt_plot, new_rtt[np.newaxis, :, :, :]], axis=1)
            at_plot = np.concatenate([at_plot, new_att[np.newaxis, :, :]], axis=1)

    return out_plot_filter, out_plot_smooth, out_plot_refined, meas_plot, truth_plot, time_vals, q_plots, q_plott, q_plotr, qt_plot, rt_plot, at_plot


def create_cov_diag(covariance, p_max):
    idx = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    cov = np.zeros([covariance.shape[0], covariance.shape[1]])
    count = 0
    for i in range(len(idx)):
        count += 1
        temp = covariance[:p_max, idx, idx]
        cov[:, count] = temp

    return cov


def comparison_plot(out_plot_X, out_plot_P, x, clean_meas, y, q_plot, q_plott, time_plotter, tstep, plot_path, ecef_ref, qt_plot, rt_plot, trans_plot):
    # ll2 = np.linspace(0, out_plot_X.shape[1] - 1, out_plot_X.shape[1])

    cov_col = 'g'
    meas_col = 'b'
    meas_col2 = 'y'
    p1_col = 'm'
    pf_col = 'k'
    truth_col = 'r'
    f_col = 'r'
    cov_t_col = 'r'

    yt = copy.copy(y[0, :, 3:])
    m = ~(yt == 0).all(1)
    yf = yt[m]
    seq = yf.shape[0]
    p_max = seq

    dim = out_plot_X.shape[2]

    out_plot_X2 = copy.copy(np.nan_to_num(out_plot_X[0, :, :])) * 1
    out_plot_P2 = copy.copy(np.nan_to_num(out_plot_P[0, :, :])) * 1
    # out_plot_F2 = copy.copy(np.nan_to_num(out_plot_F)) * 1

    x2 = copy.copy(x[0, :, :])
    y2 = copy.copy(y[0, :, :])

    q_plot2 = np.sqrt(np.power(copy.copy(q_plot[0, :, :]), 2))
    q_plott2 = np.sqrt(np.power(copy.copy(q_plott[0, :, :]), 2))
    qt_plot = np.sqrt(np.power(copy.copy(qt_plot[0, :, :]), 2))

    q_plot2 = np.sqrt(q_plot2)
    q_plott2 = np.sqrt(q_plott2)
    qt_plot = np.sqrt(qt_plot)

    # a = q_plot2[0,:p_max,0,0]
    # aa = q_plott2[0,:p_max,0,0]

    # qtt = create_cov_diag(qt_plot, p_max)

    qtt_pos_x = qt_plot[:p_max, 0, 0]
    qtt_pos_y = qt_plot[:p_max, 4, 4]
    qtt_pos_z = qt_plot[:p_max, 8, 8]
    qtt_vel_x = qt_plot[:p_max, 1, 1]
    qtt_vel_y = qt_plot[:p_max, 5, 5]
    qtt_vel_z = qt_plot[:p_max, 9, 9]
    qtt_acc_x = qt_plot[:p_max, 2, 2]
    qtt_acc_y = qt_plot[:p_max, 6, 6]
    qtt_acc_z = qt_plot[:p_max, 10, 10]
    qtt_jer_x = qt_plot[:p_max, 3, 3]
    qtt_jer_y = qt_plot[:p_max, 7, 7]
    qtt_jer_z = qt_plot[:p_max, 11, 11]

    # trans_pos_x = np.sqrt(np.square(trans_plot[0, 0:p_max, 0]))
    # trans_pos_y = np.sqrt(np.square(trans_plot[0, 0:p_max, 4]))
    # trans_pos_z = np.sqrt(np.square(trans_plot[0, 0:p_max, 8]))
    # trans_vel_x = np.sqrt(np.square(trans_plot[0, 0:p_max, 1]))
    # trans_vel_y = np.sqrt(np.square(trans_plot[0, 0:p_max, 5]))
    # trans_vel_z = np.sqrt(np.square(trans_plot[0, 0:p_max, 9]))
    # trans_acc_x = np.sqrt(np.square(trans_plot[0, 0:p_max, 2]))
    # trans_acc_y = np.sqrt(np.square(trans_plot[0, 0:p_max, 6]))
    # trans_acc_z = np.sqrt(np.square(trans_plot[0, 0:p_max, 10]))
    # trans_jer_x = np.sqrt(np.square(trans_plot[0, 0:p_max, 3]))
    # trans_jer_y = np.sqrt(np.square(trans_plot[0, 0:p_max, 7]))
    # trans_jer_z = np.sqrt(np.square(trans_plot[0, 0:p_max, 11]))

    rtt_pos_x = np.sqrt(np.sqrt(rt_plot[0, :p_max, 0, 0]))
    rtt_pos_y = np.sqrt(np.sqrt(rt_plot[0, :p_max, 1, 1]))
    rtt_pos_z = np.sqrt(np.sqrt(rt_plot[0, :p_max, 2, 2]))

    n_cov_truth = int(q_plott.shape[2])
    n_cov = int(q_plot.shape[2])

    if n_cov_truth == 12:
        qt_pos_x = q_plott2[:p_max, 0, 0]
        qt_pos_y = q_plott2[:p_max, 4, 4]
        qt_pos_z = q_plott2[:p_max, 8, 8]
        qt_vel_x = q_plott2[:p_max, 1, 1]
        qt_vel_y = q_plott2[:p_max, 5, 5]
        qt_vel_z = q_plott2[:p_max, 9, 9]
        qt_acc_x = q_plott2[:p_max, 2, 2]
        qt_acc_y = q_plott2[:p_max, 6, 6]
        qt_acc_z = q_plott2[:p_max, 10, 10]
        qt_jer_x = q_plott2[:p_max, 3, 3]
        qt_jer_y = q_plott2[:p_max, 7, 7]
        qt_jer_z = q_plott2[:p_max, 11, 11]

    if n_cov == 12:
        q_pos_x = q_plot2[:p_max, 0, 0]
        q_pos_y = q_plot2[:p_max, 4, 4]
        q_pos_z = q_plot2[:p_max, 8, 8]
        q_vel_x = q_plot2[:p_max, 1, 1]
        q_vel_y = q_plot2[:p_max, 5, 5]
        q_vel_z = q_plot2[:p_max, 9, 9]
        q_acc_x = q_plot2[:p_max, 2, 2]
        q_acc_y = q_plot2[:p_max, 6, 6]
        q_acc_z = q_plot2[:p_max, 10, 10]
        q_jer_x = q_plot2[:p_max, 3, 3]
        q_jer_y = q_plot2[:p_max, 7, 7]
        q_jer_z = q_plot2[:p_max, 11, 11]

    tp = time_plotter[0, :p_max]
    min_t = tp[0]
    max_t = tp[-1]

    errorK = np.sqrt((out_plot_P2[:p_max, :] - y2[:p_max, :]) ** 2)  # one steep in past
    errorX = np.sqrt((out_plot_X2[:p_max, :] - y2[:p_max, :]) ** 2)  # current state estimate
    # errorF = np.sqrt((out_plot_F2[0, :p_max, :] - y2[0, 0:p_max, :]) ** 2)  # current state estimate
    errorM = np.sqrt((x2[:p_max, :] - y2[:p_max, :3]) ** 2)

    # state_pctl = pctl(errorK)
    # cov_pctl = pctl(rms_vals.mdn_ra)
    # num_pctls = len(percentiles)

    ## ACCELERATION ##
    std_dev = 1.0
    plt.figure()
    plt.interactive(False)
    plt.subplot(311)
    n = 6
    plt.plot(tp, y2[:p_max, n], truth_col, label='Truth Acc', lw=1.1)
    plt.plot(tp, out_plot_P2[:p_max, n], p1_col, label='Kalman', lw=1.1)
    plt.plot(tp, out_plot_X2[:p_max, n], pf_col, label='DNN', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n])), (np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n])))
    plt.ylim(np.min(y2[:p_max, n]), np.max(y2[:p_max, n]))
    plt.legend()
    plt.title('X Acc', fontsize=12)
    plt.ylabel('M / S^2', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.subplot(312)
    n=7
    plt.plot(tp, y2[:p_max, n], truth_col, label='Truth Acc', lw=1.1)
    plt.plot(tp, out_plot_P2[:p_max, n], 'm', label='Kalman', lw=1.1)
    plt.plot(tp, out_plot_X2[:p_max, n] * 1, 'k', label='DNN', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n])), np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n]))
    plt.ylim(np.min(y2[:p_max, n]), np.max(y2[:p_max, n]))
    plt.legend()
    plt.title('Y Acc', fontsize=12)
    plt.ylabel('M / S^2', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.subplot(313)
    n=8
    plt.plot(tp, y2[0:p_max, n] * 1, truth_col, label='Truth Acc', lw=1.1)
    plt.plot(tp, out_plot_P2[0:p_max, n], 'm', label='Kalman', lw=1.1)
    plt.plot(tp, out_plot_X2[0:p_max, n] * 1, 'k', label='DNN', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n])), np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n]))
    plt.ylim(np.min(y2[:p_max, n]), np.max(y2[:p_max, n]))
    plt.legend()
    plt.title('Z Acc', fontsize=12)
    plt.ylabel('M / S^2', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_acc_results2.png')
    plt.close()

    ## JERK ##
    std_dev = 1.0
    plt.figure()
    plt.interactive(False)
    plt.subplot(311)
    n=9
    plt.plot(tp, y2[:p_max, n], truth_col, label='Truth Jerk', lw=1.1)
    plt.plot(tp, out_plot_P2[:p_max, n], p1_col, label='Kalman', lw=1.1)
    plt.plot(tp, out_plot_X2[:p_max, n], pf_col, label='DNN', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, 9]) + std_dev * np.std(out_plot_X2[0:p_max, 9])), (np.mean(out_plot_X2[0:p_max, 9]) + std_dev * np.std(out_plot_X2[0:p_max, 9])))
    plt.ylim(np.min(y2[:p_max, n]), np.max(y2[:p_max, n]))
    plt.legend()
    plt.title('X Jerk', fontsize=12)
    plt.ylabel('M / S^3', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.subplot(312)
    n=10
    plt.plot(tp, y2[:p_max, n] * 1, truth_col, label='Truth Jerk', lw=1.1)
    plt.plot(tp, out_plot_P2[0:p_max, n], 'm', label='Kalman', lw=1.1)
    plt.plot(tp, out_plot_X2[0:p_max, n], 'k', label='DNN', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, 10]) + std_dev * np.std(out_plot_X2[0:p_max, 10])), np.mean(out_plot_X2[0:p_max, 10]) + std_dev * np.std(out_plot_X2[0:p_max, 10]))
    plt.ylim(np.min(y2[:p_max, n]), np.max(y2[:p_max, n]))
    plt.legend()
    plt.title('Y Jerk', fontsize=12)
    plt.ylabel('M / S^3', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.subplot(313)
    n=11
    plt.plot(tp, y2[0:p_max, n], truth_col, label='Truth Jerk', lw=1.1)
    plt.plot(tp, out_plot_P2[0:p_max, n], 'm', label='Kalman', lw=1.1)
    plt.plot(tp, out_plot_X2[0:p_max, n], 'k', label='DNN', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, 11]) + std_dev * np.std(out_plot_X2[0:p_max, 11])), np.mean(out_plot_X2[0:p_max, 11]) + std_dev * np.std(out_plot_X2[0:p_max, 11]))
    plt.ylim(np.min(y2[:p_max, n]), np.max(y2[:p_max, n]))
    plt.legend()
    plt.title('Z Jerk', fontsize=12)
    plt.ylabel('M / S^3', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_Jerk_results2.png')
    plt.close()

    ############################################################################################################################
    idx_pos = 0
    idx_vel = 3
    std_dev = 6.0
    alpha_rms = 0.8
    start_idx = int(errorK.shape[1] * 0.1)

    plt.figure()
    plt.subplot(221)
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorK[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Position')
    plt.plot(tp, qt_pos_x, cov_col, label='$\Sigma$ Position', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + std_dev * np.std(errorK[start_idx:, idx_pos]))
    plt.title('X Position Augmented Kalman', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(222)
    plt.plot(tp, errorK[:, idx_vel], p1_col, alpha=alpha_rms, label='RMSE Velocity')
    plt.plot(tp, qt_vel_x, cov_col, label='$\Sigma$ Velocity', lw=1.5)
    plt.legend()
    plt.title('X Velocity Augmented Kalman', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + std_dev * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.subplot(223)
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Position')
    # plt.plot(tp, errorF[:, idx_pos], f_col, alpha=alpha_rms, label='RMSE PositionT')
    plt.plot(tp, q_pos_x, cov_col, label='$\Sigma$ Position', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_pos]) + 6.0 * np.std(errorX[:, idx_pos]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + std_dev * np.std(errorK[start_idx:, idx_pos]))
    plt.title('X Position DNN', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')
    
    plt.subplot(224)
    plt.plot(tp, errorX[:, idx_vel], pf_col, alpha=alpha_rms, label='RMSE Velocity')
    # plt.plot(tp, errorF[:, idx_vel], f_col, alpha=alpha_rms, label='RMSE VelocityT')
    plt.plot(tp, q_vel_x, cov_col, label='$\Sigma$ Velocity', lw=1.5)
    plt.legend()
    plt.title('X Velocity DNN', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_vel]) + 6.0 * np.std(errorX[:, idx_vel]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + std_dev * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')
    
    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_X_cov_comparison.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    idx_pos = 1
    idx_vel = 4
    plt.figure()
    plt.subplot(221)
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorK[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Position')
    plt.plot(tp, qt_pos_y, cov_col, label='$\Sigma$ Position', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + std_dev * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Y Position Augmented Kalman', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(222)
    plt.plot(tp, errorK[:, idx_vel], p1_col, alpha=alpha_rms, label='RMSE Velocity')
    plt.plot(tp, qt_vel_y, cov_col, label='$\Sigma$ Velocity', lw=1.5)
    plt.legend()
    plt.title('Y Velocity Augmented Kalman', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + std_dev * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.subplot(223)
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Position')
    # plt.plot(tp, errorF[:, idx_pos], f_col, alpha=alpha_rms, label='RMSE PositionT')
    plt.plot(tp, q_pos_y, cov_col, label='$\Sigma$ Position', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_pos]) + 6.0 * np.std(errorX[:, idx_pos]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + std_dev * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Y Position DNN', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(224)
    plt.plot(tp, errorX[:, idx_vel], pf_col, alpha=alpha_rms, label='RMSE Velocity')
    # plt.plot(tp, errorF[:, idx_vel], f_col, alpha=alpha_rms, label='RMSE VelocityT')
    plt.plot(tp, q_vel_y, cov_col, label='$\Sigma$ Velocity', lw=1.5)
    plt.legend()
    plt.title('Y Velocity DNN', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_vel]) + 6.0 * np.std(errorX[:, idx_vel]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + std_dev * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_Y_cov_comparison.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    idx_pos = 2
    idx_vel = 5
    plt.figure()
    plt.subplot(221)
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorK[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Position')
    plt.plot(tp, qt_pos_z, cov_col, label='$\Sigma$ Position', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Z Position Augmented Kalman', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(222)
    plt.plot(tp, errorK[:, idx_vel], p1_col, alpha=alpha_rms, label='RMSE Velocity')
    plt.plot(tp, qt_vel_z, cov_col, label='$\Sigma$ Velocity', lw=1.5)
    plt.legend()
    plt.title('Z Velocity Augmented Kalman', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.subplot(223)
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Position')
    # plt.plot(tp, errorF[:, idx_pos], f_col, alpha=alpha_rms, label='RMSE PositionT')
    plt.plot(tp, q_pos_z, cov_col, label='$\Sigma$ Position', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_pos]) + 6.0 * np.std(errorX[:, idx_pos]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Z Position DNN', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(224)
    plt.plot(tp, errorX[:, idx_vel], pf_col, alpha=alpha_rms, label='RMSE Velocity')
    # plt.plot(tp, errorF[:, idx_vel], f_col, alpha=alpha_rms, label='RMSE VelocityT')
    plt.plot(tp, q_vel_z, cov_col, label='$\Sigma$ Velocity', lw=1.5)
    plt.legend()
    plt.title('Z Velocity DNN', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_vel]) + 6.0 * np.std(errorX[:, idx_vel]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_Z_cov_comparison.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    idx_pos = 6
    idx_vel = idx_pos + 3
    plt.figure()
    plt.subplot(221)
    # plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorK[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Acceleration')
    plt.plot(tp, qt_acc_x, cov_col, label='$\Sigma$ Acceleration', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('X Acceleration Augmented Kalman', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(222)
    plt.plot(tp, errorK[:, idx_vel], p1_col, alpha=alpha_rms, label='RMSE Jerk')
    plt.plot(tp, qt_jer_x, cov_col, label='$\Sigma$ Jerk', lw=1.5)
    plt.legend()
    plt.title('X Jerk Augmented Kalman', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.subplot(223)
    # plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Acceleration')
    # plt.plot(tp, errorF[:, idx_pos], f_col, alpha=alpha_rms, label='RMSE PositionT')
    plt.plot(tp, q_acc_x, cov_col, label='$\Sigma$ Acceleration', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_pos]) + 6.0 * np.std(errorX[:, idx_pos]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('X Acceleration', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(224)
    plt.plot(tp, errorX[:, idx_vel], pf_col, alpha=alpha_rms, label='RMSE Jerk')
    # plt.plot(tp, errorF[:, idx_vel], f_col, alpha=alpha_rms, label='RMSE VelocityT')
    plt.plot(tp, q_jer_x, cov_col, label='$\Sigma$ Jerk', lw=1.5)
    plt.legend()
    plt.title('X Jerk DNN', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_vel]) + 6.0 * np.std(errorX[:, idx_vel]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_X_cov_comparisonaj.png')
    plt.close()
    ############################################################################################################################
    
    ############################################################################################################################
    idx_pos = 7
    idx_vel = idx_pos + 3
    plt.figure()
    plt.subplot(221)
    # plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorK[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Acceleration')
    plt.plot(tp, qt_acc_y, cov_col, label='$\Sigma$ Acceleration', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Y Acceleration Augmented Kalman', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(222)
    plt.plot(tp, errorK[:, idx_vel], p1_col, alpha=alpha_rms, label='RMSE Jerk')
    plt.plot(tp, qt_jer_y, cov_col, label='$\Sigma$ Jerk', lw=1.5)
    plt.legend()
    plt.title('Y Jerk Augmented Kalman', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.subplot(223)
    # plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Acceleration')
    # plt.plot(tp, errorF[:, idx_pos], f_col, alpha=alpha_rms, label='RMSE PositionT')
    plt.plot(tp, q_acc_y, cov_col, label='$\Sigma$ Acceleration', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_pos]) + 6.0 * np.std(errorX[:, idx_pos]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Y Acceleration', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(224)
    plt.plot(tp, errorX[:, idx_vel], pf_col, alpha=alpha_rms, label='RMSE Jerk')
    # plt.plot(tp, errorF[:, idx_vel], f_col, alpha=alpha_rms, label='RMSE VelocityT')
    plt.plot(tp, q_jer_y, cov_col, label='$\Sigma$ Jerk', lw=1.5)
    plt.legend()
    plt.title('Y Jerk DNN', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_vel]) + 6.0 * np.std(errorX[:, idx_vel]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_Y_cov_comparisonaj.png')
    plt.close()
    ############################################################################################################################
    
    ############################################################################################################################
    idx_pos = 8
    idx_vel = idx_pos+3
    plt.figure()
    plt.subplot(221)
    # plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorK[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Acceleration')
    plt.plot(tp, qt_acc_z, cov_col, label='$\Sigma$ Acceleration', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Z Acceleration Augmented Kalman', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(222)
    plt.plot(tp, errorK[:, idx_vel], p1_col, alpha=alpha_rms, label='RMSE Jerk')
    plt.plot(tp, qt_jer_z, cov_col, label='$\Sigma$ Jerk', lw=1.5)
    plt.legend()
    plt.title('Z Jerk Augmented Kalman', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.subplot(223)
    # plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Acceleration')
    # plt.plot(tp, errorF[:, idx_pos], f_col, alpha=alpha_rms, label='RMSE PositionT')
    plt.plot(tp, q_acc_z, cov_col, label='$\Sigma$ Acceleration', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_pos]) + 6.0 * np.std(errorX[:, idx_pos]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Z Position Acceleration', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(224)
    plt.plot(tp, errorX[:, idx_vel], pf_col, alpha=alpha_rms, label='RMSE Jerk')
    # plt.plot(tp, errorF[:, idx_vel], f_col, alpha=alpha_rms, label='RMSE VelocityT')
    plt.plot(tp, q_jer_z, cov_col, label='$\Sigma$ Jerk', lw=1.5)
    plt.legend()
    plt.title('Z Jerk DNN', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_vel]) + 6.0 * np.std(errorX[:, idx_vel]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_Z_cov_comparisonaj.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    idx_pos = 0
    idx_vel = 1

    std_dev = 1.
    plt.figure()
    plt.subplot(231)
    plt.plot(tp, qtt_pos_x, cov_col, label='Q $\Sigma$ X', lw=1.5)
    # plt.plot(tp, trans_pos_x, pf_col, label='Trans X', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_pos_x) + std_dev * np.std(trans_pos_x))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ X', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(232)
    plt.plot(tp, qtt_pos_y, cov_col, label='Q $\Sigma$ Y', lw=1.5)
    # plt.plot(tp, trans_pos_y, pf_col, label='Trans Y', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_pos_y) + std_dev * np.std(trans_pos_y))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ Y', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(233)
    plt.plot(tp, qtt_pos_z, cov_col, label='Q $\Sigma$ Z', lw=1.5)
    # plt.plot(tp, trans_pos_z, pf_col, label='Trans Z', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_pos_z) + std_dev * np.std(trans_pos_z))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ Z', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(234)
    plt.plot(tp, qtt_vel_x, cov_col, label='Q $\Sigma$ Xd', lw=1.5)
    # plt.plot(tp, trans_vel_x, pf_col, label='Trans Xd', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_vel_x) + std_dev * np.std(trans_vel_x))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ Xd', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(235)
    plt.plot(tp, qtt_vel_y, cov_col, label='Q $\Sigma$ Yd', lw=1.5)
    # plt.plot(tp, trans_vel_y, pf_col, label='Trans Yd', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_vel_y) + std_dev * np.std(trans_vel_y))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ Yd', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(236)
    plt.plot(tp, qtt_vel_z, cov_col, label='Q $\Sigma$ Zd', lw=1.5)
    # plt.plot(tp, trans_vel_z, pf_col, label='Trans Zd', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_vel_z) + std_dev * np.std(trans_vel_z))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ Zd', fontsize=12, fontweight='bold')
    # plt.plot(tp, trans_vel_z, pf_col, label='Trans Zd', lw=1.5)
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_Q_cov1_comparison.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    idx_pos = 0
    idx_vel = 1

    std_dev = 2.0

    plt.figure()
    plt.subplot(231)
    plt.plot(tp, qtt_acc_x, cov_col, label='Q $\Sigma$ Xdd', lw=1.5)
    # plt.plot(tp, trans_acc_x, pf_col, label='Trans Xdd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_acc_x) + std_dev * np.std(trans_acc_x))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ X', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(232)
    plt.plot(tp, qtt_acc_y, cov_col, label='Q $\Sigma$ Ydd', lw=1.5)
    # plt.plot(tp, trans_acc_y, pf_col, label='Trans Ydd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_acc_y) + std_dev * np.std(trans_acc_y))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ Y', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(233)
    plt.plot(tp, qtt_acc_z, cov_col, label='Q $\Sigma$ Zdd', lw=1.5)
    # plt.plot(tp, trans_acc_z, pf_col, label='Trans Zdd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_acc_z) + std_dev * np.std(trans_acc_z))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ Z', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(234)
    plt.plot(tp, qtt_jer_x, cov_col, label='Q $\Sigma$ Xddd', lw=1.5)
    # plt.plot(tp, trans_jer_x, pf_col, label='Trans Xddd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_jer_x) + std_dev * np.std(trans_jer_x))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ Xd', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(235)
    plt.plot(tp, qtt_jer_y, cov_col, label='Q $\Sigma$ Yddd', lw=1.5)
    # plt.plot(tp, trans_jer_y, pf_col, label='Trans Yddd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_jer_y) + std_dev * np.std(trans_jer_y))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ Yd', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(236)
    plt.plot(tp, qtt_jer_z, cov_col, label='Q $\Sigma$ Zddd', lw=1.5)
    # plt.plot(tp, trans_jer_z, pf_col, label='Trans Zddd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_jer_z) + std_dev * np.std(trans_jer_z))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ Zd', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_Q_cov2_comparison.png')
    plt.close()
    ############################################################################################################################

    plt.figure()
    plt.subplot(311)
    plt.plot(tp, rtt_pos_x, cov_col, label='R $\Sigma$ X', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ X', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(312)
    plt.plot(tp, rtt_pos_y, cov_col, label='R $\Sigma$ Y', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ Y', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(313)
    plt.plot(tp, rtt_pos_z, cov_col, label='R $\Sigma$ Z', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model $\Sigma$ Z', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_R_cov_comparison.png')
    plt.close()
    ############################################################################################################################


def output_plots(out_plot_X, out_plot_P, out_plot_refined, x, y, q_plott, q_plots, q_plotr, time_plotter, plot_path, qt_plot, rt_plot, plot_filename, sequence_length):

    cov_col = 'g'
    cov_col2 = 'r'
    meas_col = 'b'
    meas_col2 = 'y'
    p1_col = 'k'
    pf_col = 'm'
    pr_col = 'c'
    truth_col = 'r'
    f_col = 'r'
    # cov_t_col = 'r'

    npy = str.split(plot_filename, '/')
    vidx0 = ['Location' in t for t in npy]
    vidx = vidx0.index(True)
    npy = npy[vidx]
    npy = npy.replace("Location", "")
    npy = npy.replace("[", "")
    npy = npy.replace("]", "")
    loc = str.split(npy, ',')
    # location = [float(loc[0]), float(loc[1]), float(loc[2])]
    sensor_type = loc[-1]

    npyt = str.split(plot_filename, '/')[-1]
    npyt = str.split(npyt, '_')[3]
    traj = str.split(npyt, '.tsv')[0]

    image_header = 'Tracjectory: ' + traj + ' Location: ' + '< ' + loc[0] + ', ' + loc[1] + ', ' + loc[2] + ' >' + ' SensorType: ' + sensor_type
    yt = copy.copy(y[0, :, 3:])
    m = ~(yt == 0).all(1)
    yf = yt[m]
    seq = yf.shape[0]
    p_max = seq

    if sequence_length < 500:
        plot_smooth = False
    else:
        plot_smooth = True

    dim = out_plot_X.shape[2]

    out_plot_X2 = copy.copy(np.nan_to_num(out_plot_X[0, :, :])) * 1
    out_plot_P2 = copy.copy(np.nan_to_num(out_plot_P[0, :, :])) * 1
    out_plot_refined2 = copy.copy(np.nan_to_num(out_plot_refined[0, :, :])) * 1
    # out_plot_F2 = copy.copy(np.nan_to_num(out_plot_F)) * 1

    x2 = copy.copy(x[0, :, :]) * 1
    y2 = copy.copy(y[0, :, :]) * 1

    q_plott2 = np.sqrt(np.power(copy.copy(q_plott[0, :, :]), 2))
    q_plots2 = np.sqrt(np.power(copy.copy(q_plots[0, :, :]), 2))
    q_plotr2 = np.sqrt(np.power(copy.copy(q_plotr[0, :, :]), 2))

    qt_plot = np.sqrt(np.sqrt(np.power(copy.copy(qt_plot[0, :, :]), 2)))

    q_plott2 = np.sqrt(q_plott2)
    q_plots2 = np.sqrt(q_plots2)
    q_plotr2 = np.sqrt(q_plotr2)

    qtt_pos_x = qt_plot[:p_max, 0, 0]
    qtt_pos_y = qt_plot[:p_max, 4, 4]
    qtt_pos_z = qt_plot[:p_max, 8, 8]
    qtt_vel_x = qt_plot[:p_max, 1, 1]
    qtt_vel_y = qt_plot[:p_max, 5, 5]
    qtt_vel_z = qt_plot[:p_max, 9, 9]
    qtt_acc_x = qt_plot[:p_max, 2, 2]
    qtt_acc_y = qt_plot[:p_max, 6, 6]
    qtt_acc_z = qt_plot[:p_max, 10, 10]
    qtt_jer_x = qt_plot[:p_max, 3, 3]
    qtt_jer_y = qt_plot[:p_max, 7, 7]
    qtt_jer_z = qt_plot[:p_max, 11, 11]

    rtt_pos_x = np.sqrt(rt_plot[0, :p_max, 0, 0])
    rtt_pos_y = np.sqrt(rt_plot[0, :p_max, 1, 1])
    rtt_pos_z = np.sqrt(rt_plot[0, :p_max, 2, 2])

    # n_cov_truth = int(q_plott.shape[2])
    # n_cov = int(q_plot.shape[2])

    # if n_cov_truth == 12:
    #     qt_pos_x = q_plott2[:p_max, 0, 0]
    #     qt_pos_y = q_plott2[:p_max, 4, 4]
    #     qt_pos_z = q_plott2[:p_max, 8, 8]
    #     qt_vel_x = q_plott2[:p_max, 1, 1]
    #     qt_vel_y = q_plott2[:p_max, 5, 5]
    #     qt_vel_z = q_plott2[:p_max, 9, 9]
    #     qt_acc_x = q_plott2[:p_max, 2, 2]
    #     qt_acc_y = q_plott2[:p_max, 6, 6]
    #     qt_acc_z = q_plott2[:p_max, 10, 10]
    #     qt_jer_x = q_plott2[:p_max, 3, 3]
    #     qt_jer_y = q_plott2[:p_max, 7, 7]
    #     qt_jer_z = q_plott2[:p_max, 11, 11]

    # if n_cov == 12:
    q_pos_x = q_plott2[:p_max, 0, 0]
    q_pos_y = q_plott2[:p_max, 4, 4]
    q_pos_z = q_plott2[:p_max, 8, 8]
    q_vel_x = q_plott2[:p_max, 1, 1]
    q_vel_y = q_plott2[:p_max, 5, 5]
    q_vel_z = q_plott2[:p_max, 9, 9]
    q_acc_x = q_plott2[:p_max, 2, 2]
    q_acc_y = q_plott2[:p_max, 6, 6]
    q_acc_z = q_plott2[:p_max, 10, 10]
    q_jer_x = q_plott2[:p_max, 3, 3]
    q_jer_y = q_plott2[:p_max, 7, 7]
    q_jer_z = q_plott2[:p_max, 11, 11]

    q_pos_xr = q_plotr2[:p_max, 0, 0]
    q_pos_yr = q_plotr2[:p_max, 4, 4]
    q_pos_zr = q_plotr2[:p_max, 8, 8]
    q_vel_xr = q_plotr2[:p_max, 1, 1]
    q_vel_yr = q_plotr2[:p_max, 5, 5]
    q_vel_zr = q_plotr2[:p_max, 9, 9]
    q_acc_xr = q_plotr2[:p_max, 2, 2]
    q_acc_yr = q_plotr2[:p_max, 6, 6]
    q_acc_zr = q_plotr2[:p_max, 10, 10]
    q_jer_xr = q_plotr2[:p_max, 3, 3]
    q_jer_yr = q_plotr2[:p_max, 7, 7]
    q_jer_zr = q_plotr2[:p_max, 11, 11]

    q_pos = np.sqrt(q_pos_xr ** 2 + q_pos_yr ** 2 + q_pos_zr ** 2)
    q_vel = np.sqrt(q_vel_xr ** 2 + q_vel_yr ** 2 + q_vel_zr ** 2)

    tp = time_plotter[0, :p_max]
    min_t = tp[0]
    max_t = tp[-1]

    # diff_posK = np.sqrt(np.sum(((out_plot_P2[:p_max, :3]-y2[:p_max, :3])**2), axis=1))
    # diff_posX = np.sqrt(np.sum(((out_plot_X2[:p_max, :3]-y2[:p_max, :3])**2), axis=1))
    diff_posR = np.sqrt(np.sum(((out_plot_refined2[:p_max, :3]-y2[:p_max, :3])**2), axis=1))

    # diff_velK = np.sqrt(np.sum(((out_plot_P2[:p_max, 3:6] - y2[:p_max, 3:6]) ** 2), axis=1))
    # diff_velX = np.sqrt(np.sum(((out_plot_X2[:p_max, 3:6] - y2[:p_max, 3:6]) ** 2), axis=1))
    diff_velR = np.sqrt(np.sum(((out_plot_refined2[:p_max, 3:6] - y2[:p_max, 3:6]) ** 2), axis=1))

    diff_posM = np.sqrt(np.sum(((x2[:p_max, :3] - y2[:p_max, :3]) ** 2), axis=1))

    errorK = np.sqrt((out_plot_P2[:p_max, :] - y2[:p_max, :]) ** 2)
    errorX = np.sqrt((out_plot_X2[:p_max, :] - y2[:p_max, :]) ** 2)
    errorR = np.sqrt((out_plot_refined2[:p_max, :] - y2[:p_max, :]) ** 2)
    errorM = np.sqrt((x2[:p_max, :] - y2[:p_max, :3]) ** 2)

    ## ACCELERATION ##
    std_dev = 1.0
    plt.figure()
    plt.interactive(False)
    plt.subplot(311)
    n = 6
    plt.plot(tp, y2[:p_max, n], truth_col, label='Truth Acceleration', lw=1.1)
    if plot_smooth:
        plt.plot(tp, out_plot_P2[:p_max, n], pf_col, label='Smoothed', lw=1.1)
    plt.plot(tp, out_plot_X2[:p_max, n], p1_col, label='Filtered', lw=1.1)
    plt.plot(tp, out_plot_refined2[:p_max, n], pr_col, label='Refined', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n])), (np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n])))
    plt.ylim(np.min(y2[:p_max, n]), np.max(y2[:p_max, n]))
    plt.legend()
    plt.title('X Acceleration', fontsize=12)
    plt.ylabel('M / $^2$', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.subplot(312)
    n = 7
    plt.plot(tp, y2[:p_max, n], truth_col, label='Truth Acceleration', lw=1.1)
    if plot_smooth:
        plt.plot(tp, out_plot_P2[:p_max, n], pf_col, label='Smoothed', lw=1.1)
    plt.plot(tp, out_plot_X2[:p_max, n], p1_col, label='Filtered', lw=1.1)
    plt.plot(tp, out_plot_refined2[:p_max, n] * 1, pr_col, label='Refined', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n])), np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n]))
    plt.ylim(np.min(y2[:p_max, n]), np.max(y2[:p_max, n]))
    plt.legend()
    plt.title('Y Acceleration', fontsize=12)
    plt.ylabel('M / $^2$', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.subplot(313)
    n = 8
    plt.plot(tp, y2[:p_max, n], truth_col, label='Truth Acceleration', lw=1.1)
    if plot_smooth:
        plt.plot(tp, out_plot_P2[:p_max, n], pf_col, label='Smoothed', lw=1.1)
    plt.plot(tp, out_plot_X2[:p_max, n], p1_col, label='Filtered', lw=1.1)
    plt.plot(tp, out_plot_refined2[:p_max, n], pr_col, label='Refined', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n])), np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n]))
    plt.ylim(np.min(y2[:p_max, n]), np.max(y2[:p_max, n]))
    plt.legend()
    plt.title('Z Acceleration', fontsize=12)
    plt.ylabel('M / $^2$', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_acc_results.png')
    plt.close()

    ## JERK ##
    std_dev = 1.0
    plt.figure()
    plt.interactive(False)
    plt.subplot(311)
    n = 9
    plt.plot(tp, y2[:p_max, n], truth_col, label='Truth Jerk', lw=1.1)
    if plot_smooth:
        plt.plot(tp, out_plot_P2[:p_max, n], pf_col, label='Smoothed', lw=1.1)
    plt.plot(tp, out_plot_X2[:p_max, n], p1_col, label='Filtered', lw=1.1)
    plt.plot(tp, out_plot_refined2[:p_max, n], pr_col, label='Refined', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, 9]) + std_dev * np.std(out_plot_X2[0:p_max, 9])), (np.mean(out_plot_X2[0:p_max, 9]) + std_dev * np.std(out_plot_X2[0:p_max, 9])))
    plt.ylim(np.min(y2[:p_max, n]), np.max(y2[:p_max, n]))
    plt.legend()
    plt.title('X Jerk', fontsize=12)
    plt.ylabel('M / $^3$', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.subplot(312)
    n = 10
    plt.plot(tp, y2[:p_max, n], truth_col, label='Truth Jerk', lw=1.1)
    if plot_smooth:
        plt.plot(tp, out_plot_P2[:p_max, n], pf_col, label='Smoothed', lw=1.1)
    plt.plot(tp, out_plot_X2[:p_max, n], p1_col, label='Filtered', lw=1.1)
    plt.plot(tp, out_plot_refined2[:p_max, n], pr_col, label='Refined', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, 10]) + std_dev * np.std(out_plot_X2[0:p_max, 10])), np.mean(out_plot_X2[0:p_max, 10]) + std_dev * np.std(out_plot_X2[0:p_max, 10]))
    plt.ylim(np.min(y2[:p_max, n]), np.max(y2[:p_max, n]))
    plt.legend()
    plt.title('Y Jerk', fontsize=12)
    plt.ylabel('M / $^3$', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.subplot(313)
    n = 11
    plt.plot(tp, y2[:p_max, 11], truth_col, label='Truth Jerk', lw=1.1)
    if plot_smooth:
        plt.plot(tp, out_plot_P2[:p_max, n], pf_col, label='Smoothed', lw=1.1)
    plt.plot(tp, out_plot_X2[:p_max, n], p1_col, label='Filtered', lw=1.1)
    plt.plot(tp, out_plot_refined2[:p_max, n], pr_col, label='Refined', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, 11]) + std_dev * np.std(out_plot_X2[0:p_max, 11])), np.mean(out_plot_X2[0:p_max, 11]) + std_dev * np.std(out_plot_X2[0:p_max, 11]))
    plt.ylim(np.min(y2[:p_max, n]), np.max(y2[:p_max, n]))
    plt.legend()
    plt.title('Z Jerk', fontsize=12)
    plt.ylabel('M / $^3$', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_Jerk_results.png')
    plt.close()

    ############################################################################################################################

    ## POSITION ##
    std_dev = 6.0
    alpha_rms = 0.8
    start_idx = int(errorK.shape[1] * 0.1)

    plt.figure()
    plt.subplot(311)
    idx_pos = 0
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, errorR[:, idx_pos], pr_col, alpha=alpha_rms, label='RMSE Refined')
    if plot_smooth:
        plt.plot(tp, errorK[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Smooth')
    plt.plot(tp, q_pos_x, cov_col, label='$\Sigma$ Position', lw=1.5)
    plt.plot(tp, q_pos_xr, cov_col2, label='$\Sigma$ Position Refined', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(q_pos_x[start_idx:]) + std_dev * np.std(q_pos_x[start_idx:]))
    plt.title('X Position RMSE', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')

    plt.subplot(312)
    idx_pos = 1
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, errorR[:, idx_pos], pr_col, alpha=alpha_rms, label='RMSE Refined')
    if plot_smooth:
        plt.plot(tp, errorK[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Smooth')
    plt.plot(tp, q_pos_y, cov_col, label='$\Sigma$ Position', lw=1.5)
    plt.plot(tp, q_pos_yr, cov_col2, label='$\Sigma$ Position Refined', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(q_pos_y[start_idx:]) + std_dev * np.std(q_pos_y[start_idx:]))
    plt.title('Y Position RMSE', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')

    plt.subplot(313)
    idx_pos = 2
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, errorR[:, idx_pos], pr_col, alpha=alpha_rms, label='RMSE Refined')
    if plot_smooth:
        plt.plot(tp, errorK[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Smooth')
    plt.plot(tp, q_pos_z, cov_col, label='$\Sigma$ Position', lw=1.5)
    plt.plot(tp, q_pos_zr, cov_col2, label='$\Sigma$ Position Refined', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(q_pos_z[start_idx:]) + std_dev * np.std(q_pos_z[start_idx:]))
    plt.title('Z Position RMSE', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')

    plt.suptitle(image_header, fontsize=12, fontweight='bold')
    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_position_error.png')
    plt.close()
    ############################################################################################################################

    ## VELOCITY ##
    std_dev = 1.0
    alpha_rms = 0.8
    start_idx = int(errorK.shape[1] * 0.1)

    plt.figure()
    plt.subplot(311)
    idx_pos = 3
    plt.plot(tp, errorX[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, errorR[:, idx_pos], pr_col, alpha=alpha_rms, label='RMSE Refined')
    if plot_smooth:
        plt.plot(tp, errorK[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Smooth')
    plt.plot(tp, q_vel_x, cov_col, label='$\Sigma$ Velocity', lw=1.5)
    plt.plot(tp, q_vel_xr, cov_col2, label='$\Sigma$ Velocity Refined', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(q_vel_x[start_idx:]) + std_dev * np.std(q_vel_x[start_idx:]))
    plt.title('X Velocity RMSE', fontsize=12, fontweight='bold')
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')

    plt.subplot(312)
    idx_pos = 4
    plt.plot(tp, errorX[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, errorR[:, idx_pos], pr_col, alpha=alpha_rms, label='RMSE Refined')
    if plot_smooth:
        plt.plot(tp, errorK[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Smooth')
    plt.plot(tp, q_vel_y, cov_col, label='$\Sigma$ Velocity', lw=1.5)
    plt.plot(tp, q_vel_yr, cov_col2, label='$\Sigma$ Velocity Refined', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(q_vel_y[start_idx:]) + std_dev * np.std(q_vel_y[start_idx:]))
    plt.title('Y Velocity RMSE', fontsize=12, fontweight='bold')
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')

    plt.subplot(313)
    idx_pos = 5
    plt.plot(tp, errorX[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, errorR[:, idx_pos], pr_col, alpha=alpha_rms, label='RMSE Refined')
    if plot_smooth:
        plt.plot(tp, errorK[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Smooth')
    plt.plot(tp, q_vel_z, cov_col, label='$\Sigma$ Velocity', lw=1.5)
    plt.plot(tp, q_vel_zr, cov_col2, label='$\Sigma$ Velocity Refined', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(q_vel_z[start_idx:]) + std_dev * np.std(q_vel_z[start_idx:]))
    plt.title('Z Velocity RMSE', fontsize=12, fontweight='bold')
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')

    plt.suptitle(image_header, fontsize=12, fontweight='bold')
    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_velocity_error.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    idx_pos = 0
    idx_vel = 1
    std_dev = 1.

    plt.figure()
    plt.subplot(231)
    plt.plot(tp, qtt_pos_x, cov_col, label=r'Q $\Sigma$ X Position', lw=1.5)
    # plt.plot(tp, trans_pos_x, pf_col, label='Trans X', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_pos_x) + std_dev * np.std(trans_pos_x))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ X Position', fontsize=12, fontweight='bold')
    plt.ylabel('M$^2$', fontsize=12, fontweight='bold')

    plt.subplot(232)
    plt.plot(tp, qtt_pos_y, cov_col, label=r'Q $\Sigma$ Y Position', lw=1.5)
    # plt.plot(tp, trans_pos_y, pf_col, label='Trans Y', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_pos_y) + std_dev * np.std(trans_pos_y))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ Y Position', fontsize=12, fontweight='bold')
    plt.ylabel('M$^2$', fontsize=12, fontweight='bold')

    plt.subplot(233)
    plt.plot(tp, qtt_pos_z, cov_col, label=r'Q $\Sigma$ Z Position', lw=1.5)
    # plt.plot(tp, trans_pos_z, pf_col, label='Trans Z', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_pos_z) + std_dev * np.std(trans_pos_z))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ Z Position', fontsize=12, fontweight='bold')
    plt.ylabel('M$^2$', fontsize=12, fontweight='bold')

    plt.subplot(234)
    plt.plot(tp, qtt_vel_x, cov_col, label=r'Q $\Sigma$ X Velocity', lw=1.5)
    # plt.plot(tp, trans_vel_x, pf_col, label='Trans Xd', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_vel_x) + std_dev * np.std(trans_vel_x))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ X Velocity', fontsize=12, fontweight='bold')
    plt.ylabel('(M/s)$^2$', fontsize=12, fontweight='bold')

    plt.subplot(235)
    plt.plot(tp, qtt_vel_y, cov_col, label=r'Q $\Sigma$ Y Velocity', lw=1.5)
    # plt.plot(tp, trans_vel_y, pf_col, label='Trans Yd', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_vel_y) + std_dev * np.std(trans_vel_y))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ Y Velocity', fontsize=12, fontweight='bold')
    plt.ylabel('(M/s)$^2$', fontsize=12, fontweight='bold')

    plt.subplot(236)
    plt.plot(tp, qtt_vel_z, cov_col, label=r'Q $\Sigma$ Z Velocity', lw=1.5)
    # plt.plot(tp, trans_vel_z, pf_col, label='Trans Zd', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_vel_z) + std_dev * np.std(trans_vel_z))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ Z Velocity', fontsize=12, fontweight='bold')
    # plt.plot(tp, trans_vel_z, pf_col, label='Trans Zd', lw=1.5)
    plt.ylabel('(M/s)$^2$', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_Q_cov_posvel_comparison.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################
    idx_pos = 0
    idx_vel = 1
    std_dev = 2.0

    plt.figure()
    plt.subplot(231)
    plt.plot(tp, qtt_acc_x, cov_col, label=r'Q $\Sigma$ X Acceleration', lw=1.5)
    # plt.plot(tp, trans_acc_x, pf_col, label='Trans Xdd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_acc_x) + std_dev * np.std(trans_acc_x))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ X Acceleration', fontsize=12, fontweight='bold')
    plt.ylabel('(M/s$^2$)$^2$', fontsize=12, fontweight='bold')

    plt.subplot(232)
    plt.plot(tp, qtt_acc_y, cov_col, label=r'Q $\Sigma$ Ydd', lw=1.5)
    # plt.plot(tp, trans_acc_y, pf_col, label='Trans Ydd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_acc_y) + std_dev * np.std(trans_acc_y))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ Y Acceleration', fontsize=12, fontweight='bold')
    plt.ylabel('(M/s$^2$)$^2$', fontsize=12, fontweight='bold')

    plt.subplot(233)
    plt.plot(tp, qtt_acc_z, cov_col, label='Q $\Sigma$ Zdd', lw=1.5)
    # plt.plot(tp, trans_acc_z, pf_col, label='Trans Zdd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_acc_z) + std_dev * np.std(trans_acc_z))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ Z Acceleration', fontsize=12, fontweight='bold')
    plt.ylabel('(M/s$^2$)$^2$', fontsize=12, fontweight='bold')

    plt.subplot(234)
    plt.plot(tp, qtt_jer_x, cov_col, label=r'Q $\Sigma$ X Jerk', lw=1.5)
    # plt.plot(tp, trans_jer_x, pf_col, label='Trans Xddd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_jer_x) + std_dev * np.std(trans_jer_x))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ X Jerk', fontsize=12, fontweight='bold')
    plt.ylabel('(M/s$^3$)$^2$', fontsize=12, fontweight='bold')

    plt.subplot(235)
    plt.plot(tp, qtt_jer_y, cov_col, label=r'Q $\Sigma$ Y Jerk', lw=1.5)
    # plt.plot(tp, trans_jer_y, pf_col, label='Trans Yddd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_jer_y) + std_dev * np.std(trans_jer_y))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ Y Jerk', fontsize=12, fontweight='bold')
    plt.ylabel('(M/s$^3$)$^2$', fontsize=12, fontweight='bold')

    plt.subplot(236)
    plt.plot(tp, qtt_jer_z, cov_col, label=r'Q $\Sigma$ Z Jerk', lw=1.5)
    # plt.plot(tp, trans_jer_z, pf_col, label='Trans Zddd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_jer_z) + std_dev * np.std(trans_jer_z))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ Z Jerk', fontsize=12, fontweight='bold')
    plt.ylabel('(M/s$^3$)$^2$', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_Q_cov_accjer_comparison.png')

    plt.close()
    ############################################################################################################################

    plt.figure()
    plt.subplot(311)
    idx_pos = 0
    plt.plot(tp, rtt_pos_x, cov_col, label=r'R $\Sigma$ X', lw=1.5)
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ X', fontsize=12, fontweight='bold')
    plt.ylabel('Meters$^2$', fontsize=12, fontweight='bold')

    plt.subplot(312)
    idx_pos = 1
    plt.plot(tp, rtt_pos_y, cov_col, label=r'R $\Sigma$ Y', lw=1.5)
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ Y', fontsize=12, fontweight='bold')
    plt.ylabel('Meters$^2$', fontsize=12, fontweight='bold')

    plt.subplot(313)
    idx_pos = 2
    plt.plot(tp, rtt_pos_z, cov_col, label=r'R $\Sigma$ Z', lw=1.5)
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title(r'Model $\Sigma$ Z', fontsize=12, fontweight='bold')
    plt.ylabel('Meters$^2$', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_R_cov_comparison.png')
    plt.close()
    ############################################################################################################################
    ############################################################################################################################

    ## RMS ##
    std_dev = 1.0
    alpha_rms = 0.8
    start_idx = int(errorK.shape[1] * 0.1)

    plt.figure()
    plt.subplot(211)
    idx_pos = 0
    # plt.plot(tp, errorX[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, diff_posM, meas_col, alpha=0.3, label='RMSE Meas')
    plt.plot(tp, diff_posR, pr_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, q_pos, cov_col2, alpha=alpha_rms, label='RMS Covariance')
    # plt.plot(tp, q_vel_x, cov_col, label='$\Sigma$ Velocity', lw=1.5)
    # plt.plot(tp, q_vel_xr, cov_col2, label='$\Sigma$ Velocity Refined', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(q_vel_x[start_idx:]) + std_dev * np.std(q_vel_x[start_idx:]))
    plt.title('Position RMSE', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')

    plt.subplot(212)
    # plt.plot(tp, errorX[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, diff_velR, pr_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, q_vel, cov_col2, alpha=alpha_rms, label='RMS Covariance')
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(q_vel_x[start_idx:]) + std_dev * np.std(q_vel_x[start_idx:]))
    plt.title('Velocity RMSE', fontsize=12, fontweight='bold')
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')

    plt.suptitle(image_header, fontsize=12, fontweight='bold')
    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_overall_rms.png')
    plt.close()
    ############################################################################################################################

    ############################################################################################################################

    ## RMS ##
    std_dev = 1.0
    alpha_rms = 0.8
    start_idx = int(errorK.shape[1] * 0.1)

    plt.figure()
    plt.subplot(211)
    idx_pos = 0
    # plt.plot(tp, errorX[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, diff_posM, meas_col, alpha=0.3, label='RMSE Meas')
    plt.plot(tp, diff_posR, pr_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, q_pos, cov_col2, alpha=alpha_rms, label='RMS Covariance')
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(q_vel_x[start_idx:]) + std_dev * np.std(q_vel_x[start_idx:]))
    plt.title('Position RMSE', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.yscale('log')

    plt.subplot(212)
    # plt.plot(tp, errorX[:, idx_pos], p1_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, diff_velR, pr_col, alpha=alpha_rms, label='RMSE Filter')
    plt.plot(tp, q_vel, cov_col2, alpha=alpha_rms, label='RMS Covariance')
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(q_vel_x[start_idx:]) + std_dev * np.std(q_vel_x[start_idx:]))
    plt.title('Velocity RMSE', fontsize=12, fontweight='bold')
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.yscale('log')

    plt.suptitle(image_header, fontsize=12, fontweight='bold')
    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_overall_rms_log.png')
    plt.close()
    ############################################################################################################################
