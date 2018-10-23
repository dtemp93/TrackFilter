import copy
import numpy as np
import matplotlib
import warnings
matplotlib.use('Qt5Agg')
plt = matplotlib.pyplot

import pdb

percentiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9, 0.97])


def pctl_state(arr):
    arr[arr==0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanpercentile(arr, percentiles*100, axis=1).T


def pctl_cov(arr):
    arr[arr==0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanpercentile(arr, percentiles*100, axis=1).T
    
    
def plot_all(title, rms_vals, idx):
    rms_x = rms_vals.rms_x[:, idx] + rms_vals.mask[:, idx]

    plt.suptitle(title)
    plt.subplot(321)
    plt.title("Position Error")
    plt.ylabel("m")
    plt.xlabel("track time (s)")
    plt.plot(rms_x, rms_vals.rms_e_all[:, idx],
             'r', alpha=.9, linewidth=.6, label="rms_x_est_err")
    plt.plot(rms_x, rms_vals.rms_cv_all[:, idx],
             'y', linewidth=.75, label="rms_cv")
    plt.legend(loc='upper right')

    ax1 = plt.subplot(322)
    plt.title("Velocity Error")
    plt.ylabel("m/s")
    plt.xlabel("track time (s)")
    l1_a = ax1.plot(rms_x, rms_vals.rms_vt_all[:, idx], 'g', alpha=1,
                    linewidth=.6, label="rms_v_true")
    l1_b = ax1.plot(rms_x, rms_vals.rms_v_est_all[:, idx], 'b', alpha=.8,
                    linewidth=.6, label="rms_v_est")
    ax2 = ax1.twinx()
    l2 = ax2.plot(rms_x, rms_vals.rms_ve_all[:, idx], 'r', alpha=.9,
                  linewidth=.6, label="rms_v_est_err")
    ax2.spines['right'].set_color('r')
    lns = l1_a + l1_b + l2
    lbls = [l.get_label() for l in lns]
    ax1.legend(lns, lbls, loc='upper right')

    color_settings = cycler('color', ['g', 'b', 'c', 'r'])

    ax_x_pctl = plt.subplot(323)
    ax_x_pctl.set_prop_cycle(color_settings)
    plt.title('Position Error')
    plt.ylabel('m')
    plt.xlabel('index since maneuver start - 100')

    ax_v_pctl = plt.subplot(324)
    ax_v_pctl.set_prop_cycle(color_settings)
    plt.title('Velocity Error')
    plt.ylabel('m/s')
    plt.xlabel('index since maneuver start - 100')

    ax_mdn_x_pctl = plt.subplot(325)
    ax_mdn_x_pctl.set_prop_cycle(color_settings)
    plt.title('Normalized Mahalonobis Distance')
    plt.xlabel('index since maneuver start - 100')

    x_pctl = pctl(rms_vals.rms_e_ra)
    v_pctl = pctl(rms_vals.rms_ve_ra)
    mdn_x_pctl = pctl(rms_vals.mdn_ra)
    num_pctls = len(percentiles)
    for i in range(num_pctls):
        for p, ax in [(x_pctl, ax_x_pctl), (v_pctl, ax_v_pctl),
                      (mdn_x_pctl, ax_mdn_x_pctl)]:
            p_slice = p[:, i]
            p_slice = p_slice[~np.isnan(p_slice)]
            ax.plot(np.arange(len(p_slice)) / 10., p_slice, linewidth=0.6)
            if i == num_pctls - 1:
                ax.legend(percentiles, loc='upper right', fontsize='small')

    plt.subplot(326)
    plt.title("Acceleration")
    plt.ylabel("m/s$^2$")
    plt.xlabel("track time (s)")
    plt.plot(rms_x, rms_vals.rms_at_all[:, idx], 'y',
             linewidth=.75, label="rms_at")
    plt.legend(loc='upper right')

    plt.tight_layout(pad=.05, w_pad=.05, h_pad=.02)
    plt.subplots_adjust(top=0.875)
    plt.pause(0.05)


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

    x2 = copy.copy(x[0, :, :]) * 1
    y2 = copy.copy(y[0, :, :]) * 1

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

    rtt_pos_x = np.sqrt(rt_plot[0, 0:p_max, 0, 0])
    rtt_pos_y = np.sqrt(rt_plot[0, 0:p_max, 1, 1])
    rtt_pos_z = np.sqrt(rt_plot[0, 0:p_max, 2, 2])

    n_cov_truth = int(q_plott.shape[2])
    n_cov = int(q_plot.shape[2])

    if n_cov_truth == 12:
        qt_pos_x = q_plott2[0:p_max, 0, 0]
        qt_pos_y = q_plott2[0:p_max, 4, 4]
        qt_pos_z = q_plott2[0:p_max, 8, 8]
        qt_vel_x = q_plott2[0:p_max, 1, 1]
        qt_vel_y = q_plott2[0:p_max, 5, 5]
        qt_vel_z = q_plott2[0:p_max, 9, 9]
        qt_acc_x = q_plott2[0:p_max, 2, 2]
        qt_acc_y = q_plott2[0:p_max, 6, 6]
        qt_acc_z = q_plott2[0:p_max, 10, 10]
        qt_jer_x = q_plott2[0:p_max, 3, 3]
        qt_jer_y = q_plott2[0:p_max, 7, 7]
        qt_jer_z = q_plott2[0:p_max, 11, 11]

    if n_cov == 12:
        q_pos_x = q_plot2[0:p_max, 0, 0]
        q_pos_y = q_plot2[0:p_max, 4, 4]
        q_pos_z = q_plot2[0:p_max, 8, 8]
        q_vel_x = q_plot2[0:p_max, 1, 1]
        q_vel_y = q_plot2[0:p_max, 5, 5]
        q_vel_z = q_plot2[0:p_max, 9, 9]
        q_acc_x = q_plot2[0:p_max, 2, 2]
        q_acc_y = q_plot2[0:p_max, 6, 6]
        q_acc_z = q_plot2[0:p_max, 10, 10]
        q_jer_x = q_plot2[0:p_max, 3, 3]
        q_jer_y = q_plot2[0:p_max, 7, 7]
        q_jer_z = q_plot2[0:p_max, 11, 11]

    tp = time_plotter[0, :p_max]
    min_t = tp[0]
    max_t = tp[-1]

    # a = out_plot_P2[0, :p_max, :]
    # a0 = out_plot_X2[0, :p_max, :]
    # aa = y2[0, :p_max, :]
    # aaa = x2[0, :p_max, :]

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
    plt.plot(tp, y2[0:p_max, n], truth_col, label='Truth Acc', lw=1.1)
    plt.plot(tp, out_plot_P2[0:p_max, n], p1_col, label='Kalman', lw=1.1)
    plt.plot(tp, out_plot_X2[0:p_max, n], pf_col, label='DNN', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n])), (np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n])))
    plt.legend()
    plt.title('X Acc', fontsize=12)
    plt.ylabel('M / S^2', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.subplot(312)
    n=7
    plt.plot(tp, y2[0:p_max, n] * 1, truth_col, label='Truth Acc', lw=1.1)
    plt.plot(tp, out_plot_P2[0:p_max, n], 'm', label='Kalman', lw=1.1)
    plt.plot(tp, out_plot_X2[0:p_max, n] * 1, 'k', label='DNN', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n])), np.mean(out_plot_X2[0:p_max, n]) + std_dev * np.std(out_plot_X2[0:p_max, n]))
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
    plt.legend()
    plt.title('Z Acc', fontsize=12)
    plt.ylabel('M / S^2', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.tight_layout(pad=0)
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
    plt.plot(tp, y2[0:p_max, 9], truth_col, label='Truth Jerk', lw=1.1)
    plt.plot(tp, out_plot_P2[0:p_max, 9], p1_col, label='Kalman', lw=1.1)
    plt.plot(tp, out_plot_X2[0:p_max, 9], pf_col, label='DNN', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, 9]) + std_dev * np.std(out_plot_X2[0:p_max, 9])), (np.mean(out_plot_X2[0:p_max, 9]) + std_dev * np.std(out_plot_X2[0:p_max, 9])))
    plt.legend()
    plt.title('X Jerk', fontsize=12)
    plt.ylabel('M / S^3', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.subplot(312)
    n=10
    plt.plot(tp, y2[0:p_max, 10] * 1, truth_col, label='Truth Jerk', lw=1.1)
    plt.plot(tp, out_plot_P2[0:p_max, 10], 'm', label='Kalman', lw=1.1)
    plt.plot(tp, out_plot_X2[0:p_max, 10] * 1, 'k', label='DNN', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, 10]) + std_dev * np.std(out_plot_X2[0:p_max, 10])), np.mean(out_plot_X2[0:p_max, 10]) + std_dev * np.std(out_plot_X2[0:p_max, 10]))
    plt.legend()
    plt.title('Y Jerk', fontsize=12)
    plt.ylabel('M / S^3', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.subplot(313)
    n=11
    plt.plot(tp, y2[0:p_max, 11] * 1, truth_col, label='Truth Jerk', lw=1.1)
    plt.plot(tp, out_plot_P2[0:p_max, 11], 'm', label='Kalman', lw=1.1)
    plt.plot(tp, out_plot_X2[0:p_max, 11] * 1, 'k', label='DNN', lw=1.1)
    # plt.ylim(-(np.mean(out_plot_X2[0:p_max, 11]) + std_dev * np.std(out_plot_X2[0:p_max, 11])), np.mean(out_plot_X2[0:p_max, 11]) + std_dev * np.std(out_plot_X2[0:p_max, 11]))
    plt.legend()
    plt.title('Z Jerk', fontsize=12)
    plt.ylabel('M / S^3', fontsize=12)
    plt.xlim(min_t, max_t)

    plt.tight_layout(pad=0)
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
    plt.plot(tp, qt_pos_x, cov_col, label='Variance Position', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + std_dev * np.std(errorK[start_idx:, idx_pos]))
    plt.title('X Position Augmented Kalman', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(222)
    plt.plot(tp, errorK[:, idx_vel], p1_col, alpha=alpha_rms, label='RMSE Velocity')
    plt.plot(tp, qt_vel_x, cov_col, label='Variance Velocity', lw=1.5)
    plt.legend()
    plt.title('X Velocity Augmented Kalman', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + std_dev * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.subplot(223)
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Position')
    # plt.plot(tp, errorF[:, idx_pos], f_col, alpha=alpha_rms, label='RMSE PositionT')
    plt.plot(tp, q_pos_x, cov_col, label='Variance Position', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_pos]) + 6.0 * np.std(errorX[:, idx_pos]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + std_dev * np.std(errorK[start_idx:, idx_pos]))
    plt.title('X Position DNN', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')
    
    plt.subplot(224)
    plt.plot(tp, errorX[:, idx_vel], pf_col, alpha=alpha_rms, label='RMSE Velocity')
    # plt.plot(tp, errorF[:, idx_vel], f_col, alpha=alpha_rms, label='RMSE VelocityT')
    plt.plot(tp, q_vel_x, cov_col, label='Variance Velocity', lw=1.5)
    plt.legend()
    plt.title('X Velocity DNN', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_vel]) + 6.0 * np.std(errorX[:, idx_vel]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + std_dev * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')
    
    plt.tight_layout(pad=0)
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
    plt.plot(tp, qt_pos_y, cov_col, label='Variance Position', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + std_dev * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Y Position Augmented Kalman', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(222)
    plt.plot(tp, errorK[:, idx_vel], p1_col, alpha=alpha_rms, label='RMSE Velocity')
    plt.plot(tp, qt_vel_y, cov_col, label='Variance Velocity', lw=1.5)
    plt.legend()
    plt.title('Y Velocity Augmented Kalman', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + std_dev * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.subplot(223)
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Position')
    # plt.plot(tp, errorF[:, idx_pos], f_col, alpha=alpha_rms, label='RMSE PositionT')
    plt.plot(tp, q_pos_y, cov_col, label='Variance Position', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_pos]) + 6.0 * np.std(errorX[:, idx_pos]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + std_dev * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Y Position DNN', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(224)
    plt.plot(tp, errorX[:, idx_vel], pf_col, alpha=alpha_rms, label='RMSE Velocity')
    # plt.plot(tp, errorF[:, idx_vel], f_col, alpha=alpha_rms, label='RMSE VelocityT')
    plt.plot(tp, q_vel_y, cov_col, label='Variance Velocity', lw=1.5)
    plt.legend()
    plt.title('Y Velocity DNN', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_vel]) + 6.0 * np.std(errorX[:, idx_vel]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + std_dev * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=0)
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
    plt.plot(tp, qt_pos_z, cov_col, label='Variance Position', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Z Position Augmented Kalman', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(222)
    plt.plot(tp, errorK[:, idx_vel], p1_col, alpha=alpha_rms, label='RMSE Velocity')
    plt.plot(tp, qt_vel_z, cov_col, label='Variance Velocity', lw=1.5)
    plt.legend()
    plt.title('Z Velocity Augmented Kalman', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.subplot(223)
    plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Position')
    # plt.plot(tp, errorF[:, idx_pos], f_col, alpha=alpha_rms, label='RMSE PositionT')
    plt.plot(tp, q_pos_z, cov_col, label='Variance Position', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_pos]) + 6.0 * np.std(errorX[:, idx_pos]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Z Position DNN', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(224)
    plt.plot(tp, errorX[:, idx_vel], pf_col, alpha=alpha_rms, label='RMSE Velocity')
    # plt.plot(tp, errorF[:, idx_vel], f_col, alpha=alpha_rms, label='RMSE VelocityT')
    plt.plot(tp, q_vel_z, cov_col, label='Variance Velocity', lw=1.5)
    plt.legend()
    plt.title('Z Velocity DNN', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_vel]) + 6.0 * np.std(errorX[:, idx_vel]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=0)
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
    plt.plot(tp, qt_acc_x, cov_col, label='Variance Acceleration', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('X Acceleration Augmented Kalman', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(222)
    plt.plot(tp, errorK[:, idx_vel], p1_col, alpha=alpha_rms, label='RMSE Jerk')
    plt.plot(tp, qt_jer_x, cov_col, label='Variance Jerk', lw=1.5)
    plt.legend()
    plt.title('X Jerk Augmented Kalman', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.subplot(223)
    # plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Acceleration')
    # plt.plot(tp, errorF[:, idx_pos], f_col, alpha=alpha_rms, label='RMSE PositionT')
    plt.plot(tp, q_acc_x, cov_col, label='Variance Acceleration', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_pos]) + 6.0 * np.std(errorX[:, idx_pos]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('X Acceleration', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(224)
    plt.plot(tp, errorX[:, idx_vel], pf_col, alpha=alpha_rms, label='RMSE Jerk')
    # plt.plot(tp, errorF[:, idx_vel], f_col, alpha=alpha_rms, label='RMSE VelocityT')
    plt.plot(tp, q_jer_x, cov_col, label='Variance Jerk', lw=1.5)
    plt.legend()
    plt.title('X Jerk DNN', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_vel]) + 6.0 * np.std(errorX[:, idx_vel]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=0)
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
    plt.plot(tp, qt_acc_y, cov_col, label='Variance Acceleration', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Y Acceleration Augmented Kalman', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(222)
    plt.plot(tp, errorK[:, idx_vel], p1_col, alpha=alpha_rms, label='RMSE Jerk')
    plt.plot(tp, qt_jer_y, cov_col, label='Variance Jerk', lw=1.5)
    plt.legend()
    plt.title('Y Jerk Augmented Kalman', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.subplot(223)
    # plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Acceleration')
    # plt.plot(tp, errorF[:, idx_pos], f_col, alpha=alpha_rms, label='RMSE PositionT')
    plt.plot(tp, q_acc_y, cov_col, label='Variance Acceleration', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_pos]) + 6.0 * np.std(errorX[:, idx_pos]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Y Acceleration', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(224)
    plt.plot(tp, errorX[:, idx_vel], pf_col, alpha=alpha_rms, label='RMSE Jerk')
    # plt.plot(tp, errorF[:, idx_vel], f_col, alpha=alpha_rms, label='RMSE VelocityT')
    plt.plot(tp, q_jer_y, cov_col, label='Variance Jerk', lw=1.5)
    plt.legend()
    plt.title('Y Jerk DNN', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_vel]) + 6.0 * np.std(errorX[:, idx_vel]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=0)
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
    plt.plot(tp, qt_acc_z, cov_col, label='Variance Acceleration', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Z Acceleration Augmented Kalman', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(222)
    plt.plot(tp, errorK[:, idx_vel], p1_col, alpha=alpha_rms, label='RMSE Jerk')
    plt.plot(tp, qt_jer_z, cov_col, label='Variance Jerk', lw=1.5)
    plt.legend()
    plt.title('Z Jerk Augmented Kalman', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.subplot(223)
    # plt.plot(tp, errorM[:, idx_pos], meas_col, alpha=0.3, label='RMSE Measurement')
    plt.plot(tp, errorX[:, idx_pos], pf_col, alpha=alpha_rms, label='RMSE Acceleration')
    # plt.plot(tp, errorF[:, idx_pos], f_col, alpha=alpha_rms, label='RMSE PositionT')
    plt.plot(tp, q_acc_z, cov_col, label='Variance Acceleration', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_pos]) + 6.0 * np.std(errorX[:, idx_pos]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Z Position Acceleration', fontsize=12, fontweight='bold')
    plt.ylabel('Meters', fontsize=12, fontweight='bold')

    plt.subplot(224)
    plt.plot(tp, errorX[:, idx_vel], pf_col, alpha=alpha_rms, label='RMSE Jerk')
    # plt.plot(tp, errorF[:, idx_vel], f_col, alpha=alpha_rms, label='RMSE VelocityT')
    plt.plot(tp, q_jer_z, cov_col, label='Variance Jerk', lw=1.5)
    plt.legend()
    plt.title('Z Jerk DNN', fontsize=12, fontweight='bold')
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorX[:, idx_vel]) + 6.0 * np.std(errorX[:, idx_vel]))
    plt.ylim(0, np.mean(errorK[start_idx:, idx_vel]) + 6.0 * np.std(errorK[start_idx:, idx_vel]))
    plt.ylabel('Meters per Second', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=0)
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
    plt.plot(tp, qtt_pos_x, cov_col, label='Q Variance X', lw=1.5)
    # plt.plot(tp, trans_pos_x, pf_col, label='Trans X', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_pos_x) + std_dev * np.std(trans_pos_x))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance X', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(232)
    plt.plot(tp, qtt_pos_y, cov_col, label='Q Variance Y', lw=1.5)
    # plt.plot(tp, trans_pos_y, pf_col, label='Trans Y', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_pos_y) + std_dev * np.std(trans_pos_y))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance Y', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(233)
    plt.plot(tp, qtt_pos_z, cov_col, label='Q Variance Z', lw=1.5)
    # plt.plot(tp, trans_pos_z, pf_col, label='Trans Z', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_pos_z) + std_dev * np.std(trans_pos_z))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance Z', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(234)
    plt.plot(tp, qtt_vel_x, cov_col, label='Q Variance Xd', lw=1.5)
    # plt.plot(tp, trans_vel_x, pf_col, label='Trans Xd', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_vel_x) + std_dev * np.std(trans_vel_x))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance Xd', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(235)
    plt.plot(tp, qtt_vel_y, cov_col, label='Q Variance Yd', lw=1.5)
    # plt.plot(tp, trans_vel_y, pf_col, label='Trans Yd', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_vel_y) + std_dev * np.std(trans_vel_y))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance Yd', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(236)
    plt.plot(tp, qtt_vel_z, cov_col, label='Q Variance Zd', lw=1.5)
    # plt.plot(tp, trans_vel_z, pf_col, label='Trans Zd', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_vel_z) + std_dev * np.std(trans_vel_z))
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance Zd', fontsize=12, fontweight='bold')
    # plt.plot(tp, trans_vel_z, pf_col, label='Trans Zd', lw=1.5)
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=0)
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
    plt.plot(tp, qtt_acc_x, cov_col, label='Q Variance Xdd', lw=1.5)
    # plt.plot(tp, trans_acc_x, pf_col, label='Trans Xdd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_acc_x) + std_dev * np.std(trans_acc_x))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance X', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(232)
    plt.plot(tp, qtt_acc_y, cov_col, label='Q Variance Ydd', lw=1.5)
    # plt.plot(tp, trans_acc_y, pf_col, label='Trans Ydd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_acc_y) + std_dev * np.std(trans_acc_y))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance Y', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(233)
    plt.plot(tp, qtt_acc_z, cov_col, label='Q Variance Zdd', lw=1.5)
    # plt.plot(tp, trans_acc_z, pf_col, label='Trans Zdd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_acc_z) + std_dev * np.std(trans_acc_z))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance Z', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(234)
    plt.plot(tp, qtt_jer_x, cov_col, label='Q Variance Xddd', lw=1.5)
    # plt.plot(tp, trans_jer_x, pf_col, label='Trans Xddd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_jer_x) + std_dev * np.std(trans_jer_x))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance Xd', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(235)
    plt.plot(tp, qtt_jer_y, cov_col, label='Q Variance Yddd', lw=1.5)
    # plt.plot(tp, trans_jer_y, pf_col, label='Trans Yddd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_jer_y) + std_dev * np.std(trans_jer_y))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance Yd', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(236)
    plt.plot(tp, qtt_jer_z, cov_col, label='Q Variance Zddd', lw=1.5)
    # plt.plot(tp, trans_jer_z, pf_col, label='Trans Zddd', lw=1.5)
    # if ~np.all(trans_pos_x == 0):
    #     plt.ylim(0, np.mean(trans_jer_z) + std_dev * np.std(trans_jer_z))
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance Zd', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_Q_cov2_comparison.png')
    plt.close()
    ############################################################################################################################

    plt.figure()
    plt.subplot(311)
    plt.plot(tp, rtt_pos_x, cov_col, label='R Variance X', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance X', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(312)
    plt.plot(tp, rtt_pos_y, cov_col, label='R Variance Y', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance Y', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.subplot(313)
    plt.plot(tp, rtt_pos_z, cov_col, label='R Variance Z', lw=1.5)
    plt.legend()
    plt.xlim(min_t, max_t)
    # plt.ylim(0, np.mean(errorK[start_idx:, idx_pos]) + 6.0 * np.std(errorK[start_idx:, idx_pos]))
    plt.title('Model Variance Z', fontsize=12, fontweight='bold')
    plt.ylabel('Meters^2', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.pause(0.05)
    plt.savefig(plot_path + '/' + str(p_max) + '_R_cov_comparison.png')
    plt.close()
    ############################################################################################################################
