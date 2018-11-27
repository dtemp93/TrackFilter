import pandas as pd
import os
from natsort import natsorted
import numpy as np
import random
import copy


def eci_2_ecef(eci_data, time):
    ws = np.float64(7292115.0e-11)
    # time = eci_data[:, 0, np.newaxis]
    pos = eci_data[:, :3]
    vel = eci_data[:, 3:6]

    cwdt = np.cos(ws*time)
    swdt = np.sin(ws*time)

    x = cwdt * pos[:, 0, np.newaxis] + swdt * pos[:, 1, np.newaxis]
    y = -swdt * pos[:, 0, np.newaxis] + cwdt * pos[:, 1, np.newaxis]
    z = pos[:, 2, np.newaxis]

    ecef_pos = np.concatenate([x, y, z], axis=1)

    xd = cwdt * vel[:, 0, np.newaxis] + swdt * vel[:, 1, np.newaxis] + ws*pos[:, 1, np.newaxis]
    yd = -swdt * vel[:, 0, np.newaxis] + cwdt * vel[:, 1, np.newaxis] - ws*pos[:, 0, np.newaxis]
    zd = vel[:, 2, np.newaxis]

    ecef_vel = np.concatenate([xd, yd, zd], axis=1)

    ecef_state = np.concatenate([time, ecef_pos, ecef_vel], axis=1)
    return ecef_state


def ecef_2_eci(ecef_data, time):
    ws = np.float64(7292115.0e-11)
    # time = ecef_data[: 0, np.newaxis

    for idx in range(time.shape[0]):
        eci2ecefm = np.zeros([time.shape[1], 3, 3])

        pos_ecef = ecef_data[idx, :, :3, np.newaxis]
        vel_ecef = ecef_data[idx, :, 3:6, np.newaxis]
        acc_ecef = ecef_data[idx, :, 6:9, np.newaxis]
        jer_ecef = ecef_data[idx, :, 9:, np.newaxis]

        cwdt = np.cos(ws * time[idx])
        swdt = np.sin(ws * time[idx])

        eci2ecefm[:, 0, 0] = cwdt[:, 0]
        eci2ecefm[:, 0, 1] = swdt[:, 0]
        eci2ecefm[:, 1, 0] = -swdt[:, 0]
        eci2ecefm[:, 1, 1] = cwdt[:, 0]
        eci2ecefm[:, 2, 2] = np.ones_like(cwdt)[:, 0]

        ecef2ecim = np.transpose(eci2ecefm, axes=[0, 2, 1])

        zvv = np.zeros([time.shape[1], 3, 1])
        # zvv[:, 2] = ws * (np.ones_like(time[idx]) - np.ones_like(time[idx]) * time[idx]/86400)
        zvv[:, 2] = ws * np.ones_like(time[idx])

        pos_eci = np.matmul(ecef2ecim, pos_ecef)

        vel_eci = np.matmul(ecef2ecim, vel_ecef) + np.expand_dims(np.cross(zvv[:, :, 0], pos_ecef[:, :, 0]), axis=2)
        # vel_eci = np.matmul(ecef2ecim, vel_ecef + np.expand_dims(np.cross(zvv[:, :, 0], pos_ecef[:, :, 0]), axis=2))

        acc_eci = np.matmul(ecef2ecim, acc_ecef) + np.expand_dims(np.cross(zvv[:, :, 0], np.cross(zvv[:, :, 0], pos_ecef[:, :, 0])) + 2 * np.cross(zvv[:, :, 0], vel_ecef[:, :, 0]), axis=2)
        # acc_eci = np.matmul(ecef2ecim, acc_ecef + np.expand_dims(np.cross(zvv[:, :, 0], np.cross(zvv[:, :, 0], pos_ecef[:, :, 0])) + 2 * np.cross(zvv[:, :, 0], vel_ecef[:, :, 0]), axis=2))

        jer_eci = np.matmul(ecef2ecim, jer_ecef) + np.expand_dims(np.cross(zvv[:, :, 0], np.cross(zvv[:, :, 0], np.cross(zvv[:, :, 0], pos_ecef[:, :, 0]))) + 2 * np.cross(zvv[:, :, 0], acc_ecef[:, :, 0]), axis=2)
        # jer_eci = np.matmul(ecef2ecim, jer_ecef + np.expand_dims(np.cross(zvv[:, :, 0], np.cross(zvv[:, :, 0], np.cross(zvv[:, :, 0], pos_ecef[:, :, 0]))) + 2 * np.cross(zvv[:, :, 0], acc_ecef[:, :, 0]), axis=2))

        eci_state = np.concatenate([pos_eci, vel_eci, acc_eci, jer_eci], axis=1)

    return eci_state


def translate_trajectory(target_lla, traj):

    lat = np.deg2rad(target_lla[0])
    lon = np.deg2rad(target_lla[1])
    alt = target_lla[2]

    a = 6378137
    e_sq = 0.00669437999014132
    chi = a / np.sqrt(1 - e_sq * (np.sin(lat)) ** 2)

    xs = (chi + alt) * np.cos(lat) * np.cos(lon)
    ys = (chi + alt) * np.cos(lat) * np.sin(lon)
    zs = (alt + chi * (1 - e_sq)) * np.sin(lat)

    ecef_ref = np.array([xs, ys, zs])  # This is correct

    x = traj[0, :, 1]
    y = traj[0, :, 2]
    z = traj[0, :, 3]

    traj_dist2 = np.sqrt(x**2 + y**2 + z**2)
    am = np.ones_like(traj_dist2) * a
    hit_index2 = np.argmin(np.abs(traj_dist2-am))
    hit2_ref = traj[0, hit_index2, 1:4]
    hit_diff = ecef_ref-hit2_ref

    new_pos = np.ones_like(traj[0, :, 1:4]) * hit_diff + traj[0, :, 1:4]
    new_pos = np.where(traj[0,:,0, np.newaxis]*np.ones_like(new_pos) == 0.0, np.zeros_like(new_pos), new_pos)
    new_traj = np.concatenate([traj[0, :, 0, np.newaxis], new_pos, traj[0, :, 4:]], axis=1)

    return new_traj


def normalize_data_earth_meas(meas, lati, loni, alti):
    lat = np.deg2rad(lati)
    lon = np.deg2rad(loni)
    alt = alti

    # batch, row, col = meas.shape

    time = copy.copy(meas[:, :, 0])
    dt = np.diff(time, axis=1)
    dt = np.concatenate([np.zeros(shape=[dt.shape[0], 1])*0.1, dt], axis=1)
    dt = np.where(dt > 50, 0.0, dt)

    # meas[:, :, 2] = np.where(meas[:, :, 2] > 180, meas[:, :, 2] - 360, meas[:, :, 2])

    R = np.expand_dims(meas[:, :, 1], axis=2)
    A = np.expand_dims(np.deg2rad(meas[:, :, 2]), axis=2)
    E = np.expand_dims(np.deg2rad(meas[:, :, 3]), axis=2)

    rsi = np.expand_dims(meas[:, :, 4], axis=2)
    asi = np.expand_dims(meas[:, :, 5], axis=2) * np.pi/180
    esi = np.expand_dims(meas[:, :, 6], axis=2) * np.pi/180
    snr = np.expand_dims(meas[:, :, 7], axis=2)

    meas_raem = np.concatenate([R, A, E], axis=2)
    meas_raes = np.concatenate([rsi, asi, esi], axis=2)
    meas_rae = np.concatenate([meas_raem, meas_raes, snr], axis=2)

    east = R * np.sin(A) * np.cos(E)
    north = R * np.cos(E) * np.cos(A)
    up = R * np.sin(E)

    meas_enu = np.concatenate([east, north, up], axis=2)

    cosPhi = np.cos(lat)
    sinPhi = np.sin(lat)
    cosLambda = np.cos(lon)
    sinLambda = np.sin(lon)

    tv = cosPhi * up - sinPhi * north
    wv = sinPhi * up + cosPhi * north
    uv = cosLambda * tv - sinLambda * east
    vv = sinLambda * tv + cosLambda * east

    meas_uvw = np.concatenate([uv, vv, wv], axis=2)

    a = 6378137
    e_sq = 0.00669437999014132
    chi = a / np.sqrt(1 - e_sq * (np.sin(lat) ** 2))

    x = (chi + alt) * np.cos(lat) * np.cos(lon)
    y = (chi + alt) * np.cos(lat) * np.sin(lon)
    z = (chi + alt - e_sq*chi) * np.sin(lat)

    ecef_ref = np.expand_dims(np.array([x, y, z]), axis=0)  # This is correct

    ecef = (np.ones_like(meas_uvw) * ecef_ref) + meas_uvw

    timet = np.expand_dims(time, axis=2)
    timeb = np.where(timet > 0., np.ones_like(timet), np.zeros_like(timet))
    dt = np.expand_dims(dt, axis=2)
    dt[:, 0, 0] *= 0

    if len(ecef.shape) < 3:
        ecef = ecef[np.newaxis, :, :]

    meas_ecef = np.concatenate([timet, ecef], axis=2)
    return meas_ecef, meas_rae, meas_uvw, meas_enu, np.squeeze(ecef_ref), timeb


def moving_mean(A0, N):

    A = pd.DataFrame(A0[0, :, :])
    ma = A.rolling(N, axis=0).mean()
    A = ma.values
    A = np.where(np.isnan(A), A0[0, :, :], A)
    # A_out = np.concatenate([A0[0, :N, :], A], axis=0)
    # A[0, :] = A0[0, 0, :]
    A_out = A[np.newaxis, :, :]

    return A_out


def pad_arrays(A, max_len):
    z = np.zeros(shape=[max_len, A.shape[1]])
    if np.shape(A)[0] > max_len:
        d1 = np.shape(A)[0] - max_len
        A = A[d1:, :]
    l = np.shape(A)[0]
    z[-l:, :] = A
    A0 = np.expand_dims(z, axis=0)

    return A0


def extend_states(y_traintt, HZ):
    zero_3 = np.zeros(shape=[1, 3])
    zero_31 = np.zeros(shape=[1, 1, 3])
    zero_1 = np.zeros(shape=[1, 1, 1]) * HZ

    time = y_traintt[0, :, 0]
    time = time[np.newaxis, :, np.newaxis]
    dt = np.abs(np.diff(time, axis=1))

    # first_step = np.nonzero(time)[1][0]
    # dt[0, first_step - 1, :] = HZ
    dt = np.concatenate([dt, zero_1], axis=1)

    acc_est = np.diff(y_traintt[0, :, -3:], axis=0)
    acc_est = np.concatenate([acc_est, zero_3], axis=0)

    firstNZ = np.nonzero(acc_est)[0][0]
    acc_est = np.expand_dims(np.divide(acc_est, np.squeeze(dt, axis=0), out=np.zeros_like(acc_est), where=np.squeeze(dt, axis=0) != 0), axis=0)
    acc_est[0, firstNZ, :] = copy.copy(acc_est[0, firstNZ + 1, :])

    acc_est = moving_mean(acc_est, N=5)

    jerk_est = np.diff(acc_est, axis=1)
    jerk_est = np.concatenate([jerk_est, zero_31], axis=1)

    firstNZJ = np.nonzero(jerk_est)[1][0]

    jerk_est[:, firstNZJ, :] = copy.copy(jerk_est[:, firstNZJ + 1, :])
    jerk_est = np.divide(jerk_est, dt, out=np.zeros_like(jerk_est), where=dt != 0)

    jerk_est = moving_mean(jerk_est, N=5)

    # meanj = np.mean(jerk_est, axis=1)
    # stdj = np.std(jerk_est, axis=1)

    # jerk_est = np.where(np.abs(jerk_est) > meanj + 4 * stdj, 3 * stdj, jerk_est)

    y_traintt = np.concatenate([y_traintt, acc_est], axis=2)
    y_traintt = np.concatenate([y_traintt, jerk_est], axis=2)

    # posm = y_traintt[0, :, 1:4]
    # velm = y_traintt[0, :, 4:7]
    # accm = y_traintt[0, :, 7:10]
    # jerm = y_traintt[0, :, 10:13]

    # np.where(np.abs(jerk_est * 6378137) > 500, np.mean(jerk_est, axis=1, keepdims=True), jerk_est)

    # plt.figure()
    # plt.subplot(311)
    # plt.plot(time[0, first_step:, :], velm[first_step:, 0])
    # plt.subplot(312)
    # plt.plot(time[0, first_step:, :], velm[first_step:, 1])
    # plt.subplot(313)
    # plt.plot(time[0, first_step:, :], velm[first_step:, 2])
    # plt.pause(0.01)
    #
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(time[0, first_step:, :], accm[first_step:, 0])
    # plt.subplot(312)
    # plt.plot(time[0, first_step:, :], accm[first_step:, 1])
    # plt.subplot(313)
    # plt.plot(time[0, first_step:, :], accm[first_step:, 2])
    # plt.pause(0.01)

    # y_traintt = np.repeat(y_traintt, 3, axis=1)
    # temp = y_traintt[0, :, :]
    # y_traintt = np.expand_dims(temp[temp[:, 0].argsort()], axis=0)
    # y_traintt = np.concatenate([zero_10, y_traintt], axis=1)

    # from scipy.signal import savgol_filter
    #
    # y_traintt[0, :, 10:13] = savgol_filter(y_traintt[0, :, 10:13], 9, 3, delta=0.1, axis=0)

    # plt.figure()
    # plt.subplot(311)
    # plt.plot(time[0, first_step:, :], jerm[first_step:, 0], 'b')
    # plt.plot(time[0, first_step:, :], jerm2[first_step:, 0], 'r')
    # plt.subplot(312)
    # plt.plot(time[0, first_step:, :], jerm[first_step:, 1], 'b')
    # plt.plot(time[0, first_step:, :], jerm2[first_step:, 1], 'r')
    # plt.subplot(313)
    # plt.plot(time[0, first_step:, :], jerm[first_step:, 2], 'b')
    # plt.plot(time[0, first_step:, :], jerm2[first_step:, 2], 'r')
    # plt.pause(0.01)

    return y_traintt


class DataServerPrePro:
    def __init__(self, file_dir_train, file_dir_test):
        self._count_train = -1
        self._count_test = -1
        self.file_dir_train = file_dir_train
        self.file_dir_test = file_dir_test

        self.file_dir_train_list = list()
        self.file_dir_test_list = list()

        for i, npy in enumerate(os.listdir(self.file_dir_train)):
            self.file_dir_train_list.append(self.file_dir_train + npy)

        random.shuffle(self.file_dir_train_list)

        for i, npy in enumerate(os.listdir(self.file_dir_test)):
            self.file_dir_test_list.append(self.file_dir_test + npy)

        self.num_examples_train = len(self.file_dir_train_list)
        self.num_examples_test = len(self.file_dir_test_list)
        self._index_in_epoch_train = -1
        self._index_in_epoch_test = -1

    def load(self, batch_size=512, constant=False, test=False):

        if test is False:
            if constant is False:
                self._index_in_epoch_train += 1
            else:
                self._index_in_epoch_train = 0

            self._count_train += 1
            if self._index_in_epoch_train >= self.num_examples_train:
                self._count_train = 0
                self._index_in_epoch_train = -1
                random.shuffle(self.file_dir_train_list)

            data_file = self.file_dir_train_list[self._index_in_epoch_train]

        else:
            if constant is False:
                self._index_in_epoch_test += 1
            else:
                self._index_in_epoch_test = 0

            self._count_test += 1
            if self._index_in_epoch_test >= self._index_in_epoch_test:
                self._count_test = 0
                self._index_in_epoch_test = -1

            data_file = self.file_dir_test_list[self._index_in_epoch_test]

        hf = h5py.File(data_file, 'r')
        x_data = hf.get('x_data')
        x_data = np.array(x_data)
        y_data = hf.get('y_data')
        y_data = np.array(y_data)
        ecef_ref = hf.get('ecef_data')
        ecef_ref = np.array(ecef_ref)
        lla_data = hf.get('lla_data')
        lla_data = np.array(lla_data)
        hf.close()

        # shuf = np.arange(x_data.shape[0])
        # np.random.shuffle(shuf)
        # x_data = x_data[shuf]
        # y_data = y_data[shuf]
        # ecef_ref = ecef_ref[shuf]
        # lla_data = lla_data[shuf]

        x_data = x_data[:, :2000, :]
        y_data = y_data[:, :2000, :]
        ecef_ref = ecef_ref[:, :2000, :]
        lla_data = lla_data[:, :2000, :]

        if x_data.shape[1] > batch_size:
            x_data = x_data[:batch_size, :, :]
            y_data = y_data[:batch_size, :, :]
            ecef_ref = ecef_ref[:batch_size, :, :]
            lla_data = lla_data[:batch_size, :, :]

        return x_data, y_data, ecef_ref, lla_data


class DataServerLive:
    def __init__(self, data_dir, meas_dir, state_dir, state_dir_rae='', decimate_data=False):

        self.data_dir = data_dir

        self.state_dir = data_dir + '/' + state_dir
        # self.state_dir_rae = state_dir_rae

        self.meas_dir_train = data_dir + '/Train/' + meas_dir
        self.meas_dir_test = data_dir + '/Test/' + meas_dir

        self.decimate_data = decimate_data
        self._index_in_epoch = 0
        self._count = 0

        self.meas_list_train = []
        self.meas_list_test = []

        self.state_list = []

        self.state_list_rae = []
        self.sensor_list = []
        self.file_size = []

        # for i, npy in enumerate(os.listdir(self.meas_dir)):
        #     npy = npy.replace("Location", "")
        #     npy = npy.replace("[", "")
        #     npy = npy.replace("]", "")
        #     loc = str.split(npy, ',')
        #     location = [float(loc[0]), float(loc[1]), float(loc[2])]
        #     sensor_type = loc[-1]
        #     self.sensor_list.append(location)

        for i, npy in enumerate(os.listdir(self.state_dir)):
            self.state_list.append(self.state_dir + npy)

        # for i, npy in enumerate(os.listdir(self.state_dir_rae)):
        #     self.state_list_rae.append(self.state_dir_rae + npy)

        for i, npy in enumerate(os.listdir(self.meas_dir_train)):
            for j, npy2 in enumerate(os.listdir(self.meas_dir_train + npy)):
                self.meas_list_train.append(self.meas_dir_train + npy + '/' + npy2)
                # self.file_size.append(os.path.getsize(self.meas_dir_train + npy + '/' + npy2))

        for i, npy in enumerate(os.listdir(self.meas_dir_test)):
            for j, npy2 in enumerate(os.listdir(self.meas_dir_test + npy)):
                self.meas_list_test.append(self.meas_dir_test + npy + '/' + npy2)

        # index = index_natsorted(self.file_size, reverse=False)
        # file_size = order_by_index(self.file_size, index)
        # file_size.reverse()
        # file_size_new = file_size[-int(len(file_size)*.66667):]
        # index = index_natsorted(file_size_new, reverse=False)
        # self.meas_list0 = order_by_index(self.meas_list, index)

        # self.state_list0 = natsorted(self.state_list)
        self.state_list0 = self.state_list
        # self.state_list_rae0 = natsorted(self.state_list_rae)
        # self.meas_list0 = self.meas_list0[-int(len(file_size)*.66667):]

        self.num_examples_train = int(len(self.meas_list_train))
        self.num_examples_test = int(len(self.meas_list_test))

        random.shuffle(self.meas_list_train)
        random.shuffle(self.meas_list_test)

        state_header = str.split(self.state_list0[0], '/')[-1]
        state_header = str.split(state_header, '_')

        self.state_header = state_header[0] + '_' + state_header[1] + '_' + state_header[2] + '_'

    def load(self, batch_size=250, constant=False, test=False, max_seq_len=2500, HZ=10):

        max_seq_len = max_seq_len
        HZ = 1/HZ

        self.total_batches = self.num_examples_train // batch_size

        if constant is False:
            self._index_in_epoch += batch_size
        else:
            self._index_in_epoch = 0
            self.num_examples_train = 2

        start = self._index_in_epoch
        self._count += 1
        if (self._index_in_epoch + batch_size) > self.num_examples_train and constant is False:
            import random
            random.shuffle(self.meas_list_train)
            start = 0
            self._count = 0
            self._index_in_epoch = 0

        end = self._index_in_epoch + batch_size

        if test is False:
            meas_paths = self.meas_list_train[start:end]
        else:
            if constant is True:
                meas_paths = self.meas_list_test[start:end]
            else:
                import random
                meas_paths = random.sample(self.meas_list_test, batch_size)

        lla_list = [None] * len(meas_paths)
        sensor_list = [None] * len(meas_paths)
        traj_list = [None] * len(meas_paths)
        # traj_list_rae = [None] * len(meas_paths)
        traj_num = [None] * len(meas_paths)

        for i, npy0 in enumerate(meas_paths):
            npy = str.split(npy0, '/')
            vidx0 = ['Location' in t for t in npy]
            vidx = vidx0.index(True)
            npy = npy[vidx]
            npy = npy.replace("Location", "")
            npy = npy.replace("[", "")
            npy = npy.replace("]", "")
            loc = str.split(npy, ' ')
            loc = str.split(loc[1], ',')
            location = [float(loc[0]), float(loc[1]), float(loc[2])]

            if 'big' in npy0:
                sensor_type = 'big'
            elif 'med' in npy0:
                sensor_type = 'med'
            elif 'small' in npy0:
                sensor_type = 'small'
            else:
                sensor_type = 'default'

            lla_list[i] = location
            sensor_list[i] = sensor_type
            npyt = str.split(npy0, '/')[-1]
            npyt = str.split(npyt, '_')[3]
            traj = str.split(npyt, '.tsv')[0]
            traj_name = self.state_dir + self.state_header + traj + '.tsv'
            # traj_name_rae = self.state_dir_rae + self.state_header + traj + '.tsv'
            traj_list[i] = traj_name
            # traj_list_rae[i] = traj_name_rae
            traj_num[i] = traj

        Xleci = list()
        Xlrae = list()
        Xluvw = list()
        Xlenu = list()
        ecef_l = list()
        lla_l = list()
        Ylecf = list()
        Yleci = list()
        file_list = list()
        for i, path in enumerate(meas_paths):

            traj_file = traj_list[i]
            meas_file = path
            # traj_file_rae = traj_list_rae[i]
            cur_traj = traj_num[i]
            cur_lla = lla_list[i]
            cur_sensor = sensor_list[i]

            npy = str.split(path, '/')
            vidx0 = ['Location' in t for t in npy]
            vidx = vidx0.index(True)
            npy = npy[vidx]
            npy = npy.replace("Location", "")
            npy = npy.replace("[", "")
            npy = npy.replace("]", "")
            loc = str.split(npy, ' ')
            loc = str.split(loc[1], ',')
            location = [float(loc[0]), float(loc[1]), float(loc[2])]
            # sensor_type = loc[-1]

            if 'big' in path:
                sensor_type = 'big'
            elif 'med' in path:
                sensor_type = 'med'
            elif 'small' in path:
                sensor_type = 'small'
            else:
                sensor_type = 'default'

            assert location == cur_lla

            assert sensor_type == cur_sensor

            npyt = str.split(path, '/')[-1]
            npyt = str.split(npyt, '_')[3]
            trajn = str.split(npyt, '.tsv')[0]

            if cur_traj != trajn:
                print('mismatched trajectories')
                print(cur_traj)

            lat = cur_lla[0]
            lon = cur_lla[1]
            alt = cur_lla[2]

            ## Load Data
            X = pd.read_csv(path, header=None)
            if X.shape[1] < 4:
                X = pd.read_table(path, header=None)

            Y = pd.read_csv(traj_file, header=None)
            if Y.shape[1] < 4:
                Y = pd.read_table(traj_file, header=None)

            X.drop(X[X[3] <=0].index, inplace=True)
            X.drop(X[X[1] <= X[4]**2].index, inplace=True)

            X[0] = np.round(X[0], 2)
            Y[0] = np.round(Y[0], 2)

            RAE_ECI_join = X.merge(Y, on=0, how='outer')
            RAE_ECI_join = RAE_ECI_join.dropna(how='any').values

            X = RAE_ECI_join[:, :8]
            Yeci = np.concatenate([X[:, 0, np.newaxis], RAE_ECI_join[:, -6:]], axis=1)

            Yecf = eci_2_ecef(Yeci[:, 1:], Yeci[:, 0, np.newaxis])

            Yecf = extend_states(Yecf[np.newaxis, :, :], HZ)
            Yecf = Yecf[0, :, :]

            Yeci = extend_states(Yeci[np.newaxis, :, :], HZ)
            Yeci = Yeci[0, :, :]

            # Make sure trajectory is translated
            # time_iny = copy.copy(Yeci[:, 0])

            # Y = translate_trajectory([0, 0, 0], Y[np.newaxis, :, :])
            # time_outy = copy.copy(Y[:, 0])
            # if ~np.all(time_iny == time_outy):
            #     print(time_outy)

            ## Decimate data
            if self.decimate_data:
                import random
                n_sample = random.randint(90, 100)

                idx = list()
                for j in range(X.shape[0]):
                    idx.append(j)

                time = X[:, 0]
                # man_time = np.squeeze(s[:, :, 0] == 1, axis=0)
                # not_man_time = np.squeeze(s[:, :, 0] == 0, axis=0)
                #
                # man_count = np.sum(man_time)
                n_samplet = int((n_sample / 100) * len(time))
                # idx1 = list(compress(idx, man_time))
                # idxt = list(compress(idx, time))
                sample_idx = np.random.choice(idx, size=n_samplet, replace=False)
                sample_idx = np.array(natsorted(sample_idx))

                # sample_total = natsorted(idx1 + samplet)

                X = X[sample_idx, :]
                Yecf = Yecf[sample_idx, :]
                Yeci = Yeci[sample_idx, :]

            X0 = pad_arrays(X, max_seq_len)
            Y0ecf = pad_arrays(Yecf, max_seq_len)
            Y0eci = pad_arrays(Yeci, max_seq_len)

            x_data, x_rae, x_uvw, x_enu, ecef_ref, timeb = normalize_data_earth_meas(X0, lat, lon, alt)

            zz = (Y0ecf == 0).all(2)
            zz = zz[:, :, np.newaxis]
            x_data = np.where(zz, np.zeros_like(x_data), x_data)

            # pos range = -1.25 : 1.25 RE
            # vel range = -0.00157 : 0.00157 RE
            # acc range = -3.136e-5 : 3.136e-5 RE
            # jer range = -3.136e-5 : 3.136e-5 RE
            # sen_loc = np.ones(shape=[x_data.shape[0], x_data.shape[1], 3]) * ecef_ref / 6378137
            # x_dataf = np.concatenate([x_data, sen_loc], axis=2) * timeb

            Xleci.append(x_data)
            Xlrae.append(x_rae)
            Xluvw.append(x_uvw)
            Xlenu.append(x_enu)
            Ylecf.append(Y0ecf)
            Yleci.append(Y0eci)
            file_list.append(meas_file)
            ecef_l.append(ecef_ref[np.newaxis, :])
            lla_l.append(np.array(cur_lla)[np.newaxis, :])

            del ecef_ref, zz, x_data, X0, Y0ecf, Y0eci

        meas_data_eci = np.concatenate(Xleci, axis=0)
        meas_data_rae = np.concatenate(Xlrae, axis=0)
        meas_data_uvw = np.concatenate(Xluvw, axis=0)
        meas_data_enu = np.concatenate(Xlenu, axis=0)
        traj_data_eci = np.concatenate(Yleci, axis=0)
        traj_data_ecf = np.concatenate(Ylecf, axis=0)
        ecef_ref_data = np.concatenate(ecef_l, axis=0)
        lla_data = np.concatenate(lla_l, axis=0)

        # for i in range(traj_data.shape[0]):
        #     y_traintt = np.expand_dims(traj_data[i, :, :], axis=0)
        #     y_traintt = np.expand_dims(traj_data[i, :, :], axis=0)
        #     # s_traintt = np.expand_dims(s_traint[i, :-1, :], axis=0)
        #     x_traintt_eci = np.expand_dims(meas_data_eci[i, :, :], axis=0)
        #     x_traintt_rae = np.expand_dims(meas_data_rae[i, :, :], axis=0)
        #     x_traintt_uvw = np.expand_dims(meas_data_uvw[i, :, :], axis=0)
        #     x_traintt_enu = np.expand_dims(meas_data_enu[i, :, :], axis=0)
        #
        #     time_x = x_traintt_eci[:, :, 0].T
        #     time_y = copy.copy(y_traintt[:, :, 0].T) * 1
        #     tt = np.concatenate([time_x, time_y], axis=1)
        #     dtt = np.abs(time_x-time_y)
        #
        #     if ~np.all(dtt <= 1e-6):
        #         times_y = np.nonzero(time_y)[0]
        #         nzy = times_y[0]
        #         minty = time_y[nzy]
        #         maxty = np.max(time_y)
        #         times_x = np.nonzero(time_x)[0]
        #         nzx = times_x[0]
        #         mintx = time_x[nzx]
        #         maxtx = np.max(time_x)
        #
        #         # common_times = ismember(times_x, times_y)
        #
        #         maxt = np.round(np.min([maxtx, maxty]), 1)
        #         mint = np.round(np.max([mintx, minty]), 1)
        #
        #         bools_x = np.logical_and(time_x <= maxt, time_x >= mint)
        #         bools_y = np.logical_and(time_y <= maxt, time_y >= mint)
        #
        #         overall = np.logical_and(bools_x, bools_y)
        #
        #         # idx_max = [i+1 for i, x in enumerate(bools_max) if x]
        #         # idx_min = [i for i, x in enumerate(bools_min) if x]
        #
        #         if not np.any(overall):
        #             print(' ')
        #             print('ERROR FOR FILE ')
        #             print(traj_list[i])
        #             print(' ')
        #             # idx_max
        #         else:
        #
        #             # print(traj_list[i])
        #             xtemp1 = x_traintt_eci[0, np.array(bools_x)[:, 0], :]
        #             xtemp2 = x_traintt_rae[0, np.array(bools_x)[:, 0], :]
        #             xtemp3 = x_traintt_uvw[0, np.array(bools_x)[:, 0], :]
        #             xtemp4 = x_traintt_enu[0, np.array(bools_x)[:, 0], :]
        #             ytemp = y_traintt[0, np.array(bools_y)[:, 0], :]
        #
        #             ytemp[:, 0] *= 1
        #
        #             tx = list(np.round(xtemp1[:, 0], 2))
        #             ty = list(np.round(ytemp[:, 0], 2))
        #
        #             txb = [None] * len(tx)
        #             tyb = [None] * len(ty)
        #
        #             for iii in range(len(tyb)):
        #                 if ty[iii] not in tx:
        #                     tyb[iii] = False
        #                 else:
        #                     tyb[iii] = True
        #
        #             for iii in range(len(txb)):
        #                 if tx[iii] not in ty:
        #                     txb[iii] = False
        #                 else:
        #                     txb[iii] = True
        #
        #             try:
        #                 xtemp1 = xtemp1[np.array(txb), :]
        #                 xtemp2 = xtemp2[np.array(txb), :]
        #                 xtemp3 = xtemp3[np.array(txb), :]
        #                 xtemp4 = xtemp4[np.array(txb), :]
        #                 ytemp = ytemp[np.array(tyb), :]
        #             except:
        #                 pass
        #
        #             # assert(xtemp.shape[0] == ytemp.shape[0])
        #
        #             z = np.zeros(shape=[max_seq_len, 4])
        #             l = np.shape(xtemp1)[0]
        #             z[-l:, :] = xtemp1
        #             del x_traintt_eci
        #             x_traintt_eci = z[np.newaxis, :, :]
        #
        #             z = np.zeros(shape=[max_seq_len, 3])
        #             l = np.shape(xtemp1)[0]
        #             z[-l:, :] = xtemp2
        #             del x_traintt_rae
        #             x_traintt_rae = z[np.newaxis, :, :]
        #
        #             z = np.zeros(shape=[max_seq_len, 3])
        #             l = np.shape(xtemp1)[0]
        #             z[-l:, :] = xtemp3
        #             del x_traintt_uvw
        #             x_traintt_uvw = z[np.newaxis, :, :]
        #
        #             z = np.zeros(shape=[max_seq_len, 3])
        #             l = np.shape(xtemp1)[0]
        #             z[-l:, :] = xtemp4
        #             del x_traintt_enu
        #             x_traintt_enu = z[np.newaxis, :, :]
        #
        #             z = np.zeros(shape=[max_seq_len, 13])
        #             l = np.shape(ytemp)[0]
        #             z[-l:, :] = ytemp
        #             del y_traintt
        #             y_traintt = z[np.newaxis, :, :]
        #
        #             time_x = x_traintt_eci[:, :, 0].T
        #             time_y = y_traintt[:, :, 0].T
        #             # tt = np.concatenate([time_x, time_y], axis=1)
        #             dtt = np.abs(time_x - time_y)
        #             if ~np.all(dtt <= 1e-6):
        #                 print(' ')
        #                 print('Times still not properly aligned ')
        #                 print(traj_list[i])
        #                 print(' ')
        #
        #     y_traintt = y_traintt[:, :, 1:]
        #
        #     y = np.nan_to_num(y_traintt)
        #     # s = np.nan_to_num(s_traintt)
        #     x_eci = np.nan_to_num(x_traintt_eci)
        #     x_rae = np.nan_to_num(x_traintt_rae)
        #     x_uvw = np.nan_to_num(x_traintt_uvw)
        #     x_enu = np.nan_to_num(x_traintt_enu)
        #
        #     y = np.expand_dims(y[0, :, :], axis=0)
        #     # s = np.expand_dims(s[0, :, :1], axis=0)
        #     x_eci = np.expand_dims(x_eci[0, :, :], axis=0)
        #     x_rae = np.expand_dims(x_rae[0, :, :], axis=0)
        #     x_uvw = np.expand_dims(x_uvw[0, :, :], axis=0)
        #     x_enu = np.expand_dims(x_enu[0, :, :], axis=0)
        #
        #     y_train.append(y)
        #     # s_train.append(s)
        #     x_train_eci.append(x_eci)
        #     x_train_rae.append(x_rae)
        #     x_train_uvw.append(x_uvw)
        #     x_train_enu.append(x_enu)

        x_data_out = np.concatenate([meas_data_eci, meas_data_rae, meas_data_enu, meas_data_uvw], axis=2)
        y_data_out = traj_data_ecf

        return x_data_out, y_data_out, traj_data_eci, self._count, self.total_batches, ecef_ref_data, lla_data, file_list


if __name__ == "__main__":

    data_dir = 'D:/TrackFilterData/AdvancedBroad'
    meas_dir = 'NoiseRAE/'
    state_dir = 'Translate/'

    # meas_file = r'D:\TrackFilterData\AdvancedBroad\Train\NoiseRAE\Location [0.00001,0.04490,8.00000] Az270 big\traj_noise_rae_003139.tsv'

    ds = DataServerLive(data_dir=data_dir, meas_dir=meas_dir, state_dir=state_dir)

    x_data, y_data, y_data_eci, batch_number, total_batches, ecef_ref, lla_data, meas_list = ds.load(batch_size=2, constant=True,
                                                                                                     test=False, max_seq_len=1000, HZ=25)

    y_data_eci = y_data_eci[:, :, 1:]
    y_eci_new = ecef_2_eci(y_data[:, :, 1:], y_data[:, :, 0, np.newaxis])

    a = y_data_eci[-1, :, :]
    aa = y_eci_new[:, :, 0]

    import matplotlib.pyplot as plt

    for i in range(12):
        plt.figure()
        plt.plot(y_data[-1, :, 0], np.sqrt(np.square(a[:, i] - aa[:, i])))

    plt.show()

    max_range = 100
    from propagation_utils import *
    from helper import *

    for i in range(max_range):
        print('Loading Batch ' + str(i) + ' out of ' + str(max_range))
        bs = 2
        x0, y0, batch_number, total_batches, ecef_ref, lla_data, meas_list = ds.load(batch_size=bs, constant=False)

        x0 = np.concatenate([x0[:, :, 0, np.newaxis], x0[:, :, 4:10]], axis=2)  # rae measurements

        y_uvw = y0[:, :, :3] - np.ones_like(y0[:, :, :3]) * ecef_ref[:, np.newaxis, :]