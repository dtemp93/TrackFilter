import numpy as np
import pandas as pd
import os
from natsort import index_natsorted, order_by_index, natsorted
import copy
import matplotlib.pyplot as plt
import h5py

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = True
    return [bind.get(itm, False) for itm in a]


def normalize_data_earth_meas2(meas, lati, loni, alti, num_meas=3):
    lat = np.deg2rad(lati)
    lon = np.deg2rad(loni)
    alt = alti

    batch, row, col = meas.shape

    time = copy.copy(meas[:, :, 0])
    dt = np.diff(time, axis=1)
    dt = np.concatenate([np.ones(shape=[dt.shape[0], 1])*0.1, dt], axis=1)

    R = np.expand_dims(meas[:, :, 1], axis=2)
    A = np.expand_dims(np.deg2rad(meas[:, :, 2]), axis=2)
    E = np.expand_dims(np.deg2rad(meas[:, :, 3]), axis=2)

    # phi = (np.pi / 2) - A
    # theta = (np.pi / 2) - E

    up = R * np.sin(E)
    east = R * np.cos(E) * np.sin(A)
    north = R * np.cos(E) * np.cos(A)

    # ENU = np.concatenate([east, north, up], axis=2)
    cosPhi = np.cos(lat)
    sinPhi = np.sin(lat)
    cosLambda = np.cos(lon)
    sinLambda = np.sin(lon)

    t = cosPhi * up - sinPhi * north
    w = sinPhi * up + cosPhi * north

    u = cosLambda * t - sinLambda * east
    v = sinLambda * t + cosLambda * east

    a = 6378137
    e_sq = 0.00669437999014
    # chi = np.sqrt(1 - e_sq * (np.sin(lat)) ** 2)

    Rp = a / np.sqrt(1 - e_sq * np.cos(lat) ** 2)

    x = (Rp + alt) * np.cos(lat) * np.cos(lon)
    y = (Rp + alt) * np.cos(lat) * np.sin(lon)
    z = (Rp + alt - e_sq*Rp) * np.sin(lat)

    # x = (a / chi + alt) * np.cos(lat) * np.cos(lon)
    # y = (a / chi + alt) * np.cos(lat) * np.sin(lon)
    # z = (alt + a * (1 - e_sq) / chi) * np.sin(lat)

    ecef_ref = np.expand_dims(np.array([x, y, z]), axis=0)  # This is correct

    x2 = ecef_ref[0, 0] + u
    y2 = ecef_ref[0, 1] + v
    z2 = ecef_ref[0, 2] + w

    ecef = np.concatenate([x2, y2, z2], axis=2)

    # ecef2enu = np.zeros(shape=[3, 3])
    #
    # ecef2enu[0, 0] = -np.sin(lon)
    # ecef2enu[0, 1] = np.cos(lon)
    # ecef2enu[1, 0] = -np.sin(lat) * np.cos(lon)
    # ecef2enu[1, 1] = -np.sin(lat) * np.sin(lon)
    # ecef2enu[1, 2] = np.cos(lat)
    #
    # ecef2enu[2, 0] = np.cos(lat) * np.cos(lon)
    # ecef2enu[2, 1] = np.cos(lat) * np.sin(lon)
    # ecef2enu[2, 2] = np.sin(lat)
    #
    # ecef2enu = np.expand_dims(ecef2enu, axis=0)
    # ecef2enu = np.expand_dims(ecef2enu, axis=1)
    # ecef2enu = np.repeat(ecef2enu, repeats=[batch], axis=0)
    # ecef2enu = np.repeat(ecef2enu, repeats=[row], axis=1)

    # ecef = np.squeeze(np.transpose(np.matmul(np.transpose(ecef2enu, [0, 1, 3, 2]), np.expand_dims(ENU, axis=3)), [0, 1, 3, 2]) + ecef_ref)

    # w = 0.729211515E-04  # earth rotation rate in radian/s
    # theta2 = time*w
    theta2 = 0

    ecef_mat = np.zeros(shape=[int(batch), int(row), int(num_meas), int(num_meas)])
    ecef_mat[:, :, 0, 0] = np.cos(theta2)
    ecef_mat[:, :, 0, 1] = np.sin(theta2)
    ecef_mat[:, :, 1, 0] = -np.sin(theta2)
    ecef_mat[:, :, 1, 1] = np.cos(theta2)
    ecef_mat[:, :, 2, 2] = np.ones_like(ecef_mat[:, :, 2, 2])

    eci = np.squeeze(np.matmul(ecef_mat, np.expand_dims(ecef, axis=3)))

    eci = eci / 6378137

    timet = np.expand_dims(time, axis=2)
    timeb = np.where(timet > 0., np.ones_like(timet), np.zeros_like(timet))
    dt = np.expand_dims(dt, axis=2)
    dt[:, 0, 0] *= 0

    meas = np.concatenate([timet, eci], axis=2)

    return meas, np.squeeze(ecef_ref), timeb


def translate_trajectory(target_lla, traj):

    lat = np.deg2rad(target_lla[0])
    lon = np.deg2rad(target_lla[1])
    alt = target_lla[2]

    # batch, row, col = traj.shape

    a = 6378137
    e_sq = 0.00669437999014132
    chi = a / np.sqrt(1 - e_sq * (np.sin(lat)) ** 2)

    xs = (chi + alt) * np.cos(lat) * np.cos(lon)
    ys = (chi + alt) * np.cos(lat) * np.sin(lon)
    zs = (alt + chi * (1 - e_sq)) * np.sin(lat)

    ecef_ref = np.array([xs, ys, zs])  # This is correct

    # w = 0.729211515E-04  # earth rotation rate in radian/s
    # # theta2 = time*w
    # theta2 = 0
    #
    # ecef_mat = np.zeros(shape=[row, 3, 3])
    # ecef_mat[:, 0, 0] = np.cos(theta2)
    # ecef_mat[:, 0, 1] = np.sin(theta2)
    # ecef_mat[:, 1, 0] = -np.sin(theta2)
    # ecef_mat[:, 1, 1] = np.cos(theta2)
    # ecef_mat[:, 2, 2] = np.ones_like(ecef_mat[:, 2, 2])
    #
    # pos_eci = traj[0, :, 1:4]
    # vel_eci = traj[0, :, 4:7]
    # acc_eci = traj[0, :, 7:]
    #
    # pos_ecef = np.squeeze(np.matmul(np.matrix.transpose(ecef_mat, [0, 2, 1]), np.expand_dims(pos_eci, axis=3)))
    #
    # vp1 = vel_eci[:, 0, np.newaxis] + w * pos_ecef[:, 1, np.newaxis]
    # vp2 = vel_eci[:, 1, np.newaxis] - w * pos_ecef[:, 0, np.newaxis]
    # vp3 = vel_eci[:, 2, np.newaxis]
    #
    # vel_ecef = np.concatenate([vp1, vp2, vp3], axis=1)

    x = traj[0, :, 1]
    y = traj[0, :, 2]
    z = traj[0, :, 3]

    traj_dist2 = np.sqrt(x**2 + y**2 + z**2)
    am = np.ones_like(traj_dist2) * a
    hit_index2 = np.argmin(np.abs(traj_dist2-am))
    hit2_ref = traj[0, hit_index2, 1:4]
    hit_diff = ecef_ref-hit2_ref

    new_pos = np.ones_like(traj[0, :, 1:4]) * hit_diff + traj[0, :, 1:4]

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

    R = np.expand_dims(meas[:, :, 1], axis=2)
    A = np.expand_dims(np.deg2rad(meas[:, :, 2]), axis=2)
    E = np.expand_dims(np.deg2rad(meas[:, :, 3]), axis=2)

    meas_rae = np.concatenate([R, A, E], axis=2)
    # az = (np.pi / 2) - A
    # el = (np.pi / 2) - E

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
    # ENU = np.concatenate([east, north, up], axis=2)

    a = 6378137
    e_sq = 0.00669437999014132
    chi = a / np.sqrt(1 - e_sq * (np.sin(lat) ** 2))

    x = (chi + alt) * np.cos(lat) * np.cos(lon)
    y = (chi + alt) * np.cos(lat) * np.sin(lon)
    z = (chi + alt - e_sq*chi) * np.sin(lat)

    ecef_ref = np.expand_dims(np.array([x, y, z]), axis=0)  # This is correct

    ecef = (np.ones_like(meas_uvw) * ecef_ref) + meas_uvw

    # trans_mat = np.zeros(shape=[3, 3])
    #
    # trans_mat[0, 0] = -np.sin(lon)
    # trans_mat[0, 1] = -np.sin(lat) * np.cos(lon)
    # trans_mat[0, 2] = np.cos(lat) * np.cos(lon)
    #
    # trans_mat[1, 0] = np.cos(lon)
    # trans_mat[1, 1] = -np.sin(lat) * np.sin(lon)
    # trans_mat[1, 2] = np.cos(lat) * np.sin(lon)
    #
    # trans_mat[2, 1] = np.cos(lat)
    # trans_mat[2, 2] = np.sin(lat)
    #
    # # trans_mat = np.transpose(trans_mat)
    #
    # trans_mat = np.expand_dims(trans_mat, axis=0)
    # trans_mat = np.expand_dims(trans_mat, axis=1)
    # trans_mat = np.repeat(trans_mat, repeats=[batch], axis=0)
    # trans_mat = np.repeat(trans_mat, repeats=[row], axis=1)
    #
    # ecef = np.squeeze(np.transpose(np.matmul(np.transpose(trans_mat, [0, 1, 3, 2]), np.expand_dims(ENU, axis=3)), [0, 1, 3, 2]) + ecef_ref)

    # w = 0.729211515E-04  # earth rotation rate in radian/s
    # theta2 = time*w
    # theta2 = 0

    # ecef_mat = np.zeros(shape=[batch, row, num_meas, num_meas])
    # ecef_mat[:, :, 0, 0] = np.cos(theta2)
    # ecef_mat[:, :, 0, 1] = np.sin(theta2)
    # ecef_mat[:, :, 1, 0] = -np.sin(theta2)
    # ecef_mat[:, :, 1, 1] = np.cos(theta2)
    # ecef_mat[:, :, 2, 2] = np.ones_like(ecef_mat[:, :, 2, 2])
    #
    # eci = np.squeeze(np.matmul(np.matrix.transpose(ecef_mat, [0, 1, 3, 2]), np.expand_dims(ecef, axis=3)))

    eci = ecef

    timet = np.expand_dims(time, axis=2)
    timeb = np.where(timet > 0., np.ones_like(timet), np.zeros_like(timet))
    dt = np.expand_dims(dt, axis=2)
    dt[:, 0, 0] *= 0

    if len(eci.shape) < 3:
        eci = eci[np.newaxis, :, :]

    meas_eci = np.concatenate([timet, eci], axis=2)
    # meas_rae = np.concatenate([timet, rae])
    return meas_eci, meas_rae, meas_uvw, meas_enu, np.squeeze(ecef_ref), timeb


class DataServerLive:
    def __init__(self, meas_dir, state_dir, state_dir_rae=''):

        self.state_dir = state_dir
        # self.state_dir_rae = state_dir_rae
        self.meas_dir = meas_dir
        self._index_in_epoch = 0
        self._count = 0
        self.meas_list = []
        self.state_list = []
        self.state_list_rae = []
        self.sensor_list = []
        self.file_size = []

        for i, npy in enumerate(os.listdir(self.meas_dir)):
            npy = npy.replace("Location", "")
            npy = npy.replace("[", "")
            npy = npy.replace("]", "")
            loc = str.split(npy, ',')
            location = [float(loc[0]), float(loc[1]), float(loc[2])]
            self.sensor_list.append(location)

        for i, npy in enumerate(os.listdir(self.state_dir)):
            self.state_list.append(self.state_dir + npy)

        # for i, npy in enumerate(os.listdir(self.state_dir_rae)):
        #     self.state_list_rae.append(self.state_dir_rae + npy)

        for i, npy in enumerate(os.listdir(self.meas_dir)):
            for j, npy2 in enumerate(os.listdir(self.meas_dir + npy)):
                self.meas_list.append(self.meas_dir + npy + '/' + npy2)
                self.file_size.append(os.path.getsize(self.meas_dir + npy + '/' + npy2))

        index = index_natsorted(self.file_size, reverse=False)
        file_size = order_by_index(self.file_size, index)
        # file_size_new = file_size[-int(len(file_size)*.66667):]
        # index = index_natsorted(file_size_new, reverse=False)
        self.meas_list0 = order_by_index(self.meas_list, index)
        self.state_list0 = natsorted(self.state_list)
        # self.state_list_rae0 = natsorted(self.state_list_rae)
        self.meas_list0 = self.meas_list0[-int(len(file_size)*.66667):]
        self._num_examples0 = len(self.meas_list0)

        self.num_train = int(self._num_examples0)
        # self.num_test = int(0.2*self._num_examples0)

        self.meas_list_train = self.meas_list0[:self.num_train]
        self.meas_list_test = self.meas_list0[self.num_train:]

        self._num_examples_train = len(self.meas_list_train)

        state_header = str.split(self.state_list0[0], '/')[-1]
        state_header = str.split(state_header, '_')

        self.state_header = state_header[0] + '_' + state_header[1] + '_' + state_header[2] + '_'
        # self.state_header = state_header[0] + '_' + state_header[1] + '_'

    def load(self, batch_size=250, constant=False, test=False, max_seq_len=6750, HZ=25):

        max_seq_len = max_seq_len
        HZ = 1/HZ

        # self._total_batches = self._num_examples_train // batch_size

        # lla_list = list()
        # traj_list = list()
        # traj_num = list()

        if constant is False:
            self._index_in_epoch += batch_size
        else:
            self._index_in_epoch = 0

        start = self._index_in_epoch
        self._count += 1
        if (self._index_in_epoch + batch_size) > self._num_examples_train:
            # seed = np.random.randint(100000)
            # np.random.seed(seed)
            # np.random.shuffle(self._meas)
            # np.random.seed(seed)
            # np.random.shuffle(self._state)
            start = 0
            self._count = 0
            self._index_in_epoch = 0

        end = self._index_in_epoch + batch_size

        if test is False:
            meas_paths = self.meas_list_train[start:end]
        else:
            if constant is True:
                meas_paths = self.meas_list_train[start:end]
            else:
                import random
                meas_paths = random.sample(self.meas_list_test, batch_size)

        lla_list = [None] * len(meas_paths)
        traj_list = [None] * len(meas_paths)
        # traj_list_rae = [None] * len(meas_paths)
        traj_num = [None] * len(meas_paths)

        for i, npy0 in enumerate(meas_paths):
            npy = str.split(npy0, '/')[5]
            npy = npy.replace("Location", "")
            npy = npy.replace("[", "")
            npy = npy.replace("]", "")
            loc = str.split(npy, ',')
            location = [float(loc[0]), float(loc[1]), float(loc[2])]
            lla_list[i] = location
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
        Yl = list()
        Yl_rae = list()
        for i, path in enumerate(meas_paths):

            if i % 50 == 0:
                print('Finished File ' + str(i) + ' out of ' + str(len(meas_paths)))

            traj_file = traj_list[i]
            # traj_file_rae = traj_list_rae[i]
            cur_traj = traj_num[i]
            cur_lla = lla_list[i]

            npy = str.split(path, '/')[5]
            npy = npy.replace("Location", "")
            npy = npy.replace("[", "")
            npy = npy.replace("]", "")
            loc = str.split(npy, ',')
            location = [float(loc[0]), float(loc[1]), float(loc[2])]

            assert location == cur_lla

            npyt = str.split(path, '/')[-1]
            npyt = str.split(npyt, '_')[3]
            trajn = str.split(npyt, '.tsv')[0]

            # print(trajn + '    ' + cur_traj)
            if cur_traj != trajn:
                print('mismatched trajectories')
                cur_traj

            lat = cur_lla[0]
            lon = cur_lla[1]
            alt = cur_lla[2]

            X = pd.read_csv(path, header=None)
            if X.shape[1] < 4:
                X = pd.read_table(path)
            # X = pd.DataFrame.as_matrix(X)
            X = X.values
            # X = X[25:, :]
            z = np.zeros(shape=[max_seq_len, 4])
            if np.shape(X)[0] > max_seq_len:
                d1 = np.shape(X)[0] - max_seq_len
                X = X[d1:, :]
            l = np.shape(X)[0]
            z[-l:, :] = X
            X0 = np.expand_dims(z, axis=0)

            Y = pd.read_csv(traj_file, header=None)
            if Y.shape[1] < 4:
                Y = pd.read_table(traj_file)
            # Y = pd.DataFrame.as_matrix(Y)
            Y = Y.values
            # Y = Y[25:, :]
            z = np.zeros(shape=[max_seq_len, Y.shape[1]])
            if np.shape(Y)[0] > max_seq_len:
                d1 = np.shape(Y)[0] - max_seq_len
                Y = Y[d1:, :]
            l = np.shape(Y)[0]
            z[-l:, :] = Y
            Y0 = np.expand_dims(z, axis=0)

            idx = list()
            for j in range(Y0.shape[0]):
                idx.append(j)

            # Y = pd.read_csv(traj_file_rae, header=None)
            # if Y.shape[1] < 2:
            #     Y = pd.read_table(traj_file_rae)
            # Y = pd.DataFrame.as_matrix(Y)
            # # Y = Y[:20000, :]
            # z = np.zeros(shape=[max_seq_len, Y.shape[1]])
            # if np.shape(Y)[0] > max_seq_len:
            #     d1 = np.shape(Y)[0] - max_seq_len
            #     Y = Y[d1:, :]
            # l = np.shape(Y)[0]
            # # X = X[:20000, :]
            # # try:
            # z[-l:, :] = Y
            # # except:
            # #     pdb.set_trace()
            # #     pass
            # Y0_rae = np.expand_dims(z, axis=0)

            # Decimate the data here
            # man_time = np.squeeze(s[:, :, 0] == 1, axis=0)
            # not_man_time = np.squeeze(s[:, :, 0] == 0, axis=0)
            #
            # man_count = np.sum(man_time)
            # n_samplet = n_sample - man_count
            # idx1 = list(compress(idx, man_time))
            # idxt = list(compress(idx, not_man_time))
            # samplet = random.sample(idxt, n_samplet)
            # sample_total = natsorted(idx1 + samplet)

            # ss = np.random.randint(1000, 1500)
            # man_count = int(np.sum(man_time))

            # idx1 = list(compress(idx, man_time))
            # total = int(n_sample)
            # tright = total - int(ss) - man_count
            # m1sidx = int(idx1[0]) - int(ss)
            # m2sidx = m1sidx + man_count + tright + int(ss)
            # n_samplet = n_sample - man_count
            # m1eidx = int(idx1[-1] + 50 / 0.1)
            # idxt = list(compress(idx, not_man_time[i]))
            # samplet = random.sample(idxt, n_samplet)
            # sample_total = natsorted(idx1)

            time_inx = copy.copy(X0[0, :, 0])
            x_data, x_rae, x_uvw, x_enu, ecef_ref, timeb = normalize_data_earth_meas(X0, lat, lon, alt)
            time_outx = copy.copy(x_data[0, :, 0])

            if ~np.all(time_inx == time_outx):
                time_outx

            time_iny = copy.copy(Y0[0, :, 0])
            Y0 = translate_trajectory([0, 0, 0], Y0)
            time_outy = copy.copy(Y0[:, 0])
            if ~np.all(time_iny == time_outy):
                time_outy

            y_data = Y0[np.newaxis, :, :]

            # time_iny = copy.copy(Y0_rae[0, :, 0])
            # # Y0 = translate_trajectory([0, 0, 0], Y0_rae)
            # time_outy = copy.copy(Y0_rae[:, 0])
            # if ~np.all(time_iny == time_outy):
            #     time_outy
            #
            # y_data_rae = Y0[np.newaxis, :, :]

            zz = (y_data == 0).all(2)
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
            Yl.append(y_data)
            # Yl_rae.append(y_data_rae)
            ecef_l.append(ecef_ref[np.newaxis, :])
            lla_l.append(np.array(cur_lla)[np.newaxis, :])

            del ecef_ref, z, zz, x_data, y_data, X0, Y0

        meas_data_eci = np.concatenate(Xleci, axis=0)
        meas_data_rae = np.concatenate(Xlrae, axis=0)
        meas_data_uvw = np.concatenate(Xluvw, axis=0)
        meas_data_enu = np.concatenate(Xlenu, axis=0)
        traj_data = np.concatenate(Yl, axis=0)
        ecef_ref_data = np.concatenate(ecef_l, axis=0)
        lla_data = np.concatenate(lla_l, axis=0)

        # FN = list(itertools.chain.from_iterable(FN0))

        # for i in range(traj_data.shape[0]):
        #     temp = copy.copy(traj_data[i])
        #     traj_data[i] = temp[temp[:, 0].argsort()]
        #     del temp

        # # s_traint = np.repeat(s_traint, 3, axis=1)
        # for i in range(s_traint.shape[0]):
        #     temp = copy.copy(s_traint[i])
        #     s_traint[i] = temp[temp[:, 0].argsort()]
        #     del temp

        # for i in range(meas_data.shape[0]):
        #     temp = copy.copy(meas_data[i])
        #     meas_data[i] = temp[temp[:, 0].argsort()]
        #     del temp

        y_train = []
        x_train_eci = []
        x_train_rae = []
        x_train_uvw = []
        x_train_enu = []
        # s_train = []
        # n_sample = random.randint(x_traint.shape[1]*0.98, x_traint.shape[1])

        for i in range(traj_data.shape[0]):
            y_traintt = np.expand_dims(traj_data[i, :, :7], axis=0)
            # s_traintt = np.expand_dims(s_traint[i, :-1, :], axis=0)
            x_traintt_eci = np.expand_dims(meas_data_eci[i, :, :], axis=0)
            x_traintt_rae = np.expand_dims(meas_data_rae[i, :, :], axis=0)
            x_traintt_uvw = np.expand_dims(meas_data_uvw[i, :, :], axis=0)
            x_traintt_enu = np.expand_dims(meas_data_enu[i, :, :], axis=0)

            zero_3 = np.zeros(shape=[1, 3])
            zero_31 = np.zeros(shape=[1, 1, 3])
            zero_1 = np.zeros(shape=[1, 1, 1]) * HZ

            time = y_traintt[0, :, 0]
            time = time[np.newaxis, :, np.newaxis]
            dt = np.abs(np.diff(time, axis=1))

            first_step = np.nonzero(time)[1][0]

            dt[0, first_step-1, :] = HZ

            dt = np.concatenate([zero_1, dt], axis=1)

            acc_est = np.diff(y_traintt[0, :, -3:], axis=0)
            acc_est = np.concatenate([acc_est, zero_3], axis=0)

            firstNZ = np.nonzero(acc_est)[0][0]
            acc_est = np.expand_dims(np.divide(acc_est, np.squeeze(dt, axis=0), out=np.zeros_like(acc_est), where=np.squeeze(dt, axis=0) != 0), axis=0)
            acc_est[0, firstNZ, :] = copy.copy(acc_est[0, firstNZ + 1, :])

            jerk_est = np.diff(acc_est, axis=1)
            jerk_est = np.concatenate([jerk_est, zero_31], axis=1)

            # a = jerk_est[0, :, :] * 6378137
            # a0 = acc_est[0, :, :] * 6378137

            firstNZJ = np.nonzero(jerk_est)[1][0]

            jerk_est[:, firstNZJ, :] = copy.copy(jerk_est[:, firstNZJ + 1, :])
            jerk_est = np.divide(jerk_est, dt, out=np.zeros_like(jerk_est), where=dt != 0)

            # meanj = np.mean(jerk_est, axis=1)
            # stdj = np.std(jerk_est, axis=1)

            # jerk_est = np.where(np.abs(jerk_est) > meanj + 4 * stdj, 3 * stdj, jerk_est)

            y_traintt = np.concatenate([y_traintt, acc_est], axis=2)
            y_traintt = np.concatenate([y_traintt, jerk_est], axis=2)

            # posm = y_traintt[0, :, 1:4]*6378137
            # velm = y_traintt[0, :, 4:7]
            # accm = y_traintt[0, :, 7:10]
            # jerm = y_traintt[0, :, 10:13]*6378137

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

            time_x = x_traintt_eci[:, :, 0].T
            time_y = copy.copy(y_traintt[:, :, 0].T) * 1
            tt = np.concatenate([time_x, time_y], axis=1)
            dtt = np.abs(time_x-time_y)

            if ~np.all(dtt <= 1e-6):
                times_y = np.nonzero(time_y)[0]
                nzy = times_y[0]
                minty = time_y[nzy]
                maxty = np.max(time_y)
                times_x = np.nonzero(time_x)[0]
                nzx = times_x[0]
                mintx = time_x[nzx]
                maxtx = np.max(time_x)

                # common_times = ismember(times_x, times_y)

                maxt = np.round(np.min([maxtx, maxty]), 1)
                mint = np.round(np.max([mintx, minty]), 1)

                bools_x = np.logical_and(time_x <= maxt, time_x >= mint)
                bools_y = np.logical_and(time_y <= maxt, time_y >= mint)

                overall = np.logical_and(bools_x, bools_y)

                # idx_max = [i+1 for i, x in enumerate(bools_max) if x]
                # idx_min = [i for i, x in enumerate(bools_min) if x]

                if not np.any(overall):
                    print(' ')
                    print('ERROR FOR FILE ')
                    print(traj_list[i])
                    print(' ')
                    # idx_max
                else:

                    # print(traj_list[i])
                    xtemp1 = x_traintt_eci[0, np.array(bools_x)[:, 0], :]
                    xtemp2 = x_traintt_rae[0, np.array(bools_x)[:, 0], :]
                    xtemp3 = x_traintt_uvw[0, np.array(bools_x)[:, 0], :]
                    xtemp4 = x_traintt_enu[0, np.array(bools_x)[:, 0], :]
                    ytemp = y_traintt[0, np.array(bools_y)[:, 0], :]

                    ytemp[:, 0] *= 1

                    tx = list(np.round(xtemp1[:, 0], 2))
                    ty = list(np.round(ytemp[:, 0], 2))

                    txb = [None] * len(tx)
                    tyb = [None] * len(ty)

                    for i in range(len(tyb)):
                        if ty[i] not in tx:
                            tyb[i] = False
                        else:
                            tyb[i] = True

                    for i in range(len(txb)):
                        if tx[i] not in ty:
                            txb[i] = False
                        else:
                            txb[i] = True

                    try:
                        xtemp1 = xtemp1[np.array(txb), :]
                        xtemp2 = xtemp2[np.array(txb), :]
                        xtemp3 = xtemp3[np.array(txb), :]
                        xtemp4 = xtemp4[np.array(txb), :]
                        ytemp = ytemp[np.array(tyb), :]
                    except:
                        xtemp1
                        xtemp2
                        xtemp3
                        xtemp4

                    # assert(xtemp.shape[0] == ytemp.shape[0])

                    z = np.zeros(shape=[max_seq_len, 4])
                    l = np.shape(xtemp1)[0]
                    z[-l:, :] = xtemp1
                    del x_traintt_eci
                    x_traintt_eci = z[np.newaxis, :, :]

                    z = np.zeros(shape=[max_seq_len, 3])
                    l = np.shape(xtemp1)[0]
                    z[-l:, :] = xtemp2
                    del x_traintt_rae
                    x_traintt_rae = z[np.newaxis, :, :]

                    z = np.zeros(shape=[max_seq_len, 3])
                    l = np.shape(xtemp1)[0]
                    z[-l:, :] = xtemp3
                    del x_traintt_uvw
                    x_traintt_uvw = z[np.newaxis, :, :]

                    z = np.zeros(shape=[max_seq_len, 3])
                    l = np.shape(xtemp1)[0]
                    z[-l:, :] = xtemp4
                    del x_traintt_enu
                    x_traintt_enu = z[np.newaxis, :, :]

                    z = np.zeros(shape=[max_seq_len, 13])
                    l = np.shape(ytemp)[0]
                    z[-l:, :] = ytemp
                    del y_traintt
                    y_traintt = z[np.newaxis, :, :]

                    time_x = x_traintt_eci[:, :, 0].T
                    time_y = y_traintt[:, :, 0].T
                    # tt = np.concatenate([time_x, time_y], axis=1)
                    dtt = np.abs(time_x - time_y)
                    if ~np.all(dtt <= 1e-6):
                        print(' ')
                        print('Times still not properly aligned ')
                        print(traj_list[i])
                        print(' ')

            y_traintt = y_traintt[:, :, 1:]

            y = np.nan_to_num(y_traintt)
            # s = np.nan_to_num(s_traintt)
            x_eci = np.nan_to_num(x_traintt_eci)
            x_rae = np.nan_to_num(x_traintt_rae)
            x_uvw = np.nan_to_num(x_traintt_uvw)
            x_enu = np.nan_to_num(x_traintt_enu)

            # man_time = np.squeeze(s[:, :, 0] == 1, axis=0)
            # not_man_time = np.squeeze(s[:, :, 0] == 0, axis=0)
            #
            # man_count = np.sum(man_time)
            # n_samplet = n_sample - man_count
            # idx1 = list(compress(idx, man_time))
            # idxt = list(compress(idx, not_man_time))
            # samplet = random.sample(idxt, n_samplet)
            # sample_total = natsorted(idx1 + samplet)

            # ss = np.random.randint(1000, 1500)
            # man_count = int(np.sum(man_time))

            # idx1 = list(compress(idx, man_time))
            # total = int(n_sample)
            # tright = total - int(ss) - man_count
            # m1sidx = int(idx1[0]) - int(ss)
            # m2sidx = m1sidx + man_count + tright + int(ss)
            # n_samplet = n_sample - man_count
            # m1eidx = int(idx1[-1] + 50 / 0.1)
            # idxt = list(compress(idx, not_man_time[i]))
            # samplet = random.sample(idxt, n_samplet)
            # sample_total = natsorted(idx1)
            y = np.expand_dims(y[0, :, :], axis=0)
            # s = np.expand_dims(s[0, :, :1], axis=0)
            x_eci = np.expand_dims(x_eci[0, :, :], axis=0)
            x_rae = np.expand_dims(x_rae[0, :, :], axis=0)
            x_uvw = np.expand_dims(x_uvw[0, :, :], axis=0)
            x_enu = np.expand_dims(x_enu[0, :, :], axis=0)

            y_train.append(y)
            # s_train.append(s)
            x_train_eci.append(x_eci)
            x_train_rae.append(x_rae)
            x_train_uvw.append(x_uvw)
            x_train_enu.append(x_enu)

        x_train_eci = np.concatenate(x_train_eci, axis=0)
        x_train_rae = np.concatenate(x_train_rae, axis=0)
        x_train_uvw = np.concatenate(x_train_uvw, axis=0)
        x_train_enu = np.concatenate(x_train_enu, axis=0)
        y_train_out = np.concatenate(y_train, axis=0)

        # y_train = normalize_data_earth_state(np.concatenate(y_train, axis=0))
        # s_train = np.concatenate(s_train, axis=0)

        # FND = OrderedDict.fromkeys(FN)
        # del y, x, y_traintt, x_traintt
        x_data_out = np.concatenate([x_train_eci, x_train_rae, x_train_enu, x_train_uvw], axis=2)

        return x_data_out, y_train_out, self._count, [], ecef_ref_data, lla_data, meas_paths


if __name__ == "__main__":

    meas_dir = 'D:/TrackFilterData/Delivery_13/5k25hz_oop_broad_data/NoiseRAE/'
    state_dir = 'D:/TrackFilterData/Delivery_13/5k25hz_oop_broad_data/Translate/'

    ds = DataServerLive(meas_dir=meas_dir, state_dir=state_dir)
    iters = ds._num_examples0 // 512
    from propagation_utils import *
    from helper2 import *

    for _ in range(iters):
        x0, y0, batch_number, total_batches, ecef_ref, lla_data, meas_list = ds.load(batch_size=512, constant=False, test=False, max_seq_len=2750, HZ=25)

        ecef_ref = np.ones_like(x0[:, :, :3]) * ecef_ref[:, np.newaxis, :]
        lla_data = np.ones_like(x0[:, :, :3]) * lla_data[:, np.newaxis, :]
        # FND = OrderedDict.fromkeys(meas_list)

        fname = 'oop_maneuver_batch_' + str(_) + '.h5'
        hf = h5py.File(fname, 'w')
        hf.create_dataset('x_data', data=x0)
        hf.create_dataset('y_data', data=y0)
        hf.create_dataset('ecef_data', data=ecef_ref)
        hf.create_dataset('lla_data', data=lla_data)
        hf.close()

        del x0, y0, ecef_ref, lla_data