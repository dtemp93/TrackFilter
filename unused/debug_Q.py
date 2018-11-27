import numpy
from slim_helper_1 import *
from slim_helper_2 import *
from slim_propagation_utils import *

import matplotlib.pyplot as plt

sj_vals = [1e-6, 1e-3, 1e-2, 1e-1, 1, 5, 10, 25, 100, 250, 500]
time_vals = [1e-6, 1e-3, 1e-1, 0.25, 0.5, 0.75, 0.9, 1.0]

dt = 1/25

batch_size = 2

I_4z = np.eye(4, dtype=np.float64) * 0
I_3z = np.eye(3, dtype=np.float64) * 0

I_4z = np.tile(I_4z[np.newaxis, :, :], [batch_size, 1, 1])
I_3z = np.tile(I_3z[np.newaxis, :, :], [batch_size, 1, 1])

zb = np.zeros([batch_size, 4, 2], dtype=np.float64)
om = np.ones([batch_size, 1, 1], dtype=np.float64)
zm = np.zeros([batch_size, 1, 1], dtype=np.float64)
omp = np.ones([1, 1], np.float64)
zmp = np.zeros([1, 1], np.float64)

dt = dt * om

num_state = 12
num_meas = 3

Qt_l = list()
At_l = list()
sj_l = list()
tc_l = list()

for ii in range(len(sj_vals)):
    for jj in range(len(time_vals)):

        Qt, At, Bt, At2 = get_QP_np(dt, om, zm, I_3z, I_4z, zb,
                                            dimension=int(num_state / 3),
                                            sjix=om * sj_vals[ii] ** 2,
                                            sjiy=om * sj_vals[ii] ** 2,
                                            sjiz=om * sj_vals[ii] ** 2,
                                            aji=om * time_vals[jj])
        cur_Qt = Qt[0, :, :]
        cur_At = At2[0, :, :]

        try:
            chol = np.linalg.cholesky(cur_Qt)
        except:
            print('Failed Cholesky ')
            print('SJ was : ' + str(sj_vals[ii]))
            print('TC was : ' + str(time_vals[jj]))
            print(' ')
            cur_Qt
            # emadt = np.exp(-aj * dt)
            # (1 / (2 * aj3)) * (4 * emadt + (2 * aj * dt) - (np.exp(-2 * aj * dt)) - 3)

        cur_Qt_rs = np.reshape(np.diagonal(cur_Qt), [12])
        cur_At_rs = np.reshape(cur_At[:4, :4], [16])

        Qt_l.append(cur_Qt_rs)
        At_l.append(cur_At_rs)
        sj_l.append(sj_vals[ii])
        tc_l.append(time_vals[jj])

all_Qt = np.stack(Qt_l, axis=0)
all_At = np.stack(At_l, axis=0)
all_sj = np.stack(sj_l, axis=0)
all_tc = np.stack(tc_l, axis=0)

plt.figure()
plt.subplot(411)
plt.plot(all_tc, all_At[:, -1])
# plt.pause(0.01)
plt.show()
plt.pause(0.1)
