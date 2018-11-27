import numpy
import natsort
import os
import glob
import shutil

for M in range(2):

    new_data_dir = 'D:/TrackFilterData/DualBroad'

    if M == 1:
        data_dir = 'D:/TrackFilterData/OOPBroad'
    else:
        data_dir = 'D:/TrackFilterData/AdvancedBroad'

    meas_dir = '/NoiseRAE/'
    state_dir = '/Translate/'

    # meas_list = os.listdir(data_dir + meas_dir)
    meas_list = list()
    for i, npy in enumerate(os.listdir(data_dir + meas_dir)):
        for j, npy2 in enumerate(os.listdir(data_dir + meas_dir + npy)):
            meas_list.append(data_dir + meas_dir + npy + '/' + npy2)

    state_list = os.listdir(data_dir + state_dir)

    # Move States
    for ii, npy0 in enumerate(state_list):

        if ii % 100 == 0:
            print('finished ' + str(ii) + 'out of ' + str(len(state_list)))

        original_filename = data_dir + state_dir + npy0
        new_filename = new_data_dir + state_dir + npy0

        npyt = str.split(npy0, '/')[-1]
        npyt = str.split(npyt, '_')[3]
        traj = int(str.split(npyt, '.tsv')[0])

        if M == 1:
            if traj % 2 == 0:
                shutil.copy2(original_filename, new_filename)
        elif M == 0:
            # pass
            if traj % 2 == 0:
                pass
            else:
                shutil.copy2(original_filename, new_filename)

    # Move Measurements
    for iii, npy0 in enumerate(meas_list):

        if iii % 100 == 0:
            print('finished ' + str(iii) + 'out of ' + str(len(meas_list)))

        npya = str.split(npy0, '/')
        npyb = npya[4] + '/' + npya[5]

        npy = str.split(npy0, '/')
        vidx0 = ['Location' in t for t in npy]
        vidx = vidx0.index(True)
        npy = npy[vidx]
        npy = npy.replace("Location", "")
        npy = npy.replace("[", "")
        npy = npy.replace("]", "")
        loc = str.split(npy, ',')
        location = [float(loc[0]), float(loc[1]), float(loc[2])]
        sensor_type = loc[-1]
        # lla_list[i] = location
        # sensor_list[i] = sensor_type
        npyt = str.split(npy0, '/')[-1]
        npyt = str.split(npyt, '_')[3]
        traj = int(str.split(npyt, '.tsv')[0])

        original_filename = data_dir + meas_dir + npyb
        new_filename = new_data_dir + meas_dir + npyb

        if M == 1:
            if traj % 2 == 0:
                shutil.copy2(original_filename, new_filename)
        elif M == 0:
            if traj % 2 == 0:
                pass
            else:
                shutil.copy2(original_filename, new_filename)
