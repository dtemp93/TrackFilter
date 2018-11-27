from noise_format import *

import numpy as np
import pandas as pd
import os
import sys

base_dir = 'D:/TrackFilterData/AdvancedBroad/NoiseRAE/'

all_files = os.listdir(base_dir)
for i, directory in enumerate(all_files):

    if 'error' not in directory:
        mdir = base_dir + directory
        edir = mdir.replace('NoiseRAE', 'SigmaRAE')

        print('In dir ' + str(i) + ' out of ' + str(len(all_files)))

        get_sigmas(mdir, edir, mdir + ' error')
