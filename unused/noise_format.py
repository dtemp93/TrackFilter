# -*- coding: utf-8 -*-
"""
A script that takes in the path to a directory of measurement files 
and the path to the corresponding dorectory of error files

and creates a directory full of measurement + noise files.  The format of the output files is:
    time, range, azimuth, acceleration, range sigma, azimuth sigma, elevation sigma, and signal-to-noise ratio.

Created on Fri Nov  9 13:18:06 2018

@author: oelbert
"""

# measured_dir = sys.argv[1]
# error_dir = sys.argv[2]
# out_dir = measured_dir+' errors'
import pdb
import os
import sys
import pandas as pd
import numpy as np


def get_sigmas(measured_dir, error_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if 'big' in error_dir:
        usc=1./3.
        bw=1.e6
        th=1.*np.pi/180.
    elif 'med' in error_dir:
        usc=1.
        bw = 5.e5
        th=2.*np.pi/180.
    else:
        usc=1.
        bw = 2.e5
        th=2.*np.pi/180.

    c=299792458.
    km=1.33

    fmlist = os.listdir(measured_dir)
    felist = os.listdir(error_dir)

    for i in range(len(fmlist)):
        if fmlist[i].endswith('.tsv'):

            try:
                outfile = out_dir+'/'+fmlist[i]
                mdat = pd.read_csv(measured_dir + '/' + fmlist[i],header=None, sep='\t')
                tt = mdat.values[:,0]
                rval = mdat.values[:,1]
                azval = mdat.values[:,2]
                elval = mdat.values[:,3]

                edat=pd.read_csv(error_dir + '/' + felist[i],header=None, sep='\t')
                snrs = edat.values[:,4]
                arg = 1./(2.*(10.**(snrs/10.)))
                r_sig = (c/2.)/bw * np.sqrt(3*arg/(np.pi**2)+0.05**2)
                u_sig = usc*np.sin(th)*np.sqrt((arg/km**2 + 1/18.75**2))
                v_sig = np.sin(th)*np.sqrt((arg/km**2 + 1/18.75**2))
                esig = np.arcsin(v_sig)
                asig = 180.*np.arcsin(u_sig/np.cos(esig))/np.pi
                esig = esig*180./np.pi

                findat = pd.DataFrame(list(zip(tt, rval, azval, elval, r_sig, asig, esig, snrs)))
                findat.to_csv(outfile, sep='\t', header=None, index=False)
            except:
                pass