#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14 Feb 2024

Get the regressors to a click stimulus

@author: fotisdr, tom, tong
"""

import numpy as np
import cochlea
from mne.filter import resample
from joblib import Parallel, delayed
import ic_cn2018 as nuclei
import re
from expyfun.io import read_wav, write_hdf5
import scipy.signal as signal

# import matplotlib.pyplot as plt

# %% Define functions
def findstring(ref, check):
    r = re.compile("(?:" + "|".join(check) + ")*$")
    if r.match(ref) is not None:
        return True
    return False

def get_rates(stim_up, cf):
    fs_up = int(100e3)
    return(np.array(cochlea.run_zilany2014_rate(stim_up,
                                                fs_up,
                                                anf_types='hsr',
                                                cf=cf,
                                                species='human',
                                                cohc=1,
                                                cihc=1))[:, 0])


def anm(stim, fs_in, stim_pres_db, parallel=True, n_jobs=-1, fs_up=100e3,
        stim_gen_rms=0.01, cf_low=125, cf_high=16e3, shift_cfs=False,
        shift_vals=None):
    """
     fs_up: scalar
        the sampling frequency of the AN model
     stim_gen_rms: float
        the RMS reference of the original stimulus
     shift_cfs: boolean
        shift each CF indpendently so maximum values align at zero
     shift_vals: array-like
        the values (in seconds) by which to shift each cf if shift_cfs == True

     Returns the ANM firing rates summed across channels using the model sampling
     frequency (fs_up). 
    """
    # Resample your stimuli to a higher fs for the model
    stim_up = resample(stim, fs_up, fs_in, npad='auto', n_jobs=n_jobs)
    # Put stim in correct units
    # scalar to put it in units of pascals (double-checked and good)
    sine_rms_at_0db = 20e-6
    db_conv = ((sine_rms_at_0db / stim_gen_rms) * 10 ** (stim_pres_db / 20.))
    stim_up = db_conv * stim_up

    # run the model
    dOct = 1. / 6
    cfs = 2 ** np.arange(np.log2(cf_low), np.log2(cf_high + 1), dOct)
    anf_rates_up = np.zeros([len(cfs), len(stim)*fs_up//fs_in])

    if parallel:
        anf_rates_up = Parallel(n_jobs=n_jobs)([delayed(get_rates)(stim_up, cf)
                                               for cf in cfs])
    else:
        for cfi, cf in enumerate(cfs):
            anf_rates_up[cfi] = get_rates(stim_up, cf)
    # convert into a numpy array
    anf_rates = np.array(anf_rates_up)

    # shift w1 by 1ms if not shifting each cf
    final_shift = int(fs_up*0.001)
    # Optionally, shift each cf independently
    if shift_cfs:
        final_shift = 0  # don't shift everything after aligning channels at 0
        if shift_vals is None:
            # default shift_cfs values (based on 75 dB click)
            shift_vals = np.array([0.0046875, 0.0045625, 0.00447917, 0.00435417, 0.00422917, 0.00416667,
                                   0.00402083, 0.0039375, 0.0038125, 0.0036875, 0.003625, 0.00354167, 0.00341667,
                                   0.00327083, 0.00316667, 0.0030625, 0.00302083, 0.00291667, 0.0028125,
                                   0.0026875, 0.00258333, 0.00247917, 0.00239583, 0.0023125, 0.00220833,
                                   0.00210417, 0.00204167, 0.002, 0.001875, 0.00185417, 0.00175, 0.00170833, 0.001625,
                                   0.0015625, 0.0015, 0.00147917, 0.0014375, 0.00135417, 0.0014375, 0.00129167,
                                   0.00129167, 0.00125, 0.00122917])

        # Allow fewer CFs while still using defaults
        if len(cfs) != len(shift_vals):
            ref_cfs = 2 ** np.arange(np.log2(125), np.log2(16e3 + 1), 1/6)
            picks = [cf in np.round(cfs, 3) for cf in np.round(ref_cfs, 3)]
            shift_vals = shift_vals[picks]

        # Ensure the number of shift values matches the number of cfs
        msg = 'Number of CFs does not match number of known shift values'
        assert(len(shift_vals) == len(cfs)), msg
        lags = np.round(shift_vals * fs_in).astype(int)

        # Shift each channel
        for cfi in range(len(cfs)):
            anf_rates[cfi] = np.roll(anf_rates[cfi], -lags[cfi])
            anf_rates[cfi, -lags[cfi]:] = anf_rates[cfi, -(lags[cfi]+1)]

    # Shift, scale, and sum
    M1 = nuclei.M1
    anm = M1*anf_rates.sum(0)
    anm = np.roll(anm, final_shift)
    anm[:final_shift] = anm[final_shift+1]
    return(anm)

# %% Parameters
stim_fs = 48000
anm_fs = 100000
ic_fs = 24414.0625 / 32
stim_pres_db = 65
t_mus = 0.1
eeg_fs = 10000
n_epoch = 1

# %% File paths
bids_root = '/hdd/data/ds004356/' #Music_vs_Speech_ABR/' # EEG-BIDS root path
audio_file_root = bids_root + 'stimuli/' # Present files root path
regressor_root = bids_root + 'regressors/'

# %% Stim types
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
n_jobs = 1

# # %% stimulus generation
len_eeg = int(t_mus*eeg_fs)
# Music x_in
x_in_click_pos = np.zeros((n_epoch, len_eeg))
x_in_click_neg = np.zeros((n_epoch, len_eeg))
# Load wav file +/-
temp = np.zeros(int(0.1*stim_fs)) # 100 ms total, 100 us click
stimrange = range(100, 100 + 1)
temp[stimrange] = 2e-5 * 2 * np.sqrt(2) * 10**(stim_pres_db/20.) # calibrate

ei = 0

# rect
temp_pos = np.fmax(temp, np.zeros(temp.shape))
temp_neg = - np.fmin(temp, np.zeros(temp.shape))
# print(temp.shape)
temp_pos_rsmp = resample(temp_pos, down=stim_fs/eeg_fs, npad='auto')
# print(temp_pos_rsmp.shape)
temp_neg_rsmp = resample(temp_neg, down=stim_fs/eeg_fs, npad='auto')
x_in_click_pos[ei, :] = temp_pos_rsmp[:len_eeg]
x_in_click_neg[ei, :] = temp_neg_rsmp[:len_eeg]

write_hdf5(regressor_root + 'rect/click_x_in.hdf5',
  dict(x_in_click_pos=x_in_click_pos,
        x_in_click_neg=x_in_click_neg,
        stim_fs=stim_fs), overwrite=True)

# IHC/ANM stimulus calibration
db_conv = ((2e-5 / 0.01) * 10 ** (stim_pres_db / 20.))
temp = temp / db_conv

# ANM
x_in_click_pos = np.zeros((n_epoch, len_eeg))
x_in_click_neg = np.zeros((n_epoch, len_eeg))
# print(temp.shape)
waves_pos = anm(temp, stim_fs, stim_pres_db, n_jobs=n_jobs)
# print(waves_pos.shape)
waves_pos_resmp = resample(waves_pos, down=anm_fs/eeg_fs)
# print(waves_pos_resmp.shape)
# Generate ANM
waves_neg = anm(-temp, stim_fs, stim_pres_db, n_jobs=n_jobs)
waves_neg_resmp = resample(waves_neg, down=anm_fs/eeg_fs) # resample as eeg_fs
x_in_click_pos[ei, :] = waves_pos_resmp[:len_eeg] # added this to match the shapes
x_in_click_neg[ei, :] = waves_neg_resmp[:len_eeg]

write_hdf5(regressor_root + '/ANM/click_x_in.hdf5',
           dict(x_in_click_pos=x_in_click_pos,
                x_in_click_neg=x_in_click_neg,
                fs=eeg_fs), overwrite=True)

