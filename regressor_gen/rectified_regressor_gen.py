#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:32:21 2022

@author: tshan@urmc-sh.rochester.edu

This script can be used to create the half-wave rectified regressor.
"""
import numpy as np
import scipy.signal as signal
import mne
from numpy.fft import fft, ifft
from expyfun.io import write_hdf5, read_hdf5, read_wav
import os

# %% Parameters
# Stim param
stim_fs = 48000 # stimulus sampling frequency
t_mus = 12 # music or speech trial length
# EEG param
eeg_fs = 10000 # eeg sampling frequency
#%% Preprocess rectified x_in
n_epoch = 40
len_eeg = int(t_mus*eeg_fs)
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
# %% File paths
bids_root = '/hdd/data/ds004356/' # EEG-BIDS root path
audio_file_root = bids_root + 'stimuli/' # Present files root path
regressor_root = bids_root + 'regressors/' # Path to extract the regressors
# Make folder if it doesn't exist
if not os.path.exists(regressor_root + 'rect/'):
    os.mkdir(regressor_root + 'rect/')

# Music x_in
x_in_music_pos = dict(acoustic=np.zeros((n_epoch, len_eeg)),
                      classical=np.zeros((n_epoch, len_eeg)),
                      hiphop=np.zeros((n_epoch, len_eeg)),
                      jazz=np.zeros((n_epoch, len_eeg)),
                      metal=np.zeros((n_epoch, len_eeg)),
                      pop=np.zeros((n_epoch, len_eeg)))
x_in_music_neg = dict(acoustic=np.zeros((n_epoch, len_eeg)),
                      classical=np.zeros((n_epoch, len_eeg)),
                      hiphop=np.zeros((n_epoch, len_eeg)),
                      jazz=np.zeros((n_epoch, len_eeg)),
                      metal=np.zeros((n_epoch, len_eeg)),
                      pop=np.zeros((n_epoch, len_eeg)))
for ti in music_types:
    for ei in range(n_epoch):
        temp, rt = read_wav(audio_file_root + ti + "/" +
                            ti + "{0:03d}".format(ei) + ".wav")
        temp_pos = np.fmax(temp, np.zeros(temp.shape))
        temp_neg = - np.fmin(temp, np.zeros(temp.shape))
        temp_pos_rsmp = mne.filter.resample(temp_pos, down=stim_fs/eeg_fs)
        temp_neg_rsmp = mne.filter.resample(temp_neg, down=stim_fs/eeg_fs)
        x_in_music_pos[ti][ei, :] = temp_pos_rsmp
        x_in_music_neg[ti][ei, :] = temp_neg_rsmp
write_hdf5(regressor_root + 'rect/music_x_in.hdf5',
          dict(x_in_music_pos=x_in_music_pos,
                x_in_music_neg=x_in_music_neg,
                stim_fs=stim_fs), overwrite=True)

# Speech x_in
x_in_speech_pos = dict(chn_aud=np.zeros((n_epoch, len_eeg)),
                      eng_aud=np.zeros((n_epoch, len_eeg)),
                      interview=np.zeros((n_epoch, len_eeg)),
                      lecture=np.zeros((n_epoch, len_eeg)),
                      news=np.zeros((n_epoch, len_eeg)),
                      talk=np.zeros((n_epoch, len_eeg)))
x_in_speech_neg = dict(chn_aud=np.zeros((n_epoch, len_eeg)),
                      eng_aud=np.zeros((n_epoch, len_eeg)),
                      interview=np.zeros((n_epoch, len_eeg)),
                      lecture=np.zeros((n_epoch, len_eeg)),
                      news=np.zeros((n_epoch, len_eeg)),
                      talk=np.zeros((n_epoch, len_eeg)))
for ti in speech_types:
    for ei in range(n_epoch):
        temp, rt = read_wav(audio_file_root + ti + "/" +
                            ti + "{0:03d}".format(ei) + ".wav")
        temp_pos = np.fmax(temp, np.zeros(temp.shape))
        temp_neg = - np.fmin(temp, np.zeros(temp.shape))
        temp_pos_rsmp = mne.filter.resample(temp_pos, down=stim_fs/eeg_fs)
        temp_neg_rsmp = mne.filter.resample(temp_neg, down=stim_fs/eeg_fs)
        x_in_speech_pos[ti][ei, :] = temp_pos_rsmp
        x_in_speech_neg[ti][ei, :] = temp_neg_rsmp

write_hdf5(regressor_root + 'rect/speech_x_in.hdf5',
          dict(x_in_speech_pos=x_in_speech_pos,
                x_in_speech_neg=x_in_speech_neg,
                stim_fs=stim_fs), overwrite=True)
