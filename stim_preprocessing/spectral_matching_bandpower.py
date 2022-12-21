#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:12:25 2021

@author: tong
"""

import numpy as np
from expyfun.io import read_wav
from mne.time_frequency import psd_array_multitaper, psd_array_welch
from scipy.signal import spectrogram, tukey
import scipy.signal as sig
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
import mne

# Parameters

nep_mus = 48
nep_sp = 48
fs = 44100
fs_out = 48000
dur = 60
len_mus = dur*fs
n_epoch = 8

fullset_path = '/Muspeech_Dataset_2020_2/'
music_path = fullset_path + 'music/'
speech_path = fullset_path + 'speech/'

# Loading signal
music_sig = np.zeros((nep_mus, len_mus))
speech_sig = np.zeros((nep_sp, len_mus))

music_gen = os.listdir(music_path)
speech_gen = os.listdir(speech_path)
#music_gen.remove('piano')

for k in range(len(music_gen)):
    for i in range(n_epoch):
        print(['loading', music_gen[k], i])
        temp, fs = read_wav(music_path + music_gen[k] + '/' +
                            music_gen[k] + '00' + str(i) + '.wav')
        music_sig[k*n_epoch+i, :] = temp
for k in range(len(speech_gen)):
    for i in range(n_epoch):
        print(['loading', speech_gen[k], i])
        temp, fs = read_wav(speech_path + speech_gen[k] + '/' +
                            speech_gen[k] + '00' + str(i) + '.wav')
        speech_sig[k*n_epoch+i, :] = temp


# %% try splitting into third octave bands

def split_bands(x, order=6):
    hp_freqs = 2 ** np.arange(np.log2(50), np.log2(fs / 2), 1 / 3.)
    n_bands = len(hp_freqs)
    x_bands = np.zeros((n_bands + 1, x.shape[-1]))
    x_bands[0] = x
    for bi in np.arange(n_bands):
        b, a = sig.butter(order, hp_freqs[bi] / (fs / 2), btype='highpass')
        x_bands[bi + 1] = sig.filtfilt(b, a, x)
    x_bands[:-1] = x_bands[:-1] - x_bands[1:]
    return x_bands

# %% Spectral matching
music_bands_var = np.zeros((28,))
speech_bands_var = np.zeros((28,))
# Averaged variance for each band
# Music
for i in range(nep_mus):
    music_bands_var = music_bands_var + np.reshape(np.var(split_bands(music_sig[i, :]), axis=-1, keepdims=True), -1)
music_bands_var_av = music_bands_var/nep_mus
# Speech
for i in range(nep_sp):
    speech_bands_var = speech_bands_var + np.reshape(np.var(split_bands(speech_sig[i, :]), axis=-1, keepdims=True), -1)
speech_bands_var_av = speech_bands_var/nep_mus
# Overall averaged variance
all_bands_var_av = (music_bands_var_av + speech_bands_var_av) / 2
all_bands_var_av = np.reshape(all_bands_var_av, (28, 1))

# Matching each band variance of each stim with the average band variance
# Music 
for i in range(nep_mus):
    music_bands_new = split_bands(music_sig[i, :]) * (all_bands_var_av / np.var(split_bands(music_sig[i, :]), axis=-1, keepdims=True)) ** 0.5
    music_new = music_bands_new.sum(0)
    music_new = mne.filter.resample(music_new, up=fs_out/fs)
    wavfile.write(fullset_path + "spectral_match_bandpower/" +
                  music_gen[i//n_epoch] + '/' + music_gen[i//n_epoch] +
                  '00' + str(i % n_epoch) + '.wav',
                  fs_out, music_new)
# Speech
for i in range(nep_sp):
    speech_bands_new = split_bands(speech_sig[i, :]) * (all_bands_var_av / np.var(split_bands(speech_sig[i, :]), axis=-1, keepdims=True)) ** 0.5
    speech_new = speech_bands_new.sum(0)
    speech_new = mne.filter.resample(speech_new, up=fs_out/fs)
    wavfile.write(fullset_path + "spectral_match_bandpower/" +
                  speech_gen[i//n_epoch] + '/' + speech_gen[i//n_epoch] +
                  '00' + str(i % n_epoch) + '.wav',
                  fs_out, speech_new)
