#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:57:04 2022

@author: tshan@urmc-sh.rochester.edu
"""

import numpy as np
import scipy.signal as signal
from numpy.fft import irfft, rfft, fft, ifft
import matplotlib.pyplot as plt
import h5py

from expyfun.io import read_wav, read_tab
from expyfun.io import write_hdf5, read_hdf5
import mne


import matplotlib.pyplot as plt

# %% Define Filtering Functions

def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


# %% Parameters
# Anlysis
is_click = True # if derive click ABR
is_ABR = True # if derive only ABR channels
Bayesian = True # Bayesian averaging
# Stim param
stim_fs = 48000 # stimulus sampling frequency
t_click = 60 # click trial length
click_rate = 40
# EEG param
eeg_n_channel = 2 # total channel of ABR
eeg_fs = 10000 # eeg sampling frequency
eeg_f_hp = 1 # high pass cutoff
#%% Subject
subject_list = ['subject001', 'subject002', 'subject004' , 'subject003'
                'subject005', 'subject006', 'subject007', 'subject008',
                'subject009', 'subject010', 'subject011', 'subject012',
                'subject013', 'subject015', 'subject016', 'subject017',
                'subject018', 'subject019', 'subject020', 'subject022', 
                'subject023', 'subject024']
subject_list_2 = ['subject003','subject019'] # subject with 2 eeg runs
subject_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19,
             20, 22, 23, 24]
# %% File paths
exp_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicABR/diverse_dataset/'
bids_root = '/Music_vs_Speech_ABR/' # EEG-BIDS root path
audio_file_root = "/present_files/" # Present files root path
# %% Analysis
for subject, subject_id in zip(subject_list, subject_ids):
    if subject in subject_list_2:
        run = '01'
        eeg_vhdr = bids_root + 'sub-'+f'{subject_id:02}'+'/eeg/'+'sub-'+f'{subject_id:02}'+'_task-MusicvsSpeech_run-'+run+'_eeg.vhdr'
    else:
        eeg_vhdr = bids_root + 'sub-'+f'{subject_id:02}'+'/eeg/'+'sub-'+f'{subject_id:02}'+'_task-MusicvsSpeech_eeg.vhdr'
    eeg_raw = mne.io.read_raw_brainvision(eeg_vhdr, preload=True)
    if is_ABR:
        channels = ['EP1', 'EP2']
    eeg_raw = eeg_raw.pick_channels(channels)
    events, event_dict = mne.events_from_annotations(eeg_raw)
    # Read Events, correct for tube delay
    events, event_dict = mne.events_from_annotations(eeg_raw)
    events_2trig = np.zeros((0, 3)).astype(int)
    events_new = np.zeros((1, 3)).astype(int)
    events_new[0, :] = events[1, :]
    index = []
    for i in range(len(events)-1):
        if events[i, 2] == 2:
            index += [i]
            events_2trig = np.append(events_2trig, [events[i, :]], axis=0)
            events_new = np.append(events_new, [events[i+1, :]], axis=0)
    events_2trig = np.append(events_2trig, [events[-1, :]], axis=0)
    for i in range(len(events_new)):
        if i < 10:
            events_new[i, 2] = 1
        else:
            events_new[i, 2] = 2
    # Correct fs_eeg
    time_diff = events_2trig[:10, 0] - events_new[:10, 0]
    eeg_fs_n = round(np.mean(time_diff)/(60-0.02), 2)
    # EEG Preprocessing
    print('Filtering raw EEG data...')
    # High-pass filter
    eeg_raw._data = butter_highpass_filter(eeg_raw._data, eeg_f_hp, eeg_fs_n)
    
    # Notch filter
    notch_freq = np.arange(60, 180, 540)
    notch_width = 5
    
    for nf in notch_freq:
        bn, an = signal.iirnotch(nf / (eeg_fs_n / 2.), float(nf) / notch_width)
        eeg_raw._data = signal.lfilter(bn, an, eeg_raw._data)
    
    # Epoching click
    print('Epoching EEG click data...')
    epochs_click = mne.Epochs(eeg_raw, events_new, tmin=0,
                              tmax=(t_click - 1/stim_fs + 1),
                              event_id=1, baseline=None,
                              preload=True, proj=False)
    epoch_click = epochs_click.get_data()
    # Load click wave file
    n_epoch_click = 10
    x_in = np.zeros((n_epoch_click, int(t_click * eeg_fs)), dtype=float)
    for ei in range(n_epoch_click):
        stim, fs_stim = read_wav(audio_file_root + "click/" +
                                 'click{0:03d}'.format(ei) + '.wav')
        stim_abs = np.abs(stim)
        click_times = [(np.where(np.diff(s) > 0)[0] + 1) /
                       float(fs_stim) for s in stim_abs] # Read click event
        click_inds = [(ct * eeg_fs).astype(int) for ct in click_times]
        x_in[ei, click_inds] = 1 # generate click train as x_in

    # Get x_out
    len_eeg = int(eeg_fs * t_click)
    x_out = np.zeros((n_epoch_click, 2, len_eeg))
    for i in range(n_epoch_click):
        x_out_i = epoch_click[i, :, 0:int(eeg_fs_n*t_click)]
        x_out[i, :, :] = mne.filter.resample(x_out_i, eeg_fs, eeg_fs_n)
    x_out = np.mean(x_out, axis=1) # average the two channels
    # Derive ABR
    print('Deriving ABR through cross-correlation...')
    t_start, t_stop = -200e-3, 600e-3 # ABR showed range
    # FFT
    x_in_fft = fft(x_in, axis=-1)
    x_out_fft = fft(x_out, axis=-1)
    # Cross Correlation in frequency domain
    cc = np.real(ifft(x_out_fft * np.conj(x_in_fft)))
    abr = np.mean(cc, axis=0) # average across 10 trials
    abr /= (click_rate*t_click) # real unit value
    # Concatanate click ABR response as [-200, 600] ms lag range
    abr_response = np.concatenate((abr[int(t_start*eeg_fs):],
                                   abr[0:int(t_stop*eeg_fs)]))
    # generate time vector
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)

    # # Saving Click Response
    # print('Saving click response...')
    # write_hdf5('/' + subject + '_crosscorr_click.hdf5',
    #            dict(click_abr_response=abr_response,
    #                 lags=lags), overwrite=True)
