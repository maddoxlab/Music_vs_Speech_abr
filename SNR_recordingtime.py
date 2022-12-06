#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:58:03 2022

@author: tong
"""

import numpy as np
import scipy.signal as signal
from numpy.fft import irfft, rfft, fft, ifft
import matplotlib.pyplot as plt
import h5py

from expyfun.io import read_wav, read_tab
from expyfun.io import write_hdf5, read_hdf5
import mne
import time

import matplotlib.pyplot as plt

# %% Function


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

def abr_SNR(abr_data, lags, abr_time_window=15, noise_range=[-200, -20]):
    """
    abr_time_window: time range in ms define as abr, 15 ms by default
    noise_range: prestim time range to calculate noise level, [-200, -20] ms by default
    """    
    ind_abr = np.where((lags>=0) & (lags<abr_time_window))
    abr_var = np.var(abr_data[ind_abr])
    noise_seg_num = int((noise_range[1]-noise_range[0]) / abr_time_window)
    noise_var = 0
    for i in range(noise_seg_num):
        ind_noise = np.where((lags>=(noise_range[0]+abr_time_window*i)) & (lags<(noise_range[0]+abr_time_window*(i+1))))
        noise_var += np.var(abr_data[ind_noise])
    noise_var = noise_var / noise_seg_num # averaging the var of noise
    SNR = 10*np.log10((abr_var - noise_var)/noise_var)
    return SNR

# %% Parameters
# Anlysis
is_click = False
is_ABR = True
Bayesian = True
# Stim param
stim_fs = 48000
stim_db = 65
t_click = 60
t_mus = 12
# EEG param
eeg_n_channel = 2
eeg_fs = 10000
eeg_f_hp = 1
n_epoch_cat = int(8*60/t_mus)*6 # in each category (all music/all speech)
n_epoch = int(8*60/12)  # number of epoch in each type
len_eeg = int(t_mus*eeg_fs)
# %%
record_time = np.arange(1*12, t_mus*(n_epoch*6+1), 12) # time in s, 1 unit = 12 s/epoch
#%% Subject list
subject_list = ['subject001', 'subject002', 'subject003', 'subject004' ,
                'subject005', 'subject006', 'subject007', 'subject008',
                'subject009', 'subject010', 'subject011', 'subject012',
                'subject013', 'subject015', 'subject016', 'subject017',
                'subject018', 'subject019', 'subject020', 'subject022',
                'subject023', 'subject024']
subject_list_2 = ['subject003', 'subject019']
subject_num = len(subject_list)
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]

regressor='ANM'

snr_music = np.zeros((subject_num, n_epoch_cat))
snr_music_bp = np.zeros((subject_num, n_epoch_cat))
snr_speech = np.zeros((subject_num, n_epoch_cat))
snr_speech_bp = np.zeros((subject_num, n_epoch_cat))
#%% START LOOPING SUBJECTS
for subject in subject_list:
    start_time = time.time()
    si = subject_list.index(subject)
    #%% DATA NEED TO COMPUTE
    abr_music = np.zeros((n_epoch_cat, 8000))
    abr_music_bp = np.zeros((n_epoch_cat, 8000))

    abr_speech = np.zeros((n_epoch_cat, 8000))
    abr_speech_bp = np.zeros((n_epoch_cat, 8000))
    # %% Loading and filtering TRUE EEG data
    eeg_path = "/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/" + subject
    if subject in subject_list_2:
        eeg_vhdr = eeg_path + "/music_diverse_beh_" + subject[-3:] + "_1.vhdr"
    else:
        eeg_vhdr = eeg_path + "/music_diverse_beh_" + subject[-3:] + ".vhdr"
    
    eeg_raw = mne.io.read_raw_brainvision(str(eeg_vhdr), preload=True)
    if is_ABR:
        channels = ['EP1', 'EP2']
    eeg_raw = eeg_raw.pick_channels(channels)
    # Read Events
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
    time_diff = events_2trig[10:, 0] - events_new[10:, 0]
    eeg_fs_n = np.mean(time_diff)/11.98
    eeg_fs_n = 10000.25
    
    # EEG Preprocessing
    print('Filtering raw EEG data...')
    # High-pass filter
    eeg_raw._data = butter_highpass_filter(eeg_raw._data, eeg_f_hp, eeg_fs_n)
    
    # Notch filter
    notch_freq = np.arange(60, 301, 120)
    notch_width = 5
    
    for nf in notch_freq:
        bn, an = signal.iirnotch(nf / (eeg_fs_n / 2.), float(nf) / notch_width)
        eeg_raw._data = signal.lfilter(bn, an, eeg_raw._data)
    # %% Epoch params
    # general experiment
    n_type_music = 6  # number of music types
    n_type_speech = 6  # number of speech type
    n_epoch_total = (n_type_music + n_type_speech) * n_epoch
    
    tab_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/subject005/005_2021-06-14 15_41_08.295636.tab'
    tab = read_tab(tab_path, group_start='trial_id',
                   group_end=None, return_params=False)
    file_all_list = []
    type_list = []
    cat_list = []
    for ti in np.arange(10, len(tab)):
        type_name = tab[ti]['trial_id'][0][0].split(",")[2].split(": ")[1]
        type_list += [type_name]
        if type_name in music_types:
            cat_list += ["music"]
        elif type_name in speech_types:
            cat_list += ["speech"]
        piece = tab[ti]['trial_id'][0][0].split(",")[3].split(": ")[1]
        file_all_list += [type_name + piece]
    
    music_ind = [i for i, x in enumerate(cat_list) if x == "music"]
    speech_ind = [i for i, x in enumerate(cat_list) if x == "speech"]
        # Epoching
    print('Epoching EEG data...')
    epochs = mne.Epochs(eeg_raw, events, event_id=1, tmin=0,
                        tmax=(t_mus - 1/stim_fs + 1),
                        baseline=None, preload=True)
    epoch = epochs.get_data()
    if subject in subject_list_2:
        epoch = epoch[0:480,:]
    else:
        epoch = epoch[10:490,:]
    
    music_epochs = epoch[music_ind]
    music_file_list = [file_all_list[i] for i in music_ind]
    speech_epochs = epoch[speech_ind]
    speech_file_list = [file_all_list[i] for i in speech_ind]
    
    # %% Deriving ABR for different number of epochs
    x_in_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/present_files/' + regressor + '/'
    t_start = -0.2
    t_stop = 0.6
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
    music_data = read_hdf5(x_in_path + 'music_x_in.hdf5')
    sperch_data = read_hdf5(x_in_path + 'speech_x_in.hdf5')
    # Looping from 1 epoch average to 240 epoch average
    for n in range(n_epoch_cat):
        n_epoch_run = n+1
        ######## MUSIC ABR #########
        print(subject + ": epoch run number " + str(n_epoch_run) + ": Music")
        # x in
        x_in_music_pos = np.zeros((n_epoch_run, len_eeg))
        x_in_music_neg = np.zeros((n_epoch_run, len_eeg))
        curr_music_file_list = music_file_list[:n_epoch_run]
        for ei in range(n_epoch_run):
            t = curr_music_file_list[ei][:-3]
            num = int(curr_music_file_list[ei][-3:])
            x_in_music_pos[ei, :] = music_data['x_in_music_pos'][t][num,:]
            x_in_music_neg[ei, :] = music_data['x_in_music_neg'][t][num,:]
        # x out
        x_out_music = np.zeros((n_epoch_run, eeg_n_channel, len_eeg))
        for ei in range(n_epoch_run):
            eeg_temp = music_epochs[ei, :, :]
            eeg_temp = mne.filter.resample(eeg_temp, down=eeg_fs_n/eeg_fs)
            x_out_music[ei, :, :] = eeg_temp[:, 0:len_eeg]
        x_out_music = np.mean(x_out_music, axis=1)
        # x_in fft
        x_in_music_pos_fft = fft(x_in_music_pos)
        x_in_music_neg_fft = fft(x_in_music_neg)
        # x_out fft
        x_out_music_fft = fft(x_out_music)
        if Bayesian:
            ivar = 1 / np.var(x_out_music, axis=1)
            weight = ivar/np.nansum(ivar)
        # TRF
        denom_pos = np.mean(x_in_music_pos_fft * np.conj(x_in_music_pos_fft), axis=0)
        denom_neg = np.mean(x_in_music_neg_fft * np.conj(x_in_music_neg_fft), axis=0)
        w_pos = []
        w_neg = []
        for ei in range(n_epoch_run):
            w_i_pos = (weight[ei] * np.conj(x_in_music_pos_fft[ei, :]) *
                       x_out_music_fft[ei, :]) / denom_pos
            w_i_neg = (weight[ei] * np.conj(x_in_music_neg_fft[ei, :]) *
                       x_out_music_fft[ei, :]) / denom_neg
            w_pos += [w_i_pos]
            w_neg += [w_i_neg]
        w_music = (ifft(np.array(w_pos).sum(0)).real + ifft(np.array(w_neg).sum(0)).real) / 2
        abr_music[n,:] = np.roll(np.concatenate((w_music[int(t_start*eeg_fs):],
                            w_music[0:int(t_stop*eeg_fs)])),int(2.75*eeg_fs/1000))
        abr_music_bp[n,:] = butter_bandpass_filter(abr_music[n,:], 1, 1500, eeg_fs, order=1)
        snr_music[si,n] = abr_SNR(abr_music[n,:], lags, abr_time_window=15, noise_range=[-200, -20])
        snr_music_bp[si,n] = abr_SNR(abr_music_bp[n,:], lags, abr_time_window=15, noise_range=[-200, -20])
        
        ######## SPEECH ABR #########
        print(subject + ": epoch run number " + str(n_epoch_run) + ": Speech")
        # x in
        x_in_speech_pos = np.zeros((n_epoch_run, len_eeg))
        x_in_speech_neg = np.zeros((n_epoch_run, len_eeg))
        curr_speech_file_list = speech_file_list[:n_epoch_run]
        for ei in range(n_epoch_run):
            t = curr_speech_file_list[ei][:-3]
            num = int(curr_speech_file_list[ei][-3:])
            x_in_speech_pos[ei, :] = sperch_data['x_in_speech_pos'][t][num,:]
            x_in_speech_neg[ei, :] = sperch_data['x_in_speech_neg'][t][num,:]
        # x out
        x_out_speech = np.zeros((n_epoch_run, eeg_n_channel, len_eeg))
        for ei in range(n_epoch_run):
            eeg_temp = speech_epochs[ei, :, :]
            eeg_temp = mne.filter.resample(eeg_temp, down=eeg_fs_n/eeg_fs)
            x_out_speech[ei, :, :] = eeg_temp[:, 0:len_eeg]
        x_out_speech = np.mean(x_out_speech, axis=1)
        # x_in fft
        x_in_speech_pos_fft = fft(x_in_speech_pos)
        x_in_speech_neg_fft = fft(x_in_speech_neg)
        # x_out fft
        x_out_speech_fft = fft(x_out_speech)
        if Bayesian:
            ivar = 1 / np.var(x_out_speech, axis=1)
            weight = ivar/np.nansum(ivar)
        # TRF
        denom_pos = np.mean(x_in_speech_pos_fft * np.conj(x_in_speech_pos_fft), axis=0)
        denom_neg = np.mean(x_in_speech_neg_fft * np.conj(x_in_speech_neg_fft), axis=0)
        w_pos = []
        w_neg = []
        for ei in range(n_epoch_run):
            w_i_pos = (weight[ei] * np.conj(x_in_speech_pos_fft[ei, :]) *
                       x_out_speech_fft[ei, :]) / denom_pos
            w_i_neg = (weight[ei] * np.conj(x_in_speech_neg_fft[ei, :]) *
                       x_out_speech_fft[ei, :]) / denom_neg
            w_pos += [w_i_pos]
            w_neg += [w_i_neg]
        w_speech = (ifft(np.array(w_pos).sum(0)).real + ifft(np.array(w_neg).sum(0)).real) / 2
        abr_speech[n,:] = np.roll(np.concatenate((w_speech[int(t_start*eeg_fs):],
                            w_speech[0:int(t_stop*eeg_fs)])),int(2.75*eeg_fs/1000))
        abr_speech_bp[n,:] = butter_bandpass_filter(abr_speech[n,:], 1, 1500, eeg_fs, order=1)
        snr_speech[si,n] = abr_SNR(abr_speech[n,:], lags, abr_time_window=15, noise_range=[-200, -20])
        snr_speech_bp[si,n] = abr_SNR(abr_speech_bp[n,:], lags, abr_time_window=15, noise_range=[-200, -20])
        
        print("RUN--- %s seconds ---" % (time.time() - start_time))

    write_hdf5(eeg_path + '/' + subject + '_abr_response_reg' + regressor + '_by_numEpoch.hdf5',
               dict(abr_music=abr_music, abr_music_bp=abr_music_bp, 
                    abr_speech=abr_speech, abr_speech_bp=abr_speech_bp,lags=lags), overwrite=True)
    
    print("SUBJECT TIME--- %s seconds ---" % (time.time() - start_time))
    
write_hdf5("/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/abr_snr_"+regressor+"_by_numEpoch.hdf5",
           dict(snr_music=snr_music, snr_music_bp=snr_music_bp,
                snr_speech=snr_speech, snr_speech_bp=snr_speech_bp), overwrite=True)

