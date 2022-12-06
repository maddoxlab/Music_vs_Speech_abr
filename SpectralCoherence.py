#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral coherence

Created on Sun Dec 12 21:18:02 2021

@author: tong
"""

import numpy as np
import scipy.signal as sig
from numpy.fft import irfft, rfft, fft, ifft
import matplotlib.pyplot as plt
import h5py
from itertools import combinations

from expyfun.io import read_wav, read_tab
from expyfun.io import write_hdf5, read_hdf5
import mne


import matplotlib.pyplot as plt

# %% Function

def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=1, axis=-1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = sig.lfilter(b, a, data, axis)
    return y


def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=1, axis=-1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sig.lfilter(b, a, data, axis)
    return y


def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1, axis=-1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, data, axis)
    return y

def coherence(x, y, fs, window_size=0.5, absolute=True):
    '''
    import scipy.signal
    Inputs:
        x, y: input signals
        window_size: time chunk size, default 0.5s
        fs: sampling frequency
        magnitude: if True, return absolute real values, otherwise return complex values
    Outputs:
        coh: coherence vector
        freq: frequency vector
    '''
    f, t, Zxx = sig.stft(x, fs=fs, window='boxcar', nperseg=window_size*fs, noverlap=0, return_onesided=True, boundary=None)
    f, t, Zyy = sig.stft(y, fs=fs, window='boxcar', nperseg=window_size*fs, noverlap=0, return_onesided=True, boundary=None)
    n_slice = len(x) / (window_size*fs) 
    Num = np.sum(np.conjugate(Zxx) * Zyy, axis=-1) / n_slice
    Den = np.sqrt((np.sum(np.conjugate(Zxx) * Zxx, axis=-1) / n_slice)*(np.sum(np.conjugate(Zyy) * Zyy, axis=-1) / n_slice))
    if absolute:
        coh = abs(Num) / Den
    else:
        coh = Num / Den
    freq = f
    return coh, freq


# %% Parameters
# Stim param
stim_fs = 48000
stim_db = 65
t_mus = 12
# EEG param
eeg_f_hp = 1
eeg_n_channel = 2
eeg_fs = 10000
n_epoch = int(8*60/12)
len_eeg = int(t_mus*eeg_fs)
# %% LOADING DATA
subject_list = ['subject001', 'subject002', 'subject003', 'subject004' ,
                'subject005', 'subject006', 'subject007', 'subject008',
                'subject009', 'subject010', 'subject011', 'subject012',
                'subject013', 'subject015', 'subject016', 'subject017',
                'subject018', 'subject019', 'subject020', 'subject022',
                'subject023', 'subject024']
subject_list_2 = ['subject003', 'subject019']
exp_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/'
subject_data_path = [exp_path + i + '/' for i in subject_list]
subject_num = len(subject_list)

music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
regressors = ['rec', 'IHC', 'ANM']
overall = False
# %% Coherence params
dur_slice = 0.2
n_slices = int(t_mus / dur_slice)
len_slice = int(dur_slice * eeg_fs)
n_bands = int((eeg_fs / 2) * dur_slice + 1)
# %% Compute coherence
for regressor in regressors:
    predicted_eeg_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/predicted_eeg/'
    is_ABR = True
    
    coh_music = dict(acoustic=np.zeros((subject_num, n_bands), dtype='complex_'),
                     classical=np.zeros((subject_num, n_bands), dtype='complex_'),
                     hiphop=np.zeros((subject_num, n_bands), dtype='complex_'),
                     jazz=np.zeros((subject_num, n_bands), dtype='complex_'),
                     metal=np.zeros((subject_num, n_bands), dtype='complex_'),
                     pop=np.zeros((subject_num, n_bands), dtype='complex_'))
    
    coh_speech = dict(chn_aud=np.zeros((subject_num, n_bands), dtype='complex_'),
                      eng_aud=np.zeros((subject_num, n_bands), dtype='complex_'),
                      interview=np.zeros((subject_num, n_bands), dtype='complex_'),
                      lecture=np.zeros((subject_num, n_bands), dtype='complex_'),
                      news=np.zeros((subject_num, n_bands), dtype='complex_'),
                      talk=np.zeros((subject_num, n_bands), dtype='complex_'))
    
    for subject in subject_list:
        si = subject_list.index(subject)
        ###### TRUE EEG #####
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
            bn, an = sig.iirnotch(nf / (eeg_fs_n / 2.), float(nf) / notch_width)
            eeg_raw._data = sig.lfilter(bn, an, eeg_raw._data)
         # %% Epoch params
        # general experiment
        n_type_music = 6  # number of music types
        n_type_speech = 6  # number of speech types
        n_epoch = int(8*60/12)  # number of epoch in each type
        n_epoch_total = (n_type_music + n_type_speech) * n_epoch
        
        tab_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/subject005/005_2021-06-14 15_41_08.295636.tab'
        tab = read_tab(tab_path, group_start='trial_id',
                       group_end=None, return_params=False)
        file_all_list = []
        type_list = []
        for ti in np.arange(10, len(tab)):
            type_name = tab[ti]['trial_id'][0][0].split(",")[2].split(": ")[1]
            type_list += [type_name]
            piece = tab[ti]['trial_id'][0][0].split(",")[3].split(": ")[1]
            file_all_list += [type_name + piece]
        
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
        # Indexing
        eeg_epi = dict(acoustic=np.zeros(n_epoch),
                       classical=np.zeros(n_epoch),
                       hiphop=np.zeros(n_epoch),
                       jazz=np.zeros(n_epoch),
                       metal=np.zeros(n_epoch),
                       pop=np.zeros(n_epoch),
                       chn_aud=np.zeros(n_epoch),
                       eng_aud=np.zeros(n_epoch),
                       interview=np.zeros(n_epoch),
                       lecture=np.zeros(n_epoch),
                       news=np.zeros(n_epoch),
                       talk=np.zeros(n_epoch))
        
        for epi in range(len(file_all_list)):
            stim_type = file_all_list[epi][0:-3]
            stim_ind = int(file_all_list[epi][-3:])
            eeg_epi[stim_type][stim_ind] = epi
            
        for ti in music_types:
            print(ti)
            ##### TRUE EEG #####
            x_out = np.zeros((n_epoch, eeg_n_channel, len_eeg))
            for ei in range(n_epoch):
                eeg_temp = epoch[int(eeg_epi[ti][ei]), :, :]
                eeg_temp = mne.filter.resample(eeg_temp, down=eeg_fs_n/eeg_fs)
                x_out[ei, :, :] = eeg_temp[:, 0:len_eeg]
            x_out = np.mean(x_out, axis=1) #x_out now shaped as (n_epoch, len_eeg)
            x_out_all = np.reshape(x_out, -1)
            ##### PREDICTED EEG #####
            if overall:
                data = read_hdf5(predicted_eeg_path + regressor + '_predict_x_out_overall.hdf5')
            else:
                data = read_hdf5(predicted_eeg_path + regressor + '_predict_x_out_0.hdf5')
            out_music_predicted = data['out_music_predicted'][ti]
            out_music_predicted_all = np.reshape(out_music_predicted, -1)
            coh_music[ti][si, :], freq = coherence(x_out_all, out_music_predicted_all, fs=eeg_fs, window_size=dur_slice, absolute=False)
        
        for ti in speech_types:
            print(ti)
            ##### TRUE EEG #####
            x_out = np.zeros((n_epoch, eeg_n_channel, len_eeg))
            for ei in range(n_epoch):
                eeg_temp = epoch[int(eeg_epi[ti][ei]), :, :]
                eeg_temp = mne.filter.resample(eeg_temp, down=eeg_fs_n/eeg_fs)
                x_out[ei, :, :] = eeg_temp[:, 0:len_eeg]
            x_out = np.mean(x_out, axis=1) #x_out now shaped as (n_epoch, len_eeg)
            x_out_all = np.reshape(x_out, -1)
            ##### PREDICTED EEG #####
            if overall:
                data = read_hdf5(predicted_eeg_path + regressor + '_predict_x_out_overall.hdf5')
            else:
                data = read_hdf5(predicted_eeg_path + regressor + '_predict_x_out_0.hdf5')
            out_speech_predicted = data['out_speech_predicted'][ti]
            out_speech_predicted_all = np.reshape(out_speech_predicted, -1)
            coh_speech[ti][si, :], freq = coherence(x_out_all, out_speech_predicted_all, fs=eeg_fs, window_size=dur_slice, absolute=False)
        
    # %%
    write_hdf5(predicted_eeg_path + regressor + '_coherence_02.hdf5',
               dict(coh_music=coh_music,
                    coh_speech=coh_speech,
                    dur_slice=dur_slice,
                    freq=freq), overwrite=True)


# %% Noise floor calculation
import scipy.stats as st
import random
n_subject = len(subject_list)
iternum=10000
coh_01_samp_average = np.zeros((iternum, 501), dtype='complex_')
coh_1_samp_average =np.zeros((iternum, 5001), dtype='complex_')

for i in range(iternum):
    print(i)
    coh_01_all= np.zeros(501)
    coh_1_all = np.zeros(5001)
    for si in range(n_subject):
        random_predicted_eeg = np.reshape(np.random.rand(n_epoch, len_eeg), -1)
        random_true_eeg = np.reshape(np.random.rand(n_epoch, len_eeg), -1)
        coh_01, freq_01 = coherence(random_predicted_eeg, random_true_eeg, fs=eeg_fs, window_size=0.1, absolute=False)
        coh_01_all += abs(coh_01)
        coh_1, freq_1 = coherence(random_predicted_eeg, random_true_eeg, fs=eeg_fs, window_size=1, absolute=False)
        coh_1_all += abs(coh_1)
    
    coh_01_samp_average[i, :] = coh_01_all / n_subject
    coh_1_samp_average[i, :] = coh_1_all / n_subject


coh_noise_floor_01 = np.mean(coh_01_samp_average, axis=0)
coh_01_lb = np.percentile(coh_01_samp_average, q=0, axis=0)
coh_01_ub = np.percentile(coh_01_samp_average, q=95, axis=0)

coh_noise_floor_1 = np.mean(coh_1_samp_average, axis=0)
coh_1_lb = np.percentile(coh_1_samp_average, q=0, axis=0)
coh_1_ub = np.percentile(coh_1_samp_average, q=95, axis=0)

write_hdf5(predicted_eeg_path + '/coherence_noise_floor.hdf5',
           dict(coh_noise_floor_01=coh_noise_floor_01,
                coh_01_lb=coh_01_lb,
                coh_01_ub=coh_01_ub,
                coh_noise_floor_1=coh_noise_floor_1,
                coh_1_lb=coh_1_lb,
                coh_1_ub=coh_1_ub,
                freq_01=freq_01,
                freq_1=freq_1), 
            overwrite=True)


