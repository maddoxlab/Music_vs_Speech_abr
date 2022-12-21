#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:07:42 2022

@author: tshan@urmc-sh.rochester.edu
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
from numpy.fft import fft, ifft
from scipy.stats import pearsonr
from scipy.stats import wilcoxon
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
is_ABR = True # if derive only ABR
Bayesian = True # Bayesian averaging
# Stim param
stim_fs = 48000 # stimulus sampling frequency
t_click = 60 # click trial length
t_mus = 12 # music or speech trial length
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
bids_root = '/Music_vs_Speech_ABR/' # EEG-BIDS root path
audio_file_root = '/present_files/' # Present files root path
regressor_root = '/regressors/'
# %% Analysis
for subject, subject_id in zip(subject_list, subject_ids):
    # %% Loading and filtering EEG data
    if subject in subject_list_2:
        run = '02'
        eeg_vhdr = bids_root + 'sub-'+f'{subject_id:02}'+'/eeg/'+'sub-'+f'{subject_id:02}'+'_task-MusicvsSpeech_run-'+run+'_eeg.vhdr'
    else:
        eeg_vhdr = bids_root + 'sub-'+f'{subject_id:02}'+'/eeg/'+'sub-'+f'{subject_id:02}'+'_task-MusicvsSpeech_eeg.vhdr'
    eeg_raw = mne.io.read_raw_brainvision(eeg_vhdr, preload=True)
    if is_ABR:
        channels = ['EP1', 'EP2']
    eeg_raw = eeg_raw.pick_channels(channels)
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
    if subject in subject_list_2:
        time_diff = events_2trig[:480, 0] - events_new[0:480, 0]
    else:
        time_diff = events_2trig[10:490, 0] - events_new[10:490, 0]
    eeg_fs_n = round(np.mean(time_diff)/(12-0.02), 2)
    
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

    # %% Epoch params
    # general experiment
    n_type_music = 6  # number of music types
    n_type_speech = 6  # number of speech types
    n_epoch = 40  # number of epoch in each type
    n_epoch_total = (n_type_music + n_type_speech) * n_epoch
    
    if subject in subject_list_2:
        run = '02'
        events_file_name = bids_root + 'sub-'+f'{subject_id:02}'+'/eeg/'+'sub-'+f'{subject_id:02}'+'_task-MusicvsSpeech_run-'+run+'_events.tsv'
        start_trial = 0
    else:
        events_file_name = bids_root + 'sub-'+f'{subject_id:02}'+'/eeg/'+'sub-'+f'{subject_id:02}'+'_task-MusicvsSpeech_events.tsv'
        start_trial = 10
    events_df = pd.read_csv(events_file_name, sep='\t')
    file_all_list = []
    for ti in np.arange(start_trial, len(events_df)):
        type_name = events_df['trial_type'][ti]
        piece =  events_df['number_trial'][ti]
        file_all_list += [type_name + f'{piece:03}']
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
    # %% Epoch indexing
    music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
    speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
    types = music_types + speech_types
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
    # Get epoch number for every types
    for epi in range(len(file_all_list)):
        stim_type = file_all_list[epi][0:-3]
        stim_ind = int(file_all_list[epi][-3:])
        eeg_epi[stim_type][stim_ind] = epi
    # %% Odd-Even average
    len_eeg = int(t_mus*eeg_fs)
    t_start = -0.2
    t_stop = 0.6
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
    print("Odd-even ave")
    # Every two epochs assigned to one group, e.g., epoch 0&1 -> odd, epoch 2&3 -> even
    odd_epoch = []
    even_epoch = []    
    for ei in range(n_epoch):
        if ei//2%2==0:
            odd_epoch += [ei]
        else:
            even_epoch += [ei]
    # MUSIC
    w_music_odd = dict(acoustic=np.zeros(len_eeg),
                       classical=np.zeros(len_eeg),
                       hiphop=np.zeros(len_eeg),
                       jazz=np.zeros(len_eeg),
                       metal=np.zeros(len_eeg),
                       pop=np.zeros(len_eeg))
    abr_music_odd = dict(acoustic=np.zeros(8000),
                         classical=np.zeros(8000),
                         hiphop=np.zeros(8000),
                         jazz=np.zeros(8000),
                         metal=np.zeros(8000),
                         pop=np.zeros(8000))
    w_music_even = dict(acoustic=np.zeros(len_eeg),
                        classical=np.zeros(len_eeg),
                        hiphop=np.zeros(len_eeg),
                        jazz=np.zeros(len_eeg),
                        metal=np.zeros(len_eeg),
                        pop=np.zeros(len_eeg))
    abr_music_even = dict(acoustic=np.zeros(8000),
                          classical=np.zeros(8000),
                          hiphop=np.zeros(8000),
                          jazz=np.zeros(8000),
                          metal=np.zeros(8000),
                          pop=np.zeros(8000))
    for ti in music_types:
        print(ti)
        data = read_hdf5(regressor_root + 'ANM/music_x_in.hdf5')
        # Load x_in
        x_in_pos = data['x_in_music_pos'][ti]
        x_in_pos_odd = x_in_pos[odd_epoch,:]
        x_in_pos_even = x_in_pos[even_epoch,:]
        x_in_neg = data['x_in_music_neg'][ti]
        x_in_neg_odd = x_in_neg[odd_epoch,:]
        x_in_neg_even = x_in_neg[even_epoch,:]
        # Load x_out
        x_out = np.zeros((n_epoch, eeg_n_channel, len_eeg))
        for ei in range(n_epoch):
            eeg_temp = epoch[int(eeg_epi[ti][ei]), :, :]
            eeg_temp = mne.filter.resample(eeg_temp, down=eeg_fs_n/eeg_fs)
            x_out[ei, :, :] = eeg_temp[:, 0:len_eeg]
        x_out = np.mean(x_out, axis=1)
        x_out_odd = x_out[odd_epoch,:]
        x_out_even = x_out[even_epoch,:]
        # x_in fft
        x_in_pos_fft_odd = fft(x_in_pos_odd)
        x_in_neg_fft_odd = fft(x_in_neg_odd)
        x_in_pos_fft_even = fft(x_in_pos_even)
        x_in_neg_fft_even = fft(x_in_neg_even)
        # x_out fft
        x_out_fft_odd = fft(x_out_odd)
        x_out_fft_even = fft(x_out_even)
        if Bayesian:
            ivar_odd = 1 / np.var(x_out_odd, axis=1)
            weight_odd = ivar_odd/np.nansum(ivar_odd)
            ivar_even = 1 / np.var(x_out_even, axis=1)
            weight_even = ivar_even/np.nansum(ivar_even)
            
        # TRF-odd
        print(ti,"-odd")
        denom_pos_odd = np.mean(x_in_pos_fft_odd * np.conj(x_in_pos_fft_odd), axis=0)
        denom_neg_odd = np.mean(x_in_neg_fft_odd * np.conj(x_in_neg_fft_odd), axis=0)
        w_pos_odd = []
        w_neg_odd = []
        for eo in range(int(n_epoch/2)):
            w_i_pos_odd = (weight_odd[eo] * np.conj(x_in_pos_fft_odd[eo, :]) *
                           x_out_fft_odd[eo, :]) / denom_pos_odd
            w_i_neg_odd = (weight_odd[eo] * np.conj(x_in_neg_fft_odd[eo, :]) *
                           x_out_fft_odd[eo, :]) / denom_neg_odd
            w_pos_odd += [w_i_pos_odd]
            w_neg_odd += [w_i_neg_odd]
        w_music_odd[ti] = (ifft(np.array(w_pos_odd).sum(0)).real + ifft(np.array(w_neg_odd).sum(0)).real) / 2
        abr_music_odd[ti] = np.roll(np.concatenate((w_music_odd[ti][int(t_start*eeg_fs):],
                                w_music_odd[ti][0:int(t_stop*eeg_fs)])), int(2.75*eeg_fs/1000))
        
        # TRF-even
        print(ti,"-even")
        denom_pos_even = np.mean(x_in_pos_fft_even * np.conj(x_in_pos_fft_even), axis=0)
        denom_neg_even = np.mean(x_in_neg_fft_even * np.conj(x_in_neg_fft_even), axis=0)
        w_pos_even = []
        w_neg_even = []
        for ee in range(int(n_epoch/2)):
            w_i_pos_even = (weight_even[ee] * np.conj(x_in_pos_fft_even[ee, :]) *
                            x_out_fft_even[ee, :]) / denom_pos_even
            w_i_neg_even = (weight_even[ee] * np.conj(x_in_neg_fft_even[ee, :]) *
                            x_out_fft_even[ee, :]) / denom_neg_even
            w_pos_even += [w_i_pos_even]
            w_neg_even += [w_i_neg_even]
        w_music_even[ti] = (ifft(np.array(w_pos_even).sum(0)).real + ifft(np.array(w_neg_even).sum(0)).real) / 2
        abr_music_even[ti] = np.roll(np.concatenate((w_music_even[ti][int(t_start*eeg_fs):], 
                              w_music_even[ti][0:int(t_stop*eeg_fs)])), int(2.75*eeg_fs/1000))
    
    # SPEECH
    w_speech_odd = dict(chn_aud=np.zeros(len_eeg),
                       eng_aud=np.zeros(len_eeg),
                       interview=np.zeros(len_eeg),
                       lecture=np.zeros(len_eeg),
                       news=np.zeros(len_eeg),
                       talk=np.zeros(len_eeg))
    abr_speech_odd = dict(chn_aud=np.zeros(8000),
                          eng_aud=np.zeros(8000),
                          interview=np.zeros(8000),
                          lecture=np.zeros(8000),
                          news=np.zeros(8000),
                          talk=np.zeros(8000))
    w_speech_even = dict(chn_aud=np.zeros(len_eeg),
                         eng_aud=np.zeros(len_eeg),
                         interview=np.zeros(len_eeg),
                         lecture=np.zeros(len_eeg),
                         news=np.zeros(len_eeg),
                         talk=np.zeros(len_eeg))
    abr_speech_even = dict(chn_aud=np.zeros(8000),
                           eng_aud=np.zeros(8000),
                           interview=np.zeros(8000),
                           lecture=np.zeros(8000),
                           news=np.zeros(8000),
                           talk=np.zeros(8000))
    
    for ti in speech_types:
        print(ti)
        data = read_hdf5(regressor_root + 'ANM/speech_x_in.hdf5')
        # Load x_in
        x_in_pos = data['x_in_speech_pos'][ti]
        x_in_pos_odd = x_in_pos[odd_epoch,:]
        x_in_pos_even = x_in_pos[even_epoch,:]
        x_in_neg = data['x_in_speech_neg'][ti]
        x_in_neg_odd = x_in_neg[odd_epoch,:]
        x_in_neg_even = x_in_neg[even_epoch,:]
        # Load x_out
        x_out = np.zeros((n_epoch, eeg_n_channel, len_eeg))
        for ei in range(n_epoch):
            eeg_temp = epoch[int(eeg_epi[ti][ei]), :, :]
            eeg_temp = mne.filter.resample(eeg_temp, down=eeg_fs_n/eeg_fs)
            x_out[ei, :, :] = eeg_temp[:, 0:len_eeg]
        x_out = np.mean(x_out, axis=1)
        x_out_odd = x_out[odd_epoch,:]
        x_out_even = x_out[even_epoch,:]
        # x_in fft
        x_in_pos_fft_odd = fft(x_in_pos_odd)
        x_in_neg_fft_odd = fft(x_in_neg_odd)
        x_in_pos_fft_even = fft(x_in_pos_even)
        x_in_neg_fft_even = fft(x_in_neg_even)
        # x_out fft
        x_out_fft_odd = fft(x_out_odd)
        x_out_fft_even = fft(x_out_even)
        if Bayesian:
            ivar_odd = 1 / np.var(x_out_odd, axis=1)
            weight_odd = ivar_odd/np.nansum(ivar_odd)
            ivar_even = 1 / np.var(x_out_even, axis=1)
            weight_even = ivar_even/np.nansum(ivar_even)
            
        # TRF-odd
        print(ti,"-odd")
        denom_pos_odd = np.mean(x_in_pos_fft_odd * np.conj(x_in_pos_fft_odd), axis=0)
        denom_neg_odd = np.mean(x_in_neg_fft_odd * np.conj(x_in_neg_fft_odd), axis=0)
        w_pos_odd = []
        w_neg_odd = []
        for eo in range(int(n_epoch/2)):
            w_i_pos_odd = (weight_odd[eo] * np.conj(x_in_pos_fft_odd[eo, :]) *
                           x_out_fft_odd[eo, :]) / denom_pos_odd
            w_i_neg_odd = (weight_odd[eo] * np.conj(x_in_neg_fft_odd[eo, :]) *
                           x_out_fft_odd[eo, :]) / denom_neg_odd
            w_pos_odd += [w_i_pos_odd]
            w_neg_odd += [w_i_neg_odd]
        w_speech_odd[ti] = (ifft(np.array(w_pos_odd).sum(0)).real +
                            ifft(np.array(w_neg_odd).sum(0)).real) / 2
        abr_speech_odd[ti] = np.roll(np.concatenate((w_speech_odd[ti][int(t_start*eeg_fs):],
                                w_speech_odd[ti][0:int(t_stop*eeg_fs)])),
                                int(2.75*eeg_fs/1000))
        
        # TRF-even
        print(ti,"-even")
        denom_pos_even = np.mean(x_in_pos_fft_even * np.conj(x_in_pos_fft_even), axis=0)
        denom_neg_even = np.mean(x_in_neg_fft_even * np.conj(x_in_neg_fft_even), axis=0)
        w_pos_even = []
        w_neg_even = []
        for ee in range(int(n_epoch/2)):
            w_i_pos_even = (weight_even[ee] * np.conj(x_in_pos_fft_even[ee, :]) *
                            x_out_fft_even[ee, :]) / denom_pos_even
            w_i_neg_even = (weight_even[ee] * np.conj(x_in_neg_fft_even[ee, :]) *
                            x_out_fft_even[ee, :]) / denom_neg_even
            w_pos_even += [w_i_pos_even]
            w_neg_even += [w_i_neg_even]
        w_speech_even[ti] = (ifft(np.array(w_pos_even).sum(0)).real + 
                             ifft(np.array(w_neg_even).sum(0)).real) / 2
        abr_speech_even[ti] = np.roll(np.concatenate((w_speech_even[ti][int(t_start*eeg_fs):],
                                w_speech_even[ti][0:int(t_stop*eeg_fs)])),
                                int(2.75*eeg_fs/1000))
    
    # # %% SAVE FILE
    # write_hdf5('/' + subject + '_abr_odd_even_ANM.hdf5',
    #            dict(abr_music_odd=abr_music_odd,abr_music_even=abr_music_even,
    #                 abr_speech_odd=abr_speech_odd, abr_speech_even=abr_speech_even,
    #                 lags=lags), overwrite=True)
# %% Music-Speech ABR morphology correlation
subject_num = len(subject_list)
exp_path = '/music_spech_abr/'
subject_data_path = [exp_path + i + '/' for i in subject_list]
# Music and speech overall
abr_music_all = dict(acoustic=np.zeros((subject_num, 8000)),
                     classical=np.zeros((subject_num, 8000)),
                     hiphop=np.zeros((subject_num, 8000)),
                     jazz=np.zeros((subject_num, 8000)),
                     metal=np.zeros((subject_num, 8000)),
                     pop=np.zeros((subject_num, 8000)))
abr_allmusic_all = np.zeros((subject_num, 8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regANM.hdf5')
    lags = data['lags']
    for ti in music_types:
        abr_music_all[ti][i] = data['abr_music'][ti]
        abr_allmusic_all[i] += abr_music_all[ti][i]
    abr_allmusic_all[i] = abr_allmusic_all[i] / len(music_types)

abr_speech_all = dict(chn_aud=np.zeros((subject_num, 8000)),
                      eng_aud=np.zeros((subject_num, 8000)),
                      interview=np.zeros((subject_num, 8000)),
                      lecture=np.zeros((subject_num, 8000)),
                      news=np.zeros((subject_num, 8000)),
                      talk=np.zeros((subject_num, 8000)))
abr_allspeech_all = np.zeros((subject_num, 8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regANM.hdf5')
    lags = data['lags']
    for ti in speech_types:
        abr_speech_all[ti][i] = data['abr_speech'][ti]
        abr_allspeech_all[i] += abr_speech_all[ti][i]
    abr_allspeech_all[i] = abr_allspeech_all[i] / len(speech_types)
# Lag 0-15 ms
ind_abr = np.where((lags>=0) & (lags<15))
mscorr = np.zeros(subject_num)
msr = np.zeros(subject_num)
for i in range(subject_num):   
    mscorr[i], msr[i] = pearsonr(abr_allmusic_all[i,ind_abr].reshape(-1), abr_allspeech_all[i,ind_abr].reshape(-1))

mscorr_med = np.median(mscorr)

# %% Odd-even morphology correlation
# Odd epochs
abr_music_all_odd = dict(acoustic=np.zeros((subject_num, 8000)),
                         classical=np.zeros((subject_num, 8000)),
                         hiphop=np.zeros((subject_num, 8000)),
                         jazz=np.zeros((subject_num, 8000)),
                         metal=np.zeros((subject_num, 8000)),
                         pop=np.zeros((subject_num, 8000)))
abr_allmusic_all_odd = np.zeros((subject_num, 8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regANM.hdf5')
    lags = data['lags']
    for ti in music_types:
        abr_music_all_odd[ti][i] = data['abr_music_odd'][ti]
        abr_allmusic_all_odd[i] += abr_music_all_odd[ti][i]
    abr_allmusic_all_odd[i] = abr_allmusic_all_odd[i] / len(music_types)

abr_speech_all_odd = dict(chn_aud=np.zeros((subject_num, 8000)),
                          eng_aud=np.zeros((subject_num, 8000)),
                          interview=np.zeros((subject_num, 8000)),
                          lecture=np.zeros((subject_num, 8000)),
                          news=np.zeros((subject_num, 8000)),
                          talk=np.zeros((subject_num, 8000)))
abr_allspeech_all_odd = np.zeros((subject_num, 8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regANM.hdf5')
    lags = data['lags']
    for ti in speech_types:
        abr_speech_all_odd[ti][i] = data['abr_speech_odd'][ti]
        abr_allspeech_all_odd[i] += abr_speech_all_odd[ti][i]
    abr_allspeech_all_odd[i] = abr_allspeech_all_odd[i] / len(speech_types)

abr_odd = np.zeros((subject_num, 8000))
for i in range(subject_num):
    abr_odd[i,:] = (abr_allmusic_all_odd[i,:] + abr_allspeech_all_odd[i,:]) / 2

# Even epochs
abr_music_all_even = dict(acoustic=np.zeros((subject_num, 8000)),
                          classical=np.zeros((subject_num, 8000)),
                          hiphop=np.zeros((subject_num, 8000)),
                          jazz=np.zeros((subject_num, 8000)),
                          metal=np.zeros((subject_num, 8000)),
                          pop=np.zeros((subject_num, 8000)))
abr_allmusic_all_even = np.zeros((subject_num, 8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regANM.hdf5')
    lags = data['lags']
    for ti in music_types:
        abr_music_all_even[ti][i] = data['abr_music_even'][ti]
        abr_allmusic_all_even[i] += abr_music_all_even[ti][i]
    abr_allmusic_all_even[i] = abr_allmusic_all_even[i] / len(music_types)

abr_speech_all_even = dict(chn_aud=np.zeros((subject_num, 8000)),
                           eng_aud=np.zeros((subject_num, 8000)),
                           interview=np.zeros((subject_num, 8000)),
                           lecture=np.zeros((subject_num, 8000)),
                           news=np.zeros((subject_num, 8000)),
                           talk=np.zeros((subject_num, 8000)))
abr_allspeech_all_even = np.zeros((subject_num, 8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regANM.hdf5')
    lags = data['lags']
    for ti in speech_types:
        abr_speech_all_even[ti][i] = data['abr_speech_even'][ti]
        abr_allspeech_all_even[i] += abr_speech_all_even[ti][i]
    abr_allspeech_all_even[i] = abr_allspeech_all_even[i] / len(speech_types)

abr_even = np.zeros((subject_num, 8000))
for i in range(subject_num):
    abr_even[i,:] = (abr_allmusic_all_even[i,:] + abr_allspeech_all_even[i,:]) / 2

# Oddd-even epoch correlation
oecorr = np.zeros(subject_num)
oer = np.zeros(subject_num)
for i in range(subject_num):   
    oecorr[i], oer[i] = pearsonr(abr_odd[i,ind_abr].reshape(-1), abr_even[i,ind_abr].reshape(-1))

oecorr_med = np.median(oecorr)

# %% Stats
stats, p = wilcoxon(mscorr, oecorr, alternatibe='greater')
# %% Plot
dpi=300
fig = plt.figure(dpi=dpi)
plt.rcParams['hatch.linewidth'] = 1
fig.set_size_inches(3.5, 3)
plt.hist(mscorr, bins=np.arange(0.5, 1.025, 0.025), color='C4', alpha=0.6, label='Music-Speech')
plt.vlines(mscorr_med, 0, 5, linestyle='dashed', color='C4', linewidth=1.3, label='Median')
plt.hist(oecorr, bins=np.arange(0.5, 1.025, 0.025), histtype='step', color='C0', alpha=1, label='Null', hatch='///')
plt.vlines(oecorr_med, 0, 5, linestyle='dashed', color='C0',linewidth=1.3, label='Median')
plt.ylim(0,5.5)
plt.xlabel("Correlation Coefficient (Pearson's r)")
plt.ylabel("Count")
plt.legend(loc="upper left", fontsize=8)
plt.grid(alpha=0.5,zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
