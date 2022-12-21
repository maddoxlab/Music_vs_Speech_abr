#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:36:02 2021

@author: tong
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
from numpy.fft import fft, ifft
from expyfun.io import write_hdf5, read_hdf5
import mne

"""
This script is used for deriving ABR using deconvolution with different regressors.
The regressors (half-wave rectified stimulus waveform, IHC, and ANM) were pre-generated.
(refer to rectified_regressor_gen.py and IHC_ANM_regressor_gen.py)
"""
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
audio_file_root = '/present_files/' # Present files waveforms root path
regressor_root = '/regressors/' # Regressor files root path
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
    # %% Analysis 
    # Regressor
    regressor_list = ['rect', 'IHC', 'ANM'] # half-wave rectified stimulus, IHC and ANM regressors
    for regressor in regressor_list:
        # For music response
        len_eeg = int(t_mus*eeg_fs)
        data = read_hdf5(regressor_root + regressor + '/music_x_in.hdf5')
        t_start = -0.2
        t_stop = 0.6
        lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
        
        w_music = dict(acoustic=np.zeros(len_eeg),
                      classical=np.zeros(len_eeg),
                      hiphop=np.zeros(len_eeg),
                      jazz=np.zeros(len_eeg),
                      metal=np.zeros(len_eeg),
                      pop=np.zeros(len_eeg))
        
        abr_music = dict(acoustic=np.zeros(8000),
                        classical=np.zeros(8000),
                        hiphop=np.zeros(8000),
                        jazz=np.zeros(8000),
                        metal=np.zeros(8000),
                        pop=np.zeros(8000))
        
        for ti in music_types:
            print(ti)
            n_epoch = 40
            # Load x_in
            x_in_pos = data['x_in_music_pos'][ti]
            x_in_neg = data['x_in_music_neg'][ti]
            # Load x_out
            x_out = np.zeros((n_epoch, eeg_n_channel, len_eeg))
        
            for ei in range(n_epoch):
                eeg_temp = epoch[int(eeg_epi[ti][ei]), :, :]
                eeg_temp = mne.filter.resample(eeg_temp, down=eeg_fs_n/eeg_fs)
                x_out[ei, :, :] = eeg_temp[:, 0:len_eeg]
            x_out = np.mean(x_out, axis=1)
                
            # x_in fft
            x_in_pos_fft = fft(x_in_pos)
            x_in_neg_fft = fft(x_in_neg)
            # x_out fft
            x_out_fft = fft(x_out)
            if Bayesian:
                ivar = 1 / np.var(x_out, axis=1)
                weight = ivar/np.nansum(ivar)
            # TRF
            denom_pos = np.mean(x_in_pos_fft * np.conj(x_in_pos_fft), axis=0)
            denom_neg = np.mean(x_in_neg_fft * np.conj(x_in_neg_fft), axis=0)
            w_pos = []
            w_neg = []
            for ei in range(n_epoch):
                w_i_pos = (weight[ei] * np.conj(x_in_pos_fft[ei, :]) *
                          x_out_fft[ei, :]) / denom_pos
                w_i_neg = (weight[ei] * np.conj(x_in_neg_fft[ei, :]) *
                          x_out_fft[ei, :]) / denom_neg
                w_pos += [w_i_pos]
                w_neg += [w_i_neg]
            w_music[ti] = (ifft(np.array(w_pos).sum(0)).real +
                          ifft(np.array(w_neg).sum(0)).real) / 2
            abr_music[ti] = np.concatenate((w_music[ti][int(t_start*eeg_fs):],
                                            w_music[ti][0:int(t_stop*eeg_fs)]))
            # shift ABR for IHC and ANM regressor
            if regressor in ['IHC', 'ANM']:
                abr_music[ti] = np.roll(abr_music[ti], int(2.75*eeg_fs/1000))
        
        # For speech response
        data = read_hdf5(regressor_root + regressor + '/speech_x_in.hdf5')
        t_start = -0.2
        t_stop = 0.6
        lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
        
        w_speech = dict(chn_aud=np.zeros(len_eeg),
                        eng_aud=np.zeros(len_eeg),
                        interview=np.zeros(len_eeg),
                        lecture=np.zeros(len_eeg),
                        news=np.zeros(len_eeg),
                        talk=np.zeros(len_eeg))
        
        abr_speech = dict(chn_aud=np.zeros(8000),
                          eng_aud=np.zeros(8000),
                          interview=np.zeros(8000),
                          lecture=np.zeros(8000),
                          news=np.zeros(8000),
                          talk=np.zeros(8000))
        
        for ti in speech_types:
            print(ti)
            n_epoch = 40
            # Load x_in
            x_in_pos = data['x_in_speech_pos'][ti]
            x_in_neg = data['x_in_speech_neg'][ti]
            # Load x_out
            x_out = np.zeros((n_epoch, eeg_n_channel, len_eeg))
            for ei in range(n_epoch):
                eeg_temp = epoch[int(eeg_epi[ti][ei]), :, :]
                eeg_temp = mne.filter.resample(eeg_temp, down=eeg_fs_n/eeg_fs)
                x_out[ei, :, :] = eeg_temp[:, 0:len_eeg]
            x_out = np.mean(x_out, axis=1)
            
            # x_in fft
            x_in_pos_fft = fft(x_in_pos)
            x_in_neg_fft = fft(x_in_neg)
            # x_out fft
            x_out_fft = fft(x_out)
            if Bayesian:
                ivar = 1 / np.var(x_out, axis=1)
                weight = ivar/np.nansum(ivar)
            # TRF
            denom_pos = np.mean(x_in_pos_fft * np.conj(x_in_pos_fft), axis=0)
            denom_neg = np.mean(x_in_neg_fft * np.conj(x_in_neg_fft), axis=0)
            w_pos = []
            w_neg = []
            for ei in range(n_epoch):
                w_i_pos = (weight[ei] * np.conj(x_in_pos_fft[ei, :]) *
                          x_out_fft[ei, :]) / denom_pos
                w_i_neg = (weight[ei] * np.conj(x_in_neg_fft[ei, :]) *
                          x_out_fft[ei, :]) / denom_neg
                w_pos += [w_i_pos]
                w_neg += [w_i_neg]
            w_speech[ti] = (ifft(np.array(w_pos).sum(0)).real +
                            ifft(np.array(w_neg).sum(0)).real) / 2
            abr_speech[ti] = np.concatenate((w_speech[ti][int(t_start*eeg_fs):],
                                            w_speech[ti][0:int(t_stop*eeg_fs)]))
            # shift ABR for IHC and ANM regressor
            if regressor in ['IHC', 'ANM']:
                abr_music[ti] = np.roll(abr_music[ti], int(2.75*eeg_fs/1000))
        
        # %% bandpassing
        abr_music_bp = dict(acoustic=np.zeros(8000),
                            classical=np.zeros(8000),
                            hiphop=np.zeros(8000),
                            jazz=np.zeros(8000),
                            metal=np.zeros(8000),
                            pop=np.zeros(8000))
        abr_music_ave = np.zeros(8000,)
        for ti in music_types:
            abr_music_bp[ti] = butter_bandpass_filter(abr_music[ti], 1, 1500, eeg_fs, order=1)
            abr_music_ave += abr_music_bp[ti]
        abr_music_ave = abr_music_ave / len(music_types)
        
        abr_speech_bp = dict(chn_aud=np.zeros(8000),
                            eng_aud=np.zeros(8000),
                            interview=np.zeros(8000),
                            lecture=np.zeros(8000),
                            news=np.zeros(8000),
                            talk=np.zeros(8000))
        abr_speech_ave = np.zeros(8000,)
        for ti in speech_types:
            abr_speech_bp[ti] = butter_bandpass_filter(abr_speech[ti], 1, 1500, eeg_fs, order=1)
            abr_speech_ave += abr_speech_bp[ti]
        abr_speech_ave = abr_speech_ave / len(music_types)
        
        # write_hdf5('/' + subject + '_abr_response_' + regressor + '.hdf5',
        #           dict(w_music=w_music, abr_music=abr_music,
        #                 w_speech=w_speech, abr_speech=abr_speech,
        #                 abr_music_ave=abr_music_ave, abr_speech_ave=abr_speech_ave,
        #                 lags=lags), overwrite=True)
        
