#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:10:01 2021

@author: tong
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
from numpy.fft import fft, ifft
from expyfun.io import write_hdf5, read_hdf5
import mne

"""
This script is used for deriving cortical response using deconvolution with different regressors.
The regressors (half-wave rectified stimulus waveform, IHC, and ANM) were pre-generated.
(refer to rectified_regressor_gen.py and IHC_ANM_regressor_gen.py)

We cut the first 2s and the last 30 ms to reduce the onset and offset effect
"""
# %% Define Filtering Functions

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
is_cort = True # if derive only ABR
Bayesian = True # Bayesian averaging
onset_cut = 2
offset_cut=0.03
# Stim param
stim_fs = 48000 # stimulus sampling frequency
t_click = 60 # click trial length
t_mus = 12 # music or speech trial length
# EEG param
eeg_n_channel = 2 # total channel of ABR
eeg_fs = 10000 # eeg sampling frequency
eeg_f_hp = 1 # high pass cutoff
# %% Cortical Channels
channel_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
                 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
                 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2',
                  'F4', 'F8', 'Fp2']
channel_groups = [channel_names[i:i+eeg_n_channel] for i in range(0, len(channel_names), eeg_n_channel)] 
ref_channels = ['P7', 'P8']
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
# %% Regressor
"""
Change this variables among three regressors ["rect", "IHC", "ANM"] for 
half-wave rectified stimulus wavefors, IHC and ANM
"""
regressor = "ANM" 
# %% Analysis
for subject, subject_id in zip(subject_list, subject_ids):
    if subject in subject_list_2:
        run = '02'
        eeg_vhdr = bids_root + 'sub-'+f'{subject_id:02}'+'/eeg/'+'sub-'+f'{subject_id:02}'+'_task-MusicvsSpeech_run-'+run+'_eeg.vhdr'
    else:
        eeg_vhdr = bids_root + 'sub-'+f'{subject_id:02}'+'/eeg/'+'sub-'+f'{subject_id:02}'+'_task-MusicvsSpeech_eeg.vhdr'
    eeg_raw = mne.io.read_raw_brainvision(eeg_vhdr, preload=True)
        
    len_eeg = int((t_mus-onset_cut-offset_cut)*eeg_fs)
    n_epoch = 40
    
    w_music = dict(acoustic=np.zeros((len(channel_names), len_eeg)),
                   classical=np.zeros((len(channel_names), len_eeg)),
                   hiphop=np.zeros((len(channel_names), len_eeg)),
                   jazz=np.zeros((len(channel_names), len_eeg)),
                   metal=np.zeros((len(channel_names), len_eeg)),
                   pop=np.zeros((len(channel_names), len_eeg)))
    
    response_music = dict(acoustic=np.zeros((len(channel_names), 8000)),
                          classical=np.zeros((len(channel_names), 8000)),
                          hiphop=np.zeros((len(channel_names), 8000)),
                          jazz=np.zeros((len(channel_names), 8000)),
                          metal=np.zeros((len(channel_names), 8000)),
                          pop=np.zeros((len(channel_names), 8000)))
    
    
    w_speech = dict(chn_aud=np.zeros((len(channel_names), len_eeg)),
                    eng_aud=np.zeros((len(channel_names), len_eeg)),
                    interview=np.zeros((len(channel_names), len_eeg)),
                    lecture=np.zeros((len(channel_names), len_eeg)),
                    news=np.zeros((len(channel_names), len_eeg)),
                    talk=np.zeros((len(channel_names), len_eeg)))
    
    response_speech = dict(chn_aud=np.zeros((len(channel_names), 8000)),
                      eng_aud=np.zeros((len(channel_names), 8000)),
                      interview=np.zeros((len(channel_names), 8000)),
                      lecture=np.zeros((len(channel_names), 8000)),
                      news=np.zeros((len(channel_names), 8000)),
                      talk=np.zeros((len(channel_names), 8000)))
    
    #%% Iterating chennel_groups
    for chi in range(len(channel_groups)):
    
        #%% Loading specific channels and filtering EEG data
        eeg_raw = mne.io.read_raw_brainvision(eeg_vhdr, preload=True)
        # Downsampling
        #eeg_raw = mne.filter.resample(eeg_raw, down=eeg_fs/fs_down)
    
        if is_cort:
            channels = channel_groups[chi]
        print(channels)
        eeg_raw = eeg_raw.pick_channels(channels + ref_channels)
        # Reference
        eeg_raw = eeg_raw.set_eeg_reference(ref_channels=ref_channels)
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
        # %% Deconvolution
        # For music response
        data = read_hdf5(regressor_root + regressor + '/music_x_in.hdf5')
        t_start = -0.2
        t_stop = 0.6
        lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
            
        for ti in music_types:
            print(ti)
            n_epoch = 40
            # Load x_in
            x_in_pos = data['x_in_music_pos'][ti][:, onset_cut*eeg_fs:int((t_mus-offset_cut)*eeg_fs)]
            x_in_neg = data['x_in_music_neg'][ti][:, onset_cut*eeg_fs:int((t_mus-offset_cut)*eeg_fs)]
            # Load x_out
            x_out = np.zeros((n_epoch, eeg_n_channel, len_eeg))
        
            for ei in range(n_epoch):
                eeg_temp = epoch[int(eeg_epi[ti][ei]), :, :]
                eeg_temp = mne.filter.resample(eeg_temp, down=eeg_fs_n/eeg_fs)
                x_out[ei, :, :] = eeg_temp[:, onset_cut*eeg_fs:int((t_mus-offset_cut)*eeg_fs)]
                
            # x_in fft
            x_in_pos_fft = fft(x_in_pos)
            x_in_pos_fft_n = np.tile(x_in_pos_fft.reshape((1, n_epoch, len_eeg)), (eeg_n_channel, 1, 1))
            x_in_neg_fft = fft(x_in_neg)
            x_in_neg_fft_n = np.tile(x_in_neg_fft.reshape((1, n_epoch, len_eeg)), (eeg_n_channel, 1, 1))
            # x_out fft
            x_out_fft = fft(x_out)
            if Bayesian:
                ivar = 1 / np.var(x_out, axis=-1)
                weight = ivar/np.nansum(ivar, axis=0)
            # TRF
            denom_pos = np.mean(x_in_pos_fft * np.conj(x_in_pos_fft), axis=0)
            denom_neg = np.mean(x_in_neg_fft * np.conj(x_in_neg_fft), axis=0)
            w_pos = []
            w_neg = []
            for ei in range(n_epoch):
                w_i_pos = (weight[ei, :].reshape((eeg_n_channel, 1)) * np.conj(x_in_pos_fft_n[:, ei, :]) *
                           x_out_fft[ei, :, :]) / denom_pos
                w_i_neg = (weight[ei, :].reshape((eeg_n_channel, 1)) * np.conj(x_in_neg_fft_n[:, ei, :]) *
                           x_out_fft[ei, :, :]) / denom_neg
                w_pos += [w_i_pos]
                w_neg += [w_i_neg]
            w_music[ti][chi*eeg_n_channel:chi*eeg_n_channel+eeg_n_channel, :] = (ifft(np.array(w_pos).sum(0)).real +
                                                                                 ifft(np.array(w_neg).sum(0)).real) / 2
            response_music[ti][chi*eeg_n_channel:chi*eeg_n_channel+eeg_n_channel, :] = np.roll(np.concatenate((w_music[ti][chi*eeg_n_channel:chi*eeg_n_channel+eeg_n_channel, int(t_start*eeg_fs):],
                                                                                   w_music[ti][chi*eeg_n_channel:chi*eeg_n_channel+eeg_n_channel, 0:int(t_stop*eeg_fs)]), axis=-1),  int(3.4*eeg_fs/1000))

        
        # For speech response
        data = read_hdf5(regressor_root + regressor + '/speech_x_in.hdf5')
        t_start = -0.2
        t_stop = 0.6
        lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
        
        
        for ti in speech_types:
            print(ti)
            n_epoch = 40
            # Load x_in
            x_in_pos = data['x_in_speech_pos'][ti][:, onset_cut*eeg_fs:int((t_mus-offset_cut)*eeg_fs)]
            x_in_neg = data['x_in_speech_neg'][ti][:, onset_cut*eeg_fs:int((t_mus-offset_cut)*eeg_fs)]
            # Load x_out
            x_out = np.zeros((n_epoch, eeg_n_channel, len_eeg))
            for ei in range(n_epoch):
                eeg_temp = epoch[int(eeg_epi[ti][ei]), :, :]
                eeg_temp = mne.filter.resample(eeg_temp, down=eeg_fs_n/eeg_fs)
                x_out[ei, :, :] = eeg_temp[:, onset_cut*eeg_fs:int((t_mus-offset_cut)*eeg_fs)]
    
            
            # x_in fft
            x_in_pos_fft = fft(x_in_pos)
            x_in_pos_fft_n = np.tile(x_in_pos_fft.reshape((1, n_epoch, len_eeg)), (eeg_n_channel, 1, 1))
            x_in_neg_fft = fft(x_in_neg)
            x_in_neg_fft_n = np.tile(x_in_neg_fft.reshape((1, n_epoch, len_eeg)), (eeg_n_channel, 1, 1))
            # x_out fft
            x_out_fft = fft(x_out)
            if Bayesian:
                ivar = 1 / np.var(x_out, axis=-1)
                weight = ivar/np.nansum(ivar, axis=0)
            # TRF
            denom_pos = np.mean(x_in_pos_fft * np.conj(x_in_pos_fft), axis=0)
            denom_neg = np.mean(x_in_neg_fft * np.conj(x_in_neg_fft), axis=0)
            w_pos = []
            w_neg = []
            for ei in range(n_epoch):
                w_i_pos = (weight[ei, :].reshape((eeg_n_channel, 1)) * np.conj(x_in_pos_fft_n[:, ei, :]) *
                           x_out_fft[ei, :, :]) / denom_pos
                w_i_neg = (weight[ei, :].reshape((eeg_n_channel, 1)) * np.conj(x_in_neg_fft_n[:, ei, :]) *
                           x_out_fft[ei, :, :]) / denom_neg
                w_pos += [w_i_pos]
                w_neg += [w_i_neg]
            w_speech[ti][chi*eeg_n_channel:chi*eeg_n_channel+eeg_n_channel, :] = (ifft(np.array(w_pos).sum(0)).real +
                            ifft(np.array(w_neg).sum(0)).real) / 2
            response_speech[ti][chi*eeg_n_channel:chi*eeg_n_channel+eeg_n_channel, :] = np.roll(np.concatenate((w_speech[ti][chi*eeg_n_channel:chi*eeg_n_channel+eeg_n_channel, int(t_start*eeg_fs):],
                                                  w_speech[ti][chi*eeg_n_channel:chi*eeg_n_channel+eeg_n_channel, 0:int(t_stop*eeg_fs)]), axis=-1), int(3.4*eeg_fs/1000))
        
        del  x_in_pos_fft_n, x_in_neg_fft_n, denom_pos, denom_neg, w_i_pos, w_i_neg, x_in_pos_fft, x_out_fft, weight, ivar, epoch, eeg_temp

    # %% Save file
    # write_hdf5(eeg_path + '/' + subject + '_cortical_response_regANM_noonoff-2-30m.hdf5',
    #        dict(w_music=w_music, response_music=response_music,
    #             w_speech=w_speech, response_speech=response_speech,
    #             lags=lags), overwrite=True)
    