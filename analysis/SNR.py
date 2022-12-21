#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 11:16:44 2022

@author: tshan@urmc-sh.rochester.edu
"""
import numpy as np
import matplotlib.pyplot as plt
from expyfun.io import read_hdf5
# %% Function

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
is_click = True
is_ABR = True

# Stim param
stim_fs = 48000
stim_db = 65
t_click = 60
t_mus = 12
# EEG param
eeg_n_channel = 2
eeg_fs = 10000

abr_time_window = 15
# %% LOADING DATA
subject_list = ['subject001', 'subject002', 'subject003', 'subject004' ,
                'subject005', 'subject006', 'subject007', 'subject008',
                'subject009', 'subject010', 'subject011', 'subject012',
                'subject013', 'subject015', 'subject016', 'subject017',
                'subject018', 'subject019', 'subject020', 'subject022',
                'subject023', 'subject024']
exp_path = '/music_speech_abr/'
subject_data_path = [exp_path + i + '/' for i in subject_list]
subject_num = len(subject_list)
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
# %% Rectified
abr_allmusic_all = np.zeros((subject_num, 8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regrec.hdf5')
    lags = data['lags']
    abr_allmusic_all[i] = data['abr_music_ave'] *1e6
abr_allmusic_ave = np.sum(abr_allmusic_all, axis=0) / subject_num
abr_allmusic_err = np.std(abr_allmusic_all, axis=0) / np.sqrt(subject_num)

abr_allspeech_all = np.zeros((subject_num, 8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regrec.hdf5')
    lags = data['lags']
    abr_allspeech_all[i] = data['abr_speech_ave'] *1e6
abr_allspeech_ave = np.sum(abr_allspeech_all, axis=0) / subject_num
abr_allspeech_err = np.std(abr_allspeech_all, axis=0) / np.sqrt(subject_num)
############ SNR ############
SNR_rectified_music = np.zeros((subject_num,))
SNR_rectified_speech = np.zeros((subject_num,))
for i in range(subject_num):
    SNR_rectified_music[i] = abr_SNR(abr_allmusic_all[i], lags, abr_time_window=abr_time_window, noise_range=[-200, -20])
    SNR_rectified_speech[i] = abr_SNR(abr_allspeech_all[i], lags, abr_time_window=abr_time_window, noise_range=[-200, -20])

SNR_rectified_music_ave = abr_SNR(abr_allmusic_ave, lags, abr_time_window=abr_time_window, noise_range=[-200, -20])
SNR_rectified_speech_ave = abr_SNR(abr_allspeech_ave, lags, abr_time_window=abr_time_window, noise_range=[-200, -20])
# %% ANM
abr_allmusic_all = np.zeros((subject_num, 8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regANM.hdf5')
    lags = data['lags']
    abr_allmusic_all[i] = data['abr_music_ave']
abr_allmusic_ave = np.sum(abr_allmusic_all, axis=0) / subject_num
abr_allmusic_err = np.std(abr_allmusic_all, axis=0) / np.sqrt(subject_num)

abr_allspeech_all = np.zeros((subject_num, 8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regANM.hdf5')
    lags = data['lags']
    abr_allspeech_all[i] = data['abr_speech_ave']
abr_allspeech_ave = np.sum(abr_allspeech_all, axis=0) / subject_num
abr_allspeech_err = np.std(abr_allspeech_all, axis=0) / np.sqrt(subject_num)
############ SNR ############
SNR_ANM_music = np.zeros((subject_num,))
SNR_ANM_speech = np.zeros((subject_num,))
for i in range(subject_num):
    SNR_ANM_music[i] = abr_SNR(abr_allmusic_all[i], lags, abr_time_window=abr_time_window, noise_range=[-200, -20])
    SNR_ANM_speech[i] = abr_SNR(abr_allspeech_all[i], lags, abr_time_window=abr_time_window, noise_range=[-200, -20])

SNR_ANM_music_ave = abr_SNR(abr_allmusic_ave, lags, abr_time_window=abr_time_window, noise_range=[-200, -20])
SNR_ANM_speech_ave = abr_SNR(abr_allspeech_ave, lags, abr_time_window=abr_time_window, noise_range=[-200, -20])
# %% IHC
abr_allmusic_all = np.zeros((subject_num, 8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regIHC.hdf5')
    lags = data['lags']
    abr_allmusic_all[i] = data['abr_music_ave'] *1e8
abr_allmusic_ave = np.sum(abr_allmusic_all, axis=0) / subject_num
abr_allmusic_err = np.std(abr_allmusic_all, axis=0) / np.sqrt(subject_num)

abr_allspeech_all = np.zeros((subject_num, 8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regIHC.hdf5')
    lags = data['lags']
    abr_allspeech_all[i] = data['abr_speech_ave'] *1e8
abr_allspeech_ave = np.sum(abr_allspeech_all, axis=0) / subject_num
abr_allspeech_err = np.std(abr_allspeech_all, axis=0) / np.sqrt(subject_num)
############ SNR ############
SNR_IHC_music = np.zeros((subject_num,))
SNR_IHC_speech = np.zeros((subject_num,))
for i in range(subject_num):
    SNR_IHC_music[i] = abr_SNR(abr_allmusic_all[i], lags, abr_time_window=abr_time_window, noise_range=[-200, -20])
    SNR_IHC_speech[i] = abr_SNR(abr_allspeech_all[i], lags, abr_time_window=abr_time_window, noise_range=[-200, -20])

SNR_IHC_music_ave = abr_SNR(abr_allmusic_ave, lags, abr_time_window=abr_time_window, noise_range=[-200, -20])
SNR_IHC_speech_ave = abr_SNR(abr_allspeech_ave, lags, abr_time_window=abr_time_window, noise_range=[-200, -20])

# %% SNR adjust plotting
# SNR corrected by the number of subjects
sub_corr = 10*np.log10(subject_num)
barWidth = 0.25
br1 = np.arange(2)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig = plt.figure(dpi=300)
fig.set_size_inches(3.5, 3)
plt.grid(alpha=0.5,zorder=0)
plt.bar(br1, [SNR_rectified_music_ave-sub_corr,SNR_rectified_speech_ave-sub_corr], color='C0', width=barWidth, label='Half-wave Rectified',zorder=3)
plt.bar(br2, [SNR_IHC_music_ave-sub_corr,SNR_IHC_speech_ave-sub_corr], color='C4', width=barWidth, label='IHC',zorder=3)
plt.bar(br3, [SNR_ANM_music_ave-sub_corr,SNR_ANM_speech_ave-sub_corr], color='C2', width=barWidth, label='ANM',zorder=3)
plt.xlabel('Stimulus')
plt.ylabel('SNR (dB)')
plt.ylim(-15,30)
plt.xticks([r + barWidth for r in range(2)], ['Music', 'Speech'])
plt.hlines(0, -0.25, 1.75, linestyles='solid', linewidth=2, color='k')
for si in range(subject_num):
    plt.plot([br1[0],br2[0], br3[0]],[SNR_rectified_music[si],SNR_IHC_music[si], SNR_ANM_music[si]], ".-",markersize=2, linewidth=0.5, c='k', alpha=0.3+0.03*si,zorder=4)
    plt.plot([br1[1],br2[1], br3[1]],[SNR_rectified_speech[si],SNR_IHC_speech[si], SNR_ANM_speech[si]], ".-",markersize=2,linewidth=0.5, c='k', alpha=0.3+0.03*si,zorder=4)
lg = plt.legend(fontsize=7, bbox_to_anchor=(1.03, 1.0), loc='upper left')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)