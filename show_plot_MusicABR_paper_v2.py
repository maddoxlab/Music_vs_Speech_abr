#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:10:56 2022

@author: tong
"""

import numpy as np
import scipy.signal as signal
from numpy.fft import irfft, rfft, fft, ifft
import matplotlib.pyplot as plt
import h5py
from scipy.stats import pearsonr
from scipy.stats import wilcoxon

from expyfun.io import read_wav, read_tab
from expyfun.io import write_hdf5, read_hdf5
import mne

import seaborn as sns

import matplotlib.pyplot as plt
from cycler import cycler

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

#%% FIG SETTING
dpi = 300
data_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/'
figure_path = '/home/tong/AMPLab/MusicABR/diverse_dataset/paper_figures/'

plt.rc('axes', titlesize=9, labelsize=8)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('axes', prop_cycle=cycler(color=["#4477AA","#66CCEE","#228833","#CCBB44","#EE6677","#AA3377","#BBBBBB"]))
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

# %% LOADING DATA
subject_list = ['subject001', 'subject002', 'subject003', 'subject004' ,
                'subject005', 'subject006', 'subject007', 'subject008',
                'subject009', 'subject010', 'subject011', 'subject012',
                'subject013', 'subject015', 'subject016', 'subject017',
                'subject018', 'subject019', 'subject020', 'subject022',
                'subject023', 'subject024']
exp_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/'
figure_path=  '/home/tong/AMPLab/MusicABR/diverse_dataset/paper_figures/'
subject_data_path = [exp_path + i + '/' for i in subject_list]
subject_num = len(subject_list)
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]

# %% Click
click_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    click_response_data = read_hdf5(data_path + subject + '/' + subject + '_crosscorr_click.hdf5')
    click_response_all[subi] = click_response_data['click_abr_response'] / (40*60) * 1e6
    lags = click_response_data['lags']
    
click_response_ave = np.sum(click_response_all, axis=0) / subject_num
click_response_err = np.std(click_response_all, axis=0) / np.sqrt(subject_num)

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, click_response_ave, c='k', linewidth=1, label='Click')
plt.fill_between(lags, click_response_ave-click_response_err, click_response_ave+click_response_err, alpha=0.6, color='k', linewidth=0)
plt.xlim(-10, 30)
plt.ylim(-0.9,0.73)
plt.ylabel('Potential (μV)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-0.9,0.73, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=7)
plt.tight_layout()
#plt.annotate('N=22', [-9, 0.5], fontsize=8)
plt.annotate('I', [3.4, 0.12], fontsize=7)
plt.annotate('III', [4.6, 0.206], fontsize=7)
plt.annotate('V', [7.3, 0.71], fontsize=7)
plt.savefig(figure_path + 'cross_corr_clickABR.tiff', dpi=dpi, format='tiff')

# Individual response all
fig, ax = plt.subplots(6, 4, sharex=True, sharey=True, figsize=(7.5,8.5))
for si in range(subject_num):
    row_num=(si)//4
    col_num=(si)%4
    line1, = ax[row_num, col_num].plot(lags, click_response_all[si], c='k', linewidth=1, label='click')
    ax[row_num, col_num].set_xlim([-10, 30])
    ax[row_num, col_num].set_ylim([-0.9,1])
    ax[row_num, col_num].tick_params(axis='both', which="major", labelsize=7)
    ax[row_num, col_num].yaxis.offsetText.set_fontsize(7)
    ax[row_num, col_num].spines['top'].set_visible(False)
    ax[row_num, col_num].spines['right'].set_visible(False)
    ax[row_num, col_num].vlines(0,-0.9,1, linestyles='solid', color='k', linewidth=1, alpha=0.5)
    ax[row_num, col_num].hlines(0,-10,30, linestyles='solid', color='k', linewidth=1, alpha=0.5)
    ax[row_num, col_num].grid(alpha=0.5)
ax[5, 2].set_visible(False)
ax[5, 3].set_visible(False)
fig.legend(handles=[line1], loc='lower right', bbox_to_anchor=(0.95, 0.15), fontsize=9)
fig.subplots_adjust(wspace=0.2, hspace=0.7)
fig.suptitle("Individual Response (Click)", fontsize=10)
fig.supxlabel('Time (ms)', fontsize=9)
fig.supylabel('Potential (μV)', fontsize=9)
plt.tight_layout()
plt.savefig(figure_path + 'cross_corr_click_individual.tiff', dpi=dpi, format='tiff')
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

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, abr_allmusic_ave, c='C0', linewidth=1, label='Music')
plt.fill_between(lags, abr_allmusic_ave-abr_allmusic_err, abr_allmusic_ave+abr_allmusic_err, alpha=0.6, color='C0',ec=None, linewidth=0)
plt.plot(lags, abr_allspeech_ave, c='C1', linewidth=1, linestyle='--', label='Speech')
plt.fill_between(lags, abr_allspeech_ave-abr_allspeech_err, abr_allspeech_ave+abr_allspeech_err, alpha=0.6, color='C1', ec=None, linewidth=0)
plt.xlim(-10, 30)
plt.ylim(-0.2,0.4)
plt.yticks(visible=False)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-0.2,0.4, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=8, loc='lower right')
plt.annotate('N=22', [-9, 0.3], fontsize=8)
plt.tight_layout()
plt.savefig(figure_path + 'ABR_rect.tiff', dpi=dpi, format='tiff')

#Cortical
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, abr_allmusic_ave, c='C0', linewidth=0.8, label='Music')
plt.fill_between(lags, abr_allmusic_ave-abr_allmusic_err, abr_allmusic_ave+abr_allmusic_err, alpha=0.6, color='C0',ec=None, linewidth=0)
plt.plot(lags, abr_allspeech_ave, c='C1', linewidth=0.8, linestyle='--', label='Speech')
plt.fill_between(lags, abr_allspeech_ave-abr_allspeech_err, abr_allspeech_ave+abr_allspeech_err, alpha=0.6, color='C1', ec=None, linewidth=0)
plt.xlim(-50, 300)
plt.ylim(-0.4,0.4)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.yticks(visible=False)
plt.grid(alpha=0.5)
plt.vlines(0,-0.4,0.4, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=8, loc='lower right')
plt.annotate('N=22', [-45, 0.3], fontsize=8)
plt.tight_layout()
plt.savefig(figure_path + 'ABR_cortical_rect.tiff', dpi=dpi, format='tiff')

# Individual response all
fig, ax = plt.subplots(6, 4, sharex=True, sharey=True, figsize=(7.5,8.5))
for si in range(subject_num):
    row_num=(si)//4
    col_num=(si)%4
    line1, = ax[row_num, col_num].plot(lags, abr_allmusic_all[si], c='C0', linewidth=1, label='Music')
    line2, = ax[row_num, col_num].plot(lags, abr_allspeech_all[si], c='C1', linewidth=1, label='Speech', linestyle='--')
    ax[row_num, col_num].set_xlim([-10, 30])
    ax[row_num, col_num].set_ylim([-0.4,0.6])
    ax[row_num, col_num].tick_params(axis='both', which="major", labelsize=7)
    ax[row_num, col_num].yaxis.offsetText.set_fontsize(7)
    for tick in ax[row_num, col_num].yaxis.get_major_ticks():
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    ax[row_num, col_num].spines['top'].set_visible(False)
    ax[row_num, col_num].spines['right'].set_visible(False)
    ax[row_num, col_num].vlines(0,-0.4,0.6, linestyles='solid', color='k', linewidth=1, alpha=0.5)
    ax[row_num, col_num].hlines(0,-10,30, linestyles='solid', color='k', linewidth=1, alpha=0.5)
    ax[row_num, col_num].grid(alpha=0.5)
ax[5, 2].set_visible(False)
ax[5, 3].set_visible(False)
fig.legend(handles=[line1, line2], loc='lower right', bbox_to_anchor=(0.95, 0.15), fontsize=9)
fig.subplots_adjust(wspace=0.2, hspace=0.7)
fig.suptitle("Individual Response (Half-wave Rectified)", fontsize=10)
fig.supxlabel('Time (ms)', fontsize=9)
fig.supylabel('Magnitude (AU)', fontsize=9)
plt.tight_layout()
plt.savefig(figure_path + 'ABR_rect_individual.tiff', dpi=dpi, format='tiff')

# Good example: subject018; bad example: subject012
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
ax1 = plt.subplot(211)
plt.plot(lags, abr_allmusic_all[11], c='C0', linewidth=1)
plt.plot(lags, abr_allspeech_all[11], c='C1', linewidth=1, linestyle='--')
plt.xlim(-10, 30)
plt.ylim(-0.2,0.5)
plt.yticks(visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('Magnitude (AU)')
plt.grid(alpha=0.5)
plt.vlines(0,-60,80, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax1.set_title('Subject 12')

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(lags, abr_allmusic_all[17], c='C0', linewidth=1)
plt.plot(lags, abr_allspeech_all[17], c='C1', linewidth=1, linestyle='--')
plt.xlim(-10, 30)
plt.ylim(-0.2,0.5)
plt.yticks(visible=False)
plt.xlabel('Time (ms)')
plt.ylabel('Magnitude (AU)')
#fig.text(0.01, 0.5, 'Magnitude (AU)' ,va='center', rotation='vertical', fontsize=9)
plt.grid(alpha=0.5)
plt.vlines(0,-0.2,0.5, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax2.set_title('Subject 18')
fig.tight_layout()
plt.savefig(figure_path +'ABR_rect_subject012018.tiff', dpi=dpi, format='tiff')

############ SNR ############
abr_time_window = 15
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

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, abr_allmusic_ave, c='C2', linewidth=1, label='Music')
plt.fill_between(lags, abr_allmusic_ave-abr_allmusic_err, abr_allmusic_ave+abr_allmusic_err, alpha=0.6, color='C2',ec=None, linewidth=0)
plt.plot(lags, abr_allspeech_ave, c='C3', linewidth=1, linestyle='--', label='Speech')
plt.fill_between(lags, abr_allspeech_ave-abr_allspeech_err, abr_allspeech_ave+abr_allspeech_err, alpha=0.6, color='C3', ec=None, linewidth=0)
plt.xlim(-10, 30)
plt.ylim(-60,80)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.yticks(visible=False)
plt.grid(alpha=0.5)
plt.vlines(0,-60,80, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=8, loc='lower right')
plt.annotate('N=22', [-9, 50], fontsize=8)
plt.annotate('I', [3.4, 15], fontsize=7)
plt.annotate('III', [5.5, 32], fontsize=7)
plt.annotate('V', [7.5, 56], fontsize=7)
plt.tight_layout()
plt.savefig(figure_path + 'ABR_ANM.tiff', dpi=dpi, format='tiff')

#Cortical
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, abr_allmusic_ave, c='C2', linewidth=0.8, label='Music')
plt.fill_between(lags, abr_allmusic_ave-abr_allmusic_err, abr_allmusic_ave+abr_allmusic_err, alpha=0.6, color='C2',ec=None, linewidth=0)
plt.plot(lags, abr_allspeech_ave, c='C3', linewidth=0.8, linestyle='--', label='Speech')
plt.fill_between(lags, abr_allspeech_ave-abr_allspeech_err, abr_allspeech_ave+abr_allspeech_err, alpha=0.6, color='C3', ec=None, linewidth=0)
plt.xlim(-50, 300)
plt.ylim(-60,80)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.yticks(visible=False)
plt.grid(alpha=0.5)
plt.vlines(0,-60,80, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=8, loc='lower right')
plt.annotate('N=22', [-45, 50], fontsize=8)
plt.tight_layout()
plt.savefig(figure_path + 'ABR_cortical_ANM.tiff', dpi=dpi, format='tiff')

# Individual response all
fig, ax = plt.subplots(6, 4, sharex=True, sharey=True, figsize=(7.5,8.5))
for si in range(subject_num):
    row_num=(si)//4
    col_num=(si)%4
    line1, = ax[row_num, col_num].plot(lags, abr_allmusic_all[si], c='C2', linewidth=1, label='Music')
    line2, = ax[row_num, col_num].plot(lags, abr_allspeech_all[si], c='C3', linewidth=1, label='Speech', linestyle='--')
    ax[row_num, col_num].set_xlim([-10, 30])
    ax[row_num, col_num].set_ylim([-60,90])
    ax[row_num, col_num].tick_params(axis='both', which="major", labelsize=7)
    for tick in ax[row_num, col_num].yaxis.get_major_ticks():
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    ax[row_num, col_num].yaxis.offsetText.set_fontsize(7)
    ax[row_num, col_num].spines['top'].set_visible(False)
    ax[row_num, col_num].spines['right'].set_visible(False)
    ax[row_num, col_num].vlines(0,-60,90, linestyles='solid', color='k', linewidth=1, alpha=0.5)
    ax[row_num, col_num].hlines(0,-10,30, linestyles='solid', color='k', linewidth=1, alpha=0.5)
    ax[row_num, col_num].grid(alpha=0.5)
ax[5, 2].set_visible(False)
ax[5, 3].set_visible(False)
fig.legend(handles=[line1, line2], loc='lower right', bbox_to_anchor=(0.95, 0.15), fontsize=9)
fig.subplots_adjust(wspace=0.2, hspace=0.7)
fig.suptitle("Individual Response (ANM)", fontsize=10)
fig.supxlabel('Time (ms)', fontsize=9)
fig.supylabel('Magnitude (AU)', fontsize=9)
plt.tight_layout()
plt.savefig(figure_path + 'ABR_ANM_individual.tiff', dpi=dpi, format='tiff')

# Good example: subject018; bad example: subject012
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
ax1 = plt.subplot(211)
plt.plot(lags, abr_allmusic_all[11], c='C2', linewidth=1)
plt.plot(lags, abr_allspeech_all[11], c='C3', linewidth=1, linestyle='--')
plt.xlim(-10, 30)
plt.ylim(-60,80)
plt.yticks(visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('Magnitude (AU)')
plt.grid(alpha=0.5)
plt.vlines(0,-60,80, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax1.set_title('Subject 12')

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(lags, abr_allmusic_all[17], c='C2', linewidth=1)
plt.plot(lags, abr_allspeech_all[17], c='C3', linewidth=1, linestyle='--')
plt.xlim(-10, 30)
plt.ylim(-60,80)
plt.xlabel('Time (ms)')
plt.ylabel('Magnitude (AU)')
plt.yticks(visible=False)
#fig.text(0.01, 0.5, 'Magnitude (AU)' ,va='center', rotation='vertical', fontsize=9)
plt.grid(alpha=0.5)
plt.vlines(0,-60,80, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax2.set_title('Subject 18')
fig.tight_layout()
plt.savefig(figure_path +'ABR_ANM_subject012018.tiff', dpi=dpi, format='tiff')

############ SNR ############
abr_time_window = 15
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

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, abr_allmusic_ave, c='C4', linewidth=1, label='Music')
plt.fill_between(lags, abr_allmusic_ave-abr_allmusic_err, abr_allmusic_ave+abr_allmusic_err, alpha=0.6, color='C4',ec=None, linewidth=0)
plt.plot(lags, abr_allspeech_ave, c='C5', linewidth=1, linestyle='--', label='Speech')
plt.fill_between(lags, abr_allspeech_ave-abr_allspeech_err, abr_allspeech_ave+abr_allspeech_err, alpha=0.6, color='C5', ec=None, linewidth=0)
plt.xlim(-10, 30)
plt.ylim(-7,8)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.yticks(visible=False)
plt.grid(alpha=0.5)
plt.vlines(0,-7,8, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=8, loc='lower right')
plt.annotate('N=22', [-9, 6], fontsize=8)
plt.tight_layout()
plt.savefig(figure_path + 'ABR_IHC.tiff', dpi=dpi, format='tiff')

#Cortical
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, abr_allmusic_ave, c='C4', linewidth=0.8, label='Music')
plt.fill_between(lags, abr_allmusic_ave-abr_allmusic_err, abr_allmusic_ave+abr_allmusic_err, alpha=0.6, color='C4',ec=None, linewidth=0)
plt.plot(lags, abr_allspeech_ave, c='C5', linewidth=0.8, linestyle='--', label='Speech')
plt.fill_between(lags, abr_allspeech_ave-abr_allspeech_err, abr_allspeech_ave+abr_allspeech_err, alpha=0.6, color='C5', ec=None, linewidth=0)
plt.xlim(-50, 300)
plt.ylim(-7,8)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.yticks(visible=False)
plt.grid(alpha=0.5)
plt.vlines(0,-7,8, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=8, loc='lower right')
plt.annotate('N=22', [-45, 6], fontsize=8)
plt.tight_layout()
plt.savefig(figure_path + 'ABR_cortical_IHC.tiff', dpi=dpi, format='tiff')

# Individual response all
fig, ax = plt.subplots(6, 4, sharex=True, sharey=True, figsize=(7.5,8.5))
for si in range(subject_num):
    row_num=(si)//4
    col_num=(si)%4
    line1, = ax[row_num, col_num].plot(lags, abr_allmusic_all[si], c='C4', linewidth=1, label='Music')
    line2, = ax[row_num, col_num].plot(lags, abr_allspeech_all[si], c='C5', linewidth=1, label='Speech', linestyle='--')
    ax[row_num, col_num].set_xlim([-10, 30])
    ax[row_num, col_num].set_ylim([-10,12])
    ax[row_num, col_num].tick_params(axis='both', which="major", labelsize=7)
    for tick in ax[row_num, col_num].yaxis.get_major_ticks():
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    ax[row_num, col_num].yaxis.offsetText.set_fontsize(7)
    ax[row_num, col_num].spines['top'].set_visible(False)
    ax[row_num, col_num].spines['right'].set_visible(False)
    ax[row_num, col_num].vlines(0,-10,12, linestyles='solid', color='k', linewidth=1, alpha=0.5)
    ax[row_num, col_num].hlines(0,-10,30, linestyles='solid', color='k', linewidth=1, alpha=0.5)
    ax[row_num, col_num].grid(alpha=0.5)
ax[5, 2].set_visible(False)
ax[5, 3].set_visible(False)
fig.legend(handles=[line1, line2], loc='lower right', bbox_to_anchor=(0.95, 0.15), fontsize=9)
fig.subplots_adjust(wspace=0.2, hspace=0.7)
fig.suptitle("Individual Response (IHC)", fontsize=10)
fig.supxlabel('Time (ms)', fontsize=9)
fig.supylabel('Magnitude (AU)', fontsize=9)
plt.tight_layout()
plt.savefig(figure_path + 'ABR_IHC_individual.tiff', dpi=dpi, format='tiff')

# Good example: subject018; bad example: subject012
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
ax1 = plt.subplot(211)
plt.plot(lags, abr_allmusic_all[11], c='C4', linewidth=1)
plt.plot(lags, abr_allspeech_all[11], c='C5', linewidth=1, linestyle='--')
plt.xlim(-10, 30)
plt.ylim(-7,9)
plt.yticks(visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('Magnitude (AU)')
plt.grid(alpha=0.5)
plt.vlines(0,-7,9, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax1.set_title('Subject 12')

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(lags, abr_allmusic_all[17], c='C4', linewidth=1)
plt.plot(lags, abr_allspeech_all[17], c='C5', linewidth=1, linestyle='--')
plt.xlim(-10, 30)
plt.ylim(-7,9)
plt.yticks(visible=False)
plt.xlabel('Time (ms)')
plt.ylabel('Magnitude (AU)')
#fig.text(0.01, 0.5, 'Magnitude (AU)' ,va='center', rotation='vertical', fontsize=9)
plt.grid(alpha=0.5)
plt.vlines(0,-7,9, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax2.set_title('Subject 18')
fig.tight_layout()
plt.savefig(figure_path +'ABR_IHC_subject012018.tiff', dpi=dpi, format='tiff')

############ SNR ############
abr_time_window = 15
SNR_IHC_music = np.zeros((subject_num,))
SNR_IHC_speech = np.zeros((subject_num,))
for i in range(subject_num):
    SNR_IHC_music[i] = abr_SNR(abr_allmusic_all[i], lags, abr_time_window=abr_time_window, noise_range=[-200, -20])
    SNR_IHC_speech[i] = abr_SNR(abr_allspeech_all[i], lags, abr_time_window=abr_time_window, noise_range=[-200, -20])

SNR_IHC_music_ave = abr_SNR(abr_allmusic_ave, lags, abr_time_window=abr_time_window, noise_range=[-200, -20])
SNR_IHC_speech_ave = abr_SNR(abr_allspeech_ave, lags, abr_time_window=abr_time_window, noise_range=[-200, -20])

# %% ANM EACH TYPE 12 types
abr_music_all = dict(acoustic=np.zeros((subject_num, 8000)),
                     classical=np.zeros((subject_num, 8000)),
                     hiphop=np.zeros((subject_num, 8000)),
                     jazz=np.zeros((subject_num, 8000)),
                     metal=np.zeros((subject_num, 8000)),
                     pop=np.zeros((subject_num, 8000)))
abr_music_ave = dict(acoustic=np.zeros(8000),
                     classical=np.zeros(8000),
                     hiphop=np.zeros(8000),
                     jazz=np.zeros(8000),
                     metal=np.zeros(8000),
                     pop=np.zeros(8000))
abr_music_ave_bp = dict(acoustic=np.zeros(8000),
                     classical=np.zeros(8000),
                     hiphop=np.zeros(8000),
                     jazz=np.zeros(8000),
                     metal=np.zeros(8000),
                     pop=np.zeros(8000))
for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regANM.hdf5')
    lags = data['lags']
    for ti in music_types:
        abr_music_all[ti][i] = data['abr_music'][ti]
for ti in music_types:
    abr_music_ave[ti] = np.sum(abr_music_all[ti], axis=0) / subject_num
    abr_music_ave_bp[ti] = butter_bandpass_filter(abr_music_ave[ti], 1, 1500, eeg_fs)

abr_speech_all = dict(chn_aud=np.zeros((subject_num, 8000)),
                      eng_aud=np.zeros((subject_num, 8000)),
                      interview=np.zeros((subject_num, 8000)),
                      lecture=np.zeros((subject_num, 8000)),
                      news=np.zeros((subject_num, 8000)),
                      talk=np.zeros((subject_num, 8000)))
abr_speech_ave = dict(chn_aud=np.zeros(8000),
                      eng_aud=np.zeros(8000),
                      interview=np.zeros(8000),
                      lecture=np.zeros(8000),
                      news=np.zeros(8000),
                      talk=np.zeros(8000))
abr_speech_ave_bp = dict(chn_aud=np.zeros(8000),
                      eng_aud=np.zeros(8000),
                      interview=np.zeros(8000),
                      lecture=np.zeros(8000),
                      news=np.zeros(8000),
                      talk=np.zeros(8000))

for i in range(subject_num):
    data = read_hdf5(subject_data_path[i] + subject_list[i]
            + '_abr_response_regANM.hdf5')
    lags = data['lags']
    for ti in speech_types:
        abr_speech_all[ti][i] = data['abr_speech'][ti]
for ti in speech_types:
    abr_speech_ave[ti] = np.sum(abr_speech_all[ti], axis=0) / subject_num
    abr_speech_ave_bp[ti] = butter_bandpass_filter(abr_speech_ave[ti], 1, 1500, eeg_fs)

# Plot
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
for ti in music_types:
    ind = music_types.index(ti)
    if ind==0:
        a = 0.5
    else:
        a = 0.1*ind + 0.5
    plt.plot(lags, abr_music_ave_bp[ti]+75, c='C2', alpha=a, label='Music', linewidth=0.8)
for ti in speech_types:
    ind = speech_types.index(ti)
    if ind==0:
        a = 0.5
    else:
        a = 0.1*ind + 0.5
    plt.plot(lags, abr_speech_ave_bp[ti], c='C3', linestyle='-', alpha=a, label='Speech', linewidth=0.8)
plt.xlim(-10, 30)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.ylim(-50, 150)
plt.yticks(visible=False)
plt.grid(alpha=0.5)
plt.vlines(0,-50,150, linestyles='solid', color='k', linewidth=1, alpha=0.5)
plt.hlines(0,-10, 30, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.hlines(75,-10, 30, linestyles='solid',color='k', linewidth=1, alpha=0.5)
plt.text(-9, 90, 'Music', fontsize=9, c='C2')
plt.text(-9, 12, 'Speech', fontsize=9, c='C3')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.tight_layout()
plt.savefig(figure_path+'ABR_ANM_alltypes.tiff', dpi=dpi, format='tiff')

# %% ANM Latency Distribution
click_latencies_lp = np.array([8.1,7.8,7.7,7.7,7.7,7.3,7.5,7.4,7.9,7.2,8.1,7.4,8.0,7.7,7.6,7.7,7.8,7.6,7.6,7.8,7.5,9.1])
music_latencies_lp = np.array([8.3,8.4,8.2,7.8,8.4,7.7,8.3,7.8,8.4,7.4,8.5,8.2,8.2,8.2,7.8,8.3,7.8,7.6,8.1,7.8,7.6,8.5])
speech_latencies_lp = np.array([8.1,8.3,8.4,7.9,8.5,7.6,8.1,7.8,8.2,7.5,8.5,8.0,8.4,8.0,7.8,8.1,7.8,7.7,7.9,8.3,7.7,8.5])

click_latencies_ave = np.mean(click_latencies_lp)
click_latencies_err = np.std(click_latencies_lp) / np.sqrt(subject_num)
music_latencies_ave = np.mean(music_latencies_lp)
music_latencies_err = np.std(music_latencies_lp) / np.sqrt(subject_num)
speech_latencies_ave = np.mean(speech_latencies_lp)
speech_latencies_err = np.std(speech_latencies_lp) / np.sqrt(subject_num)

# Plot for three types
classes = ['click', 'music', 'speech']
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
for s in subject_list:
    si = subject_list.index(s)
    line1, = plt.plot(classes, [click_latencies_lp[si], music_latencies_lp[si], speech_latencies_lp[si]], 'o-', c='grey', markersize=1, linewidth=0.8, label="Individual")
line2 = plt.errorbar(classes, [click_latencies_ave, music_latencies_ave, speech_latencies_ave], 
             yerr=[click_latencies_err, music_latencies_err, speech_latencies_err],color='k', marker='s', linestyle='-', capsize=4,markersize=4, linewidth=1, label="Averaged")
plt.xlabel('Stimulus type')
plt.ylabel("Latency (ms)")
plt.grid(alpha=0.5)
plt.legend(handles=[line1, line2], loc="upper right", fontsize=7)
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path + "latnecy_dist.tiff", format='tiff')

#%% # %% ANM Cortical 32-CHANNEL
channel_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
                 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
                 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2',
                  'F4', 'F8', 'Fp2']

music_response_all = np.zeros((subject_num,len(music_types),32,8000))
for sn in range(subject_num):
    data = read_hdf5(subject_data_path[sn] + subject_list[sn] + '_cortical_response_regANM_noonoff-2-30m.hdf5')
    lags = data['lags']    
    for ti in music_types:
        tn = music_types.index(ti)
        temp = butter_lowpass_filter(data['response_music'][ti], 100, eeg_fs)
        music_response_all[sn,tn,:,:] = temp
music_response_ave = np.average(music_response_all, axis=0)
music_response_ave_class = np.average(music_response_ave, axis=0)

# For speech response
speech_response_all = np.zeros((subject_num,len(speech_types),32,8000))
for sn in range(subject_num):
    data = read_hdf5(subject_data_path[sn] + subject_list[sn] + '_cortical_response_regANM_noonoff-2-30m.hdf5')
    lags = data['lags']    
    for ti in speech_types:
        tn = speech_types.index(ti)
        temp = butter_lowpass_filter(data['response_speech'][ti], 100, eeg_fs)
        speech_response_all[sn,tn,:,:] = temp
speech_response_ave = np.average(speech_response_all, axis=0)
speech_response_ave_class = np.average(speech_response_ave, axis=0)

#### Plotting
montage = mne.channels.make_standard_montage("standard_1020")
info = mne.create_info(channel_names, sfreq=10000, ch_types='eeg')
times = np.arange(-0.2, 0.6, 0.05)

Evoked_music = mne.EvokedArray(music_response_ave_class, info, tmin=-0.2)
Evoked_music.set_montage(montage)
Evoked_music.nave = 40*6

plt.figure(dpi=dpi)
Evoked_music.plot_joint(times = [9.3e-3, 15.4e-3, 20.5e-3, 28e-3, 35e-3, 51e-3, 95e-3, 150e-3],
                        ts_args=dict(ylim=dict(eeg=[-45, 60]), xlim=[-20, 250], units='Magnitude (AU)', time_unit='ms', gfp=True, scalings=dict(eeg=1)),
                        topomap_args=dict(vmin=-45, vmax=60, time_unit='ms', scalings=dict(eeg=1)), title=None)

plt.savefig(figure_path + 'Cortical_32channel_ANM_Music.tiff', dpi=dpi, format='tiff')

Evoked_speech = mne.EvokedArray(speech_response_ave_class, info, tmin=-0.2)
Evoked_speech.set_montage(montage)
Evoked_speech.nave = 40*6
Evoked_speech.plot_joint(times = [9.3e-3, 15.4e-3, 20.5e-3, 28e-3, 35e-3, 51e-3, 95e-3, 150e-3],
                        ts_args=dict(ylim=dict(eeg=[-45, 60]), xlim=[-20, 250], units='Magnitude (AU)', time_unit='ms', gfp=True, scalings=dict(eeg=1)),
                        topomap_args=dict(vmin=-45, vmax=60, time_unit='ms', scalings=dict(eeg=1)), title=None)

plt.savefig(figure_path + 'Cortical_32channel_ANM_Speech.tiff', dpi=dpi, format='tiff')
# %%SNR plotting
sub_corr = 10*np.log10(subject_num)
barWidth = 0.25
br1 = np.arange(2)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.grid(alpha=0.5,zorder=0)
plt.bar(br1, [SNR_rectified_music_ave,SNR_rectified_speech_ave], color='C0', width=barWidth, label='Half-wave Rectified',zorder=3)
plt.bar(br2, [SNR_IHC_music_ave,SNR_IHC_speech_ave], color='C4', width=barWidth, label='IHC',zorder=3)
plt.bar(br3, [SNR_ANM_music_ave,SNR_ANM_speech_ave], color='C2', width=barWidth, label='ANM',zorder=3)
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
plt.savefig(figure_path + 'SNR-90ms_ind.tiff', dpi=dpi, format='tiff', bbox_extra_artists=(lg,), bbox_inches='tight')

# SNR adjust
sub_corr = 10*np.log10(subject_num)
barWidth = 0.25
br1 = np.arange(2)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig = plt.figure(dpi=dpi)
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
plt.savefig(figure_path + 'SNR-15ms_ind_adjust.tiff', dpi=dpi, format='tiff', bbox_extra_artists=(lg,), bbox_inches='tight')
# %% SNR by number of Eooch ANM
snr_data = read_hdf5("/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/abr_snr_ANM_by_numEpoch.hdf5")
snr_music = snr_data["snr_music"]
snr_music_bp = snr_data["snr_music_bp"]
snr_speech = snr_data["snr_speech"]
snr_speech_bp = snr_data["snr_speech_bp"]

epoch_num_vec = np.arange(1, 241, 1)
pecentiles = [0, 5, 25, 50, 75, 95, 100]

# Bandpassed
snr_music_av = np.nanmean(snr_music_bp, axis=0)
snr_music_pt = np.nanpercentile(snr_music_bp, pecentiles, axis=0)

snr_speech_av = np.nanmean(snr_speech_bp, axis=0)
snr_speech_pt = np.nanpercentile(snr_speech_bp, pecentiles, axis=0)
# Portion of subject >= 0 dB SNR
music_portion = np.zeros(240)
speech_portion = np.zeros(240)
music_sub_list = []
speech_sub_list = []
for i in range(240):
    curr_sub = [ind for ind, snr in enumerate(snr_music_bp[:, i]) if snr >= 0]
    for sub in curr_sub:
        if sub not in music_sub_list:
            music_sub_list.append(sub)
            print(sub)
    music_portion[i] = len(music_sub_list) / subject_num
    print(music_sub_list)
    curr_sub = [ind for ind, snr in enumerate(snr_speech_bp[:, i]) if snr >= 0]
    for sub in curr_sub:
        if sub not in speech_sub_list:
            speech_sub_list.append(sub)
    speech_portion[i] = len(speech_sub_list) / subject_num
# By minutes
min_num_vec = epoch_num_vec*12/60

fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.step(min_num_vec, music_portion, label="Music", color="C2")
plt.step(min_num_vec, speech_portion, label="Speech", color="C3", linestyle="--")
plt.xlabel('Recording time (min)')
plt.ylabel('Proortion of subjects $\geq$ 0 dB SNR')
plt.xlim(0,20)
plt.yticks(np.arange(0,1.1,0.1))
plt.xticks(np.arange(0,21,2))
plt.grid(alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=7,loc='lower right')
#plt.annotate('N=22', [8,0],fontsize=8)
plt.tight_layout()
plt.savefig(figure_path+'ABR_SNR_byMin_portion.tiff', dpi=dpi, format='tiff', bbox_extra_artists=(lg,))
# %% Correlation plotting
# Plot class kernel
regressors = ['rec', 'IHC', 'ANM']
predicted_eeg_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/predicted_eeg/'
cc_music = np.zeros((len(regressors), subject_num))
cc_speech = np.zeros((len(regressors), subject_num))
for reg in regressors:
    ri = regressors.index(reg)
    data = read_hdf5(predicted_eeg_path + reg + '_corr_new.hdf5')
    out_music_cc = data['corr_music']
    out_speech_cc = data['corr_speech']
    
    for t in music_types:
        cc_music[ri] += out_music_cc[t][:,0]
    cc_music[ri] /= len(music_types)
    for t in speech_types:
        cc_speech[ri] += out_speech_cc[t][:,0]
    cc_speech[ri] /= len(music_types)

cc_music_ave = np.average(cc_music, axis=1)
cc_speech_ave = np.average(cc_speech, axis=1)

#Plot
barWidth = 0.25
br1 = np.arange(2)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.bar(br1, [cc_music_ave[0],cc_speech_ave[0]], color='C0', width=barWidth, label='Half-wave Rectified',zorder=3)
plt.bar(br2, [cc_music_ave[1],cc_speech_ave[1]], color='C4', width=barWidth, label='IHC',zorder=3)
plt.bar(br3, [cc_music_ave[2],cc_speech_ave[2]], color='C2', width=barWidth, label='ANM',zorder=3)
plt.xlabel('Stimulus')
plt.ylabel("Correlation Coefficient")
plt.ylim(-0.003, 0.055)
plt.xticks([r + barWidth for r in range(2)], ['Music', 'Speech'])
for si in range(subject_num):
    plt.plot([br1[0],br2[0], br3[0]],[cc_music[0][si],cc_music[1][si], cc_music[2][si]], ".-", markersize=2, linewidth=0.5, c='k',zorder=4,alpha=0.3+0.03*si)
    plt.plot([br1[1],br2[1], br3[1]],[cc_speech[0][si],cc_speech[1][si], cc_speech[2][si]], ".-", markersize=2,linewidth=0.5, c='k',zorder=4,alpha=0.3+0.03*si)
lg = plt.legend(fontsize=7, bbox_to_anchor=(1.03, 1.0), loc='upper left')
plt.grid(alpha=0.5,zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path+'predict_real_corr_class.tiff', dpi=dpi, format='tiff', bbox_extra_artists=(lg,), bbox_inches='tight')

# Plot overall kernel
regressors = ['rec', 'IHC', 'ANM']
predicted_eeg_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/predicted_eeg/'
cc_music = np.zeros((len(regressors), subject_num))
cc_speech = np.zeros((len(regressors), subject_num))
for reg in regressors:
    ri = regressors.index(reg)
    data = read_hdf5(predicted_eeg_path + reg + '_corr_new_overall.hdf5')
    out_music_cc = data['corr_music']
    out_speech_cc = data['corr_speech']
    
    for t in music_types:
        cc_music[ri] += out_music_cc[t][:,0]
    cc_music[ri] /= len(music_types)
    for t in speech_types:
        cc_speech[ri] += out_speech_cc[t][:,0]
    cc_speech[ri] /= len(music_types)

cc_music_ave = np.average(cc_music, axis=1)
cc_speech_ave = np.average(cc_speech, axis=1)

#Plot
barWidth = 0.25
br1 = np.arange(2)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.bar(br1, [cc_music_ave[0],cc_speech_ave[0]], color='C0', width=barWidth, label='Half-wave Rectified',zorder=3)
plt.bar(br2, [cc_music_ave[1],cc_speech_ave[1]], color='C4', width=barWidth, label='IHC',zorder=3)
plt.bar(br3, [cc_music_ave[2],cc_speech_ave[2]], color='C2', width=barWidth, label='ANM',zorder=3)
plt.xlabel('Stimulus')
plt.ylabel("Correlation Coefficient (Pearson's r)")
plt.xticks([r + barWidth for r in range(2)], ['Music', 'Speech'])
plt.ylim(-0.003, 0.055)
for si in range(subject_num):
    plt.plot([br1[0],br2[0], br3[0]],[cc_music[0][si],cc_music[1][si], cc_music[2][si]], ".-", markersize=2, linewidth=0.5, c='k',zorder=4,alpha=0.3+0.03*si)
    plt.plot([br1[1],br2[1], br3[1]],[cc_speech[0][si],cc_speech[1][si], cc_speech[2][si]], ".-", markersize=2,linewidth=0.5, c='k',zorder=4,alpha=0.3+0.03*si)
lg = plt.legend(fontsize=7, bbox_to_anchor=(1.03, 1.0), loc='upper left')
plt.grid(alpha=0.5,zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path+'predict_real_corr_overall.tiff', dpi=dpi, format='tiff', bbox_extra_artists=(lg,), bbox_inches='tight')

# %% COHERENCE
#Loading noise floor
predicted_eeg_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/predicted_eeg/'
noise_floor_data = read_hdf5(predicted_eeg_path + '/coherence_noise_floor.hdf5')
coh_noise_floor_01 = noise_floor_data['coh_noise_floor_01']
coh_01_lb =noise_floor_data['coh_01_lb']
coh_01_ub =noise_floor_data['coh_01_ub']
coh_noise_floor_02 = noise_floor_data['coh_noise_floor_02']
coh_02_lb =noise_floor_data['coh_02_lb']
coh_02_ub =noise_floor_data['coh_02_ub']
coh_noise_floor_1 = noise_floor_data['coh_noise_floor_1']
coh_1_lb =noise_floor_data['coh_1_lb']
coh_1_ub =noise_floor_data['coh_1_ub']
# Coherence params class dur 1
dur_slice = 1
n_slices = int(t_mus / dur_slice)
len_slice = int(dur_slice * eeg_fs)
n_bands = int((eeg_fs / 2) * dur_slice + 1)
# Lists
classes = ['music', 'speech']
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
regressors = ['rec', 'IHC', 'ANM']
# Plot 12 types averaged
coh_abs_music_ave = np.zeros((len(regressors), len(music_types), n_bands))
coh_abs_speech_ave = np.zeros((len(regressors), len(speech_types), n_bands))
for reg in regressors:
    ri = regressors.index(reg)
    data = read_hdf5(predicted_eeg_path + reg + '_coherence_1.hdf5')
    coh_music = data['coh_music']
    coh_speech = data['coh_speech']
    freq = data['freq']
    
    for t in music_types:
        ti = music_types.index(t)
        coh_abs_music_ave[ri, ti, :] = np.average(abs(coh_music[t]), axis=0)
    for t in speech_types:
        ti = speech_types.index(t)
        coh_abs_speech_ave[ri, ti, :] = np.average(abs(coh_speech[t]), axis=0)

coh_abs_music_ave_class = np.average(coh_abs_music_ave, axis=1)
coh_abs_speech_ave_class = np.average(coh_abs_speech_ave, axis=1)

fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(freq[1:], coh_abs_music_ave_class[0, 1:], c='C0', linewidth=1, label='Half-wave Rectified Music')
plt.plot(freq[1:], coh_abs_speech_ave_class[0, 1:], c='C1', linewidth=1, linestyle='--', label='Half-wave Rectified Speech')
plt.plot(freq[1:], coh_abs_music_ave_class[1, 1:], c='C4', linewidth=1, label='IHC Music')
plt.plot(freq[1:], coh_abs_speech_ave_class[1, 1:], c='C5', linewidth=1, linestyle='--', label='IHC Speech')
plt.plot(freq[1:], coh_abs_music_ave_class[2, 1:], c='C2', linewidth=1, label='ANM Music')
plt.plot(freq[1:], coh_abs_speech_ave_class[2, 1:], c='C3', linewidth=1, linestyle='--', label='ANM Speech')
plt.fill_between(freq[1:], np.zeros(len(freq)-1), coh_1_ub[1:], alpha=0.5, color='grey', ec=None, linewidth=0)
plt.xlim([1, 30])
plt.ylim([0.035, 0.08])
plt.tight_layout()
lg = plt.legend(fontsize=7, loc='upper right')
#plt.text(8, 0.0354,  'Noise Floor (95% CI)', fontsize=8, c='grey')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence (absolute)')
plt.grid(alpha=0.5,zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(figure_path+'coherence_class_1s.tiff', dpi=dpi, format='tiff')
# Coherence params class dur 0.1
dur_slice = .1
n_slices = int(t_mus / dur_slice)
len_slice = int(dur_slice * eeg_fs)
n_bands = int((eeg_fs / 2) * dur_slice + 1)
# Lists
classes = ['music', 'speech']
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
regressors = ['rec', 'IHC', 'ANM']
# Plot 12 types averaged
coh_abs_music_ave = np.zeros((len(regressors), len(music_types), n_bands))
coh_abs_speech_ave = np.zeros((len(regressors), len(speech_types), n_bands))
for reg in regressors:
    ri = regressors.index(reg)
    data = read_hdf5(predicted_eeg_path + reg + '_coherence_01.hdf5')
    coh_music = data['coh_music']
    coh_speech = data['coh_speech']
    freq = data['freq']
    
    for t in music_types:
        ti = music_types.index(t)
        coh_abs_music_ave[ri, ti, :] = np.average(abs(coh_music[t]), axis=0)
    for t in speech_types:
        ti = speech_types.index(t)
        coh_abs_speech_ave[ri, ti, :] = np.average(abs(coh_speech[t]), axis=0)

coh_abs_music_ave_class = np.average(coh_abs_music_ave, axis=1)
coh_abs_speech_ave_class = np.average(coh_abs_speech_ave, axis=1)

fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(freq[1:], coh_abs_music_ave_class[0, 1:], c='C0', linewidth=1, label='Half-wave Rectified Music')
plt.plot(freq[1:], coh_abs_speech_ave_class[0, 1:], c='C1', linewidth=1, linestyle='--', label='Half-wave Rectified Speech')
plt.plot(freq[1:], coh_abs_music_ave_class[1, 1:], c='C4', linewidth=1, label='IHC Music')
plt.plot(freq[1:], coh_abs_speech_ave_class[1, 1:], c='C5', linewidth=1, linestyle='--', label='IHC Speech')
plt.plot(freq[1:], coh_abs_music_ave_class[2, 1:], c='C2', linewidth=1, label='ANM Music')
plt.plot(freq[1:], coh_abs_speech_ave_class[2, 1:], c='C3', linewidth=1, linestyle='--', label='ANM Speech')
plt.fill_between(freq[1:], np.zeros(len(freq)-1), coh_01_ub[1:], alpha=0.5, color='grey', ec=None, linewidth=0)
plt.xlim([10, 300])
plt.ylim([0.01, 0.04])
plt.tight_layout()
#lg = plt.legend(fontsize=7, loc='upper right')
#plt.text(80, 0.0105,  'Noise Floor (95% CI)', fontsize=8, c='grey')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence (absolute)')
plt.grid(alpha=0.5,zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(figure_path+'coherence_class_01s.tiff', dpi=dpi, format='tiff')

# Coherence params overall dur 1
dur_slice = 1
n_slices = int(t_mus / dur_slice)
len_slice = int(dur_slice * eeg_fs)
n_bands = int((eeg_fs / 2) * dur_slice + 1)
# Lists
classes = ['music', 'speech']
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
regressors = ['rec', 'IHC', 'ANM']
# Plot 12 types averaged
coh_abs_music_ave = np.zeros((len(regressors), len(music_types), n_bands))
coh_abs_speech_ave = np.zeros((len(regressors), len(speech_types), n_bands))
for reg in regressors:
    ri = regressors.index(reg)
    data = read_hdf5(predicted_eeg_path + reg + '_coherence_overall_1.hdf5')
    coh_music = data['coh_music']
    coh_speech = data['coh_speech']
    freq = data['freq']
    
    for t in music_types:
        ti = music_types.index(t)
        coh_abs_music_ave[ri, ti, :] = np.average(abs(coh_music[t]), axis=0)
    for t in speech_types:
        ti = speech_types.index(t)
        coh_abs_speech_ave[ri, ti, :] = np.average(abs(coh_speech[t]), axis=0)

coh_abs_music_ave_class = np.average(coh_abs_music_ave, axis=1)
coh_abs_speech_ave_class = np.average(coh_abs_speech_ave, axis=1)

fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(freq[1:], coh_abs_music_ave_class[0, 1:], c='C0', linewidth=1, label='Half-wave Rectified Music')
plt.plot(freq[1:], coh_abs_speech_ave_class[0, 1:], c='C1', linewidth=1, linestyle='--', label='Half-wave Rectified Speech')
plt.plot(freq[1:], coh_abs_music_ave_class[1, 1:], c='C4', linewidth=1, label='IHC Music')
plt.plot(freq[1:], coh_abs_speech_ave_class[1, 1:], c='C5', linewidth=1, linestyle='--', label='IHC Speech')
plt.plot(freq[1:], coh_abs_music_ave_class[2, 1:], c='C2', linewidth=1, label='ANM Music')
plt.plot(freq[1:], coh_abs_speech_ave_class[2, 1:], c='C3', linewidth=1, linestyle='--', label='ANM Speech')
plt.fill_between(freq[1:], np.zeros(len(freq)-1), coh_1_ub[1:], alpha=0.5, color='grey', ec=None, linewidth=0, label='Noise floor (95% CI)')
plt.xlim([1, 30])
plt.ylim([0.035, 0.08])
plt.tight_layout()
lg = plt.legend(fontsize=7, bbox_to_anchor=(1.03, 1.0), loc='upper left')
plt.text(8, 0.039,  'Noise Floor (95% CI)', fontsize=8, c='grey')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence (absolute)')
plt.grid(alpha=0.5,zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path+'coherence_overall_1s.tiff', dpi=dpi, format='tiff', bbox_extra_artists=(lg,), bbox_inches='tight')

# Coherence params overall dur 0.1
dur_slice = .1
n_slices = int(t_mus / dur_slice)
len_slice = int(dur_slice * eeg_fs)
n_bands = int((eeg_fs / 2) * dur_slice + 1)
# Lists
classes = ['music', 'speech']
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
regressors = ['rec', 'IHC', 'ANM']
# Plot 12 types averaged
coh_abs_music_ave = np.zeros((len(regressors), len(music_types), n_bands))
coh_abs_speech_ave = np.zeros((len(regressors), len(speech_types), n_bands))
for reg in regressors:
    ri = regressors.index(reg)
    data = read_hdf5(predicted_eeg_path + reg + '_coherence_overall_01.hdf5')
    coh_music = data['coh_music']
    coh_speech = data['coh_speech']
    freq = data['freq']
    
    for t in music_types:
        ti = music_types.index(t)
        coh_abs_music_ave[ri, ti, :] = np.average(abs(coh_music[t]), axis=0)
    for t in speech_types:
        ti = speech_types.index(t)
        coh_abs_speech_ave[ri, ti, :] = np.average(abs(coh_speech[t]), axis=0)

coh_abs_music_ave_class = np.average(coh_abs_music_ave, axis=1)
coh_abs_speech_ave_class = np.average(coh_abs_speech_ave, axis=1)

fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(freq[1:], coh_abs_music_ave_class[0, 1:], c='C0', linewidth=1, label='Half-wave Rectified Music')
plt.plot(freq[1:], coh_abs_speech_ave_class[0, 1:], c='C1', linewidth=1, linestyle='--', label='Half-wave Rectified Speech')
plt.plot(freq[1:], coh_abs_music_ave_class[1, 1:], c='C4', linewidth=1, label='IHC Music')
plt.plot(freq[1:], coh_abs_speech_ave_class[1, 1:], c='C5', linewidth=1, linestyle='--', label='IHC Speech')
plt.plot(freq[1:], coh_abs_music_ave_class[2, 1:], c='C2', linewidth=1, label='ANM Music')
plt.plot(freq[1:], coh_abs_speech_ave_class[2, 1:], c='C3', linewidth=1, linestyle='--', label='ANM Speech')
plt.fill_between(freq[1:], np.zeros(len(freq)-1), coh_01_ub[1:], alpha=0.5, color='grey', ec=None, linewidth=0, label='Noise floor (95% CI)')
plt.xlim([10, 300])
plt.ylim([0.01, 0.04])
plt.tight_layout()
lg = plt.legend(fontsize=7, bbox_to_anchor=(1.03, 1.0), loc='upper left')
plt.text(80, 0.0105,  'Noise Floor (95% CI)', fontsize=8, c='grey')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence (absolute)')
plt.grid(alpha=0.5,zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path+'coherence_overall_01s.tiff', dpi=dpi, format='tiff', bbox_extra_artists=(lg,), bbox_inches='tight')

# %% Music-Speech ABR morphology correlation
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

# Odd-even morphology correlation
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

#Plot
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
plt.savefig(figure_path + 'music-speech_correlation.tiff', dpi=dpi, format='tiff')
