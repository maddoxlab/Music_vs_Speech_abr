#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICT EEG SIGNAL WITH KERNELS

Created on Tue Nov 23 09:49:29 2021

@author: tong
"""

import numpy as np
import scipy.signal as signal
from numpy.fft import irfft, rfft, fft, ifft
import matplotlib.pyplot as plt
import h5py
import scipy.stats as stats

from expyfun.io import read_wav, read_tab
from expyfun.io import write_hdf5, read_hdf5
import mne


import matplotlib.pyplot as plt

# %% Function

def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=1, axis=-1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data, axis)
    return y


def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=1, axis=-1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data, axis)
    return y


def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1, axis=-1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data, axis)
    return y

def get_abr_range(abr_response, time_lags, time_range):
    """
    input:
        abr_response: derived abr response
        time_lags: time vector
        time_range: in which time range to find the peaks [start, end] in ms
    output:
        time_vec: time vector
        response: response in the specific range
    """

    abr_response = abr_response
    start_time = time_range[0]
    end_time = time_range[1]
    ind = np.where((time_lags >= start_time) & (time_lags <=end_time))[0]
    time_vec = time_lags[ind]
    response = abr_response[ind]
    return time_vec, response

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

# %% LOADING KERNELS 
regressor = 'rec'
# Music 
abr_music_ave = np.zeros((subject_num, 8000))
for si in range(subject_num):
    data = read_hdf5(subject_data_path[si] + subject_list[si] + '_abr_response_reg' + regressor + '.hdf5')
    abr_music = data['abr_music']
    lags = data['lags']
    abr_music_ave_cs = np.zeros(8000,)
    for ti in music_types:
        abr_music_ave_cs += abr_music[ti]
    abr_music_ave[si, :] = abr_music_ave_cs / len(music_types)

abr_music_ave_class = np.average(abr_music_ave, axis=0)
# shift back 3.4 ms
abr_music_ave_class_shift = np.roll(abr_music_ave_class, int(-3.4*eeg_fs/1000))
time_vec, abr_music_ave_kernel = get_abr_range(abr_music_ave_class_shift, lags, [0, 250])

#plt.plot(time_vec, abr_music_ave_kernel)
    
abr_speech_ave = np.zeros((subject_num, 8000))
for si in range(subject_num):
    data = read_hdf5(subject_data_path[si] + subject_list[si] + '_abr_response_reg' + regressor + '.hdf5')
    abr_speech = data['abr_speech']
    abr_speech_ave_cs = np.zeros(8000,)
    for ti in speech_types:
        abr_speech_ave_cs += abr_speech[ti]
    abr_speech_ave[si, :] = abr_speech_ave_cs / len(speech_types)

abr_speech_ave_class = np.average(abr_speech_ave, axis=0)
# shift back 3.4 ms
abr_speech_ave_class_shift = np.roll(abr_speech_ave_class, int(-3.4*eeg_fs/1000))
time_vec, abr_speech_ave_kernel = get_abr_range(abr_speech_ave_class_shift, lags, [0, 250])

#plt.plot(time_vec, abr_speech_ave_kernel)
abr_overall_kernel = (abr_music_ave_kernel + abr_speech_ave_kernel) / 2
# %% LOADING REGRESSORS
overall = True
ANM_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/present_files/' + regressor + '/'
t_start = 0
t_stop = 0.25
# Music
out_music_pos = dict(acoustic=np.zeros((n_epoch, len_eeg)),
                     classical=np.zeros((n_epoch, len_eeg)),
                     hiphop=np.zeros((n_epoch, len_eeg)),
                     jazz=np.zeros((n_epoch, len_eeg)),
                     metal=np.zeros((n_epoch, len_eeg)),
                     pop=np.zeros((n_epoch, len_eeg)))
out_music_neg = dict(acoustic=np.zeros((n_epoch, len_eeg)),
                     classical=np.zeros((n_epoch, len_eeg)),
                     hiphop=np.zeros((n_epoch, len_eeg)),
                     jazz=np.zeros((n_epoch, len_eeg)),
                     metal=np.zeros((n_epoch, len_eeg)),
                     pop=np.zeros((n_epoch, len_eeg)))
out_music_predicted = dict(acoustic=np.zeros((n_epoch, len_eeg)),
                           classical=np.zeros((n_epoch, len_eeg)),
                           hiphop=np.zeros((n_epoch, len_eeg)),
                           jazz=np.zeros((n_epoch, len_eeg)),
                           metal=np.zeros((n_epoch, len_eeg)),
                           pop=np.zeros((n_epoch, len_eeg)))

for ti in music_types:
    print(ti)
    data = read_hdf5(ANM_path + 'music_x_in.hdf5')
    # Load x_in
    x_in_pos = data['x_in_music_pos'][ti]
    x_in_neg = data['x_in_music_neg'][ti]
    
#    #### convolution ####
#    #zero padding x_in
#    before_zero_sig = np.zeros((n_epoch, int(abs(t_start*eeg_fs))+1))
#    after_zero_sig = np.zeros((n_epoch, int(abs(t_stop*eeg_fs))))
#    x_in_pos_pad = np.concatenate((before_zero_sig, x_in_pos, after_zero_sig), axis=1)
#    x_in_neg_pad = np.concatenate((before_zero_sig, x_in_neg, after_zero_sig), axis=1)
#    x_out_pos = np.zeros(x_in_pos.shape)
#    x_out_neg = np.zeros(x_in_neg.shape)
#    x_predict = np.zeros(x_in_neg.shape)
#    # convolve
#    for ei in range(n_epoch):
#        x_out_pos[ei, :] = signal.convolve(x_in_pos_pad[ei, :], abr_music_ave_kernel, mode='valid')
#        x_out_neg[ei, :] = signal.convolve(x_in_neg_pad[ei, :], abr_music_ave_kernel, mode='valid')
#        x_predict[ei, :] = (x_out_pos[ei, :] + x_out_neg[ei, :]) / 2
#    out_music_pos[ti] = x_out_pos
#    out_music_neg[ti] = x_out_neg
#    out_music_predicted[ti] = x_predict
    
    #### fft ####
    # zero pad kernels
    if overall:
        abr_music_ave_kernel = abr_overall_kernel
    else:
        abr_music_ave_kernel = abr_music_ave_kernel
    music_kernel = np.zeros(t_mus*eeg_fs)
    music_kernel[int(t_start*eeg_fs):int(t_stop*eeg_fs)] = abr_music_ave_kernel
    x_out_pos = np.zeros(x_in_pos.shape)
    x_out_neg = np.zeros(x_in_neg.shape)
    x_predict = np.zeros(x_in_neg.shape)
    # fft multiplication
    for ei in range(n_epoch):
        x_out_pos[ei, :] = np.fft.ifft(np.fft.fft(x_in_pos[ei,:])*np.fft.fft(music_kernel)).real
        x_out_neg[ei, :] = np.fft.ifft(np.fft.fft(x_in_neg[ei,:])*np.fft.fft(music_kernel)).real
        x_predict[ei, :] = (x_out_pos[ei, :] + x_out_neg[ei, :]) / 2
    out_music_pos[ti] = x_out_pos
    out_music_neg[ti] = x_out_neg
    out_music_predicted[ti] = x_predict

# Speech
out_speech_pos = dict(chn_aud=np.zeros((n_epoch, len_eeg)),
                      eng_aud=np.zeros((n_epoch, len_eeg)),
                      interview=np.zeros((n_epoch, len_eeg)),
                      lecture=np.zeros((n_epoch, len_eeg)),
                      news=np.zeros((n_epoch, len_eeg)),
                      talk=np.zeros((n_epoch, len_eeg)))
out_speech_neg = dict(chn_aud=np.zeros((n_epoch, len_eeg)),
                      eng_aud=np.zeros((n_epoch, len_eeg)),
                      interview=np.zeros((n_epoch, len_eeg)),
                      lecture=np.zeros((n_epoch, len_eeg)),
                      news=np.zeros((n_epoch, len_eeg)),
                      talk=np.zeros((n_epoch, len_eeg)))
out_speech_predicted = dict(chn_aud=np.zeros((n_epoch, len_eeg)),
                            eng_aud=np.zeros((n_epoch, len_eeg)),
                            interview=np.zeros((n_epoch, len_eeg)),
                            lecture=np.zeros((n_epoch, len_eeg)),
                            news=np.zeros((n_epoch, len_eeg)),
                            talk=np.zeros((n_epoch, len_eeg)))
for ti in speech_types:
    print(ti)
    data = read_hdf5(ANM_path + 'speech_x_in.hdf5')
    # Load x_in
    x_in_pos = data['x_in_speech_pos'][ti]
    x_in_neg = data['x_in_speech_neg'][ti]
    
#    # convolution
#    #zero padding x_in
#    before_zero_sig = np.zeros((n_epoch, int(abs(t_start*eeg_fs))-1))
#    after_zero_sig = np.zeros((n_epoch, int(abs(t_stop*eeg_fs))))
#    x_in_pos_pad = np.concatenate((before_zero_sig, x_in_pos, after_zero_sig), axis=1)
#    x_in_neg_pad = np.concatenate((before_zero_sig, x_in_neg, after_zero_sig), axis=1)
#    x_out_pos = np.zeros(x_in_pos.shape)
#    x_out_neg = np.zeros(x_in_neg.shape)
#    x_predict = np.zeros(x_in_neg.shape)
#    # convolve
#    for ei in range(n_epoch):
#        x_out_pos[ei, :] = signal.convolve(x_in_pos_pad[ei, :], abr_speech_ave_kernel, mode='valid')
#        x_out_neg[ei, :] = signal.convolve(x_in_neg_pad[ei, :], abr_speech_ave_kernel, mode='valid')
#        x_predict[ei, :] = (x_out_pos[ei, :] + x_out_neg[ei, :]) / 2
#    out_speech_pos[ti] = x_out_pos
#    out_speech_neg[ti] = x_out_neg
#    out_speech_predicted[ti] = x_predict
    
    #### fft ####
    # zero pad kernels
    if overall:
        abr_speech_ave_kernel = abr_overall_kernel
    else:
        abr_speech_ave_kernel = abr_speech_ave_kernel
    speech_kernel = np.zeros(t_mus*eeg_fs)
    speech_kernel[int(t_start*eeg_fs):int(t_stop*eeg_fs)] = abr_speech_ave_kernel
    x_out_pos = np.zeros(x_in_pos.shape)
    x_out_neg = np.zeros(x_in_neg.shape)
    x_predict = np.zeros(x_in_neg.shape)
    # fft multiplication
    for ei in range(n_epoch):
        x_out_pos[ei, :] = np.fft.ifft(np.fft.fft(x_in_pos[ei,:])*np.fft.fft(speech_kernel)).real
        x_out_neg[ei, :] = np.fft.ifft(np.fft.fft(x_in_neg[ei,:])*np.fft.fft(speech_kernel)).real
        x_predict[ei, :] = (x_out_pos[ei, :] + x_out_neg[ei, :]) / 2
    out_speech_pos[ti] = x_out_pos
    out_speech_neg[ti] = x_out_neg
    out_speech_predicted[ti] = x_predict

predicted_eeg_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/predicted_eeg/'
write_hdf5(predicted_eeg_path + regressor + '_predict_x_out_overall.hdf5',
           dict(out_music_pos=out_music_pos, out_music_neg=out_music_neg, 
                out_music_predicted=out_music_predicted,
                out_speech_pos=out_speech_pos, out_speech_neg=out_speech_neg,
                out_speech_predicted=out_speech_predicted),
                overwrite=True)

