#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:36:02 2021

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
#%%
#subject_list = ['subject001', 'subject002', 'subject004' ,
#                'subject005', 'subject006', 'subject007', 'subject008',
#                'subject009', 'subject010', 'subject011', 'subject012',
#                'subject013', 'subject015', 'subject016', 'subject017',
#                'subject018', 'subject020', 'subject022', 'subject023', 'subject024']
subject_list = ['subject003','subject019']
for subject in subject_list:
    # %% Loading and filtering EEG data
    eeg_path = "/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/" + subject
    eeg_vhdr = eeg_path + "/music_diverse_beh_" + subject[-3:] + "_1.vhdr"
    
    eeg_raw = mne.io.read_raw_brainvision(eeg_vhdr, preload=True)
    #mne.viz.plot_raw_psd(eeg_raw)
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
    
    #events_new[:, 2] = 2
    
    # Correct fs_eeg
    time_diff = events_2trig[10:, 0] - events_new[10:, 0]
    eeg_fs_n = np.mean(time_diff)/11.98
    #eeg_fs_n = 10000.25
    
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
    
    # %% IF DO CLICK RESPONSE
    # Epoching click
    if is_click:
        print('Epoching EEG click data...')
        epochs_click = mne.Epochs(eeg_raw, events_new, tmin=0,
                                  tmax=(t_click - 1/stim_fs + 1),
                                  event_id=1, baseline=None,
                                  preload=True, proj=False)
        epoch_click = epochs_click.get_data()
    
        # Load click wave file
        n_epoch_click = 10
        file_path = "/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/present_files/"
    
        # Read click event
        x_in = np.zeros((n_epoch_click, int(t_click * eeg_fs)), dtype=float)
        for ei in range(n_epoch_click):
            stim, fs_stim = read_wav(file_path + "click/" +
                                     'click{0:03d}'.format(ei) + '.wav')
            stim_abs = np.abs(stim)
            click_times = [(np.where(np.diff(s) > 0)[0] + 1) /
                           float(fs_stim) for s in stim_abs]
            click_inds = [(ct * eeg_fs).astype(int) for ct in click_times]
            x_in[ei, click_inds] = 1
    
        # Get x_out
        len_eeg = int(eeg_fs * t_click)
        x_out = np.zeros((n_epoch_click, 2, len_eeg))
        for i in range(n_epoch_click):
            x_out_i = epoch_click[i, :, 0:int(eeg_fs_n*t_click)]
            x_out[i, :, :] = mne.filter.resample(x_out_i, eeg_fs, eeg_fs_n)
    
        # TRF
        print('TRF...')
        t_start, t_stop = -200e-3, 600e-3  # -30e-3, 60e-3
        n_ch_in = x_in.shape[1]
        n_ch_out = x_out.shape[1]
    
        x_out = np.mean(x_out, axis=1)
    
        x_in_fft = fft(x_in, axis=-1)
        x_out_fft = fft(x_out, axis=-1)
    
        cc = np.real(ifft(x_out_fft * np.conj(x_in_fft)))
        abr = np.mean(cc, axis=0)
    
        abr_response = np.concatenate((abr[int(t_start*eeg_fs):],
                                       abr[0:int(t_stop*eeg_fs)]))
        lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
        # Plotting
        plt.plot(lags[0:8000], abr_response)
    
        # Saving Click Response
        print('Saving click response...')
        write_hdf5(eeg_path + '/' + subject + '_crosscorr_click.hdf5',
                   dict(click_abr_response=abr_response,
                        lags=lags), overwrite=True)
    
    
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
    #epoch = epoch[10:490,:]
    epoch = epoch[0:480,:]
    # %% Epoch indexing
    file_path = "/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/present_files/"
    
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
    
    for epi in range(len(file_all_list)):
        stim_type = file_all_list[epi][0:-3]
        stim_ind = int(file_all_list[epi][-3:])
        eeg_epi[stim_type][stim_ind] = epi
    
    # %% Preprocess rectified x_in
#    len_eeg = int(t_mus*eeg_fs)
#    # Music x_in
#    x_in_music_pos = dict(acoustic=np.zeros((n_epoch, len_eeg)),
#                          classical=np.zeros((n_epoch, len_eeg)),
#                          hiphop=np.zeros((n_epoch, len_eeg)),
#                          jazz=np.zeros((n_epoch, len_eeg)),
#                          metal=np.zeros((n_epoch, len_eeg)),
#                          pop=np.zeros((n_epoch, len_eeg)))
#    x_in_music_neg = dict(acoustic=np.zeros((n_epoch, len_eeg)),
#                          classical=np.zeros((n_epoch, len_eeg)),
#                          hiphop=np.zeros((n_epoch, len_eeg)),
#                          jazz=np.zeros((n_epoch, len_eeg)),
#                          metal=np.zeros((n_epoch, len_eeg)),
#                          pop=np.zeros((n_epoch, len_eeg)))
#    for ti in music_types:
#        for ei in range(n_epoch):
#            temp, rt = read_wav(file_path + ti + "/" +
#                                ti + "{0:03d}".format(ei) + ".wav")
#            temp_pos = np.fmax(temp, np.zeros(temp.shape))
#            temp_neg = - np.fmin(temp, np.zeros(temp.shape))
#            temp_pos_rsmp = mne.filter.resample(temp_pos, down=stim_fs/eeg_fs)
#            temp_neg_rsmp = mne.filter.resample(temp_neg, down=stim_fs/eeg_fs)
#            x_in_music_pos[ti][ei, :] = temp_pos_rsmp
#            x_in_music_neg[ti][ei, :] = temp_neg_rsmp
#    write_hdf5(file_path + 'rectified/music_x_in.hdf5',
#               dict(x_in_music_pos=x_in_music_pos,
#                    x_in_music_neg=x_in_music_neg,
#                    stim_fs=stim_fs), overwrite=True)
#    
#    # Speech x_in
#    x_in_speech_pos = dict(chn_aud=np.zeros((n_epoch, len_eeg)),
#                           eng_aud=np.zeros((n_epoch, len_eeg)),
#                           interview=np.zeros((n_epoch, len_eeg)),
#                           lecture=np.zeros((n_epoch, len_eeg)),
#                           news=np.zeros((n_epoch, len_eeg)),
#                           talk=np.zeros((n_epoch, len_eeg)))
#    x_in_speech_neg = dict(chn_aud=np.zeros((n_epoch, len_eeg)),
#                           eng_aud=np.zeros((n_epoch, len_eeg)),
#                           interview=np.zeros((n_epoch, len_eeg)),
#                           lecture=np.zeros((n_epoch, len_eeg)),
#                           news=np.zeros((n_epoch, len_eeg)),
#                           talk=np.zeros((n_epoch, len_eeg)))
#    for ti in speech_types:
#        for ei in range(n_epoch):
#            temp, rt = read_wav(file_path + ti + "/" +
#                                ti + "{0:03d}".format(ei) + ".wav")
#            temp_pos = np.fmax(temp, np.zeros(temp.shape))
#            temp_neg = - np.fmin(temp, np.zeros(temp.shape))
#            temp_pos_rsmp = mne.filter.resample(temp_pos, down=stim_fs/eeg_fs)
#            temp_neg_rsmp = mne.filter.resample(temp_neg, down=stim_fs/eeg_fs)
#            x_in_speech_pos[ti][ei, :] = temp_pos_rsmp
#            x_in_speech_neg[ti][ei, :] = temp_neg_rsmp
#    
#    write_hdf5(file_path + 'rectified/speech_x_in.hdf5',
#               dict(x_in_speech_pos=x_in_speech_pos,
#                    x_in_speech_neg=x_in_speech_neg,
#                    stim_fs=stim_fs), overwrite=True)
#    
#    #del x_in_music_pos, x_in_music_neg, x_in_speech_pos, x_in_speech_neg
#    #
#    #temp, rt = read_wav(file_path + ti + "/" + ti + "{0:03d}".format(0) + ".wav")
#    #fft = np.fft.rfft(temp[0])
#    #power = np.square(np.abs(fft))
#    #freq = np.linspace(0, rt/2, len(power))
#    #
#    #plt.plot(freq, power)
#    #plt.yscale("log")
    # %% Rectified stim regressor
    # For music response
    len_eeg = int(t_mus*eeg_fs)
    data = read_hdf5(file_path + 'rec/music_x_in.hdf5')
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
    
    # For speech response
    data = read_hdf5(file_path + 'rec/speech_x_in.hdf5')
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
    
    write_hdf5(eeg_path + '/' + subject + '_abr_response_regrec.hdf5',
              dict(w_music=w_music, abr_music=abr_music,
                    w_speech=w_speech, abr_speech=abr_speech,
                    abr_music_ave=abr_music_ave, abr_speech_ave=abr_speech_ave,
                    lags=lags), overwrite=True)
    
    # %% +/- ANM | IHC regressor
    len_eeg = int(t_mus*eeg_fs)
    
    ANM_path = '/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/present_files/IHC/'
    t_start = -0.2
    t_stop = 0.6
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
    
    # For music response
    w_music = dict(acoustic=np.zeros(len_eeg),
                   classical=np.zeros(len_eeg),
                   hiphop=np.zeros(len_eeg),
                   jazz=np.zeros(len_eeg),
                   metal=np.zeros(len_eeg),
                   pop=np.zeros(len_eeg))
    
    eeg_music = dict(acoustic=np.zeros((n_epoch, len_eeg)),
                     classical=np.zeros((n_epoch, len_eeg)),
                     hiphop=np.zeros((n_epoch, len_eeg)),
                     jazz=np.zeros((n_epoch, len_eeg)),
                     metal=np.zeros((n_epoch, len_eeg)),
                     pop=np.zeros((n_epoch, len_eeg)))
    
    abr_music = dict(acoustic=np.zeros(8000),
                     classical=np.zeros(8000),
                     hiphop=np.zeros(8000),
                     jazz=np.zeros(8000),
                     metal=np.zeros(8000),
                     pop=np.zeros(8000))
    
    for ti in music_types:
        print(ti)
        data = read_hdf5(ANM_path + 'music_x_in.hdf5')
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
#        w_music[ti] = ifft(np.array(w_pos).sum(0)).real
        abr_music[ti] = np.roll(np.concatenate((w_music[ti][int(t_start*eeg_fs):],
                                w_music[ti][0:int(t_stop*eeg_fs)])),
                                int(2.75*eeg_fs/1000))
    
    # For speech response
    w_speech = dict(chn_aud=np.zeros(len_eeg),
                    eng_aud=np.zeros(len_eeg),
                    interview=np.zeros(len_eeg),
                    lecture=np.zeros(len_eeg),
                    news=np.zeros(len_eeg),
                    talk=np.zeros(len_eeg))
    
    eeg_speech = dict(chn_aud=np.zeros((n_epoch, len_eeg)),
                      eng_aud=np.zeros((n_epoch, len_eeg)),
                      interview=np.zeros((n_epoch, len_eeg)),
                      lecture=np.zeros((n_epoch, len_eeg)),
                      news=np.zeros((n_epoch, len_eeg)),
                      talk=np.zeros((n_epoch, len_eeg)))
    
    abr_speech = dict(chn_aud=np.zeros(8000),
                      eng_aud=np.zeros(8000),
                      interview=np.zeros(8000),
                      lecture=np.zeros(8000),
                      news=np.zeros(8000),
                      talk=np.zeros(8000))
    
    for ti in speech_types:
        print(ti)
        data = read_hdf5(ANM_path + 'speech_x_in.hdf5')
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
        #w_speech[ti] = ifft(np.array(w_pos).sum(0)).real
        abr_speech[ti] = np.roll(np.concatenate((w_speech[ti][int(t_start*eeg_fs):],
                                 w_speech[ti][0:int(t_stop*eeg_fs)])),
                                 int(2.75*eeg_fs/1000))

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
    
    # %% Odd-Even average
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
        data = read_hdf5(ANM_path + 'music_x_in.hdf5')
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
        data = read_hdf5(ANM_path + 'speech_x_in.hdf5')
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
    
    # %% SAVE FILE
    write_hdf5(eeg_path + '/' + subject + '_abr_response_regIHC.hdf5',
               dict(w_music=w_music, abr_music=abr_music,
                    w_speech=w_speech, abr_speech=abr_speech,
                    abr_music_bp=abr_music_bp, abr_speech_bp=abr_speech_bp,
                    abr_music_ave=abr_music_ave, abr_speech_ave=abr_speech_ave,
                    abr_music_odd=abr_music_odd,abr_music_even=abr_music_even,
                    abr_speech_odd=abr_speech_odd, abr_speech_even=abr_speech_even,
                    lags=lags), overwrite=True)
