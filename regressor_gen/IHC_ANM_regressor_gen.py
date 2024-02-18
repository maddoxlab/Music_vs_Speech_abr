#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:49:13 2021

@author: tom, tong
@contributor: fotisdr

This script can be used to create the ANM regressor.
It has only been tested with Python 2, where the cochlea 
module can be installed and run reliably. 
The ic_cn2018 script from the Verhulst2018 model needs to be
included in the folder for scaling the AN responses. 
"""

import numpy as np
from mne.filter import resample
from joblib import Parallel, delayed
import re
from expyfun.io import read_wav, write_hdf5

import cochlea
import ic_cn2018 as nuclei

# %% Define functions
def findstring(ref, check):
    r = re.compile("(?:" + "|".join(check) + ")*$")
    if r.match(ref) is not None:
        return True
    return False

def get_rates(stim_up, cf):
    fs_up = int(100e3)
    return(np.array(cochlea.run_zilany2014_rate(stim_up,
                                                fs_up,
                                                anf_types='hsr',
                                                cf=cf,
                                                species='human',
                                                cohc=1,
                                                cihc=1))[:, 0])


def anm(stim, fs_in, stim_pres_db, parallel=True, n_jobs=-1, fs_up=100e3,
        stim_gen_rms=0.01, cf_low=125, cf_high=16e3, shift_cfs=False,
        shift_vals=None):
    """
     fs_up: scalar
        the sampling frequency of the AN model
     stim_gen_rms: float
        the RMS reference of the original stimulus
     shift_cfs: boolean
        shift each CF indpendently so maximum values align at zero
     shift_vals: array-like
        the values (in seconds) by which to shift each cf if shift_cfs == True

     Returns the ANM firing rates summed across channels using the model sampling
     frequency (fs_up). 
    """
    # Resample your stimuli to a higher fs for the model
    stim_up = resample(stim, fs_up, fs_in, npad='auto', n_jobs=n_jobs)
    # Put stim in correct units
    # scalar to put it in units of pascals (double-checked and good)
    sine_rms_at_0db = 20e-6
    db_conv = ((sine_rms_at_0db / stim_gen_rms) * 10 ** (stim_pres_db / 20.))
    stim_up = db_conv * stim_up

    # run the model
    dOct = 1. / 6
    cfs = 2 ** np.arange(np.log2(cf_low), np.log2(cf_high + 1), dOct)
    anf_rates_up = np.zeros([len(cfs), len(stim)*fs_up//fs_in])

    if parallel:
        anf_rates_up = Parallel(n_jobs=n_jobs)([delayed(get_rates)(stim_up, cf)
                                               for cf in cfs])
    else:
        for cfi, cf in enumerate(cfs):
            anf_rates_up[cfi] = get_rates(stim_up, cf)
    # convert into a numpy array
    anf_rates = np.array(anf_rates_up)

    # shift w1 by 1ms if not shifting each cf
    final_shift = int(fs_up*0.001)
    # Optionally, shift each cf independently
    if shift_cfs:
        final_shift = 0  # don't shift everything after aligning channels at 0
        if shift_vals is None:
            # default shift_cfs values (based on 75 dB click)
            shift_vals = np.array([0.0046875, 0.0045625, 0.00447917, 0.00435417, 0.00422917, 0.00416667,
                                   0.00402083, 0.0039375, 0.0038125, 0.0036875, 0.003625, 0.00354167, 0.00341667,
                                   0.00327083, 0.00316667, 0.0030625, 0.00302083, 0.00291667, 0.0028125,
                                   0.0026875, 0.00258333, 0.00247917, 0.00239583, 0.0023125, 0.00220833,
                                   0.00210417, 0.00204167, 0.002, 0.001875, 0.00185417, 0.00175, 0.00170833, 0.001625,
                                   0.0015625, 0.0015, 0.00147917, 0.0014375, 0.00135417, 0.0014375, 0.00129167,
                                   0.00129167, 0.00125, 0.00122917])

        # Allow fewer CFs while still using defaults
        if len(cfs) != len(shift_vals):
            ref_cfs = 2 ** np.arange(np.log2(125), np.log2(16e3 + 1), 1/6)
            picks = [cf in np.round(cfs, 3) for cf in np.round(ref_cfs, 3)]
            shift_vals = shift_vals[picks]

        # Ensure the number of shift values matches the number of cfs
        msg = 'Number of CFs does not match number of known shift values'
        assert(len(shift_vals) == len(cfs)), msg
        lags = np.round(shift_vals * fs_in).astype(int)

        # Shift each channel
        for cfi in range(len(cfs)):
            anf_rates[cfi] = np.roll(anf_rates[cfi], -lags[cfi])
            anf_rates[cfi, -lags[cfi]:] = anf_rates[cfi, -(lags[cfi]+1)]

    # Shift, scale, and sum
    M1 = nuclei.M1
    anm = M1*anf_rates.sum(0)
    anm = np.roll(anm, final_shift)
    anm[:final_shift] = anm[final_shift+1]
    return(anm)

def model_abr(stim, fs_in, stim_pres_db, parallel=True, n_jobs=-1,
              stim_gen_rms=0.01, cf_low=125, cf_high=16e3, return_flag='abr'):
    """
    return_flag: str
     Indicates which waves of the abr to return. Defaults to 'abr' which
     returns a single abr waveform containing waves I, III, and V. Can also be
     '1', '3', or '5' to get individual waves. Combining these option will
     return a dict with the desired waveforms. e.g. '13abr' will return a
     dict with keys 'w1', 'w3', and 'abr'
    """

    return_flag = str(return_flag)
    known_flags = ['1', '3', '5', 'abr', 'rates']
    msg = ('return_flag must be a combination of the following: ' +
           str(known_flags))
    assert(findstring(return_flag, known_flags)), msg

    # Resample your stimuli to a higher fs for the model
    fs_up = int(100e3)
    stim_up = resample(stim, fs_up, fs_in, npad='auto', n_jobs=n_jobs)

    # Put stim in correct units
    # scalar to put it in units of pascals (double-checked and good)
    sine_rms_at_0db = 20e-6
    db_conv = ((sine_rms_at_0db / stim_gen_rms) * 10 ** (stim_pres_db / 20.))
    stim_up = db_conv * stim_up

    # run the model
    dOct = 1. / 6
    cfs = 2 ** np.arange(np.log2(cf_low), np.log2(cf_high + 1), dOct)
    anf_rates_up = np.zeros([len(cfs), len(stim)*fs_up//fs_in])

    if parallel:
        anf_rates_up = Parallel(n_jobs=n_jobs)([delayed(get_rates)(stim_up, cf)
                                               for cf in cfs])
    else:
        for cfi, cf in enumerate(cfs):
            anf_rates_up[cfi] = get_rates(stim_up, cf)

    # Downsample to match input fs
    anf_rates = resample(anf_rates_up, fs_in, fs_up, npad='auto',
                         n_jobs=n_jobs)

    # sum and filter to get AN and IC response, only use hsf to save time
    w3, w1 = nuclei.cochlearNuclei(anf_rates.T, anf_rates.T, anf_rates.T,
                                   1, 0, 0, fs_in)
    # filter to get IC response
    w5 = nuclei.inferiorColliculus(w3, fs_in)

    # shift, scale, and sum responses
    w1_shift = int(fs_in*0.001)
    w3_shift = int(fs_in*0.00225)
    w5_shift = int(fs_in*0.0035)
    w1 = np.roll(np.sum(w1, axis=1)*nuclei.M1, w1_shift)
    w3 = np.roll(np.sum(w3, axis=1)*nuclei.M3, w3_shift)
    w5 = np.roll(np.sum(w5, axis=1)*nuclei.M3, w5_shift)

    # clean up the roll
    w1[:w1_shift] = w1[w1_shift+1]
    w3[:w3_shift] = w3[w3_shift+1]
    w5[:w5_shift] = w5[w5_shift+1]

    # Handle output
    if return_flag == 'abr':
        return w1+w3+w5

    waves = {}
    if 'abr' in return_flag:
        waves['abr'] = w1+w3+w5
    if '1' in return_flag:
        waves['w1'] = w1
    if '3' in return_flag:
        waves['w3'] = w3
    if '5' in return_flag:
        waves['w5'] = w5
    if 'rates' in return_flag:
        waves['rates'] = anf_rates

    return waves


def get_ihc_voltage(stim_up, cf):
    fs_up = int(100e3)
    return(np.array(cochlea.zilany2014._zilany2014.run_ihc(stim_up,
                                                           cf,
                                                           fs_up,
                                                           species='human',
                                                           cohc=1,
                                                           cihc=1)))

def ihc(stim, fs_in, stim_pres_db, parallel=True, n_jobs=-1,
        stim_gen_rms=0.01, cf_low=125, cf_high=16e3):

    # Resample your stimuli to a higher fs for the model
    fs_up = int(100e3)
    stim_up = resample(stim, fs_up, fs_in, npad='auto', n_jobs=n_jobs)

    # Put stim in correct units
    # scalar to put it in units of pascals (double-checked and good)
    sine_rms_at_0db = 20e-6
    db_conv = ((sine_rms_at_0db / stim_gen_rms) * 10 ** (stim_pres_db / 20.))
    stim_up = db_conv * stim_up

    # run the model
    dOct = 1. / 6
    cfs = 2 ** np.arange(np.log2(cf_low), np.log2(cf_high + 1), dOct)
    ihc_out_up = np.zeros([len(cfs), len(stim)*fs_up//fs_in])

    if parallel:
        ihc_out_up = Parallel(n_jobs=n_jobs)([delayed(get_ihc_voltage)(stim_up, cf) for cf in cfs])
    else:
        for cfi, cf in enumerate(cfs):
            ihc_out_up[cfi] = get_ihc_voltage(stim_up, cf)

    # Downsample to match input fs
    ihc_out = resample(ihc_out_up, fs_in, fs_up, npad='auto', n_jobs=n_jobs)
    ihc_out_sum = ihc_out.sum(0)
    
    return ihc_out_sum
    

# %% Parameters
stim_fs = 48000
stim_pres_db = 65
t_mus = 12
eeg_fs = 10000
n_epoch = 40
anm_fs = 100000 # sampling rate for the ANM model
n_jobs = -1 # <= CPU cores for faster execution

# %% Stim types
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]

# %% File paths
bids_root = '/hdd/data/ds004356/' # EEG-BIDS root path
audio_file_root = bids_root + 'stimuli/' # Present files root path
regressor_root = bids_root + 'regressors/' # Path to extract the regressors
# Make folder if it doesn't exist
if not os.path.exists(regressor_root + 'ANM/'):
    os.mkdir(regressor_root + 'ANM/')
    
# # %% ANM regressir generation
len_eeg = int(t_mus*eeg_fs)
# Music x_in
x_in_music_pos = dict(acoustic=np.zeros((n_epoch, len_eeg)),
                      classical=np.zeros((n_epoch, len_eeg)),
                      hiphop=np.zeros((n_epoch, len_eeg)),
                      jazz=np.zeros((n_epoch, len_eeg)),
                      metal=np.zeros((n_epoch, len_eeg)),
                      pop=np.zeros((n_epoch, len_eeg)))
x_in_music_neg = dict(acoustic=np.zeros((n_epoch, len_eeg)),
                      classical=np.zeros((n_epoch, len_eeg)),
                      hiphop=np.zeros((n_epoch, len_eeg)),
                      jazz=np.zeros((n_epoch, len_eeg)),
                      metal=np.zeros((n_epoch, len_eeg)),
                      pop=np.zeros((n_epoch, len_eeg)))
for ti in music_types:
    for ei in range(n_epoch):
        print(ti, ei)
        # Load wav file +/-
        temp, rt = read_wav(audio_file_root + ti + "/" +
                            ti + "{0:03d}".format(ei) + ".wav")
        temp = temp[0, :]
        waves_pos = anm(temp, stim_fs, stim_pres_db, n_jobs=n_jobs)
        # print('Derived ANM responses have shape: ',waves_pos.shape)
        waves_pos_resmp = resample(waves_pos, down=anm_fs/eeg_fs) # resample as eeg_fs from the model sampling rate 
        # print('ANM resampled responses have shape: ',waves_pos_resmp.shape)
        # Generate ANM
        waves_neg = anm(-temp, stim_fs, stim_pres_db, n_jobs=n_jobs)
        waves_neg_resmp = resample(waves_neg, down=anm_fs/eeg_fs) # resample as eeg_fs from the model sampling rate 
        # Make sure the shapes are matched
        x_in_music_pos[ti][ei, :] = waves_pos_resmp[:len_eeg]
        x_in_music_neg[ti][ei, :] = waves_neg_resmp[:len_eeg]
        
# Save regressor file
write_hdf5(regressor_root + '/ANM/music_x_in.hdf5',
           dict(x_in_music_pos=x_in_music_pos,
                x_in_music_neg=x_in_music_neg,
                fs=eeg_fs), overwrite=True)

# Speech x_in
x_in_speech_pos = dict(chn_aud=np.zeros((n_epoch, len_eeg)),
                       eng_aud=np.zeros((n_epoch, len_eeg)),
                       interview=np.zeros((n_epoch, len_eeg)),
                       lecture=np.zeros((n_epoch, len_eeg)),
                       news=np.zeros((n_epoch, len_eeg)),
                       talk=np.zeros((n_epoch, len_eeg)))
x_in_speech_neg = dict(chn_aud=np.zeros((n_epoch, len_eeg)),
                       eng_aud=np.zeros((n_epoch, len_eeg)),
                       interview=np.zeros((n_epoch, len_eeg)),
                       lecture=np.zeros((n_epoch, len_eeg)),
                       news=np.zeros((n_epoch, len_eeg)),
                       talk=np.zeros((n_epoch, len_eeg)))
for ti in speech_types:
    for ei in range(n_epoch):
        print(ti, ei)
        # Load wav file +/-
        temp, rt = read_wav(audio_file_root + ti + "/" +
                            ti + "{0:03d}".format(ei) + ".wav")
        temp = temp[0, :]
        waves_pos = anm(temp, stim_fs, stim_pres_db, n_jobs=n_jobs)
        waves_pos_resmp = resample(waves_pos, down=anm_fs/eeg_fs) # resample as eeg_fs from the model sampling rate 
        # Generate ANM
        waves_neg = anm(-temp, stim_fs, stim_pres_db, n_jobs=n_jobs)
        waves_neg_resmp = resample(waves_neg, down=anm_fs/eeg_fs) # resample as eeg_fs from the model sampling rate 
        # Make sure the shapes are matched
        x_in_speech_pos[ti][ei, :] = waves_pos_resmp[:len_eeg]
        x_in_speech_neg[ti][ei, :] = waves_neg_resmp[:len_eeg]
        
# Save regressor file
write_hdf5(regressor_root + '/ANM/speech_x_in.hdf5',
           dict(x_in_speech_pos=x_in_speech_pos,
                x_in_speech_neg=x_in_speech_neg,
                fs=eeg_fs), overwrite=True)

# #%% IHC regressor generation
# # Music x_in
# x_in_music_pos = dict(acoustic=np.zeros((n_epoch, len_eeg)),
#                       classical=np.zeros((n_epoch, len_eeg)),
#                       hiphop=np.zeros((n_epoch, len_eeg)),
#                       jazz=np.zeros((n_epoch, len_eeg)),
#                       metal=np.zeros((n_epoch, len_eeg)),
#                       pop=np.zeros((n_epoch, len_eeg)))
# x_in_music_neg = dict(acoustic=np.zeros((n_epoch, len_eeg)),
#                       classical=np.zeros((n_epoch, len_eeg)),
#                       hiphop=np.zeros((n_epoch, len_eeg)),
#                       jazz=np.zeros((n_epoch, len_eeg)),
#                       metal=np.zeros((n_epoch, len_eeg)),
#                       pop=np.zeros((n_epoch, len_eeg)))
# for ti in music_types:
#     for ei in range(n_epoch):
#         print(ti, ei)
#         # Load wav file +/-
#         temp, rt = read_wav(audio_file_root + ti + "/" +
#                             ti + "{0:03d}".format(ei) + ".wav")
#         temp = temp[0, :]
#         waves_pos = ihc(temp, stim_fs, stim_pres_db, n_jobs=n_jobs)
#         waves_pos_resmp = resample(waves_pos, down=stim_fs/eeg_fs)
#         # Generate IHC
#         waves_neg = ihc(-temp, stim_fs, stim_pres_db, n_jobs=n_jobs)
#         waves_neg_resmp = resample(waves_neg, down=stim_fs/eeg_fs)  # resample as eeg_fs
        
#         x_in_music_pos[ti][ei, :] = waves_pos_resmp[:len_eeg]
#         x_in_music_neg[ti][ei, :] = waves_neg_resmp[:len_eeg]
    
# # Save regressor file
# write_hdf5(regressor_root + '/IHC/music_x_in.hdf5',
#            dict(x_in_music_pos=x_in_music_pos,
#                 x_in_music_neg=x_in_music_neg,
#                 fs=eeg_fs), overwrite=True)

# # Speech x_in
# x_in_speech_pos = dict(chn_aud=np.zeros((n_epoch, len_eeg)),
#                        eng_aud=np.zeros((n_epoch, len_eeg)),
#                        interview=np.zeros((n_epoch, len_eeg)),
#                        lecture=np.zeros((n_epoch, len_eeg)),
#                        news=np.zeros((n_epoch, len_eeg)),
#                        talk=np.zeros((n_epoch, len_eeg)))
# x_in_speech_neg = dict(chn_aud=np.zeros((n_epoch, len_eeg)),
#                        eng_aud=np.zeros((n_epoch, len_eeg)),
#                        interview=np.zeros((n_epoch, len_eeg)),
#                        lecture=np.zeros((n_epoch, len_eeg)),
#                        news=np.zeros((n_epoch, len_eeg)),
#                        talk=np.zeros((n_epoch, len_eeg)))
# for ti in speech_types:
#     for ei in range(n_epoch):
#         print(ti, ei)
#         # Load wav file +/-
#         temp, rt = read_wav(audio_file_root + ti + "/" +
#                             ti + "{0:03d}".format(ei) + ".wav")
#         temp = temp[0, :]
#         waves_pos = ihc(temp, stim_fs, stim_pres_db, n_jobs=n_jobs)
#         waves_pos_resmp = resample(waves_pos, down=stim_fs/eeg_fs)
#         # Generate IHC
#         waves_neg = ihc(-temp, stim_fs, stim_pres_db, n_jobs=n_jobs)
#         waves_neg_resmp = resample(waves_neg, down=stim_fs/eeg_fs)  # resample as eeg_fs
        
#         x_in_speech_pos[ti][ei, :] = waves_pos_resmp[:len_eeg]
#         x_in_speech_neg[ti][ei, :] = waves_neg_resmp[:len_eeg]
        
# # Save regressor file
# write_hdf5(regressor_root + '/IHC/speech_x_in.hdf5',
#            dict(x_in_speech_pos=x_in_speech_pos,
#                 x_in_speech_neg=x_in_speech_neg,
#                 fs=eeg_fs), overwrite=True)
