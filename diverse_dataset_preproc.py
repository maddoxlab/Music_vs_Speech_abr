#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 01:36:44 2020

@author: tong

Data from www.cambridge-mt.com/ms-mtk.htm
"""
import numpy as np
import os
import pandas as pd
from scipy.io import wavfile
import scipy.signal as sig
from librosa import effects
from pydub import AudioSegment
import mne

def stereo2mono(music):
    if music.shape[1] == 2:
        music_mono = np.mean(music, axis=1)
    else:
        print("File is already mono. No need to convert.")
    return np.array(music_mono, dtype='int16')

DSD_path = '/media/tong/Elements/DSD100/'
dataset_tags_path = DSD_path + 'dsd100.xlsx'
# reading tags
dataset_tags = pd.read_excel(dataset_tags_path)
# Making accompany part of music from DSD100 without vocal

source_path = DSD_path + 'Sources/'
os.listdir(source_path)
name_list = dict(Dev=[], Test=[])
for sets in ['Dev', 'Test']:
    name_list[sets] = os.listdir(source_path + sets)
for sets in ['Dev', 'Test']:
    for i in range(len(name_list[sets])):
        fs, bass = wavfile.read(source_path + sets + '/' +
                                name_list[sets][i] + '/bass.wav')
        fs, drums = wavfile.read(source_path + sets + '/' +
                                 name_list[sets][i] + '/drums.wav')
        fs, other = wavfile.read(source_path + sets + '/' +
                                 name_list[sets][i] + '/other.wav')
        accomp = bass + drums + other
        wavfile.write(source_path + sets + '/' +
                      name_list[sets][i] + '/accomp.wav', fs, accomp)

# %% Preprocessing music in 60s long

# Convert mp3 to wave

def stereo2mono(music):
    if music.shape[1] == 2:
        music_mono = np.mean(music, axis=1)
    else:
        print("File is already mono. No need to convert.")
    return np.array(music_mono, dtype='float32')


def trim_music(music, rt, top_db):
    music_trim, ind = effects.trim(music, top_db=top_db,
                                   frame_length=int(0.5*rt))
    return music_trim, ind


def trim_speech(signal, rt, top_db):
    sig_trim_ind = effects.split(signal, top_db=top_db,
                                 frame_length=int(0.5*rt))
    sig_trim = []
    for i in range(len(sig_trim_ind)):
        sig_trim_part = signal[sig_trim_ind[i, 0]:sig_trim_ind[i, 1]]
        sig_trim = np.concatenate((sig_trim, sig_trim_part))
    return sig_trim


fc = 0.1  # frequency for low-pass filtering the envelope
ref_rms = 0.01
dur = 60

for sets in ['Dev', 'Test']:
    for i in range(len(name_list[sets])):
        print([sets, name_list[sets][i]])
        fs, accomp = wavfile.read(source_path + sets + '/' +
                                  name_list[sets][i] + '/accomp.wav')
        fs, vocals = wavfile.read(source_path + sets + '/' +
                                  name_list[sets][i] + '/vocals.wav')
        music_mono = stereo2mono(accomp)
        vocals_mono = stereo2mono(vocals)
        # filter and modulate
        b, a = sig.butter(1, fc / (fs / 2.))
        env = sig.filtfilt(b, a, np.abs(music_mono))
        env += env.std() * 0.1
        gain = 1. / env
        music_norm = music_mono * gain
        
        vocals_env = sig.filtfilt(b, a, np.abs(vocals_mono))
        vocals_env += vocals_env.std() * 0.1
        vocals_gain = 1. / vocals_env
        vocals_norm = vocals_mono * vocals_gain
        
        music_norm_norm = music_norm / np.std(music_norm) * ref_rms
        vocals_norm_norm = vocals_norm / np.std(vocals_norm) * ref_rms
        
        music_trim, ind = trim_music(music_norm_norm, fs, 20)
        vocals_trim = vocals_norm_norm[ind[0]:ind[1]]
        # number of 60s long splits of the song
        n_split = len(music_trim) // (fs*dur)
        for s in range(n_split):
            wavfile.write(source_path + sets + '/' +
                          name_list[sets][i] + '/accomp_proc_' + str(s) +
                          '.wav', fs, music_trim[s*dur*fs:(s+1)*dur*fs])
            wavfile.write(source_path + sets + '/' +
                          name_list[sets][i] + '/vocals_proc_' + str(s) +
                          '.wav', fs, vocals_trim[s*dur*fs:(s+1)*dur*fs])
            print([s, ' in ', n_split])

# %% Preprocessing speech in 60s long

speech_path = '/media/tong/Elements/Speech/'
speech_list = [f for f in os.listdir(speech_path + 'mp3/') if os.path.isfile(os.path.join(speech_path + 'mp3/', f))]
for i in range(len(speech_list)):
    sound = AudioSegment.from_mp3(speech_path + 'mp3/' + speech_list[i])
    sound.export(speech_path + speech_list[i][0:-4] + '.wav', format="wav")
for i in range(len(speech_list)):
    fs, speech = wavfile.read(speech_path + speech_list[i][0:-4] + '.wav')
    print(speech_path + speech_list[i][0:-4] + '.wav')
    if len(speech.shape) == 2:
        speech_mono = stereo2mono(speech)
    else: speech_mono = speech
    speech_norm = speech_mono / np.std(speech_mono) * ref_rms
    speech_trim = trim_speech(speech_norm, fs, 20)
    n_split = len(speech_trim) // (fs*dur)
    for s in range(n_split):
        wavfile.write(speech_path + speech_list[i][0:-4] +
                      '_{0:03d}'.format(s) + '.wav', fs,
                      speech_trim[s*dur*fs:(s+1)*dur*fs])
        print([speech_list[i][0:-4], s, ' in', n_split-1])

sound_path = '/media/tong/Elements/Muspeech-Dataset/Speech/eng_aud/'
#sound_list = [f for f in os.listdir(sound_path + 'mp3/') if os.path.isfile(os.path.join(sound_path + 'mp3/', f))]
sound_list = [f for f in os.listdir(sound_path) if os.path.isfile(os.path.join(sound_path, f))]
for i in range(len(sound_list)):
    sound = AudioSegment.from_mp3(sound_path + 'mp3/' + sound_list[i])
    sound.export(sound_path + sound_list[i][0:-4] + '.wav', format="wav")
for i in range(len(sound_list)):
    fs, sound = wavfile.read(sound_path + sound_list[i][0:-4] + '.wav')
    print(sound_path + sound_list[i][0:-4] + '.wav')
    if len(sound.shape) == 2:
        sound_mono = stereo2mono(sound)
    else: sound_mono = sound
    # filter and modulate
    #sound_mono = sound_mono[0:fs*60*10]
    b, a = sig.butter(1, fc / (fs / 2.))
    env = sig.filtfilt(b, a, np.abs(sound_mono))
    env += env.std() * 0.1
    gain = 1. / env
    sound_norm = sound_mono * gain
    sound_norm_norm = sound_mono / np.std(sound_mono) * ref_rms
    #sound_trim = trim_speech(sound_norm_norm, fs, 20)
    n_split = len(sound_norm_norm) // (fs*dur)
    for s in range(n_split):
        wavfile.write(sound_path + sound_list[i][0:-4] +
                      '_{0:03d}'.format(s) + '.wav', fs,
                      sound_norm_norm[s*dur*fs:(s+1)*dur*fs])
        print([sound_list[i][0:-4], s, ' in', n_split-1])

# %% making the rt as 44100 hz
import librosa

fullset_path = '/media/tong/Elements/Muspeech_Dataset_2020/'
out_path = '/media/tong/Elements/Muspeech_Dataset_2020_1/'
music_gen = os.listdir(fullset_path + 'music/')
speech_gen = os.listdir(fullset_path + 'speech/')

fs_d = 44100
for g in music_gen:
    for root, dirs, file in os.walk(fullset_path + 'music/' + g):
        for f in file:
            wav_d,fs_d = librosa.load(os.path.join(fullset_path + 'music/' + g, f), fs_d)
            librosa.output.write_wav(os.path.join(out_path + 'music/' + g,f),wav_d,fs_d)

for g in speech_gen:
    for root, dirs, file in os.walk(fullset_path + 'speech/' + g):
        for f in file:
            wav_d,fs_d = librosa.load(os.path.join(fullset_path + 'speech/' + g, f), fs_d)
            librosa.output.write_wav(os.path.join(out_path + 'speech/' + g,f),wav_d,fs_d)

#%% If do a new stimi
sound_path = '/home/tong/Downloads/yt1s.com - Lecture 13 The Deuteronomistic History Prophets and Kings 1 and 2 Samuel.mp3'
sound = AudioSegment.from_mp3(sound_path)
sound.export(sound_path[0:-4] + '.wav', format="wav")

fs, sound = wavfile.read('/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/Muspeech-Dataset/Speech/lecture/wave/khan-fourier.wav')
if len(sound.shape) == 2:
    sound_mono = stereo2mono(sound)
else: sound_mono = sound
# filter and modulate
#sound_mono = sound_mono[0:fs*60*10]
b, a = sig.butter(1, fc / (fs / 2.))
env = sig.filtfilt(b, a, np.abs(sound_mono))
env += env.std() * 0.1
gain = 1. / env
sound_norm = sound_mono * gain

#import scipy.signal as signal
#notch_freq = np.arange(60, 120, 180)
#notch_width = 1
#for nf in notch_freq:
#    bn, an = signal.iirnotch(nf / (fs / 2.), float(nf) / notch_width)
#    sound_norm = signal.lfilter(bn, an, sound_norm)
        
sound_norm_norm = sound_norm / np.std(sound_norm) * ref_rms
sound_norm_norm = trim_speech(sound_norm_norm, fs, 20)
n_split = len(sound_norm_norm) // (fs*dur)
for s in range(n_split):
    wavfile.write('/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/Muspeech-Dataset/Speech/lecture/khan-fourier' +
                  '_{0:03d}'.format(s) + '.wav', fs,
                  sound_norm_norm[s*dur*fs:(s+1)*dur*fs])

import matplotlib.pyplot as plt
plt.magnitude_spectrum(sound_norm, Fs=44100)

#%% CHN_AUD PROCESSING
path = "/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/Muspeech-Dataset/Speech/chn_aud/AISHELL-2019C-EVAL/SPEECHDATA/wav/006/"
sound = []
for i in range(150):
    if os.path.isfile(path + "006_mic1_{0:04d}".format(i+1) + ".wav"):
        fs, temp = wavfile.read(path + "006_mic1_{0:04d}".format(i+1) + ".wav")
        sound += list(temp)

sound = np.array(sound)
if len(sound.shape) == 2:
    sound_mono = stereo2mono(sound)
else: sound_mono = sound

b, a = sig.butter(1, fc / (fs / 2.))
env = sig.filtfilt(b, a, np.abs(sound_mono))
env += env.std() * 0.1
gain = 1. / env
sound_norm = sound_mono * gain
sound_norm_norm = sound_norm / np.std(sound_norm) * ref_rms
sound_trim = trim_speech(sound_norm_norm, fs, 30)
n_split = len(sound_norm_norm) // (fs*dur)
for s in range(n_split):
    wavfile.write('/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/Muspeech-Dataset/Speech/chn_aud/AISHELL_006' +
                  '_{0:03d}'.format(s) + '.wav', fs,
                  sound_norm_norm[s*dur*fs:(s+1)*dur*fs])


#%% DO THIS AFTER SPECTRAL MATCHING!!
# FLIP EVERY TRIAL
from expyfun.stimuli import window_edges
### Chunck stim in to 12 small pieces
sound_path = "/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/Muspeech_Dataset_2020_2/spectral_match_bandpower/"
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
n_ep_each = int(60/12)

for g in music_types:
    for root, dirs, file in os.walk(sound_path + g):
        for f in file:
            fs, sound = wavfile.read(os.path.join(sound_path + g, f))
            for i in range(n_ep_each):
                if i%2 == 0:
                    sound_temp = sound[i*12*fs:(i+1)*12*fs]
                    sound_temp = window_edges(sound_temp, fs, dur=0.03)
                else:
                    sound_temp = - (sound[i*12*fs:(i+1)*12*fs])
                    sound_temp = window_edges(sound_temp, fs, dur=0.03)
                wavfile.write(sound_path + '12s/' + g + '/' + f[0:-7] + '{0:03d}'.format(int(f[-7:-4])*n_ep_each + i) + '.wav', fs, sound_temp)
                
for g in speech_types:
    for root, dirs, file in os.walk(sound_path + g):
        for f in file:
            fs, sound = wavfile.read(os.path.join(sound_path + g, f))
            for i in range(n_ep_each):
                if i%2 == 0:
                    sound_temp = sound[i*12*fs:(i+1)*12*fs]
                    sound_temp = window_edges(sound_temp, fs, dur=0.03)
                else:
                    sound_temp = - (sound[i*12*fs:(i+1)*12*fs])
                    sound_temp = window_edges(sound_temp, fs, dur=0.03)
                wavfile.write(sound_path + '12s/' + g + '/' + f[0:-7] + '{0:03d}'.format(int(f[-7:-4])*n_ep_each + i) + '.wav', fs, sound_temp)

# %% NO Flip every other trial
from expyfun.stimuli import window_edges
### Chunck stim in to 12 small pieces
sound_path = "/media/tong/Elements/AMPLab/MusicABR/diverse_dataset/Muspeech_Dataset_2020_2/spectral_match_bandpower/"
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
n_ep_each = int(60/12)

for g in music_types:
    for root, dirs, file in os.walk(sound_path + g):
        for f in file:
            fs, sound = wavfile.read(os.path.join(sound_path + g, f))
            for i in range(n_ep_each):
                sound_temp = sound[i*12*fs:(i+1)*12*fs]
                sound_temp = window_edges(sound_temp, fs, dur=0.03)
                wavfile.write(sound_path + '12s/' + g + '/' + f[0:-7] + '{0:03d}'.format(int(f[-7:-4])*n_ep_each + i) + '.wav', fs, sound_temp)
                
for g in speech_types:
    for root, dirs, file in os.walk(sound_path + g):
        for f in file:
            fs, sound = wavfile.read(os.path.join(sound_path + g, f))
            for i in range(n_ep_each):
                sound_temp = sound[i*12*fs:(i+1)*12*fs]
                sound_temp = window_edges(sound_temp, fs, dur=0.03)
                wavfile.write(sound_path + '12s/' + g + '/' + f[0:-7] + '{0:03d}'.format(int(f[-7:-4])*n_ep_each + i) + '.wav', fs, sound_temp)