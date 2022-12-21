#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 01:36:44 2020

@author: tong
"""
import numpy as np
import os
import pandas as pd
from scipy.io import wavfile
import scipy.signal as sig
from librosa import effects
from pydub import AudioSegment

"""
This script is used for preprocessing of the music and speech stimuli
"""

def stereo2mono(music):
    if music.shape[1] == 2:
        music_mono = np.mean(music, axis=1)
    else:
        print("File is already mono. No need to convert.")
    return np.array(music_mono, dtype='int16')

DSD_path = '/DSD100/'
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

# %% Preprocessing music

# Convert mp3 to wave
def stereo2mono(music):
    if music.shape[1] == 2:
        music_mono = np.mean(music, axis=1)
    else:
        print("File is already mono. No need to convert.")
    return np.array(music_mono, dtype='float32')


def trim_music(music, rt, top_db):
    """
    Parameters
    ----------
    music : priginal music files
    rt : music sampling rate
    top_db : threshold below to consider as silence

    Returns
    -------
    music_trim : trimmed signal
    ind : the index of interval responding to non-silent region
    """
    music_trim, ind = effects.trim(music, top_db=top_db,
                                   frame_length=int(0.5*rt))
    return music_trim, ind

fc = 0.1  # frequency for low-pass filtering the envelope
ref_rms = 0.01
dur = 12

for sets in ['Dev', 'Test']:
    for i in range(len(name_list[sets])):
        print([sets, name_list[sets][i]])
        fs, accomp = wavfile.read(source_path + sets + '/' +
                                  name_list[sets][i] + '/accomp.wav')
        music_mono = stereo2mono(accomp)
        # filter and modulate, flatten the envelope
        b, a = sig.butter(1, fc / (fs / 2.))
        env = sig.filtfilt(b, a, np.abs(music_mono))
        env += env.std() * 0.1
        gain = 1. / env
        music_norm = music_mono * gain
        # normalize to 0.01 rms
        music_norm_norm = music_norm / np.std(music_norm) * ref_rms 
        # trim the silence
        music_trim, ind = trim_music(music_norm_norm, fs, 20)
        # number of 60s long splits of the song
        n_split = len(music_trim) // (fs*dur)
        for s in range(n_split):
            wavfile.write(source_path + sets + '/' +
                          name_list[sets][i] + '/accomp_proc_' + str(s) +
                          '.wav', fs, music_trim[s*dur*fs:(s+1)*dur*fs])
            print([s, ' in ', n_split])

# %% Preprocessing speech
speech_path = '/Speech/'
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
    speech_norm = speech_mono / np.std(speech_mono) * ref_rms # normalize to 0.01 rms
    n_split = len(speech_norm) // (fs*dur)
    for s in range(n_split):
        wavfile.write(speech_path + speech_list[i][0:-4] +
                      '_{0:03d}'.format(s) + '.wav', fs,
                      speech_norm[s*dur*fs:(s+1)*dur*fs])
        print([speech_list[i][0:-4], s, ' in', n_split-1])
