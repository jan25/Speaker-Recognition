
import os
from features import mfcc
import scipy.io.wavfile as wav
import numpy as np

def normal(m, mini, maxi):
	return (m - mini) / (maxi - mini)

def normalize(mfcc):
	mini = 100
	maxi = -100
	for i in range(len(mfcc)):
		mini = min(mini, mfcc[i])
		maxi = max(maxi, mfcc[i])
	for i in range(len(mfcc)):
		mfcc[i] = normal(mfcc[i], mini, maxi)
	return mfcc

def pick_window(m, step):
	#r = np.random.randint(0, len(m) - step)
	r = len(m) / 2
	return m[r : r + step]


# picks 2000 mfcc from center of sample
def convert_wav_to_mfcc(wav_file, step = 120):
	(f, signal) = wav.read(wav_file)
	mf = mfcc(signal, f).flatten()
	return pick_window(mf, step)

'''
argument: input directory_name
returns [['speaker_dir_name', [mfcc_clip1, mfcc_clip2, ]], ]
'''
def wav_to_mfcc(dir_name, step = 120):
	speaker_to_mfcc = []
	for speaker in os.listdir(dir_name):
		mfcc = []
		for wav_file in os.listdir(dir_name + '/' + speaker + '/wav'):
			if wav_file.endswith('.wav'):
				full_path = dir_name + '/' + speaker + '/wav/' + wav_file
				mfcc.append(normalize(convert_wav_to_mfcc(os.path.abspath(full_path), step)))
		speaker_to_mfcc.append([speaker.split('-')[0], mfcc])
	return speaker_to_mfcc

