
import sys
import pickle
import numpy as np
import read_dir
from accuracy import *
from ann import *


size = [130, 80, 60]
iterations = 2800
rate = 0.01

names = {}

def make_arrays(mfcc, num_of_speakers, test = False):
    pat = []
    target = []

    s_num = 0 # speaker index
    for s in mfcc:
        for mfc in s[1]:
            tar = [0] * num_of_speakers
            if test:
            	s_num = names[s[0]]
            tar[s_num] = 1
            pat.append(mfc)
            target.append(tar)
        if not test:
        	names[s[0]] = s_num
        	s_num += 1
    return (pat, target)

def train(train_data):
    mfcc = read_dir.wav_to_mfcc(train_data, size[0])
    num_of_speakers = len(mfcc)
    inputs, target = make_arrays(mfcc, num_of_speakers)
    size.append(num_of_speakers)

    nn = Ann(size)
    nn.train(np.array(inputs), np.array(target), True, iterations, rate)
    print ('printing weights to file .........')
    nn.print_to_file()
    print ('printing successful\n')

    print (names)
    f = open('names', 'wb')
    pickle.dump(names, f)

def load_names():
	f = open('names', 'rb')
	d = pickle.load(f)
	for k in d:
		names[k] = d[k]
	
def test(test_data):
	mfcc = read_dir.wav_to_mfcc(test_data, size[0])
	num_of_speakers = len(mfcc)

	load_names()

	inputs, target = make_arrays(mfcc, num_of_speakers, True)

	size.append(num_of_speakers)
	nn = Ann(size)
	nn.load_w()
	print (accuracy.percent(np.array(inputs), np.array(target), nn))

def get_name(i):
	for k in names:
		if names[k] == i:
			return k
	return 'invalid speaker'

def recognize(wav_file, ns):
	load_names()

	mfcc = read_dir.normalize(read_dir.convert_wav_to_mfcc(wav_file, size[0]))

	size.append(ns)
	nn = Ann(size)
	nn.load_w()

	sindex = accuracy.sindex(nn.test(mfcc))

	print ('Speaker identified as:', get_name(sindex))


if __name__ == '__main__':
	usage = 'usage: main.py <-train -test -recog> <dir> <num_iter num_speakers>'
	options = ['-train', '-test', '-recog']
	if len(sys.argv) < 3 or sys.argv[1] not in options:
		print (usage)
		sys.exit()
	if sys.argv[1] == '-train':
		if len(sys.argv) > 3:
			iterations = int(sys.argv[3])
		train(sys.argv[2])
	elif sys.argv[1] == '-test':
		test(sys.argv[2])
	elif len(sys.argv) > 3 and sys.argv[1] == '-recog':
		recognize(sys.argv[2], int(sys.argv[3]))
	else:
		print (usage)

    
