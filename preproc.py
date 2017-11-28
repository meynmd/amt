import numpy as np
import glob
import os
import sys
import utils
import librosa
from lib import pretty_midi as pmidi

proj_dir = os.getcwd()
data_dir = 'data'
wav_dir = data_dir + '/wav'
mid_dir = data_dir + '/mid'
train_dir = data_dir + '/train'

sample_rate = 16000
hop_len = 512

# test files
os.chdir(proj_dir + '/data/test/mid')
midfiles = glob.glob('*.mid')
testfiles = []
os.chdir(proj_dir + '/data/test/wav')
for fname in midfiles:
    base_fname = fname.split('.')[0]
    testfiles += [('data/test/wav/' + wavfile, 'data/test/mid/' + fname) for wavfile in glob.glob(base_fname + '*')]

# gather train files
"""
os.chdir(proj_dir + '/data/train/mid')
midfiles = glob.glob('*.mid')
trainfiles = []
os.chdir(proj_dir + '/data/train/wav')
for fname in midfiles:
    base_fname = fname.split('.')[0]
    trainfiles += [('data/train/wav/' + wavfile, 'data/train/mid/' + fname)
                   for wavfile in glob.glob(base_fname + '*')]
"""

os.chdir(proj_dir + '/data/train/wav')
wavfiles = glob.glob('*.wav')
trainfiles = []
os.chdir(proj_dir + '/data/train/mid')
for wname in wavfiles:
    base_fname = wname.split('_')[0]
    trainfiles += [('data/train/wav/' + wname, 'data/train/mid/' + base_fname + '.mid')]


# preprocess test
xs, ys = [], []
os.chdir(proj_dir)
for wav, mid in testfiles:
    # do constant-q transform on the wav file
    wavdata = utils.load_wav(wav)
    # cqt_windows = utils.cqt_windows(wavdata, 7, hop_length=hop_len)
    cqt_windows = utils.cqt(wavdata)
    savefile = 'data/test/preprocessed/' + wav.split('/')[-1]
    np.save(savefile, cqt_windows)
    xs.append(cqt_windows)
    print ('wrote {}.npy\n dimensions: {}'.format(savefile, cqt_windows.shape), file=sys.stderr)

    pm = pmidi.PrettyMIDI(mid)
    t = librosa.frames_to_time(np.arange(cqt_windows.shape[1]), sr=sample_rate, hop_length=hop_len)
    piano_roll = pm.get_piano_roll(fs=sample_rate, times=t)
    savefile = 'data/test/preprocessed/' + mid.split( '/' )[-1]
    np.save(savefile, piano_roll)
    ys.append(piano_roll)
    print ('wrote {}.npy\n dimensions: {}'.format(savefile, piano_roll.shape), file=sys.stderr)

# preprocess training
xs, ys = [], []
os.chdir(proj_dir)
for wav, mid in trainfiles:
    # do constant-q transform on the wav file
    wavdata = utils.load_wav(wav)
    # cqt_windows = utils.cqt_windows(wavdata, 7, hop_length=hop_len)
    cqt_windows = utils.cqt(wavdata)
    savefile = 'data/train/preprocessed/' + wav.split('/')[-1]
    np.save(savefile, cqt_windows)
    xs.append(cqt_windows)
    print ('wrote {}.npy\n dimensions: {}'.format(savefile, cqt_windows.shape), file=sys.stderr)

    pm = pmidi.PrettyMIDI(mid)
    t = librosa.frames_to_time(np.arange(cqt_windows.shape[1]), sr=sample_rate, hop_length=hop_len)
    piano_roll = pm.get_piano_roll(fs=sample_rate, times=t)
    savefile = 'data/train/preprocessed/' + mid.split( '/' )[-1]
    np.save(savefile, piano_roll)
    ys.append(piano_roll)
    print ('wrote {}.npy\n dimensions: {}'.format(savefile, piano_roll.shape), file=sys.stderr)
