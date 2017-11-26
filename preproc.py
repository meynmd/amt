import numpy as np
import glob
import os
import sys
import utils
from lib import pretty_midi as pmidi

proj_dir = os.getcwd()
data_dir = 'data'
wav_dir = data_dir + '/wav'
mid_dir = data_dir + '/mid'
train_dir = data_dir + '/train'

sample_rate = 16000

os.chdir(proj_dir + '/' + mid_dir)
midfiles = glob.glob('*.mid')

trainfiles = []
os.chdir(proj_dir + '/' + wav_dir)
for fname in midfiles:
    base_fname = fname.split('.')[0]
    trainfiles += [(wav_dir + '/' + wavfile, mid_dir + '/' + fname) for wavfile in glob.glob(base_fname + '*')]

os.chdir(proj_dir)
for wav, mid in trainfiles:
    # do constant-q transform on the wav file
    wavdata = utils.load_wav(wav)
    # cqt = utils.cqt(wavdata)
    cqt = utils.cqt_windows(wavdata, 7)
    savefile = train_dir + '/' + wav.split('/')[-1]
    np.save(savefile, cqt)
    print ('wrote {}'.format(savefile), file=sys.stderr)

    labels = np.zeros((88, cqt.shape[1]))
    pm = pmidi.PrettyMIDI(mid)
    piano_roll = pm.get_piano_roll(fs=sample_rate)  # would times=cqt.shape[1] be correct?
    print ('done')



#     lines = lines[1:]
#     lines = [line.strip().split('\t') for line in lines]
#     for line in lines:
#         start_frame = int(round(frame_per_sec*float(line[0])))
#         end_frame = int(round(frame_per_sec*float(line[1])))
#         pitch = int(line[2])-21
#         for j in range(start_frame,end_frame):
#             y_data[pitch,j]=1 asdf
#     np.save(txtfile+".npy",y_data)
# print "%d / %d" % (i+1,len(txtfile_list))