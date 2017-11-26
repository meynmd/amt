import numpy as np
import glob
import os
import sys
import utils
from lib import pretty_midi as pmidi

proj_dir = os.getcwd()
data_dir = proj_dir + '/data'
wav_dir = data_dir + '/wav'
mid_dir = data_dir + '/mid'
train_dir = data_dir + 'train'

os.chdir(wav_dir)
wavfiles = [glob.glob('*.wav')]
os.chdir(mid_dir)
midfiles = [glob.glob('*.mid')]
trainfiles = [(wav_dir + '/' + fn, mid_dir + fn[:-3] + '.mid') for fn in wavfiles]

for wav, mid in trainfiles:
    wavdata = utils.load_wav(wav)
    cqt = utils.cqt(wavdata)
    savefile = open(train_dir + wav + '.npy', 'w')
    np.save(savefile, cqt)
    print ('wrote {}'.format(savefile.name), file=sys.stderr)

    labels = np.zeros((88,cqt.shape[1]))
    with open(mid) as f:
        pm = pmidi.PrettyMIDI(mid)

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