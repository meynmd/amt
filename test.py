import scipy.io.wavfile as wavfile
import numpy as np
from numpy import inf
import utils
import matplotlib.pyplot as plt
import pdb

sr,wav = wavfile.read('example2.wav')
wav = np.mean(wav,axis=1)

cqt = utils.cqt(wav)
print(cqt.min())
std_cqt = utils.standardize(cqt)
log_std_cqt = utils.standardize(cqt+1,log=True)
#log_std_cqt[log_std_cqt == -inf] = 0
pdb.set_trace()
plt.pcolormesh(std_cqt,cmap='jet')
plt.show()
plt.pcolormesh(log_std_cqt,cmap='jet')
plt.show()
