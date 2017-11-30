"""

"""
import numpy as np
import os
import torch
from torch.autograd import Variable
import scipy.io.wavfile
import librosa
import sys


def correct_perm(perms,piece_lens,padding):
  """
  A function for correcting permutation index.
  Args
    perm (numpy int array) : array of index to sample. shape = [<=batch_size]
    len_list (python list of int) : list of samples of each music piece.
    padding (tuple) : padding policy. shape = (front,end)
  Returns
    perm (numpy int array) : corrected permutation index.
  """
  cumul_lens = [sum(piece_lens[:i+1]) for i in range(len(piece_lens))]
  for i,perm in enumerate(perms):
    last_cumul_len = 0
    for j,cumul_len in enumerate(cumul_lens):
      if perm>=last_cumul_len and perm<cumul_len:
        perm+=padding[0] + sum(padding)*j
        break
      else :
        last_cumul_len = cumul_len
    perms[i] = perm

  return perms


def next_batch(inputs,labels,perms,train_len_list,window_size):
  """
  Load next batch ofdata.
  It samples batch_size amount of data where perm indicates.
  Index is manipulated to avoid to indicate padded area.
  Args
    inputs (torch.Tensor) : input data. shape = [num_samples, num_features(264)]
    labels (torch.Tensor) : label data. shape = [num_samples, ==> dim_label (88)] <===========
    perms (numpy int array) : array of index to sample. shape = [<=batch_size]
    train_len_list (numpy list) :
        list of samples of each music piece. They are separated with zero pads.
        Each music pieace is padded with 3 rows along axis 0.
        ex) [0,0,0,1,7,2,8,9,6,0,0,0,0,0,0,5,3,6,6,4,7,5,4,3,0,0,0] (inaccurate)
        shape = [180]
    window_size (python int) : window size.
  Returns
    input_batch (torch.Tensor) :
        batch of input data.
        shape = [batch_size,1(channel),window_size,num_features]
    label_batch (torch.Tensor) :
        batch of label data.
        shape = [batch_size,1,1,dim_label]
  """
  batch_size = len(perms)
  num_features = inputs.data.size()[1]
  dim_label = labels.data.size()[1]
  input_batch = Variable(torch.cuda.FloatTensor(batch_size, 1, window_size,num_features))
  label_batch = Variable(torch.cuda.FloatTensor(batch_size,dim_label))
  corrected_perms = correct_perm(perms,train_len_list,(3,3))

  for i, perm in enumerate(corrected_perms):
    while perm >= inputs.size()[0] - window_size//2:
      perm -= window_size
    input_batch[i,0,:,:] = inputs[int(perm-window_size//2) : int(perm+window_size//2+1), :]
    label_batch[i,:] = labels[perm,:]

  return input_batch,label_batch


def permutate(x,y):
  """
  Permutate x and y array in same way.
  Args
    x,y (numpy array) : x and y data. shape = (num_samples,?,?)
  Returns
    x,y (numpy array) : Random permutated x,y data. shape = (num_samples,?,?)
  """
  dtype = torch.cuda.LongTensor
  perm = torch.randperm(x.size()[0]).type(dtype)
  return x[perm],y[perm]


def standardize(data,axis=None,log=False):
  """
  Standardize given data.
  data = (data-data.mean())/data.std()
  Args
    data (numpy array) : Input data. Assumed to be rank 2. shape = (?,?)
    axis (int) : Indicator of which axis to apply standardization.
    log (bool) : Whether calculate log value or not.
  Returns
    data (numpy array) : Standardized data.
  """
  if (log==True):
    data = np.log(data)
  if (axis==None):
    data = data - data.mean()
    data = data / data.std()
  else:
    assert axis<2
    shape = list(data.shape)
    shape.pop(axis)
    if(axis==0):
      for i in range(shape[0]):
        data[:,i] = (data[:,i] - data[:,i].mean()) / data[:,i].std()
    elif(axis==1):
      for i in range(shape[1]):
        data[i,:] = (data[i,:] - data[i,:].mean()) / data[i,:].std()

  return data


def window(data,window_size):
  """
  Slice data in window_wise.
  Args
    data (numpy array) : Input data. shape = (time,num_features)
    window_size (int) : Size of window.
  Returns
    windowed_data (numpy array) : Windowed data. shape=(time,window_size,num_features)
  """

  time_length = data.shape[0]
  num_features = data.shape[1]
  is_even = window_size % 2 == 0

  # Zero padding
  pad = np.zeros([window_size/2,num_features])
  pad_minus_one = np.zeros([window_size/2-1,num_features])
  data = np.append(pad,data,axis=0)
  if is_even:
    data = np.append(data,pad_minus_one,axis=0)
  else :
    data = np.append(data,pad,axis=0)

  # Append to list
  windowed_list = []
  for i in range(time_length):
    windowed_list.append(data[i:i+window_size])

  # Merge into single numpy array.

  windowed_data = np.asarray(windowed_list)

  return windowed_data


def data_load(path, max_files=0):
  """
  Load cqt and label data and return it as list.
  Order of cqt and label in each list is guaranteed to be same by using sort().
  Args
    path (python str) : Path from which we parse data.
  Returns
    x_list (list of np array) : list of numpy array of train data.
    y_list (list of np array) : list of numpy array of test data.
  """

  f_list = sorted(os.listdir(path))
  x_list = []
  y_list = []
  x_list
  # Separate cqt and label file
  for i, f in enumerate(f_list):
    if (max_files!=0 and i>=max_files) : break
    filename = path + '/' + f
    if '.wav' in f:
      base_name = f.split('_')[0]
      mid_name = base_name + '.mid.npy'
      if mid_name in f_list:
        x_data = np.load(filename)
        y_data = np.load(path + '/' + mid_name)
        if x_data.shape[1] == y_data.shape[1]:
          x_list.append(x_data)
          y_list.append(y_data)
        else:
          print('warning: data sizes from files {} and {} do not match. Ignoring.'.format(filename, mid_name))
        # x_list.append(np.load(filename))
        # y_list.append(np.load(path + '/' + mid_name))
        print('loading {} to x list'.format(filename), file=sys.stderr)
        print( 'loading {} to y list'.format( path + '/' + mid_name ), file=sys.stderr)



  # for i, f in enumerate(f_list):
  #   if (max_files!=0 and i>=max_files) : break
  #   filename = path + '/' + f
  #   if '.wav' in f:
  #     x_list.append(np.load(filename))
  #     print('loading {} to x list'.format(filename))
  #   elif '.mid' in f:
  #     y_list.append(np.load(filename))
  #     print('loading {} to y list'.format(filename))


  return x_list, y_list


def load_wav(path,target_sr=16000):
  """
  Load .wav file.
  Args
    path (python str) : Path from which we parse data.
  Returns
    sr (python float) : sample rate. default is 16000.
    wav_resample (numpy array) : resampled wav data. shape = [len].
  """
  original_sr, wav = scipy.io.wavfile.read(path)
  wav = 0.5*(wav[:,0]+wav[:,1])
  wav_resample = librosa.core.resample(wav,original_sr,target_sr)

  return wav_resample


def cqt(wav,sr=16000,hop_length=512,n_bins=264,bins_per_octave=36):
  """
  Calculate cqt
  Args
    wav (numpy array) : loaded wavfile. shape = [len]
  Returns
    cqt_wav : cqt result. shape = [?,?]
  """
  cqt_wav=np.abs(librosa.core.cqt(y=wav,sr=sr,
                 hop_length=hop_length,fmin=librosa.core.note_to_hz('A0'),
                 n_bins=n_bins,bins_per_octave=bins_per_octave))
  return cqt_wav

def cqt_windows(wav, size_w, sr=16000,hop_length=512,n_bins=264,bins_per_octave=36):
  trans = cqt(wav, sr, hop_length, n_bins, bins_per_octave)
  trans = np.pad(trans, ((int(size_w / 2), int(size_w / 2)), (0, 0)), 'constant', constant_values=np.min(trans))
  return np.array([trans[:, i : i + size_w] for i in range(trans.shape[1] - size_w + 1)])

