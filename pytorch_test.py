from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pytorch_model
import numpy as np
import os.path
import utils

import sys


def main(args):
    proj_path = os.getcwd()
    data_path = 'data'
    test_path = data_path + '/test/preprocessed'
    model_save_path = 'model'

    save_freq = 10
    max_epoch = 5000
    max_patience = 30
    window_size = 7
    num_features = 264
    batch_size = 16

    net = torch.load(args[1])

    test_x_list, test_y_list = utils.data_load( 'data/final/preprocessed')

    train_piece_lens = []
    test_piece_lens = []

    for i in range( len( test_x_list ) ):
        # Add 1 to train data for log computability.
        # It can be inversed at post-processing phase.
        test_x_list[i] = utils.standardize( test_x_list[i] + 1, log=True ).T
        test_y_list[i] = test_y_list[i].T
        test_piece_lens.append( test_x_list[i].shape[0] )

        print( 'test loaded {}/{}'.format( i + 1, len( test_x_list ) ) )

    test_x = np.vstack( test_x_list )
    del test_x_list
    test_y = np.vstack( test_y_list )
    del test_y_list

    # For GPU computing.
    dtype = torch.cuda.FloatTensor
    test_x = Variable( torch.Tensor( test_x ).type( dtype ) )
    test_x.volatile = True
    test_y = Variable( torch.Tensor( test_y ).type( dtype ) )
    test_y.volatile = True

    min_valid_loss = float( 'inf' )
    patience = 0

    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam( net.parameters() )

    print( 'Preprocessing Completed.' )

    # Train and calculate loss value.
    prec, recall, acc = run_test( net, test_x, test_y, criterion,
        test_piece_lens, batch_size, window_size )
    f_score = 2 * prec * recall / (prec + recall)

    print ('Precision: {}\tRecall: {}\tAccuracy: {}'.format(prec, recall, acc))
    print ('F-score: {}'.format(f_score))


def run_test( net, inputs, labels, criterion, piece_lens, batch_size, window_size ):
    overall_num_samples = 0
    num_samples = sum( piece_lens )
    num_batches = num_samples // batch_size
    # num_batches = 5
    tp, fp, fn = 0, 0, 0
    for i in range(window_size // 2, inputs.data.size()[0] - window_size // 2 - 1):
        x = inputs[i - window_size // 2 : i + window_size // 2 + 1, :]
        y = labels[i].cpu().data.numpy()
        z = net(x).cpu().data.numpy()[0, :]
        z = 1 * (z.round() > 0)
        tp += y.dot(z)
        fp += np.sum(1 * (y - z < 0))
        fn += np.sum(1 * (z - y < 0))
    p = tp / float(tp + fp)
    r = tp / float(tp + fn)
    a = tp / float(tp + fp + fn)
    return p, r, a


if __name__ == '__main__':
    main(sys.argv)
