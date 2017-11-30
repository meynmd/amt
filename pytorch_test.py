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


def main():
    proj_path = os.getcwd()
    data_path = 'data'
    test_path = data_path + '/test/preprocessed'
    model_save_path = 'model'

    save_freq = 10
    max_epoch = 5000
    max_patience = 30
    window_size = 7
    num_features = 264
    batch_size = 1024

    torch.load()

    net = pytorch_model.AMT( window_size, num_features ).cuda()
    train_x_list, train_y_list = utils.data_load( train_path )
    test_x_list, test_y_list = utils.data_load( 'data/test/preprocessed', 2 )

    train_piece_lens = []
    test_piece_lens = []

    # Standardize.
    for i in range( len( train_x_list ) ):
        # Add 1 to train data for log computability.
        # It can be inversed at post-processing phase.
        # train_x_list[i] = np.pad(standardized, ((3,3),(0,0)),'constant')
        # train_y_list[i] = np.pad(train_y_list[i],((3,3),(0,0)),'constant')
        train_x_list[i] = utils.standardize( train_x_list[i] + 1, log=True ).T
        train_y_list[i] = train_y_list[i].T
        train_piece_lens.append( train_x_list[i].shape[0] )
    print( 'train loaded {}/{}'.format( i + 1, len( train_x_list ) ) )

    for i in range( len( test_x_list ) ):
        # Add 1 to train data for log computability.
        # It can be inversed at post-processing phase.
        test_x_list[i] = utils.standardize( test_x_list[i] + 1, log=True ).T
        test_y_list[i] = test_y_list[i].T
        test_piece_lens.append( test_x_list[i].shape[0] )

        # test_x_list[i] = np.pad(utils.standardize(test_x_list[i]+1,log=True),
        #                          ((3,3),(0,0)),'constant')
        # test_y_list[i] = np.pad(test_y_list[i],((3,3),(0,0)),'constant')

        print( 'test loaded {}/{}'.format( i + 1, len( test_x_list ) ) )

    train_x = np.vstack( train_x_list )
    del train_x_list
    train_y = np.vstack( train_y_list )
    del train_y_list
    test_x = np.vstack( test_x_list )
    del test_x_list
    test_y = np.vstack( test_y_list )
    del test_y_list

    # For GPU computing.
    dtype = torch.cuda.FloatTensor
    train_x = Variable( torch.Tensor( train_x ).type( dtype ) )
    train_y = Variable( torch.Tensor( train_y ).type( dtype ) )
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

    '''
    net.load_state_dict(torch.load(model_save_path+'10'))
    out = net(train_x[:10])
    print(out)
    '''
    for i in range( max_epoch ):

        # Train and calculate loss value.
        train_loss = pytorch_model.run_train( net, train_x, train_y, criterion,
                                              optimizer, train_piece_lens, batch_size,
                                              window_size ).cpu().data.numpy()
        valid_loss = pytorch_model.run_loss( net, test_x, test_y, criterion,
                                             test_piece_lens, batch_size, window_size
                                             ).cpu().data.numpy()
        if (valid_loss < min_valid_loss):
            patience = 0
            min_valid_loss = valid_loss
            torch.save( net.state_dict(), model_save_path + '_ReLU_whole_log_best' )
            print( '\nBest model is saved.***\n', file=sys.stderr )
        else:
            patience += 1
        if (patience == max_patience or i == max_epoch - 1):
            torch.save( net.state_dict(), model_save_path + '_ReLU_whole_log' + str( i + 1 ) )
            print( '\n***{}th last model is saved.***\n'.format( i + 1 ), file=sys.stderr )
            break

        print( '------{}th iteration (max:{})-----'.format( i + 1, max_epoch ), file=sys.stderr )
        print( 'train_loss : ', train_loss, file=sys.stderr )
        print( 'valid_loss : ', valid_loss, file=sys.stderr )
        print( 'patience : ', patience, file=sys.stderr )

        # print(i+1, train_loss[0], valid_loss[0])

        if (i % save_freq == save_freq - 1):
            torch.save( net.state_dict(), model_save_path + '_ReLU_whole_log' + str( i + 1 ) )
            print( '\n***{}th model is saved.***\n'.format( i + 1 ), file=sys.stderr )


if __name__ == '__main__':
    main()
