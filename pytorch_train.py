from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_model
#import amt_model as pytorch_model
import numpy as np
import os.path
import utils
import pdb
import gc
import sys


def main():
    proj_path = os.getcwd()
    data_path = 'data'
    train_path = data_path + '/train/preprocessed'
    model_save_path = 'model'

    save_freq = 1
    max_epoch = 5
    max_patience = 5
    window_size = 7
    num_features = 264
    batch_size = 128
    mb_size = 500

    net = pytorch_model.AMT( window_size, num_features ).cuda()
    train_x_list, train_y_list = utils.data_load( train_path )
    test_x_list, test_y_list = utils.data_load( 'data/test/preprocessed' )

    train_piece_lens = []
    test_piece_lens = []

    # Standardize.
    for i in range( len( train_x_list ) ):
        train_x_list[i] = utils.standardize( train_x_list[i] + 1, log=True ).T
        train_y_list[i] = train_y_list[i].T
        train_piece_lens.append( train_x_list[i].shape[0] )
    print( 'train loaded {}/{}'.format( i + 1, len( train_x_list ) ), file=sys.stderr )

    for i in range( len( test_x_list ) ):
        # Add 1 to train data for log computability.
        # It can be inversed at post-processing phase.
        test_x_list[i] = utils.standardize( test_x_list[i] + 1, log=True ).T
        test_y_list[i] = test_y_list[i].T
        test_piece_lens.append( test_x_list[i].shape[0] )

        # test_x_list[i] = np.pad(utils.standardize(test_x_list[i]+1,log=True),
        #                          ((3,3),(0,0)),'constant')
        # test_y_list[i] = np.pad(test_y_list[i],((3,3),(0,0)),'constant')

        print( 'test loaded {}/{}'.format( i + 1, len( test_x_list ) ), file=sys.stderr )

    train_x = np.vstack( train_x_list )
    del train_x_list
    train_y = np.vstack( train_y_list )
    del train_y_list
    test_x = np.vstack( test_x_list )
    del test_x_list
    test_y = np.vstack( test_y_list )
    del test_y_list

    # train_x = Variable( torch.Tensor( train_x ) )
    # train_y = Variable( torch.Tensor( train_y ) )
    # test_x = Variable( torch.Tensor( test_x ) )
    # test_x.volatile = True
    # test_y = Variable( torch.Tensor( test_y ) )
    # test_y.volatile = True

    min_valid_loss = float( 'inf' )
    patience = 0

    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam( net.parameters() )

    print( 'Preprocessing Completed.', file=sys.stderr)
    sys.stderr.flush()

    num_megabatches = train_x.data.shape[0] // mb_size
    print( '{} megabatches\n'.format(num_megabatches), file=sys.stderr )
    train_megabatches = [(train_x[k*mb_size : (k+1)*mb_size, :], train_y[k*mb_size : (k+1)*mb_size, :])
                   for k in range(num_megabatches)]
    # train_megabatches.append((train_x[num_megabatches*mb_size :, :], train_y[num_megabatches*mb_size :, :]))
    test_megabatches = [(test_x[k*mb_size : (k+1)*mb_size, :], test_y[k*mb_size : (k+1)*mb_size, :])
                        for k in range( num_megabatches )]
    # test_megabatches.append((test_x[num_megabatches*mb_size :, :], test_y[num_megabatches*mb_size :, :]))

    del train_x, train_y, test_x, test_y

    for j in range(num_megabatches):
        print( 'megabatch {}'.format(j+1) )

        for i in range( max_epoch ):
            # train_x = Variable( torch.Tensor( train_megabatches[j][0] ) )
            # train_y = Variable( torch.Tensor( train_megabatches[j][1] ) )
            # test_x = Variable( torch.Tensor( test_megabatches[j][0] ) )
            # test_y = Variable( torch.Tensor( test_megabatches[j][1] ) )

            train_x =  train_megabatches[j][0]
            train_y = train_megabatches[j][1]
            test_x =  test_megabatches[j][0]
            test_y = test_megabatches[j][1]

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
                # torch.save(net.state_dict(),model_save_path+'_ReLU_whole_log_best')
                torch.save( net, model_save_path + '/' + 'model_best.pt' )
                print( '\nBest model is saved.***\n', file=sys.stderr )
            else:
                patience += 1
            if (patience == max_patience or i == max_epoch - 1):
                # torch.save(net.state_dict(),model_save_path+'_ReLU_whole_log'+str(i+1))
                torch.save( net, model_save_path + '/model_' + str( i + 1 ) )
                print( '\n***{}th last model is saved.***\n'.format( i + 1 ), file=sys.stderr )
                break

            print( '------{}th iteration (max:{})-----'.format( i + 1, max_epoch ))
            print( 'train_loss : ', train_loss)
            print( 'valid_loss : ', valid_loss )
            print( 'patience : ', patience )

            # print(i+1, train_loss[0], valid_loss[0])

            if (i % save_freq == save_freq - 1):
                # torch.save(net.state_dict(),model_save_path+'_ReLU_whole_log'+str(i+1))
                torch.save( net, model_save_path + '/model' + str( i + 1 ) )
                print( '\n***{}th model is saved.***\n'.format( i + 1 ), file=sys.stderr )

            del train_x, train_y, test_x, test_y, train_loss, valid_loss

if __name__ == '__main__':
    main()
