from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np
import pdb
import sys
import random
import gc


class AMT( nn.Module ):
    def __init__( self, window_size=7, num_features=264 ):
        super( AMT, self ).__init__()

        print ('AMT 3-conv-layer', file=sys.stderr)

        # Model parameters.
        self.window_size = window_size
        self.num_features = num_features

        # Conv layers.
        self.conv1 = nn.Conv2d( 1, 64, (1, num_features // 2), padding=(2, 32) )
        self.conv2 = nn.Conv2d( 64, 64, (3, num_features // 8), padding=(2, 16) )
        self.conv3 = nn.Conv2d(64, 64, (5, num_features // 16), padding=(4, 8))

        # FC layers.
        self.fc1 = nn.Linear( 28288, 1000 )
        self.fc2 = nn.Linear( 1000, 200 )

        # Output layer.
        # self.fc3 = nn.Linear(200,88)
        self.fc3 = nn.Linear( 200, 128 )

    def forward( self, x ):
        x = x.view( -1, 1, self.window_size, self.num_features )
        x = F.relu( F.max_pool2d( self.conv1( x ), (1, 2) ) )
        x = F.relu( F.max_pool2d( self.conv2( x ), (1, 2), padding=(0, 1) ) )
        x = F.relu( F.max_pool2d( self.conv3( x ), (1, 2), padding=(0, 1) ) )

        x = x.view( -1, 28288 )
        x = F.sigmoid( self.fc1( x ) )
        x = F.sigmoid( self.fc2( x ) )
        x = self.fc3( x )
        return x

    # return feature map of each conv layer.
    def features( self, x ):
        self.relu1 = F.relu( F.max_pool2d( self.conv1( x ), (1, 3) ) )
        self.relu2 = F.relu( F.max_pool2d( self.conv2( self.relu1 ), (1, 3), padding=(0, 1) ) )
        return self.relu1, self.relu2

    def grams( self, x ):
        num_batches = x.size()[0]
        assert num_batches == 1  # Doesn't support batch grams yet.
        feature_list = self.features( x )
        g_list = []
        for feature in feature_list:
            a, b, c, d = feature.size()
            f = feature.view( b, c * d )
            g = torch.mm( f, f.t() )
            g.div( b * c * d )
            g_list.append( g )

        return g_list


def run_train( net, inputs, labels, criterion, optimizer,
               piece_lens, batch_size=256, window_size=7 ):
    overall_loss = 0.0
    overall_num_samples = 0
    num_samples = sum( piece_lens )
    # num_batches = num_samples // batch_size
    num_batches = 100

    perm = np.random.permutation( num_samples - window_size - 2 )
    perm = perm + np.ones( perm.shape ) * window_size // 2

    for i in range( num_batches ):
        input_batch, label_batch = utils.next_batch(
            inputs, labels, perm[i * batch_size:(i + 1) * batch_size],
            piece_lens, window_size )

        optimizer.zero_grad()
        output_batch = net( input_batch )
        loss = criterion( output_batch, label_batch )
        loss.backward()
        optimizer.step()
        overall_loss += loss * input_batch.size()[0]
        overall_num_samples += input_batch.size()[0]
        cumul_loss = overall_loss / float( overall_num_samples )
        print( 'progress : {:4d}/{:4d} loss : {:6.3f}'.format(
            i, num_batches, cumul_loss.cpu().data.numpy()[0] ), end='\r', file=sys.stderr )
        sys.stdout.flush()
        sys.stderr.flush()
        del input_batch, label_batch, output_batch, loss
        gc.collect()

    print( '', file=sys.stderr )
    mean_loss = overall_loss / float( num_samples )

    return mean_loss


def run_loss( net, inputs, labels, criterion, piece_lens, batch_size, window_size ):
    overall_loss = 0.0
    overall_num_samples = 0
    num_samples = sum( piece_lens )
    num_batches = num_samples // batch_size
    # num_batches = 5

    perm = np.random.permutation( num_samples - window_size - 2 )
    perm = perm + np.ones( perm.shape ) * window_size // 2

    for i in range( num_batches ):
        input_batch, label_batch = utils.next_batch(
            inputs, labels, perm[i * batch_size:(i + 1) * batch_size],
            piece_lens, window_size )
        output_batch = net( input_batch )
        loss = criterion( output_batch, label_batch )
        overall_loss += loss * input_batch.size()[0]
        overall_num_samples += input_batch.size()[0]
        cumul_loss = overall_loss / float( overall_num_samples )
        print( 'valid progress : {:4d}/{:4d} loss : {:6.3f}'.format(
            i, num_batches, cumul_loss.cpu().data.numpy()[0] ), end='\r', file=sys.stderr )
        sys.stdout.flush()

        del input_batch, label_batch

    print( '', file=sys.stderr )
    mean_loss = overall_loss / float( num_samples )
    return mean_loss
