'''
Copyright 2017, Yingzhen Li and Yarin Gal, All rights reserved.
Please consider citing the ICML 2017 paper if using any of this code for your research:

Yingzhen Li and Yarin Gal.
Dropout inference in Bayesian neural networks with alpha-divergences.
International Conference on Machine Learning (ICML), 2017.

'''

from keras import backend as K
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.models import Model
from keras.regularizers import l2
from keras.utils import np_utils
import numpy as np
import os, pickle, sys, time
import tensorflow as tf
#import edward.KLpq as ed
#import lasagne
# #You can find the C code in this temporary file: C:\Users\ADMINI~1\AppData\Local\Temp\theano_compilation_error_ayee232i     actual_version, force_compile, _need_reload)) ImportError: Version check of the existing lazylinker compiled file. Looking for version 0.211, but found None. Extra debug information: force_compile=False, _need_reload=True

###################################################################
# aux functions

def Dropout_mc(p):
    layer = Lambda(lambda x: K.dropout(x, p), output_shape=lambda shape: shape)
    return layer

def Identity(p):
    layer = Lambda(lambda x: x, output_shape=lambda shape: shape)
    return layer

def pW(p):
    layer = Lambda(lambda x: x*(1.0-p), output_shape=lambda shape: shape)
    return layer

def apply_layers(inp, layers):
    output = inp
    for layer in layers:
        output = layer(output)
    return output

# utility to gather variational dropout parameters
def gather_logalphas(graph):
    node_defs = [n for n in graph.as_graph_def().node if 'log_alpha' in n.name]
    tensors = [graph.get_tensor_by_name(n.name+":0") for n in node_defs]
    return tensors


def GenerateMCSamples(inp, layers, K_mc=20):
    if K_mc == 1:
        return apply_layers(inp, layers)
    output_list = []
    for _ in range(K_mc):
        output_list += [apply_layers(inp, layers)]  # THIS IS BAD!!! we create new dense layers at every call!!!!
    def pack_out(output_list):
        #output = K.pack(output_list) # K_mc x nb_batch x nb_classes
        output = K.stack(output_list) # K_mc x nb_batch x nb_classes
        return K.permute_dimensions(output, (1, 0, 2)) # nb_batch x K_mc x nb_classes
    def pack_shape(s):
        s = s[0]
        assert len(s) == 2
        return (s[0], K_mc, s[1])
    out = Lambda(pack_out, output_shape=pack_shape)(output_list)
    return out

# evaluation for classification tasks
def test_MC_dropout(model, X, Y):
    pred = model.predict(X)  # N x K x D
    pred = np.mean(pred, 1)
    acc = np.mean(np.argmax(pred, axis=-1) == np.argmax(Y, axis=-1))
    ll = np.sum(np.log(np.sum(pred * Y, -1)))
    return acc, ll

def logsumexp(x, axis=None):
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def bbalpha_softmax_cross_entropy_with_mc_logits(alpha):
    alpha = K.cast_to_floatx(alpha)
    def loss(y_true, mc_logits):
        # log(p_ij), p_ij = softmax(logit_ij)
        #assert mc_logits.ndim == 3
        mc_log_softmax = mc_logits - K.max(mc_logits, axis=2, keepdims=True)
        mc_log_softmax = mc_log_softmax - K.log(K.sum(K.exp(mc_log_softmax), axis=2, keepdims=True))
        mc_ll = K.sum(y_true * mc_log_softmax, -1)  # N x K
        K_mc = mc_ll.get_shape().as_list()[1]	# only for tensorflow
        return - 1. / alpha * (logsumexp(alpha * mc_ll, 1) + K.log(1.0 / K_mc))
    return loss

###################################################################
#  GEP LOSS
def gep_softmax_cross_entropy_with_mc_logits():
    def loss(logits, labels):
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # prior DKL part of the ELBO
        log_alphas = gather_logalphas(tf.get_default_graph())
        # print("found %i logalphas"%len(log_alphas))
        divergences = [dkl_qp(la) for la in log_alphas]
        # combine to form the ELBO
        # N = float(50000.) # only useable with cifar-10
        N = float(60000.)
        dkl = tf.reduce_sum(tf.stack(divergences))
        elbo_reg = (1. / N) * dkl
        elbo_reg = tf.identity(elbo_reg, name='elbo_reg')
        tf.add_to_collection('losses', elbo_reg)

        if not len(log_alphas) > 0:
            # add l2 loss
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            tf.add_to_collection('losses', tf.identity(5e-4 * l2_loss, name='l2'))

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss


###################################################################
# the model

def get_logit_mlp_layers(nb_layers, nb_units, p, wd, nb_classes, layers = [], \
                         dropout = 'none'):
    if dropout == 'MC':
        D = Dropout_mc
    if dropout == 'pW':
        D = pW
    if dropout == 'none':
        D = Identity

    for _ in range(nb_layers):
        layers.append(D(p))
        layers.append(Dense(nb_units, activation='relu', W_regularizer=l2(wd)))
    layers.append(D(p))
    layers.append(Dense(nb_classes, W_regularizer=l2(wd)))
    return layers

def get_logit_cnn_layers(nb_units, p, wd, nb_classes, layers = [], dropout = False):
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    if dropout == 'MC':
        D = Dropout_mc
    if dropout == 'pW':
        D = pW
    if dropout == 'none':
        D = Identity

    layers.append(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid', W_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                W_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(MaxPooling2D(pool_size=pool_size))

    layers.append(Flatten())
    layers.append(D(p))
    layers.append(Dense(nb_units, W_regularizer=l2(wd)))
    layers.append(Activation('relu'))
    layers.append(D(p))
    layers.append(Dense(nb_classes, W_regularizer=l2(wd)))
    return layers

