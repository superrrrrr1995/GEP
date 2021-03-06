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
from BBalpha_dropout import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import pdb
nb_epoch = 100
nb_batch = 10
nb_classes = 2
nb_test_mc = 10
wd = 1e-6
alpha = 0; K_mc = 10; nb_layers = 3
nb_units = 1000; p = 0
model_arch = 'mlp'
folder = 'C:/tmp/' + model_arch + '_nb_layers_' + str(nb_layers) + '_nb_units_' + str(nb_units) + '_p_' + str(p) + '/'
if not os.path.exists('C:/tmp/'):
    os.makedirs('C:/tmp/')
if not os.path.exists(folder):
    os.makedirs(folder)

file_name = folder + 'K_mc_' + str(K_mc) + '_alpha_' + str(alpha) + '.obj'
#print file_name

'''
if model_arch == 'mlp':
    nb_in = 500  #the dimensions of train data
    input_shape = (nb_in,)
else:
    img_rows, img_cols = 28, 28
    #input_shape = (1, img_rows, img_cols)
    input_shape = (img_rows, img_cols, 1)
'''
'''
###################################################################
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, *input_shape) #60000x784
X_test = X_test.reshape(10000, *input_shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)#one-hot
Y_test = np_utils.to_categorical(y_test, nb_classes)
'''

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
dire = 'C:\\Users\\Administrator\\codespace\\project\\Dropout_BBalpha-master\\dataset'
# load pima indians dataset
#X = np.loadtxt(dire + "\\archive_x.txt")
#Y = np.loadtxt(dire + "\\archive_y.txt");input_shape = (500,)
#X = np.loadtxt(dire + "\\iris.data.txt")[:, 0:3]
#Y = np.loadtxt(dire + "\\iris.data.txt")[:, 4];input_shape = (3,)
#X = np.loadtxt(dire + "\\wdbc.txt")[:, 2:31]
#Y = np.loadtxt(dire + "\\wdbc.txt")[:, 1];input_shape = (29,)
X = np.loadtxt(dire + "\\wpbc.txt")[:, 2:33]
Y = np.loadtxt(dire + "\\wpbc.txt")[:, 1];input_shape = (31,)
#X = np.loadtxt(dire + "\\Ionosphere.txt")[:, 0:33]
#Y = np.loadtxt(dire + "\\Ionosphere.txt")[:, 34];input_shape = (33,)
# # split into input (X) and output (Y) variables

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
Y_train = np_utils.to_categorical(y_train, nb_classes)#one-hot
Y_test = np_utils.to_categorical(y_test, nb_classes)

###################################################################
# compile model

inp = Input(shape=input_shape)
if model_arch == 'mlp':
    layers = get_logit_mlp_layers(nb_layers, nb_units, p, wd, nb_classes, dropout = 'MC')
else:
    layers = get_logit_cnn_layers(nb_units, p, wd, nb_classes, dropout = 'MC')
mc_logits = GenerateMCSamples(inp, layers, K_mc)

if alpha != 0:
    model = Model(input=inp, output=mc_logits)
    model.compile(optimizer='adam', loss=bbalpha_softmax_cross_entropy_with_mc_logits(alpha), metrics=['accuracy'])
    #model.compile(optimizer='sgd', loss=loss(mc_logits, labels), metrics=['accuracy'])
else:
    mc_softmax = Activation('softmax')(mc_logits)  # softmax is over last dim
    model = Model(input=inp, output=mc_softmax)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

mc_logits = GenerateMCSamples(inp, layers, nb_test_mc)
mc_softmax = Activation('softmax')(mc_logits)  # softmax is over last dim
test_model = Model(input=inp, output=mc_softmax)


###################################################################
# train the model

Y_train_dup = np.squeeze(np.concatenate(K_mc * [Y_train[:, None]], axis=1)) # N x K x D
Y_test_dup = np.squeeze(np.concatenate(K_mc * [Y_test[:, None]], axis=1)) # N x K x D

evals = {'acc_approx': [], 'acc': [], 'll': [], 'time': [], 'train_acc': [], 'train_loss': [],
         'nb_layers': nb_layers, 'nb_units': nb_units}


tic = time.clock()
train_loss = model.fit(X_train, Y_train_dup, verbose=1,
                           batch_size=nb_batch, nb_epoch=nb_epoch)
#predictions = model.predict(X_train)



scores = model.evaluate(X_train, Y_train_dup)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
toc = time.clock()
evals['acc_approx'] += [model.evaluate(X_test, Y_test_dup, verbose=0)[1]]
acc, ll = test_MC_dropout(test_model, X_test, Y_test)
evals['acc'] += [acc]
evals['ll'] += [ll]
evals['time'] += [toc - tic]
evals['train_acc'] += [train_loss.history['acc'][-1]]
evals['train_loss'] += [train_loss.history['loss'][-1]]

print('acc',evals['acc'])
print('ll',evals['ll'])
print('acc_approx',evals['acc_approx'])
print('train_acc',evals['train_acc'])
print('train_loss',evals['train_loss'])
print('time',evals['time'])
'''
        # summarize history for accuracy
        plt.plot(evals['train_acc'])
        plt.plot(evals['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("model_accuracy.png")
        #plt.show()

        # summarize history for loss
        plt.plot(evals['train_loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("model_loss.png")
        #plt.show()

        plt.plot(evals['ll'])
        plt.title('test log likelihood')
        plt.ylabel('test ll')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("test_ll.png")
        #plt.show()

        plt.plot(evals['acc_approx'])
        plt.title('accuracy approximation')
        plt.ylabel('acc_approx')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("acc_approx.png")
        #plt.show()

        plt.plot(evals['time'])
        plt.title('time')
        plt.ylabel('second')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("time.png")
        #plt.show()
    with open(file_name, 'wb') as f:
        pickle.dump(evals, f)
    
# save model for future test
file_name = folder + 'K_mc_' + str(K_mc) + '_alpha_' + str(alpha)
model.save_weights(file_name+'_weights.h5')
#print 'model weights saved to file ' + file_name + '_weights.h5'
'''
