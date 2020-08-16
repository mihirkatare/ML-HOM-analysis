import time
start_time = time.time()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import math
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from data import data
import pickle
from gen import DataGenerator
from gen2 import DG2
import scipy
from testing import test

def scheduler(epoch, lr):
  return lr * 0.5**math.floor(epoch/3)

def plotmesh(Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylabel('Turns')
    ax.set_zlabel('PhaseOsc')
    ax.set_xlabel('bunches')
    X,Y = np.meshgrid(range(0,Z.shape[1]), range(0,Z.shape[0]))
    surf = ax.plot_surface(X,Y,Z)
    plt.show()

def skipcol(arr, skip=4):
    return arr[:, ::skip]

def scaledsigmoid(z, scale=324):
    return backend.sigmoid(z) * scale

class model:
    def __init__(self):
        self.scaler = None
        self.encoder = None
    def metadata(self, ex_array, n_scans_use = 1, train_ratio = 1, skip = 4, _timebloc = None):
        self.n_ts = ex_array.shape[0]
        self.n_feat = ex_array.shape[1]
        self.ntrainscans = math.ceil(train_ratio*n_scans_use)
        self.ntestscans = n_scans_use - self.ntrainscans
        self.skip = skip
        self.timebloc = _timebloc
        if(_timebloc == None):
            self.timebloc = self.n_ts

    def conv(self, _batch_size=8, nmodes=324):
        #INPUT LAYER
        inp_e = Input(shape=(self.timebloc, self.n_feat,1), batch_size = _batch_size)

        #ENCODER LAYER
        layc1 = Conv2D(filters=16, kernel_size=(5,5), activation='relu')
        layc1_o = layc1(inp_e)

        laymp1 = MaxPool2D(pool_size=(2,2))
        laymp1_o = laymp1(layc1_o)

        layc2 = Conv2D(filters=32, kernel_size=(5,5), activation='relu')
        layc2_o = layc2(laymp1_o)

        laymp2 = MaxPool2D(pool_size=(2,2))
        laymp2_o = laymp2(layc2_o)

        layc3 = Conv2D(filters=32, kernel_size=(5,5), activation='relu')
        layc3_o = layc3(laymp2_o)

        laymp3 = MaxPool2D(pool_size=(7,7))
        laymp3_o = laymp3(layc3_o)

        layf = Flatten()
        layf_o = layf(laymp3_o)

        #Deep Static
        layd2 = Dense(128, activation = "relu")
        layd2_o = layd2(layf_o)

        #DECODER LAYER
        layOut = Dense(nmodes, activation='softmax')
        layOut_o = layOut(layd2_o)

        self.model = Model(inp_e, layOut_o)

    def conv_growth(self, _batch_size=8):
        #INPUT LAYER
        inp_e = Input(shape=(self.timebloc, self.n_feat,1), batch_size = _batch_size)

        #ENCODER LAYER
        layc1 = Conv2D(filters=16, kernel_size=(5,5), activation='relu')
        layc1_o = layc1(inp_e)

        laymp1 = MaxPool2D(pool_size=(2,2))
        laymp1_o = laymp1(layc1_o)

        layc2 = Conv2D(filters=32, kernel_size=(5,5), activation='relu')
        layc2_o = layc2(laymp1_o)

        laymp2 = MaxPool2D(pool_size=(2,2))
        laymp2_o = laymp2(layc2_o)

        layc3 = Conv2D(filters=32, kernel_size=(5,5), activation='relu')
        layc3_o = layc3(laymp2_o)

        laymp3 = MaxPool2D(pool_size=(7,7))
        laymp3_o = laymp3(layc3_o)

        layf = Flatten()
        layf_o = layf(laymp3_o)

        #Deep Static
        layd2 = Dense(128, activation = "relu")
        layd2_o = layd2(layf_o)

        #DECODER LAYER
        layOut = Dense(1)
        layOut_o = layOut(layd2_o)

        self.model = Model(inp_e, layOut_o)

    def conv_da(self, _batch_size=8, kernelsize = 3):
        #INPUT LAYER
        inp_e = Input(shape=(self.timebloc, 1), batch_size = _batch_size)

        #ENCODER LAYER
        layc1 = Conv1D(filters=32, kernel_size=kernelsize, activation='relu')
        layc1_o = layc1(inp_e)

        layc2 = Conv1D(filters=64, kernel_size=kernelsize, activation='relu')
        layc2_o = layc2(layc1_o)

        layc3 = Conv1D(filters=64, kernel_size=kernelsize, activation='relu')
        layc3_o = layc3(layc2_o)

        layc4 = Conv1D(filters=1, kernel_size=kernelsize, activation='relu')
        layc4_o = layc4(layc3_o)

        self.model = Model(inp_e, layc4_o)

    def compile(self):
        adam = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')
        # self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=[tf.keras.metrics.categorical_accuracy])

        self.model.compile(optimizer= adam, loss="mse", metrics=["mse", "mae"])
        # self.model.compile(optimizer= adam, loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        print(self.model.summary())
        if(self.encoder != None):
            self.encoder.compile(optimizer= adam, loss='mse', metrics=["mse"])
            print(self.encoder.summary())

    def scale(self, arr, scale_file=None):
        d1,d2,d3 = arr.shape
        arr = np.reshape(arr, (d1*d2, d3))
        if(scale_file == None):
            if(self.scaler == None):
                self.scaler = MinMaxScaler()
                self.scaler.fit(arr)
                joblib.dump(self.scaler, "post/minmax.pkl")
        else:
            self.scaler = joblib.load(scale_file)
        arr = self.scaler.transform(arr)
        arr = np.reshape(arr, (d1, d2, d3))
        return arr

    def indices(self, inp = None ,path = "modeSequence.txt"):
        dc = dict()
        arr = pd.read_csv(path, header=None).values[0]
        for i in range(len(arr)):
            dc[i+1] = arr[i]
        if(inp == None):
            return dc
        else:
            return dc[inp]

    def load_savedmodel(self, savedpath):
        self.model = tf.keras.models.load_model(savedpath)
        print(self.model.summary())

# bsize = 128
# dir = "C:/Users/HP/Desktop/ANL/DS2"
# indexfile = "modeSequence.txt"
# ksize = 5
#
# obj = data()
# obj.addpath(dir)
# print(obj.details())
# nskip = 4
# timebloc = None
# exarr = skipcol(obj.load_example_array(), skip = nskip)
#
# foo = model()
# foo.metadata(ex_array=exarr, n_scans_use = 6, train_ratio = 1, skip = nskip
# , _timebloc = timebloc)
#
#  #
# # doo = DataGenerator(_n_ts=foo.n_ts, _n_feat=foo.n_feat, obj=obj, _skip = foo.skip, batch_size=bsize, type="mode_growth", _timebloc = timebloc
# # , n_scans_use=7, _shuffle = False, indices_diff = indexfile, _grpath = dir + "/gr.mat")
#
# doo = DG2(_n_ts=foo.n_ts, _n_feat=foo.n_feat, obj=obj, _skip = foo.skip, batch_size=bsize, type="denac2"
# , _timebloc = timebloc, n_scans_use=6, _shuffle = False, _reduce = 16)
#
#
# #Test
# # foo.test_with_params(doo)
# test(doo, type = "denac" ,model_path = "post/model_den2.h5", item = 5772)
# #
# # # foo.autoenc(_batch_size=bsize, units = 128)
# #
# # # foo.denac2(_batch_size=bsize, units = 128)
# #
# # foo.conv(_batch_size=bsize, nmodes=324)
# # foo.conv_growth(_batch_size=bsize)
# # foo.conv_da(_batch_size=bsize, kernelsize = ksize)
# # foo.compile()
# #
# print(doo.indexes)
# print(doo.__len__())
# # a = doo.__getitem__(0)
# # print(a[0].shape, a[1].shape)
# #
# # chkpt = ModelCheckpoint("post/model_den2.h5",monitor='loss',mode='min',save_best_only=True,verbose=1)
# # # chkpt = ModelCheckpoint("post/model_growth.h5",monitor='loss',mode='min',save_best_only=True,verbose=1)
# # lrs = LearningRateScheduler(scheduler)
# # # # tb = TensorBoard(log_dir = "post", histogram_freq = 1)
# # hist = foo.model.fit(doo, epochs = 2, verbose=1, callbacks=[chkpt, lrs], max_queue_size = 20)
# # with open('post/history.pickle', 'wb') as handle:
# #     pickle.dump(hist.history, handle)
#
# # foo.encoder.save("post/encoder2.h5")
# #
# # # os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
# # # plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# #
# #
# print("--- %s seconds ---" % (time.time() - start_time))
