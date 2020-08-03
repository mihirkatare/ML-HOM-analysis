import time
start_time = time.time()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

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
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from data import data
import pickle
from gen import DataGenerator
from gen2 import DG2
import scipy

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
        self.skip = nskip
        self.timebloc = timebloc
        if(_timebloc == None):
            self.timebloc = self.n_ts
    def autoenc(self, _batch_size=8, units = 64):
        #INPUT LAYER
        inp_e = Input(shape=(self.timebloc, self.n_feat), batch_size = _batch_size)

        #ENCODER LAYER
        # layl1 = LSTM(units, return_state = True)
        layl1 = LSTM(units)
        # layl1_o, layl1_h, layl1_c = layl1(inp_e)
        layl1_o = layl1(inp_e)

        #Repeat LAYER
        layr = RepeatVector(self.timebloc)
        layr_o = layr(layl1_o)

        #Decoder LAYER
        layd1 = LSTM(units, return_sequences=True)
        # layd1_o = layd1(layr_o, initial_state = [layl1_h, layl1_c])
        layd1_o = layd1(layr_o)

        #TD Dense
        layOut = TimeDistributed(Dense(self.n_feat))
        layOut_o = layOut(layd1_o)
        self.model = Model(inp_e, layOut_o)
        self.encoder = Model(inp_e, layl1_o)

    def denac2(self, _batch_size=8, units = 64):
        #INPUT LAYER
        inp_e = Input(shape=(self.timebloc, 1), batch_size = _batch_size)

        #ENCODER LAYER
        # layl1 = LSTM(units, return_state = True)
        layl1 = LSTM(units, return_sequences=True, stateful = True)
        # layl1_o, layl1_h, layl1_c = layl1(inp_e)
        layl1_o = layl1(inp_e)

        layl2 = LSTM(64, stateful = True, return_sequences=True)
        layl2_o = layl2(layl1_o)

        # #Repeat LAYER
        # layr = RepeatVector(self.timebloc)
        # layr_o = layr(layl2_o)

        #Decoder LAYER
        layd1 = LSTM(64, return_sequences=True, stateful = True)
        # layd1_o = layd1(layr_o, initial_state = [layl1_h, layl1_c])
        layd1_o = layd1(layl2_o)

        layd2 = LSTM(units, return_sequences=True, stateful = True)
        layd2_o = layd2(layd1_o)
        #TD Dense
        layOut = TimeDistributed(Dense(1, activation='tanh'))
        layOut_o = layOut(layd2_o)
        self.model = Model(inp_e, layOut_o)

    def encdec(self, nmodes=3):
        #INPUT LAYER
        inp_e = Input(shape=(self.n_ts, self.n_feat))

        #ENCODER LAYER
        lay1 = LSTM(100)
        lay1_o = lay1(inp_e)

        #Dropout
        layd1 = Dropout(0.2)
        layd1_o = layd1(lay1_o)

        #Deep Static
        lay2 = Dense(nmodes, activation = "relu")
        lay2_o = lay2(layd1_o)

        #DECODER LAYER
        lay3 = Dense(nmodes, activation='softmax')
        lay3_o = lay3(lay2_o)
        self.model = Model(inp_e, lay3_o)

    def regr(self, nmodes=3):
        #INPUT LAYER
        inp_e = Input(shape=(self.n_ts, self.n_feat))

        #ENCODER LAYER
        layl1 = LSTM(128, return_sequences = True)
        layl1_o = layl1(inp_e)

        #ENCODER LAYER
        layl2 = LSTM(64, return_sequences = True)
        layl2_o = layl2(layl1_o)

        #ENCODER LAYER
        layl3 = LSTM(64)
        layl3_o = layl3(layl2_o)

        #Deep Static
        layd1 = Dense(256, activation = "relu")
        layd1_o = layd1(layl3_o)

        #Deep Static
        layd2 = Dense(256, activation = "relu")
        layd2_o = layd2(layd1_o)

        #DECODER LAYER
        layOut = Dense(1, activation = scaledsigmoid)
        layOut_o = layOut(layd2_o)
        self.model = Model(inp_e, layOut_o)

    def conv(self, _batch_size=8, nmodes=324):
        #INPUT LAYER
        inp_e = Input(shape=(self.timebloc, self.n_feat), batch_size = _batch_size)

        #ENCODER LAYER
        layc1 = Conv1D(filters=32, kernel_size=3, activation='relu')
        layc1_o = layc1(inp_e)

        laymp1 = MaxPool1D(pool_size=2)
        laymp1_o = laymp1(layc1_o)

        layc2 = Conv1D(filters=64, kernel_size=3, activation='relu')
        layc2_o = layc2(laymp1_o)

        laymp2 = MaxPool1D(pool_size=2)
        laymp2_o = laymp2(layc2_o)

        layf = Flatten()
        layf_o = layf(laymp2_o)

        #Deep Static
        layd1 = Dense(256, activation = "relu")
        layd1_o = layd1(layf_o)

        #Deep Static
        layd2 = Dense(256, activation = "relu")
        layd2_o = layd2(layd1_o)

        #DECODER LAYER
        layOut = Dense(nmodes, activation='softmax')
        layOut_o = layOut(layd2_o)

        self.model = Model(inp_e, layOut_o)

    def process(self, array):
        array = np.abs(skipcol(array))
        array = np.reshape(array, (1, array.shape[0], array.shape[1]))
        return array

    def compile(self):
        adam = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')
        # self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=[tf.keras.metrics.categorical_accuracy])

        # self.model.compile(optimizer= adam, loss="mse", metrics=["mse", tf.keras.metrics.KLDivergence()])
        self.model.compile(optimizer= adam, loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        print(self.model.summary())
        if(self.encoder != None):
            self.encoder.compile(optimizer= adam, loss='mse', metrics=["mse"])
            print(self.encoder.summary())

    def trainer(self, obj, modes=None):
        for i in range(self.ntrainscans):
            # X, Y = self.largedataload(obj, modes, iter=i)
            X, Y = self.largedataload_reg(obj, modes, iter=i)
            X = self.scale(X, scale_file = "post/minmax.pkl")
            print("--- %s seconds ---" % (time.time() - start_time))
            hist = self.model.fit(X, X, epochs = 10, verbose=1, batch_size=8)
            self.model.save("post/model1.h5")
            with open('post/history.pickle', 'wb') as handle:
                pickle.dump(hist.history(), handle)

            break

    def npysaver(self, obj, _iter, modes=None):
        X, Y = self.largedataload_reg(obj, modes, iter=_iter)
        X = self.scale(X, scale_file = "post/minmax.pkl")
        print("--- %s seconds ---" % (time.time() - start_time))
        np.save( "DS2npy/" + str(len(list(obj.files.keys()))) + "_X_" + str(_iter), X)
        np.save( "DS2npy/" + str(len(list(obj.files.keys()))) + "_Y_" + str(_iter), Y)

    def eval(self, mod = None):
        if(self.ntestscans == 0):
            print("Test Ratio too Low")
        else:
            for i in range(self.ntrainscans, self.ntrainscans+self.ntestscans):
                for key in obj.datasets:
                    print("actual mode:", key)
                    if(mod == None):
                        print(self.model.predict(self.process(obj.datasets[key][i])))
                    else:
                        model = keras.models.load_model(mod)
                        print(model.predict(self.process(obj.datasets[key][i])))

    def largedataload(self, obj, modes = None, iter =0):
        if(modes == None):
            modes = list(obj.files.keys())
        X = np.zeros((int(len(modes)),self.n_ts, self.n_feat), dtype=np.int16)
        Y = np.zeros((int(len(modes)), 324))
        for j in range(len(modes)):
            X[j] = pd.read_csv(obj.files[modes[j]][iter][0], header=None, dtype = np.int16, nrows= self.n_ts, skiprows=self.n_ts*(obj.files[modes[j]][iter][2])).values[:, ::self.skip]

            Y[j, modes[j]] = 1.0
        return np.abs(X), Y

    def largedataload_reg(self, obj, modes = None, iter =0):
        if(modes == None):
            modes = list(obj.files.keys())
        X = np.zeros((int(len(modes)),self.n_ts, self.n_feat), dtype=np.int16)
        Y = np.zeros((int(len(modes)), 1))
        for j in range(len(modes)):
            X[j] = pd.read_csv(obj.files[modes[j]][iter][0], header=None, dtype = np.int16, nrows= self.n_ts, skiprows=self.n_ts*(obj.files[modes[j]][iter][2])).values[:, ::self.skip]

            Y[j,0] = modes[j]
        return np.abs(X), Y

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
    def test_with_params(self, doo):
        Xi, Xc = doo.__getitem__(5)
        mod = tf.keras.models.load_model("post/model_den2.h5")
        Xo = mod.predict(Xi)
        # plt.plot(Xo[0,:,0])
        # plt.plot(Xc[0,:,0])
        # plt.plot(Xi[0,:,0])
        print(Xo[0])
        # plt.show()

bsize = 9
dir = "C:/Users/HP/Desktop/ANL/DS2"
obj = data()
obj.addpath(dir)
# print(obj.details())
nskip = 4
timebloc = None
exarr = skipcol(obj.load_example_array(), skip = nskip)
# tscans = 10
# trainperc = 0.8
# #



foo = model()
foo.metadata(ex_array=exarr, n_scans_use = 6, train_ratio = 1, skip = nskip
, _timebloc = timebloc)
#

doo = DataGenerator(_n_ts=foo.n_ts, _n_feat=foo.n_feat, obj=obj, _skip = foo.skip, batch_size=bsize, type="mode_id", _timebloc = timebloc, n_scans_use=6, _shuffle = True, indices_diff = "modeSequence.txt")
# doo = DG2(_n_ts=foo.n_ts, _n_feat=foo.n_feat, obj=obj, _skip = foo.skip, batch_size=bsize, type="denac2", _timebloc = timebloc, n_scans_use=1, _shuffle = True)

#Test
foo.test_with_params(doo)

# foo.autoenc(_batch_size=bsize, units = 128)

# foo.denac2(_batch_size=bsize, units = 128)

# foo.conv(_batch_size=bsize, nmodes=324)
# foo.compile()

# print(doo.indexes)
# a = doo.__getitem__(0)
# plotmesh(a[1][0])


# # foo.npysaver(obj, _iter= 1)
# # a,b = foo.largedataload_reg(obj, modes = None, iter=2)
# # print(a, a.shape)
# # a = np.load("DS2npy/300_X_0.npy")
# # print(a, a.shape)

# # AVERAGER
# modes = list(obj.files.keys())
# for j in range(len(modes)):
#     a = np.zeros(((5696, 324)))
#     for iter in range(6):
#         a += pd.read_csv(obj.files[modes[j]][iter][0], header=None, dtype = np.float32, nrows= foo.n_ts, skiprows=foo.n_ts*(obj.files[modes[j]][iter][2])).values[:, ::foo.skip]
#     np.savetxt("C:/Users/HP/Desktop/ANL/DS2avgs/" + str(modes[j]) + ".csv", a/6, delimiter=',' )
#     print(j)


# chkpt = ModelCheckpoint("post/model_den2.h5",monitor='loss',mode='min',save_best_only=True,verbose=1)
# tb = TensorBoard(log_dir = "post", histogram_freq = 1)
# foo.model.fit(doo, epochs = 5, verbose=1, callbacks=[chkpt], max_queue_size = 20)
# foo.encoder.save("post/encoder2.h5")




# #ANIM
# # arr = pd.read_csv("C:/Users/HP/Desktop/ANL/DS/0_1_85.csv", header=None, dtype = np.float32).values[:, ::foo.skip]
# # a = arr[0:500, 0]
# fig = plt.figure(figsize=(10,6))
# ax1 = fig.add_subplot(1,1,1)
# ax1.axis(xmin = 0, xmax = 100)
# ax1.axis(ymin = 0.25,ymax = 0.6)
#
# # #
#
# plt.plot(arr2[:,50], color="red", alpha = 0.8, label = "Noisy Data")
# plt.plot(arr1[:,50], color="b", label = "Denoised Data")
# plt.legend()
# plt.show()
#

# def animate(i):
#     ax1.plot(arr2[:,50][0:i], color = "red", alpha = 0.8, label = "Noisy Data")
#     ax1.plot(arr1[:,50][0:i], color = "blue", label = "Denoised Data")
# plt.rcParams['animation.ffmpeg_path'] = "C:/Users/HP/Desktop/ANL/ffmpeg/bin"
# ani = animation.FuncAnimation(fig, animate, frames = 100, interval=10)
# # writer = animation.FFMpegWriter()
# # ani.save('animation_video.mp4', writer=writer)
# plt.show()





# foo.trainer(obj, modes = None)

# foo.eval(mod = "C:/Users/HP/Desktop/Code/Python/ANL/model1.h5")

# tarr = skipcol(obj.datasets[0][0])
# tarr = np.reshape(tarr, (1, tarr.shape[0], tarr.shape[1]))
# ty = np.array([[0]])

# foo.model.fit(tarr, ty, verbose=1, epochs=1)

# os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
# plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')


print("--- %s seconds ---" % (time.time() - start_time))
