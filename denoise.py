from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from matplotlib import cm
# import codecs
import os
import time
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import plot_model
from auto import autop


def plotmesh(Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylabel('Turns')
    ax.set_zlabel('PhaseOsc')
    ax.set_xlabel('bunches')
    X,Y = np.meshgrid(range(0,Z.shape[1]), range(0,Z.shape[0]))
    surf = ax.plot_surface(X,Y,Z)
    plt.show()

start_time = time.time()

"""loader"""
# dict1 = loadmat(r"C:\Users\HP\Desktop\ANL\ED_md115_181121\gd.mat")
# bunches = dict1["bunches"]
# Z = np.delete(bunches, 0,0)

# fnames = ["28_1_82", "28_2_82", "28_3_82", "28_4_82"]
# dir = "C:/Users/HP/Desktop/ANL/324_bunchPattern_Data/ED_28to45_S36_82F_4sets_181210/rawdata"
#
# database = list()
# for i in fnames:
#     database.append( autop(pd.read_csv(dir + "/" + i + ".csv").values, 324, header=False, abs=False) )
#
# denoise_arr = np.zeros(np.shape(database[0].Z))
# for j in database:
#     denoise_arr = denoise_arr + j.Z
# denoise_arr = denoise_arr/len(database)
# Zc = np.tile(denoise_arr, (4,1))
# Zc = np.abs(np.reshape(Zc, (Zc.shape[0],Zc.shape[1],1))).astype("float64")
# print(Zc.shape)
#
#
# ts = np.transpose(np.tile(np.array(range(0,database[0].Z.shape[0])), (database[0].Z.shape[1],1)))
# Zlist = []
# ts = np.reshape(ts, (ts.shape[0],ts.shape[1],1))
# for j in database:
#     j.Z = np.reshape(j.Z, (j.Z.shape[0],j.Z.shape[1],1))
#     j.Z = np.append(j.Z, ts, axis=2)
#     Zlist.append(j.Z)
# Z = np.abs(np.concatenate(Zlist, axis=0)).astype("float64")
# # print(Z.shape)
# # Z = np.reshape(Z, (Z.shape[0],Z.shape[1],1))
# # print(Z[:,:,0])
# timesteps = Z.shape[1]
#
# scaler = StandardScaler()
# scaler2 = StandardScaler()
# scaler3 = StandardScaler()
#
# Z[:,:,0] = scaler.fit_transform(Z[:,:,0])
# Z[:,:,1] = scaler2.fit_transform(Z[:,:,1])
# Zc[:,:,0] = scaler3.fit_transform(Zc[:,:,0])

# print(Z[:,:,0])
# print(Z[:,:,1])
# print(Zc[:,:,0])

# Zc = np.copy(Z)
# for i in range(0, Z[:,:,0].shape[0]):
#     randex = np.random.randint(0,Z[:,:,0].shape[1], int(0.6 * Z[:,:,0].shape[1]))
#     Zc[i, randex] = np.zeros((randex.size,1))

# Zc = np.reshape(Zc, (9696,1296))
# Zc = scaler.inverse_transform(Zc)
# plotmesh(Zc)

"""ml part"""
# Z_train, Z_test, Zc_train, Zc_test = train_test_split(Z,Zc,test_size = 0.2)
#
timesteps = 5000
model = Sequential()
model.add(LSTM(50, activation='selu', input_shape=(timesteps,2)))
model.add(RepeatVector(timesteps))
model.add(LSTM(50, activation='selu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))

adam = tf.keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-9, amsgrad=False,
    name='Adam')
model.compile(optimizer=adam, loss="mse", metrics=["mse", tf.keras.metrics.MeanAbsolutePercentageError()])
print(model.summary())

# model.fit(Z_train, Zc_train, validation_data=(Z_test,Zc_test), epochs=1, verbose=1, batch_size=128)
#
# #
# #
#
# testZ = np.abs(np.copy(Zlist[0])).astype("float64")
# testZ[:,:,0] = scaler.transform(Zlist[0][:,:,0])
# testZ[:,:,1] = scaler2.transform(Zlist[0][:,:,1])
# Zhat = model.predict(testZ, verbose=1)
#
#
# # Zhat = model.predict(Z, verbose=1)
# # print(Zhat.shape)
# #
# Zhat = np.reshape(Zhat, (Zhat.shape[0],Zhat.shape[1]))
# Zhat = scaler3.inverse_transform(Zhat)
# plotmesh(Zhat)
#
# os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
# plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')

print("--- %s seconds ---" % (time.time() - start_time))
