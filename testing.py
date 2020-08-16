import time
start_time = time.time()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import os
import joblib
import tensorflow as tf
from data import data
import pickle
from gen import DataGenerator
from gen2 import DG2
from tensorflow import keras
from scipy.stats import pearsonr
# with open('post/history.pickle', 'rb') as handle:
#     a = pickle.load(handle)

# plt.plot( range(1, len(a["mae"]) +1) , a["mae"])
# plt.xlabel("Epoch")
# plt.ylabel("Mean Absolute Error Loss")
# plt.show()

def test(doo, type = "reg", model_path = None, item = None):
    if(type == "reg"):
        Xi = np.array([])
        Xr = np.array([])
        Xo = np.array([])
        print(doo.indexes)
        max = doo.__len__()
        for i in range(max):
            xi,xr = doo.__getitem__(i)
            mod = keras.models.load_model(model_path)
            xo = mod.predict(xi)
            Xi = np.append(Xi, doo.indexes[i][1])
            Xr = np.append(Xr, xr[:,0])
            Xo = np.append(Xo, xo[:,0])
        # print(Xr, Xo, Xi)
        # np.save('Xr.npy', Xr)
        # np.save('Xo.npy', Xo)
        plt.scatter(Xi,Xr, color = "b", label = "Actual Values", marker = '.')
        plt.scatter(Xi,Xo, color = "red", label = "Predicted Values", marker = '.')
        plt.xlabel("Mode Number")
        plt.ylabel("Mode Growth Rate")
        plt.legend()
        plt.show()

    if(type == "denac"):
        # dati = dict()
        # dato = dict()
        # mod = tf.keras.models.load_model(model_path)
        # print(doo.indexes)
        # print(doo.__len__())
        # for ind in range(doo.__len__()):
        #     Xi, Xc = doo.__getitem__(ind)
        #     Xo = mod.predict(Xi)
        #     sumi=0
        #     sumo=0
        #     for j in range(81):
        #         sumi = sumi + pearsonr(Xi[j,0:Xc[j,:,0].shape[0],0], Xc[j,:,0])[0]
        #         sumo = sumo + pearsonr(Xo[j,:Xc[j,:,0].shape[0],0], Xc[j,:,0])[0]
        #     if(doo.indexes[ind][1] not in list(dati.keys())):
        #         dati[doo.indexes[ind][1]] = sumi/81
        #         dato[doo.indexes[ind][1]] = sumo/81
        #     else:
        #         dati[doo.indexes[ind][1]] += sumi/81
        #         dato[doo.indexes[ind][1]] += sumo/81
        # print(dati, dato)
        # with open('dati.pickle', 'wb') as handle:
        #     pickle.dump(dati, handle)
        # with open('dato.pickle', 'wb') as handle:
        #     pickle.dump(dato, handle)

        # listsi = sorted(dati.items()) # sorted by key, return a list of tuples
        # listso = sorted(dato.items())
        # xi, yi = zip(*listsi) # unpack a list of pairs into two tuples
        # xo, yo = zip(*listso) # unpack a list of pairs into two tuples
        # plt.plot(xi, yi)
        # plt.plot(xo, yo)
        # plt.show()

        Xi, Xc = doo.__getitem__(item)
        mod = tf.keras.models.load_model(model_path)
        Xo = mod.predict(Xi)

        plt.plot(Xi[2,0:100,0], alpha = 0.5, label = "Input Raw Data")
        plt.plot(Xc[2,0:100,0], label = "Target Clean Data")
        plt.plot(Xo[2,0:100,0], label = "Output From Denoising Autoencoder")
        plt.xlabel("Turns")
        plt.ylabel("Measured Phase Oscillation Values")

        # print(Xc, Xo)
        plt.legend()
        plt.show()
        plt.show()
