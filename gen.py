import numpy as np
import tensorflow as tf
from data import data
import pandas as pd
import joblib
from scipy.io import loadmat

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, _n_ts, _n_feat, obj, _skip,batch_size=16, n_scans_use =1, type="reg", _modes=None
    , _start=0, scale_file="post/minmax.pkl", _timebloc = None, _shuffle = False, indices_diff = None, _grpath = None, x_only = False):

        if(indices_diff != None):
            self.dc = self.indices(path = indices_diff)
        if(_grpath != None):
            self.grd = self.gr(path = _grpath)
        self.n_ts = _n_ts
        self.n_feat = _n_feat
        self.bs = batch_size
        self.dataobj = obj
        self.nsu = n_scans_use
        self.load_type = type
        if(_modes == None):
            self.modes = list(self.dataobj.files.keys())
        self.nsamp = len(self.modes)
        self.splits = (self.bs*np.arange(1,np.ceil(self.nsamp / self.bs))).astype(np.int)
        self.start = _start
        self.skip = _skip
        self.scaler = joblib.load(scale_file)
        self.shuffle = _shuffle
        self.timebloc = _timebloc
        if(_timebloc == None):
            self.timebloc = self.n_ts
        self.on_epoch_end()
        self.xonly = x_only
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int( self.nsu * np.ceil(self.nsamp / self.bs) )

    def indices(self, inp = None ,path = "modeSequence.txt"):
        dc = dict()
        arr = pd.read_csv(path, header=None).values[0]
        for i in range(len(arr)):
            dc[i+1] = arr[i]
        if(inp == None):
            return dc
        else:
            return dc[inp]

    def gr(self, mode=None, path = "C:/Users/HP/Desktop/ANL/DS2/gr.mat"):
        mat = loadmat(path)
        modes = mat[list(mat.keys())[-1]][0][0][0][0]
        gr = mat[list(mat.keys())[-1]][0][0][1][0]
        grdict = dict()
        for i in range(len(modes)):
            grdict[int(modes[i])] = gr[i]
        if(mode == None):
            return grdict
        else:
            return(grdict[mode])

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = []
        for i in range(self.nsu):
            arr = np.array(self.modes)
            if(self.shuffle == True):
                np.random.shuffle(arr)
                split = np.split(arr, self.splits)
            else:
                split = np.split(arr, self.splits)
            split[-1] = np.append(split[-1], np.random.choice(np.array(self.modes), int(self.bs-split[-1].size)))
            for k in split:
                self.indexes.append([i, k])

    def __getitem__(self, index):
        'Generate one batch of data'

        seg = self.indexes[index][0]
        modes = self.indexes[index][1]

        if(self.load_type == "mode_id"):
            X = np.zeros( ( self.indexes[index][1].size, self.n_ts, self.n_feat ), dtype=np.int16 )
            Y = np.zeros(( self.indexes[index][1].size , 324))
            for j in range(len(modes)):
                X[j][:,1:] = np.abs(np.fft.fft(pd.read_csv(self.dataobj.files[modes[j]][self.start+seg][0], header=None, dtype = np.int16, nrows= self.n_ts
                , skiprows=self.n_ts*(self.dataobj.files[modes[j]][self.start+seg][2])).values[:, ::self.skip], axis=1))[:,1:]

                Y[j, self.dc[modes[j]] ] = 1.0
            return np.abs(X.reshape((X.shape[0],X.shape[1],X.shape[2],1))[:, 0:self.timebloc, :,:]), Y

        if(self.load_type == "reg"):
            X = np.zeros( ( self.indexes[index][1].size, self.n_ts, self.n_feat ), dtype=np.int16 )
            Y = np.zeros(( self.indexes[index][1].size , 1))
            for j in range(len(modes)):
                X[j] = pd.read_csv(self.dataobj.files[modes[j]][self.start+seg][0], header=None, dtype = np.int16, nrows= self.n_ts, skiprows=self.n_ts*(self.dataobj.files[modes[j]][self.start+seg][2])).values[:, ::self.skip]

                Y[j,0] = modes[j]
            return self.scale(np.abs(X)), Y

        if(self.load_type == "autoenc"):
            X = np.zeros( ( self.indexes[index][1].size, self.n_ts, self.n_feat ), dtype=np.int16 )
            for j in range(len(modes)):
                X[j] = pd.read_csv(self.dataobj.files[modes[j]][self.start+seg][0], header=None, dtype = np.int16, nrows= self.n_ts, skiprows=self.n_ts*(self.dataobj.files[modes[j]][self.start+seg][2])).values[:, ::self.skip]

            return self.scale(np.abs(X[:, 0:self.timebloc, :])), self.scale(np.abs(X[:, 0:self.timebloc, :]))

        if(self.load_type == "denac"):
            X = np.zeros( ( self.indexes[index][1].size, self.n_ts, self.n_feat ), dtype=np.int16 )
            X_clean = X
            for j in range(len(modes)):
                X[j] = pd.read_csv(self.dataobj.files[modes[j]][self.start+seg][0], header=None, dtype = np.int16, nrows= self.n_ts, skiprows=self.n_ts*(self.dataobj.files[modes[j]][self.start+seg][2])).values[:, ::self.skip]
                X_clean[j] = pd.read_csv("C:/Users/HP/Desktop/ANL/DS2avgs/" + str(modes[j]) + ".csv", header=None).values
            return self.scale(np.abs(X[:, 0:self.timebloc, :])), self.scale(np.abs(X_clean[:, 0:self.timebloc, :]))

        if(self.load_type == "fft"):
            X = np.zeros( ( self.indexes[index][1].size, self.n_ts, self.n_feat ), dtype=np.int16 )
            X_f = np.zeros( ( self.indexes[index][1].size, self.n_ts, self.n_feat ))
            for j in range(len(modes)):
                X[j] = pd.read_csv(self.dataobj.files[modes[j]][self.start+seg][0], header=None, dtype = np.int16, nrows= self.n_ts, skiprows=self.n_ts*(self.dataobj.files[modes[j]][self.start+seg][2])).values[:, ::self.skip]
                X_f[j][:,1:] = np.abs(np.fft.fft(X[j], axis=1))[:,1:]
            return self.scale(np.abs(X[:, 0:self.timebloc, :])), X_f[:, 0:self.timebloc, :]

        if(self.load_type == "mode_growth"):
            X = np.zeros( ( self.indexes[index][1].size, self.n_ts, self.n_feat ), dtype=np.int16 )
            if(self.xonly == False):
                Y = np.zeros(( self.indexes[index][1].size , 1))
            for j in range(len(modes)):
                X[j][:,1:] = np.abs(np.fft.fft(pd.read_csv(self.dataobj.files[modes[j]][self.start+seg][0], header=None, dtype = np.int16, nrows= self.n_ts
                , skiprows=self.n_ts*(self.dataobj.files[modes[j]][self.start+seg][2])).values[:, ::self.skip], axis=1))[:,1:]
                if(self.xonly == False):
                    Y[j,0] = self.grd[modes[j]]
            if(self.xonly == False):
                return np.abs((X.reshape((X.shape[0],X.shape[1],X.shape[2],1)))[:, 0:self.timebloc, :,:]), Y
            else:
                return np.abs((X.reshape((X.shape[0],X.shape[1],X.shape[2],1)))[:, 0:self.timebloc, :,:])

        return 0
    def scale(self, arr):
        d1,d2,d3 = arr.shape
        arr = np.reshape(arr, (d1*d2, d3))
        arr = self.scaler.transform(arr)
        arr = np.reshape(arr, (d1, d2, d3))
        return arr
