import numpy as np
import tensorflow as tf
from data import data
import pandas as pd
import joblib

class DG2(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, _n_ts, _n_feat, obj, _skip,batch_size=16, n_scans_use =1, type="denac2", _modes=None, _start=0, scale_file="post/minmax2.pkl"
    , _timebloc = None, _shuffle = False, _reduce = 6, _avgpath = "C:/Users/HP/Desktop/ANL/DS2avgs/", x_only = False):
        self.n_ts = _n_ts
        self.n_feat = _n_feat
        self.bs = batch_size
        self.dataobj = obj
        self.nsu = n_scans_use
        self.load_type = type
        self.reduce = _reduce
        if(_modes == None):
            self.modes = list(self.dataobj.files.keys())
        self.nsamp = len(self.modes)
        self.splits = (self.bs*np.arange(1,np.ceil(self.n_feat / self.bs))).astype(np.int)
        self.start = _start
        self.skip = _skip
        self.scaler = joblib.load(scale_file)
        self.shuffle = _shuffle
        self.timebloc = _timebloc
        if(_timebloc == None):
            self.timebloc = self.n_ts
        self.on_epoch_end()
        self.avgpath = _avgpath
        self.xonly = x_only
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int( self.nsu * self.nsamp * np.ceil(self.n_feat / self.bs) )

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = []
        for i in range(self.nsu):
            marr = np.array(self.modes)
            np.random.shuffle(marr)
            for j in marr:
                arr = np.arange(0, self.n_feat)
                if(self.shuffle == True):
                    np.random.shuffle(arr)
                    split = np.split(arr, self.splits)
                else:
                    split = np.split(arr, self.splits)
                split[-1] = np.append(split[-1], np.random.choice(arr, int(self.bs-split[-1].size)))
                for k in split:
                    self.indexes.append([i,j, k])

    def __getitem__(self, index):
        'Generate one batch of data'

        seg = self.indexes[index][0]
        mode = self.indexes[index][1]
        bunches = self.indexes[index][2]
        if(self.load_type == "denac2"):
            X = np.zeros( ( self.bs, self.n_ts, 1 ), dtype=np.int16 )
            X[:,:,0] = (pd.read_csv(self.dataobj.files[mode][self.start+seg][0], header=None,usecols=self.skip*bunches ,dtype = np.int16, nrows= self.n_ts
            , skiprows=self.n_ts*(self.dataobj.files[mode][self.start+seg][2])).values.T)
            if(self.xonly == False):
                X_clean = np.zeros( ( self.bs, self.n_ts, 1 ), dtype=np.int16 )
                X_clean[:,:,0] = (pd.read_csv(self.avgpath + str(mode) + ".csv", header=None,usecols=bunches).values.T)
                return self.scale(np.abs(X[:, 0:self.timebloc, :])), self.scale(np.abs(X_clean[:, 0:self.timebloc-self.reduce, :]))
            else:
                return self.scale(np.abs(X[:, 0:self.timebloc, :]))
            # print(X.shape, X_clean.shape)

        return 0
    def scale(self, arr):
        d1,d2,d3 = arr.shape
        arr = np.reshape(arr, (d1*d2, d3))
        arr = self.scaler.transform(arr)
        arr = np.reshape(arr, (d1, d2, d3))
        return arr
