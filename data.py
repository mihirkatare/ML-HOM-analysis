import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from os.path import isfile
import time
import warnings

class data:
    def __init__(self):
        self.paths = []
        self.files = dict()
        self.datasets = dict()
    def addpath(self, path):
        if(isinstance(path, str)):
            self.paths.append(path)
            self.addfiles(dir = path)
        if(isinstance(path, list)):
            self.paths.extend(path)
            for i in range(len(path)):
                self.addfiles(dir = path[i])

    def addfiles(self, dir = None, file = None, details = None):
        if(dir == None):
            pass
        else:
            templist = os.listdir(dir)
            for i in templist:
                if(isfile(dir + "/" + i)):
                    if(os.path.splitext(dir + "/" + i)[1] == ".csv"):
                        if(len(i.split("_")) == 3):
                            fsplit = os.path.splitext(i)[0].split("_")
                            if(int(fsplit[0]) not in self.files):
                                self.files[int(fsplit[0])] = []
                            if( fsplit[1][-4:] == "sets" ):
                                for m in range(int(fsplit[1][0:-4])):
                                    self.files[int(fsplit[0])].append([dir + "/" + i, int(fsplit[1][0:-4]), m , fsplit[2]])
                            else:
                                self.files[int(fsplit[0])].append([dir + "/" + i, 1, 0,fsplit[2]])

    def loader(self, modes = None, scans = 1):
        if(modes == None):
            modes = list(self.files.keys())
        for mode in modes:
            if(mode not in self.files):
                warnings.warn(str(mode) + " mode has no available scans")
            else:
                scans1 = scans
                if(len(self.files[mode]) < scans):
                    scans1 = len(self.files[mode])
                    warnings.warn("Insufficient Scans for Mode " + str(mode))
                    print("Available scans for Mode " + str(mode) + ": " + str(len(self.files[mode])))
                for i in range(scans1):
                    if(mode not in self.datasets):
                        self.datasets[mode] = []
                    self.datasets[mode].append(pd.read_csv(self.files[mode][i][0], header=None, dtype = np.int16).values)
                    # self.datasets[mode].append(np.genfromtxt(self.files[mode][i][0], delimiter=",", dtype = np.int16))
    def load_example_array(self):
        earr = pd.read_csv(self.files[list(self.files.keys())[0]][0][0], header=None, dtype = np.int16)
        return earr.head(int(earr.shape[0]/self.files[list(self.files.keys())[0]][0][1])).values

    def details(self):
        nscans = []
        for key in self.files.keys():
            nscans.append(len(self.files[key]))
        detail = pd.DataFrame(np.array([list(self.files.keys()), nscans]).T, columns = ["Modes", "Number of Scans"]).sort_values("Modes")
        return detail

# start_time = time.time()
#
# dir = "C:/Users/HP/Desktop/ANL/test"
# obj = data()
# obj.addpath(dir)
# print(obj.load_example_array())
# obj.loader([0], scans = 9)
# print(obj.datasets)
#
#
# print("--- %s seconds ---" % (time.time() - start_time))
