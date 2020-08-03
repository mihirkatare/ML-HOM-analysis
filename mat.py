from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.fft import dst,dct,fft, ifft
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from scipy.signal import savgol_filter as sgf
from scipy.optimize import curve_fit
import math
def model_func(x, a, k, b):
    return a * np.exp(-k*x) + b
# dat = np.loadtxt(r"C:\Users\HP\Desktop\ANL\324_bunchPattern_Data\ED_28to45_S36_82F_4sets_181210\raw.dat", encoding="latin-1")
dir = "C:/Users/HP/Desktop/ANL/324_bunchPattern_Data/ED_28to45_S36_82F_4sets_181210/rawdata"
# dict1 = loadmat(r"C:\Users\HP\Desktop\ANL\ED_md115_181121\gd.mat")

# print(dict1.keys())
# bunches = dict1["bunches"]
# print(bunches)

bunches = pd.read_csv(dir + "/28_1_82.csv").values



# dict1 = loadmat(r"C:\Users\HP\Desktop\ANL\ED_md115_181121\bunchdt.mat")
# # print(dict1.keys())
# bunches = np.abs(dict1["bunches"])
# print(bunches)

# turns = np.linspace(0,2*np.pi,num = bunches.shape[0]-1 )
# bunchdt = np.linspace(10,12,num = bunches.shape[1] )

# R, T = np.meshgrid(bunchdt, turns)
# X = R * np.cos(T)
# Y = R * np.sin(T)

bunch_value = 324

# Z = np.abs(np.delete(bunches, 0,0))

"""del"""
# Z = np.delete(bunches, 0,0)
Z = bunches
print(Z)

# scaler = MinMaxScaler()
# Z = scaler.fit_transform(Z)

# """col adjust"""
# cols = np.array(range(0, Z.shape[1], int(Z.shape[1]/bunch_value)))
# cols_inv = np.setdiff1d(np.array(range(0, Z.shape[1])), cols)
# # Z = Z[:, cols]
# Z[:,cols_inv] = np.zeros((Z.shape[0],cols_inv.size))


# for i in turns:
#     Z[i] = sgf(Z[i], 7, 4)
# Z = gaussian_filter(Z, sigma=1.0)

# modes_com = np.fft.fft(Z,axis=1)

"""processing in matlab script"""

"""FindFs"""
# M = Z.shape[0]
# len	= 2**(math.ceil(math.log(M)/math.log(2)));
# Z	= Z - np.ones((M,1))*np.mean(Z, axis = 0);
# Z	= fft(Z,axis = 0,n = len);
# # print(Z, Z.shape)
# Z	= ifft(Z,axis = 0,n = 9210);
# # print(Z, Z.shape)

modes_com = fft(Z,axis=1)
#
# # modes_com = idst(Z,axis=1 ,norm = "ortho")
# # modes = np.abs(idst(Z,axis=1 ,norm = "ortho"))
#
"""ZOM removal"""
# bv_total = np.shape(modes_com)[1]
# zom_number = bv_total/bunch_value
# zom_modes = bunch_value*np.arange(0, zom_number).astype("int")
# modes_com = np.delete(modes_com, zom_modes, axis=1)


modes = np.absolute(modes_com)
# for i in turns:
#     modes[i] = modes[i]/np.linalg.norm(modes[i], ord=2)

turns = range(0,Z.shape[0])
bunchn = range(0,Z.shape[1])
X, Y = np.meshgrid(bunchn, turns)
# X = np.delete(X, zom_modes, axis=1)
# Y = np.delete(Y, zom_modes, axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('PhaseOsc')

surf = ax.plot_surface(X,Y,abs(Z))

plt.show()

# y = modes[:,114]
# x = range(y.size)
# plt.plot(x,y)
# opt, pcov = curve_fit(model_func, x, y)
# a, k, b = opt
# y2 = model_func(x, a, k, b)
#
# plt.plot(x,y2)
# plt.show()

# maxes = np.amax(modes, axis=0)
# for i in range(0,maxes.size):
#     if(maxes[i]>300):
#         print(i)
