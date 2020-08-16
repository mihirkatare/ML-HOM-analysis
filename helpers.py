import math
import matplotlib.pyplot as plt
import numpy as np

def skipcol(arr, skip=4):
    return arr[:, ::skip]

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
