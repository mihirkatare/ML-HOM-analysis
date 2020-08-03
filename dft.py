import numpy as np
from scipy.linalg import dft
from scipy.io import loadmat
from auto import autop
import matplotlib.pyplot as plt


dict1 = loadmat(r"C:\Users\HP\Desktop\ANL\ED_mode36_24mS_181105\gd_mode36_gd_24mS.mat")
bunches = dict1["bunches"]

# obj = autop(bunches, 324, abs=True)
#
# obj.create_modes(type = "dft")
# obj.ZOMremoval()
# # obj.plot_data(type = "modes")
# for i in range(30,39):
#     plt.plot(obj.modes[i])
# plt.show()


mode = 36
index = mode-1
dft(N)[index]
