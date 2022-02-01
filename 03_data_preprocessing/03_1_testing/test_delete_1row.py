from numpy import genfromtxt
import numpy as np
import os
import pandas as pd

os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/predicted/sindelfingen")

VVeh = genfromtxt("predict_scaled_TqEng_AitmContnsHorzn_sindelfingen.csv", delimiter=',')
print(VVeh.shape)
VVeh = np.delete(VVeh, (0), axis=0)
print(VVeh.shape)
print(VVeh)

np.savetxt("predict_scaled_TqEng_AitmContnsHorzn_sindelfingen.csv", VVeh, delimiter=',')


