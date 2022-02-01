import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml
import matplotlib.pyplot as plt
import fnmatch
import joblib
import sys

base_path = "/home/ase/Dokumente/eh_basics/masterarbeit_eh/02_dataset/Mdl_Pwr_MdlBatPwr_Construct_BatConstruct/"
testdrive_path = "Sim20210225_2020-05-01_223-291_20A_Abtsgmuend_Fahrt-Huettlingen-EMode_01/"
path = base_path + testdrive_path
name = "CurBatTBas"
list = fnmatch.filter(os.listdir(path), '*' + name + '*')
print("Signal: ", list)
print("Testfahrt: ", testdrive_path)
for signal in list:
    sig = np.array(pd.read_csv(os.path.join(path, signal), delimiter=','))
# data[data==0] = float("NaN")

    # sig[sig==920] = float(1)
    # print(sig)
    # sig[sig == 255] = float("50")

    print("Shape: ", sig.shape)

    # print("Statistic: ", sig.describe())
    print("Max: ", np.max(sig))
    print("Min: ", np.min(sig))