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

base_path = "/02_dataset/Mdl_Pwr_MdlBatPwr_Construct_BatConstruct/"
testdrive_path = "Sim20210225_2019-12-20 12_31_werk_mit_routenfuehrung/"
path = base_path + testdrive_path
target_list = fnmatch.filter(os.listdir(path), "*100*")
print("Signale: ", target_list)
# testdrive_path.split("/")[-1].replace("/", "")
testdrive_path = testdrive_path.replace("/", "")
print("Testfahrt: ", testdrive_path)

min_array = []
max_array = []

for signal in target_list:
    sig = np.array(pd.read_csv(os.path.join(path, signal), delimiter=","))
    # data[data==0] = float("NaN")

    # filter ersatzwerte raus
    if signal == "label_AitmContnsHorzn_NEngTBasAry100.csv":
        sig[sig == 7650] = float(1)
    elif signal == "label_AitmContnsHorzn_VVehTBasAry100.csv":
        sig[sig == 255] = float(20)
    elif signal == "label_AitmContnsHorzn_TContnsAry100.csv":
        sig[sig == 6553.5] = float(1)
    elif signal == "label_AitmContnsHorzn_TqEngTBasAry100.csv":
        sig[sig == 920] = float(1)

    # berechne min und max werte
    mini = np.min(sig).round(2)
    maxi = np.max(sig).round(2)

    min_array.append(mini)
    max_array.append(maxi)

    print("Shape: ", sig.shape)

    # print("Statistic: ", sig.describe())
    print("Max: ", np.max(sig))
    print("Min: ", np.min(sig))

print(min_array)
print(max_array)
min_array = pd.Series(min_array)
max_array = pd.Series(max_array)
target_list = pd.Series(target_list)
values_df = pd.concat([min_array, max_array, target_list], axis=1)
print(values_df)

scaler_path = "/02_dataset/Mdl_Pwr_MdlBatPwr_Construct_BatConstruct/scaler_values/"
# values_df.to_csv(scaler_path + testdrive_path + "_scaler_values.csv", header=False, index=False)
