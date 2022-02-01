import pandas as pd
import numpy as np
import sys
import os
import fnmatch
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import genfromtxt
import yaml
from tensorflow.keras.callbacks import *
import tensorflow as tf


# Load the data
# config_Preprocess_Data_MdlPwr_LSTM.yml
with open(
    "05_neural_nets/LSTM/Train/config_Preprocess_Data_MdlPwr_LSTM.yml", "r"
) as stream:
    config = yaml.safe_load(stream)

# Read csv files

# Grundlegendes
path = "02_dataset/Mdl_Pwr/original_structure/AitmContsHorzn_sindelfingen"
horizon_path = path + "/" + config["input_names"]["sample"]["VSlopAFinVectLen"]
horizon_len = pd.read_csv(horizon_path, delimiter=",")
event_len = 200
timestep_len = horizon_len.shape[0]
horizon_len = horizon_len.iloc[:, 0].values  # numpy array

# Listen der sample & target Daten
label_list = fnmatch.filter(os.listdir(path), "*label*")
sample_list = fnmatch.filter(os.listdir(path), "*sample*")
print(len(sample_list))
print(len(label_list))

# temp = np.array(pd.read_csv(os.path.join(path, sample_list[0]), delimiter=','))
# Net_input = np.empty(np.shape(temp))
# Net_output = np.empty(np.shape(temp))
# print(Net_input.shape)


sample_array = np.ones(
    shape=(timestep_len, event_len)
)  # dummy array für append funktion
print("Shape (time_step_len, event_len): ", sample_array.shape)

for sample in sample_list:
    temp = np.array(pd.read_csv(os.path.join(path, sample), delimiter=","))
    # print(temp)
    print(sample)
    # print(np.shape(temp))

    if temp.shape[1] > 1:
        # TODO: Datenbearbeitung nach Horizon Length --> Werte auf 0 setzen!
        rows = []
        for i, d in enumerate(horizon_len):  # +1
            row = np.full(event_len, temp[i])
            row[int(d) :] = 0  # int(-1) np.nan
            rows.append(row)
        rows = np.array(rows)
        # print(rows.shape)
        # # print(rows[11:14, :5])
    else:
        rows = temp
    sample_array = np.append(sample_array, rows, axis=1)

sample_array = sample_array[:, 200:]
sample_df = pd.DataFrame(sample_array)
print(sample_df.shape)

target_array = np.ones(
    shape=(timestep_len, event_len)
)  # dummy array für append funktion
for label in label_list:
    temp = np.array(pd.read_csv(os.path.join(path, label), delimiter=","))
    # print(temp)
    print(label)
    # print(np.shape(temp))

    rows = []
    for i, d in enumerate(horizon_len):  # +1
        row = np.full(event_len, temp[i])
        row[int(d) :] = 0  # int(-1) np.nan
        rows.append(row)
    rows = np.array(rows)
    # print(rows.shape)
    # # print(rows[11:14, :5])
    target_array = np.append(target_array, rows, axis=1)

target_array = target_array[:, 200:]
print(target_array.shape)
print(target_array[10:20, :])
target_df = pd.DataFrame(target_array)
print(target_df.head())

# save data
# target_name = "targets.csv"
# target_df.to_csv(path + "/" + target_name, header=False, index=False)
#
# sample_name = "samples.csv"
# sample_df.to_csv(path + "/" + sample_name, header=False, index=False)
#
#
