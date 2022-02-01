import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml
import matplotlib.pyplot as plt
import joblib
import sys

# import yaml data
with open("03_data_preprocessing/config_Preprocess_Data_MdlPwr.yml", "r") as stream:
    config = yaml.safe_load(stream)


# Read csv files
os.chdir(config["input_path"])
VSlopAFinPosn = pd.read_csv(
    config["input_names"]["sample"]["VSlopAFinPosn"], delimiter=","
)
# SlopFinVal = pd.read_csv(config["input_names"]["sample"]["SlopFinVal"], delimiter=',')
# VFinVal = pd.read_csv(config["input_names"]["sample"]["VFinVal"], delimiter=',')
# AFinVal = pd.read_csv(config["input_names"]["sample"]["AFinVal"], delimiter=',')
#
# Tm_AmbAirP = pd.read_csv(config["input_names"]["sample"]["Tm_AmbAirP"], delimiter=',')
VSlopAFinVectLen = pd.read_csv(
    config["input_names"]["sample"]["VSlopAFinVectLen"], delimiter=","
)
# Tm_AmbAirTp = pd.read_csv(config["input_names"]["sample"]["Tm_AmbAirTp"], delimiter=',')
# MVeh = pd.read_csv(config["input_names"]["sample"]["MVeh"], delimiter=',')
# Cod_Diff_Ratio_Calc = pd.read_csv(config["input_names"]["sample"]["Cod_Diff_Ratio_Calc"],
#                                   delimiter=',')
# CurbWeight = pd.read_csv(config["input_names"]["sample"]["CurbWeight"], delimiter=',')
# WhlPA_Circumfer = pd.read_csv(config["input_names"]["sample"]["WhlPA_Circumfer"], delimiter=',')

print(VSlopAFinPosn)
print(VSlopAFinPosn.info())
print(VSlopAFinPosn.describe())

Posn = VSlopAFinPosn.to_numpy()
# Berechnung für neuen Inputkanal
print(Posn.shape)
print(Posn[4849, :])
posn_1 = Posn[:, 1:]
print(posn_1.shape)
print(posn_1[4849, :])

horz_len = VSlopAFinVectLen.to_numpy()
print(horz_len[4849])
print(horz_len.shape)

a = int(horz_len[4849, 0] - 1)
# print(a)
new_posn = Posn[4849, :a]
print(new_posn.shape)
print(new_posn)


# concat data
# sample = pd.concat([MVeh, AFinVal, SlopFinVal, VFinVal, VSlopAFinPosn,
#                     VSlopAFinVectLen, Cod_Diff_Ratio_Calc, CurbWeight, Tm_AmbAirP,
#                     Tm_AmbAirTp, WhlPA_Circumfer], axis=1)

# print(sample)
