import numpy as np
import pandas as pd
from numpy import genfromtxt
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import pathlib

# pathlib.Path('/my/directory').mkdir(parents=True, exist_ok=True)


dict_i = {
    0: "sample_VSlopAFinPosn",
    1: "sample_SlopFinVal",
    2: "sample_VFinVal",
    3: "sample_AFinVal",
    4: "label_GrSt",
    5: "label_NEng",
    6: "label_T",
    7: "label_TqEng",
    8: "label_VVeh",
}

dict = {
    0: "sample_Tm_AmbAirP",
    1: "sample_VSlopAFinVectLen",
    2: "sample_Tm_AmbAirTp",
    3: "sample_MVeh",
    4: "sample_Cod_Diff_Ratio_Calc",
    5: "sample_CurbWeight",
    6: "sample_WhlPA_Circumfer",
}

# dict = {0: "sample_MVeh"}
# dict_i = {0:"label_NEng"}

for i in dict:
    print(dict.get(i))
    os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/huettlingen")
    arr = genfromtxt(
        str(dict.get(i)) + "_AitmContnsHorzn_huettlingen.csv", delimiter=","
    )
    print("Loaded Data Shape" + str(arr.shape))

    arr = arr.reshape(-1, 1)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0.0001, 0.9999))
    # scaler.fit(arr_0)
    scaler.fit(arr)
    print("scaler_min_value " + str(scaler.data_min_))
    print("scaler_max_value " + str(scaler.data_max_))
    arr = scaler.transform(arr)
    # print(VVeh)
    print("scaled data shape" + str(arr.shape))

    # Save the scaler
    os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/huettlingen/scaler")
    scaler_filename = "scaler_" + str(dict.get(i)) + "_AitmContnsHorzn_huettlingen.save"
    joblib.dump(scaler, scaler_filename)

    # Save the scaled data as. csv
    os.chdir(
        "/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/huettlingen/scaled_data"
    )
    np.savetxt(
        "scaled_" + str(dict.get(i)) + "_AitmContnsHorzn_huettlingen.csv",
        arr,
        delimiter=",",
    )

print("Das waren die Kanäle ohne [i]")

for i in dict_i:
    print(dict_i.get(i))
    os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/huettlingen")
    arr = genfromtxt(
        str(dict_i.get(i)) + "_AitmContnsHorzn_huettlingen.csv", delimiter=","
    )
    print("Loaded Data Shape" + str(arr.shape))

    arr_0 = arr[:, 0].reshape(-1, 1)
    print("Shape of i = 0 -> Training data " + str(arr_0.shape))  # 47727, 200

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0.0001, 0.9999))
    scaler.fit(arr_0)
    print("scaler_min_value " + str(scaler.data_min_))
    print("scaler_max_value " + str(scaler.data_max_))
    arr = scaler.transform(arr)
    # print(VVeh)
    print("scaled data shape" + str(arr.shape))

    # Save the scaler
    os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/huettlingen/scaler")
    scaler_filename = (
        "scaler_" + str(dict_i.get(i)) + "_AitmContnsHorzn_huettlingen.save"
    )
    joblib.dump(scaler, scaler_filename)

    # Save the scaled data as. csv
    os.chdir(
        "/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/huettlingen/scaled_data"
    )
    np.savetxt(
        "scaled_" + str(dict_i.get(i)) + "_AitmContnsHorzn_huettlingen.csv",
        arr,
        delimiter=",",
    )
