import numpy as np
import pandas as pd
import os

# Laden der Daten
VSlopAFinVectLen = np.genfromtxt(
    "/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/sindelfingen/sample_VSlopAFinVectLen_AitmContnsHorzn_sindelfingen.csv",
    delimiter=",",
)
sample_filter_dict = {
    0: "VSlopAFinPosn",
    1: "SlopFinVal",
    2: "VFinVal",
    3: "AFinVal",
}  # TODO: Add label channels
label_filter_dict = {0: "NEng", 1: "T", 2: "TqEng", 3: "VVeh", 4: "GrSt"}
constant_dict = {
    0: "MVeh",
    1: "Tm_AmbAirP",
    2: "Tm_AmbAirTp",
    3: "VSlopAFinVectLen",
    4: "Cod_Diff_Ratio_Calc",
    5: "CurbWeight",
    6: "WhlPA_Circumfer",
}
timestep_len = len(VSlopAFinVectLen)


# # TODO: str+r --> lab/sam
# for n in sample_filter_dict:
#     channel_array = np.genfromtxt('/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/sindelfingen/sample_' + sample_filter_dict.get(n) +'_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
#     indizes_len = channel_array.shape[1]
#
#     for t in range(timestep_len):
#         horizon_len = VSlopAFinVectLen[t]
#
#         for i in range(indizes_len):
#             if i >= horizon_len-1:
#                 channel_array[t, i] = np.nan
#
#     print(channel_array[:20,:])
#     channel_array = channel_array[np.logical_not(np.isnan(channel_array))]
#     print(channel_array.shape)
# #
#     os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/sindelfingen/filter_VSlopAFinVectLen")
#     np.savetxt("sample_no_nan_" + sample_filter_dict.get(n) + "_AitmContnsHorzn_sindelfingen.csv", channel_array, delimiter=',')
# len: 3521877


# for n in constant_dict:
#     channel_array = np.genfromtxt('/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/sindelfingen/sample_' + constant_dict.get(n) +'_AitmContnsHorzn_sindelfingen.csv', delimiter=',') # TODO: sample/label
#     new_array = []
#     print(constant_dict.get(n))
#
#     for t in range(timestep_len):
#         horizon_len = int(VSlopAFinVectLen[t])
#         for i in range(horizon_len-1):
#             new_array.append(channel_array[t])
#
#     new_array = np.array(new_array)
#     print(new_array.shape)
#     print(new_array)
#
#     os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/sindelfingen/filter_VSlopAFinVectLen")
#     np.savetxt("sample_no_nan_" + constant_dict.get(n) + "_AitmContnsHorzn_sindelfingen.csv", new_array, delimiter=',') # TODO: sample/label
#


# Erzeugung der Spalten für i und t für die Zuordnung (Koordinaten)
print(VSlopAFinVectLen.shape)
print(VSlopAFinVectLen[0:5])
array = []
for t in range(timestep_len):
    horizon_len = int(VSlopAFinVectLen[t])
    if horizon_len >= 2:
        for i in range(horizon_len - 1):
            array.append([int(i), int(t)])

array = np.array(array)
print(array.shape)
print(array[:100, :100])

# 3521877
# 3474151 mit VSlopAFinVectLen-1
# diff 47726 --> Erste Zeile war vorher schon weg
os.chdir(
    "/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/sindelfingen/filter_VSlopAFinVectLen"
)
# np.savetxt("coordinates_i_t"+"_AitmContnsHorzn_sindelfingen.csv", array, delimiter=',') # TODO: sample/label


"""
    # horizon = self.VSlopAFinVectLen.to_numpy()
        # timestep_len = len(horizon)
        # array_list = [self.VSlopAFinPosn, self.SlopFinVal, self.VFinVal, self.AFinVal,
        #              self.NEng, self.T, self.TqEng, self.VVeh, self.GrSt]
        #
        # value_list = [self.Tm_AmbAirP, self.VSlopAFinVectLen,
        #              self.Tm_AmbAirTp, self.Tm_AmbAirTp,  self.MVeh, self.Cod_Diff_Ratio_Calc, self.CurbWeight, self.WhlPA_Circumfer]
        # for n in array_list:
        #     array = np.array(n)
        #     indizes_len = array.shape[1]
        #
        #     for t in range(timestep_len):
        #         horizon_len = horizon[t]
        #         print(horizon_len)
        #
        #         for i in range(indizes_len):
        #             if i >= horizon_len-1:
        #                 array[t, i] = np.nan
        #
        #     print(array[:20,:])
        #     array = array[np.logical_not(np.isnan(array))]
        #     print(array.shape)
        #     # value_list[n] = array

"""
