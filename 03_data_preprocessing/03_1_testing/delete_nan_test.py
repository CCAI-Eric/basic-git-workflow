import numpy as np
import os
import pandas as pd

# GrSt = np.genfromtxt('/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/sindelfingen/filter_VSlopAFinVectLen/label_GrSt_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
# print(GrSt)
#
# # Remove nan from numpy array
# GrSt = GrSt[np.logical_not(np.isnan(GrSt))]
# # GrSt = GrSt[~np.isnan(GrSt)]
# print("Zeilenumbruch")
# print(GrSt)
# print(GrSt.shape)
#

# #  pandas example
os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/sindelfingen/filter_VSlopAFinVectLen")
VSlopAFinPosn = pd.read_csv('sample_VSlopAFinPosn_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
SlopFinVal = pd.read_csv('sample_SlopFinVal_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
VFinVal = pd.read_csv('sample_VFinVal_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
AFinVal = pd.read_csv('sample_AFinVal_AitmContnsHorzn_sindelfingen.csv', delimiter=',')

os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/sindelfingen")

Tm_AmbAirP = pd.read_csv('sample_Tm_AmbAirP_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
VSlopAFinVectLen = pd.read_csv('sample_VSlopAFinVectLen_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
Tm_AmbAirTp = pd.read_csv('sample_Tm_AmbAirTp_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
MVeh = pd.read_csv('sample_MVeh_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
Cod_Diff_Ratio_Calc = pd.read_csv('sample_Cod_Diff_Ratio_Calc_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
CurbWeight = pd.read_csv('sample_CurbWeight_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
WhlPA_Circumfer = pd.read_csv('sample_WhlPA_Circumfer_AitmContnsHorzn_sindelfingen.csv', delimiter=',')

sample = pd.concat([MVeh, AFinVal, SlopFinVal,VFinVal,VSlopAFinPosn,VSlopAFinVectLen, Cod_Diff_Ratio_Calc, CurbWeight, Tm_AmbAirP, Tm_AmbAirTp, WhlPA_Circumfer], axis=1)
print(sample)

sample = sample.to_numpy()
print(sample.shape)

sample = sample[np.logical_not(np.isnan(sample))]
print(sample.shape)
print(sample)
