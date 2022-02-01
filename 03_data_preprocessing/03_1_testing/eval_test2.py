import numpy as np
import os
from numpy import genfromtxt

os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Daten/sindelfingen")

GrSt = genfromtxt('label_GrSt_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
NEng = genfromtxt('label_NEng_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
print(NEng.shape)  # (47727, 200)

len_dataset = len(NEng)  # 47727

sample_filter_dict = {0: "Tm_AmbAirP", 1: "VSlopAFinPosn", 2: "SlopFinVal",3: "VFinVal", 4: "AFinVal",5: "VSlopAFinVectLen",
                      6: "Tm_AmbAirTp",7: "MVeh",8: "Cod_Diff_Ratio_Calc", 9: "CurbWeight", 10: "WhlPA_Circumfer"}

sample = [MVeh[t,i], AFinVal[t,i], SlopFinVal[t,i], VFinVal[t,i], VSlopAFinPosn[t,i], VSlopAFinVectLen[t,i], Cod_Diff_Ratio_Calc[t,i],
          CurbWeight[t,i], Tm_AmbAirP[t,i], Tm_AmbAirTp[t,i], WhlPA_Circumfer[t,i]]

label = [NEng[t,i], T[t,i], TqEng[t,i], VVeh[t,i], GrSt[t,i]]


len_dataset = 10
for t in range(len_dataset):
    for i in range(200):
        sample = [GrSt[t,i], NEng[t,i]]
        print(sample)
print(sample)
print(np.shape(sample))
