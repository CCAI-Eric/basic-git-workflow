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

for t in range(len_dataset):
    pass

    for i in range(200):
        sample = []  # Leere Liste für sample Daten
        label = []

        label.append(GrSt[1, i])  # t=1
        label.append(NEng[1, i])
        label.append(VVeh[1, i])
        label.append(TqEng[1, i])

        sample.append(Tm_AmbAirP[1, i])
        sample.append(VSlopAFinPosn[1, i])
        sample.append(SlopFinVal[1, i])
        sample.append(VFinVal[1, i])
        sample.append(AFinVal[1, i])
        sample.append(VSlopAFinVectLen[1, i])
        sample.append(Tm_AmbAirTp[1, i])
        sample.append(MVeh[1, i])
        sample.append(Cod_Diff_Ratio_Calc[1, i])
        sample.append(CurbWeight[1, i])
        sample.append(WhlPA_Circumfer[1, i])

        print(label.shape)  # (4, )

        score = model.evaluate(sample, label, verbose=0) # Berechne den Score (loss, R2) von der Evaluierung
        print("Score: ", score)

        predictions = model.predict(x=sample, batch_size=1, verbose=0) # Berechne die "prediction" bzw. Outputwerte
        print(predictions.shape) # Shape sollte nun (4, ) sein

        Gang = Gang.append(predictions[0])
        Drehzahl = Drehzahl.append(predictions[1])
        Moment = Moment.append(predictions[2])
        Speed = Speed.append(predictions[3])


    print(Gang.shape)  # (200,)
    #  Renormierung