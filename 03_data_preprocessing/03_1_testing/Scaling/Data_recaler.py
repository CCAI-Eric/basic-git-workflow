from numpy import genfromtxt
import os
import joblib

dict_i = {0:"NEng", 1:"TqEng", 2:"VVeh"}
liste= []
for i in dict_i:
    print(dict_i.get(i))
    os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/predicted/sindelfingen")
    arr = genfromtxt("predict_scaled_" + str(dict_i.get(i)) + "_AitmContnsHorzn_sindelfingen.csv", delimiter=',')
    print("Loaded Data Shape" +str(arr.shape))

    os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/sindelfingen/scaler")
    scaler_filename = "scaler_label_" + str(dict_i.get(i)) +"_AitmContnsHorzn_sindelfingen.save"
    new_scaler = joblib.load(scaler_filename)

    print(new_scaler.data_min_)
    print(new_scaler.data_max_)

    arr = new_scaler.inverse_transform(arr)
    print(arr)

    os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/predicted/sindelfingen")
    # np.savetxt("predict_"+ str(dict_i.get(i)) +"_AitmContnsHorzn_sindelfingen.csv", arr, delimiter=',')
