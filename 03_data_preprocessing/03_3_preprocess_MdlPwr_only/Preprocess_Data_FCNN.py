import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml
import matplotlib.pyplot as plt
import joblib
import sys
sys.path.append(os.getcwd())

"""
Kurzbeschreibung:
1. Lade die sample und target Daten mit pandas (ohne nans)
2. Lösche die nan und erzeuge damit neue Struktur (1 Spaltenvektor)
3. Verbinde einzelne Kanäle zu gesamten Dataframe (Concat)
4. Erstelle Koordinaten (t und i) Dataframe und füge es dem gesamten df hinzu
5. Splitte Daten zu Trainings- und Testdatensatz (mit shuffle) --> danach nicht mehr shufflen, um Zuordnung für t und ich nicht zu verlieren
    5.1 Speichere Trainings und Testdatensatz als .npy oder .csv
6. Nutze MinMaxScaler auf die Trainingsdaten (sample und target)
    6.1 Speicher die scaler ab, um diese ggfs. später zu nutzen
"""
###########################################################
#  working directory: eh_masterarbeit
print(os.getcwd())

# Open config file
with open("03_data_preprocessing/config_preprocess_data_fcnn.yml", "r") as stream:
    config = yaml.safe_load(stream)

def main():
    # Instanz der Klasse
    preprocess = Preprocessing()

    # correct NEng
    preprocess.correct_NEng()
    
    # Lade .csv Daten und lösche nans
    horzn_len = preprocess.load_csv_data_and_delete_nan()

    # Erstelle Daten für Koordinaten t und i für Zuordnung Evaluierung
    preprocess.create_coordinates_ti(horzn_len)

    # Füge die einzelnen Signale zusammen zu vollständigen Datensatz
    preprocess.concat_data()

    # Splitte die Daten in Trainings- und Testdaten mit Train_test_split
    preprocess.split_data()

    # Speichere die Daten als .csv oder .npy
    preprocess.save_data()

    # Skalierung der Daten mit MinMaxScaler (speichern der Scaler mit joblib)
    preprocess.data_scaler()


class Preprocessing:
    def __init__(self):
        pass
    
    def correct_NEng(self):
        os.chdir(config["input_path"]) # TODO: Erzeugt eine nan, wodurch das Training nicht funktioniert!
        NEng = pd.read_csv(config["input_names"]["label"]["NEng"], delimiter=',')
        
        if config["correct_NEng"]:
            NEng = NEng.to_numpy()
            NEng = NEng[:, :] * config["korrekturfaktor_NEng"]
            print(NEng)
            np.savetxt("label_AitmContnsHorzn_NEngAry200.csv", NEng, delimiter=',')
    
    def load_csv_data_and_delete_nan(self):
        # load sample data
        VSlopAFinPosn = pd.read_csv(config["input_names"]["sample"]["VSlopAFinPosn"], delimiter=',')
        SlopFinVal = pd.read_csv(config["input_names"]["sample"]["SlopFinVal"], delimiter=',')
        VFinVal = pd.read_csv(config["input_names"]["sample"]["VFinVal"], delimiter=',')
        AFinVal = pd.read_csv(config["input_names"]["sample"]["AFinVal"], delimiter=',')

        Tm_AmbAirP = pd.read_csv(config["input_names"]["sample"]["Tm_AmbAirP"], delimiter=',')
        VSlopAFinVectLen = pd.read_csv(config["input_names"]["sample"]["VSlopAFinVectLen"], delimiter=',')
        Tm_AmbAirTp = pd.read_csv(config["input_names"]["sample"]["Tm_AmbAirTp"], delimiter=',')
        MVeh = pd.read_csv(config["input_names"]["sample"]["MVeh"], delimiter=',')
        Cod_Diff_Ratio_Calc = pd.read_csv(config["input_names"]["sample"]["Cod_Diff_Ratio_Calc"],
                                          delimiter=',')
        CurbWeight = pd.read_csv(config["input_names"]["sample"]["CurbWeight"], delimiter=',')
        WhlPA_Circumfer = pd.read_csv(config["input_names"]["sample"]["WhlPA_Circumfer"], delimiter=',')

        # load target data
        NEng = pd.read_csv(config["input_names"]["label"]["NEng"], delimiter=',')
        T = pd.read_csv(config["input_names"]["label"]["T"], delimiter=',')
        TqEng = pd.read_csv(config["input_names"]["label"]["TqEng"], delimiter=',')
        VVeh = pd.read_csv(config["input_names"]["label"]["VVeh"], delimiter=',')
        GrSt = pd.read_csv(config["input_names"]["label"]["GrSt"], delimiter=',')

        
        array_list = [VSlopAFinPosn, SlopFinVal, VFinVal, AFinVal, VSlopAFinVectLen,
                    Tm_AmbAirTp, Tm_AmbAirP,  MVeh, Cod_Diff_Ratio_Calc, CurbWeight, WhlPA_Circumfer,
                    NEng, T, TqEng, VVeh, GrSt]

        # delete nan from numpy arrays
        for n, array in enumerate(array_list):  # counter, value
            print("Kanal: ", n+1)
            array = array.to_numpy()
            print(array.shape)
            array = array[np.logical_not(np.isnan(array))]
            # print(array.shape)
            array_list[n] = array
            print(array_list[n].shape)


        # changed data structure to class variables
        self.VSlopAFinPosn = pd.DataFrame(array_list[0])
        self.SlopFinVal = pd.DataFrame(array_list[1])
        self.VFinVal = pd.DataFrame(array_list[2])
        self.AFinVal = pd.DataFrame(array_list[3])
        self.VSlopAFinVectLen = pd.DataFrame(array_list[4])
        self.Tm_AmbAirTp = pd.DataFrame(array_list[5])
        self.Tm_AmbAirP = pd.DataFrame(array_list[6])
        self.MVeh = pd.DataFrame(array_list[7])
        self.Cod_Diff_Ratio_Calc = pd.DataFrame(array_list[8])
        self.CurbWeight = pd.DataFrame(array_list[9])
        self.WhlPA_Circumfer = pd.DataFrame(array_list[10])
        self.NEng = pd.DataFrame(array_list[11])
        self.T = pd.DataFrame(array_list[12])
        self.TqEng = pd.DataFrame(array_list[13])
        self.VVeh = pd.DataFrame(array_list[14])
        self.GrSt = pd.DataFrame(array_list[15])

        return VSlopAFinVectLen

    def create_coordinates_ti(self, horzn_len):
        pass
        # Variables for creating coor
        VSlopAFinVectLen = horzn_len.to_numpy() # 3474151
        timestep_len = len(VSlopAFinVectLen)

        # Erzeugung der Spalten für i und t für die Zuordnung (Koordinaten)
        coordinates_array = []
        for t in range(timestep_len):
            horizon_len = int(VSlopAFinVectLen[t, 0])
            if horizon_len >= 2:
                for i in range(horizon_len): # -1
                    coordinates_array.append([int(i), int(t)])

        coordinates_array = np.array(coordinates_array) # TODO: Überprüfe die Shape 3521877
        print(coordinates_array.shape)
        print(coordinates_array[:50, :50])
        self.coords = pd.DataFrame(coordinates_array, columns=["t", "i"])
        print(self.coords.head())
        # np.savetxt("coordinates_i_t"+"_AitmContnsHorzn.csv", coordinates_array, delimiter=',') # TODO: sample/label

    def concat_data(self):
        # concat sample df
        sample = pd.concat([self.MVeh, self.AFinVal, self.SlopFinVal, self.VFinVal, self.VSlopAFinPosn,
                            self.VSlopAFinVectLen, self.Cod_Diff_Ratio_Calc, self.CurbWeight, self.Tm_AmbAirP,
                            self.Tm_AmbAirTp, self.WhlPA_Circumfer], axis=1)
        print(sample.head())

        # concat label df
        label = pd.concat([self.GrSt, self.NEng, self.T, self.TqEng, self.VVeh], axis=1)
        print(label.tail(50))

        # concat to complete df --> [coordinates, sample, label]
        dataset = pd.concat([self.coords, sample, label], axis=1)
        print(dataset)
        print(dataset.describe())

        # convert dataframe to numpy array
        data = dataset.to_numpy()
        print(data.shape)
        self.sample = data[:, :13] #TODO: Achte darauf, dass die coordinaten Spalten mitgenommen werden und erst später aussortieren!
        self.label = data[:, 13:]
        print(sample.shape)
        print(label.shape)


    def split_data(self):
        self.train_samples, self.test_samples, self.train_ground_truth, self.test_ground_truth = train_test_split(
        self.sample, self.label, test_size=config["test_size"], random_state=config["seed"], shuffle=True)

        print(self.train_samples.shape)
        print(self.test_samples.shape)


    def data_scaler(self):
        # TODO: Für das Skalieren beachte die Koordinatenspalten t und i --> Eliminiere diese!
        # TODO: ohne Skalierung auf GrSt als IF
        self.train_samples = self.train_samples[:, 2:]
        self.test_samples = self.test_samples[:, 2:]
        print(self.train_samples.shape)

        if not os.path.isdir(config["scaler_path"]):
            os.mkdir(config["scaler_path"])
        os.chdir(config["scaler_path"])

        # Einstellung des MinMaxScalers
        scaler = MinMaxScaler(feature_range=(config["scaler_feature_range"]["min"], config["scaler_feature_range"]["max"]))

        # Skaliere auf Trainingsdaten (sample)
        scaler.fit(self.train_samples.reshape(-1, config['num_inputs']))
        print(scaler.data_min_)
        print(scaler.data_max_)
        # Speichere den Scaler ab
        scaler_filename = "scaler" + config["split_data_names"]["api"] + config["split_data_names"]["train_samples"] + config["scaler_format"]
        joblib.dump(scaler, scaler_filename)

        # Skaliere auf Trainingsdaten (ground_truth)
        if config["GrSt_no_scaling"]:
            self.train_ground_truth = self.train_ground_truth[:, 1:]

        scaler.fit(self.train_ground_truth.reshape(-1, config['num_outputs']))
        print(scaler.data_min_)
        print(scaler.data_max_)
        # Speichere den Scaler ab
        scaler_filename = "scaler" + config["split_data_names"]["api"] + config["split_data_names"]["train_ground_truth"] + config["scaler_format"]
        joblib.dump(scaler, scaler_filename)


    def save_data(self):
        print(os.getcwd())
        if not os.path.isdir(config["split_data_path"]):
            os.mkdir(config["split_data_path"])
        os.chdir(config["split_data_path"])

        # Speichere die gesplitteten nud geshuffelten Daten ab!
        np.savetxt(config["split_data_names"]["train_samples"] + config["split_data_names"]["data_format"], self.train_samples, delimiter=',')
        np.savetxt(config["split_data_names"]["train_ground_truth"] + config["split_data_names"]["data_format"], self.train_ground_truth, delimiter=',')
        np.savetxt(config["split_data_names"]["test_samples"] + config["split_data_names"]["data_format"], self.test_samples, delimiter=',')
        np.savetxt(config["split_data_names"]["test_ground_truth"] + config["split_data_names"]["data_format"], self.test_ground_truth, delimiter=',')


if __name__ == '__main__':
    main()