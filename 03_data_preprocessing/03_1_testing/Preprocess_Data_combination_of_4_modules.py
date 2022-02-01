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
sys.path.append(os.getcwd())

"""
Für die Module MdlPwr, MdlBatPr, Constrct
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
with open("03_data_preprocessing/config_Preprocess_Data_fcnn_combination.yml", "r") as stream:
    config = yaml.safe_load(stream)

def main():
    # Instanz der Klasse
    preprocess = Preprocessing()

    # Lade die .csv Rohdaten und fülle alles nach horizon lenght mit 0
    preprocess.load_and_prepare_csv_data()

    preprocess.split_data()

    # preprocess.data_scaler()

    # preprocess.save_data()

class Preprocessing:
    def __init__(self):
        pass

    
    def load_and_prepare_csv_data(self):
        path = config["input_path"]
        horizon_path = path + "/" + config["horizon_signal"]
        horizon_len = pd.read_csv(horizon_path, delimiter=',')
        event_len = 200
        timestep_len = horizon_len.shape[0]
        horizon_len = horizon_len.iloc[:, 0].values  # numpy array
        print("Timestep Length: ", timestep_len)

        # Listen der sample & target Daten
        label_list = fnmatch.filter(os.listdir(path), '*label*')
        sample_list = fnmatch.filter(os.listdir(path), '*sample*')
        print("Eingangssignale: ", len(sample_list))
        print("Ausgangssignale: ", len(label_list))

        sample_array = np.ones(shape=(timestep_len, event_len))  # dummy array für append funktion
        print("Shape (time_step_len, event_len): ", sample_array.shape)

        for sample in sample_list:
            temp = np.array(pd.read_csv(os.path.join(path, sample), delimiter=','))
            # print(temp)
            print(sample)
            # print(np.shape(temp))

            if temp.shape[1] > 1:
                # TODO: Datenbearbeitung nach Horizon Length --> Werte auf 0 setzen!
                rows = []
                for i, d in enumerate(horizon_len):  # +1
                    row = np.full(event_len, temp[i])
                    row[int(d):] = 0  # int(-1) np.nan
                    rows.append(row)
                rows = np.array(rows)
                # print(rows.shape)
                # # print(rows[11:14, :5])
            else:
                rows = temp
            sample_array = np.append(sample_array, rows, axis=1)

        self.sample_array = sample_array[:, 200:]
        print(self.sample_array.shape)

        target_array = np.ones(shape=(timestep_len, event_len))  # dummy array für append funktion
        event_len = 100
        for label in label_list:
            temp = np.array(pd.read_csv(os.path.join(path, label), delimiter=','))
            # print(temp)
            print(label)
            # print(np.shape(temp))

            rows = []
            for i, d in enumerate(horizon_len):  # +1
                row = np.full(event_len, temp[i])
                row[int(d):] = 0  # int(-1) np.nan
                rows.append(row)
            rows = np.array(rows)
            # print(rows.shape)
            # # print(rows[11:14, :5])
            target_array = np.append(target_array, rows, axis=1)

        self.target_array = target_array[:, 200:]
        print(self.target_array.shape)
        print(self.target_array[10:20, :])



    def split_data(self):
        self.train_samples, self.test_samples, self.train_targets, self.test_targets = train_test_split(
        self.sample_array, self.target_array, test_size=config["test_size"], random_state=config["seed"], shuffle=True)
        # print(self.test_targets[10:20, :])
        print(self.train_samples.shape)
        print(self.test_samples.shape)

        pd.set_option('display.max_columns', None)
        test_df = pd.DataFrame(self.test_targets[100:150, :])
        print(test_df)
        # print(test_df.head())


    def data_scaler(self):
        os.chdir(config["input_path"])
        if not os.path.isdir(config["scaler_path"]):
            os.mkdir(config["scaler_path"])
        os.chdir(config["scaler_path"])

        # TODO: Prepare data for scaling --> Arrays zusammenfügen für Skalierung

        # Einstellung des MinMaxScalers
        scaler = MinMaxScaler(feature_range=(config["scaler_feature_range"]["min"], config["scaler_feature_range"]["max"]))

        # Skaliere auf Trainingsdaten (sample)
        scaler.fit(self.train_samples.reshape(-1, config['num_inputs']))
        # print(scaler.data_min_)
        # print(scaler.data_max_)

        # Skaliere die sample Daten
        self.scaled_train_samples = scaler.transform(self.train_samples)
        self.scaled_test_samples = scaler.transform(self.test_samples)

        # Speichere den Scaler ab
        scaler_filename = "scaler" + config["split_data_names"]["api"] + config["split_data_names"]["train_samples"] + config["scaler_format"]
        joblib.dump(scaler, scaler_filename)

        # Skaliere auf Trainingsdaten (target)
        scaler.fit(self.train_targets.reshape(-1, config['num_outputs']))
        # print(scaler.data_min_)
        # print(scaler.data_max_)

        # Speichere den Scaler ab
        scaler_filename = "scaler" + config["split_data_names"]["api"] + config["split_data_names"]["train_ground_truth"] + config["scaler_format"]
        joblib.dump(scaler, scaler_filename)

        # Skaliere die sample Daten
        self.scaled_train_targets = scaler.transform(self.train_targets)
        self.scaled_test_targets = scaler.transform(self.test_targets)


    def save_data(self):
        print(os.getcwd())
        if not os.path.isdir(config["split_data_path"]):
            os.mkdir(config["split_data_path"])
        os.chdir(config["split_data_path"])

        train_samples = pd.DataFrame(self.scaled_train_samples)
        train_targets = pd.DataFrame(self.scaled_train_targets)
        test_samples = pd.DataFrame(self.scaled_test_samples)
        test_targets = pd.DataFrame(self.scaled_test_targets)

        train_samples.to_csv(config["split_data_names"]["train_samples"] + config["split_data_names"]["data_format"],
                             header=False, index=False)
        test_samples.to_csv(config["split_data_names"]["test_samples"] + config["split_data_names"]["data_format"],
                             header=False, index=False)
        train_targets.to_csv(config["split_data_names"]["train_ground_truth"] + config["split_data_names"]["data_format"],
                             header=False, index=False)
        test_targets.to_csv(config["split_data_names"]["test_ground_truth"] + config["split_data_names"]["data_format"],
                             header=False, index=False)

if __name__ == '__main__':
    main()