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

"""
###########################################################
#  working directory: eh_masterarbeit
print(os.getcwd())

# Open config file
with open("03_data_preprocessing/config_Preprocess_Data_fcnn_combination_v2.yml", "r") as stream:
    config = yaml.safe_load(stream)

def main():
    # Instanz der Klasse
    preprocess = Preprocessing()

    # Lade die .csv Rohdaten und fülle alles nach horizon lenght mit nan
    preprocess.load_and_prepare_csv_data()

    # Concat sample und label data
    preprocess.concat_sample_and_target()

    # Splitte die Daten zu Trainings und Testdatensatz
    preprocess.split_data()

    # Normiere die Daten mit dem MinMaxScaler zwischen 0.001 und 1
    preprocess.data_scaler()

    # Nun werden die gesplitteten und normierten Daten betrachtet und nans durch 0 oder -1 ersetzt
    preprocess.replace_nans()

    # Speichern der verarbeiteten Daten
    preprocess.save_data()

class Preprocessing:
    def __init__(self):
        self.scaled_train_samples = []
        self.scaled_train_targets = []
        self.scaled_test_samples = []
        self.scaled_test_targets = []
        self.ersatzwert = 9999999
        self.sample_ew = -100000

    
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
                    row[int(d):] = np.nan  # int(-1) np.nan
                    rows.append(row)
                rows = np.array(rows)
                # print(rows.shape)
                # # print(rows[11:14, :5])
            else:
                rows = temp
            sample_array = np.append(sample_array, rows, axis=1)

        self.sample_array = sample_array[:, 200:]
        print(self.sample_array.shape)
        self.sample_array = pd.DataFrame(self.sample_array)


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
                # row[int(d):] = np.nan  TODO: targets werden nicht durch horizon_len beschränkt
                rows.append(row)
            rows = np.array(rows)
            # print(rows.shape)
            # # print(rows[11:14, :5])
            target_array = np.append(target_array, rows, axis=1)

        self.target_array = target_array[:, 200:]
        # print(self.target_array.shape)
        self.target_array = pd.DataFrame(self.target_array)

        # Ersatzwerte ersetzen
        self.target_array = self.target_array.replace(7650, self.ersatzwert)  # für NEng
        self.target_array = self.target_array.replace(255, self.ersatzwert)  # VVeh
        self.target_array = self.target_array.replace(920, self.ersatzwert)  # TqEng
        self.target_array = self.target_array.replace(6553.5, self.ersatzwert)  # T
        # self.target_array = self.target_array.to_numpy()
        print(self.target_array)


    def concat_sample_and_target(self):
        data_df = pd.concat([self.target_array, self.sample_array], axis=1)
        print(data_df)
        # Ersetze die nans aus den sample Daten mit Null
        data_df = data_df.fillna(self.sample_ew)
        print(data_df)
        # Ersatzwerte zu nan
        data_df = data_df.replace(self.ersatzwert, np.nan)
        print(data_df)
        data_df = data_df.dropna()  # Drop the rows where at least one element is missing
        print(data_df)  # von 9173 Zeilen --> 8643 Zeilen nur durch Ersatzwert 920 TqEng
        # 8226 Zeilen bleiben, wenn alle 4 Ersatzwerte rausgefiltert werden

        # Ändere Sample Ersatzwerte wieder zu NaN
        data_df = data_df.replace(self.sample_ew, np.nan)
        print(data_df)

        self.sample_array = data_df.iloc[:, 600:]
        self.target_array = data_df.iloc[:, :600]
        print("Shape sample Daten: ", self.sample_array.shape)
        print("Shape target Daten: ", self.target_array.shape)
        self.sample_array = self.sample_array.to_numpy()
        self.target_array = self.target_array.to_numpy()

    def split_data(self):
        self.train_samples, self.test_samples, self.train_targets, self.test_targets = train_test_split(
        self.sample_array, self.target_array, test_size=config["test_size"], random_state=config["seed"], shuffle=True)
        # print(self.test_targets[10:20, :])
        print(self.train_samples.shape)
        print(self.test_samples.shape)

        # pd.set_option('display.max_columns', None)
        # test_df = pd.DataFrame(self.test_targets)
        # test_df.to_csv(config["input_path"] + "unscaled_" + config["split_data_names"]["test_ground_truth"] + config["split_data_names"]["data_format"],
        #                      header=False, index=False)
        # print(test_df)
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
        print(os.getcwd())
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
        print(self.scaled_train_targets)


    def replace_nans(self):
        # lade die gesplitteten und normierten Daten, forme diese in Dataframes und fülle die Nans mit 0 oder -1
        # Konvertiere np. arrays zu Dataframes
        self.scaled_train_samples = pd.DataFrame(self.scaled_train_samples)
        self.scaled_train_targets = pd.DataFrame(self.scaled_train_targets)
        self.scaled_test_samples = pd.DataFrame(self.scaled_test_samples)
        self.scaled_test_targets = pd.DataFrame(self.scaled_test_targets)

        # Fülle nan values mit 0
        self.scaled_train_samples = self.scaled_train_samples.fillna(0)
        self.scaled_train_targets = self.scaled_train_targets.fillna(0)
        self.scaled_test_samples = self.scaled_test_samples.fillna(0)
        self.scaled_test_targets = self.scaled_test_targets.fillna(0)

        print(self.scaled_train_samples)

    def save_data(self):
        os.chdir("/home/ase/Dokumente/eh_basics/masterarbeit_eh/02_dataset/Kombination_MdlPwr_MdlBatPwr/original_structure/AitmContsHorzn_MdlPwr_MdlBatPwr_Cnstr_BatCnstr_2019-12-20 11_41_53gaertringen_mit_routenfuehrung")
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