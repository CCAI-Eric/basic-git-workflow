import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml
import matplotlib.pyplot as plt
import fnmatch
import joblib
import sys

sys.path.append(os.getcwd())

"""
Für die Module MdlPwr, MdlBat
Kurzbeschreibung:
1. Variante:
Vorverarbeitung für Eingangsdatenraum: 11 Inputs
Ausgangsdatenraum: 7
Daten hinter Horizontlänge werden zu NaN, dann entfernt und dadurch Matrix zu 1Dim-Vektor geflattet.
"""
###########################################################
#  working directory: eh_masterarbeit

# Open config file
with open("03_data_preprocessing/03_5_preprocess_MdlPwr_BatPwr_kombi/03_5_Var_1_Ereignis/config_preprocess_var1.yml",
          "r") as stream:
    config = yaml.safe_load(stream)


def main():
    # Instanz der Klasse
    preprocess = Preprocessing()

    # Lade die .csv Rohdaten und fülle alles nach horizon lenght mit nan
    preprocess.load_csv_data()

    # preprocess.create_hist()

    # Concat sample und label data & speichern
    preprocess.save_data()


class Preprocessing:
    def __init__(self):
        self.scaled_train_samples = []
        self.scaled_train_targets = []
        self.scaled_test_samples = []
        self.scaled_test_targets = []
        self.sample_array = []
        self.target_array = []
        self.samples_gesamtdatensatz = []
        self.targets_gesamtdatensatz = []
        self.save_name = config["dataset_name"]

    def load_csv_data(self):
        # Suche alle Datensätze im Origin Pfad
        origin_path = config["data_path"]
        dataset_list = fnmatch.filter(os.listdir(origin_path), '*Sim*')
        print("Datensätze: ", len(dataset_list))

        # Länge der einzelnen Datensätze nach Elimination der Nans
        # dataset_len_list = [3006200, 646012, 231698, 712113, 599003, 122467, 3527383]
        dataset_len_list = [3006200, 646012]
        self.samples_gesamtdatensatz = np.ones(shape=(1, 13))
        self.targets_gesamtdatensatz = np.ones(shape=(1, 7))

        for num, name in enumerate(dataset_list):
            print(name)
            path = origin_path + name
            print(path)
            horizon_path = path + "/" + config["horizon_signal"]
            horizon_len = pd.read_csv(horizon_path, delimiter=',')
            event_len = config["array_len"]  # 200
            timestep_len = horizon_len.shape[0]
            horizon_len = horizon_len.iloc[:, 0].values  # numpy array
            self.dataset_name = path.split("/Sim_")[1]
            print(self.dataset_name)
            print("Timestep Length: ", timestep_len)

            # Listen der sample & target Daten
            label_list = fnmatch.filter(os.listdir(path), '*label*')
            sample_list = fnmatch.filter(os.listdir(path), '*sample*')
            print("Eingangssignale: ", len(sample_list))
            print("Ausgangssignale: ", len(label_list))

            sample_array = np.ones(shape=(dataset_len_list[num], 1))  # TODO: Wie sollen die Zeilen ausgelesen werden?
            print("Shape (time_step_len, event_len): ", sample_array.shape)

            for sample in sample_list:
                temp = np.array(pd.read_csv(os.path.join(path, sample), delimiter=','))
                # print(temp)
                print(sample)
                # print(np.shape(temp))

                if temp.shape[1] == 1:
                    # TODO: Datenbearbeitung nach Horizon Length --> Werte NaN setzen
                    # rows = temp
                    temp = np.repeat(temp, 200, axis=1)
                    rows = []
                    for i, d in enumerate(horizon_len):  # +1
                        row = np.full(event_len, temp[i])
                        row[int(d):] = np.nan  # int(-1) np.nan
                        rows.append(row)
                    rows = np.array(rows)
                    rows = rows[np.logical_not(np.isnan(rows))]
                    rows = rows.reshape(rows.shape[0], 1)
                    print(rows.shape)

                    # print(rows.shape)
                    # print(rows[11:14, :5])
                else:
                    rows = []
                    for i, d in enumerate(horizon_len):  # +1
                        row = np.full(event_len, temp[i])
                        row[int(d):] = np.nan  # int(-1) np.nan
                        rows.append(row)
                    rows = np.array(rows)
                    rows = rows[np.logical_not(np.isnan(rows))]
                    rows = rows.reshape(rows.shape[0], 1)
                    print(rows.shape)

                sample_array = np.append(sample_array, rows, axis=1)

            self.sample_array = sample_array[:, 1:]
            print("Sample Shape: ", self.sample_array.shape)
            self.sample_array = pd.DataFrame(self.sample_array)

            target_array = np.ones(shape=(dataset_len_list[num], 1))  # TODO: Wie sollen die Zeilen ausgelesen werden?

            for label in label_list:
                temp = np.array(pd.read_csv(os.path.join(path, label), delimiter=','))
                # print(temp)
                print(label)
                # print(np.shape(temp))

                rows = temp
                rows = []
                for i, d in enumerate(horizon_len):  # +1
                    row = np.full(event_len, temp[i])
                    row[int(d):] = np.nan  # TODO: targets werden nicht durch horizon_len beschränkt
                    rows.append(row)
                rows = np.array(rows)
                rows = rows[np.logical_not(np.isnan(rows))]
                rows = rows.reshape(rows.shape[0], 1)
                print(rows.shape)
                # print(rows[11:14, :5])
                target_array = np.append(target_array, rows, axis=1)

            self.target_array = target_array[:, 1:]
            # print(self.target_array.shape)
            self.target_array = pd.DataFrame(self.target_array)
            print("Target shape: ", self.target_array.shape)

            # Übergebe Daten des einzelnen Datensatzes dem Array für Gesamtdatensatz
            print(self.targets_gesamtdatensatz.shape)
            self.samples_gesamtdatensatz = np.append(self.samples_gesamtdatensatz, self.sample_array, axis=0)
            self.targets_gesamtdatensatz = np.append(self.targets_gesamtdatensatz, self.target_array, axis=0)

        # Entferne dummy Zeile aus Gesamtdatensatz
        self.samples_gesamtdatensatz = self.samples_gesamtdatensatz[1:, :]
        self.targets_gesamtdatensatz = self.targets_gesamtdatensatz[1:, :]

        print("Shape samples Gesamtdatensatz: ", self.samples_gesamtdatensatz.shape)
        print("Shape targets Gesamtdatensatz: ", self.targets_gesamtdatensatz.shape)

    def create_hist(self):
        target_arr_0 = np.array(self.target_array)
        target_arr_0 = target_arr_0[:, 0:1399:200]
        col = ["CurBat", "StEdrv", "GrSt", "NEng", "VVeh", "TqEng", "T"]
        # df.hist(column='Test1', bins=[0, .5, .75, 1]);
        target_arr_0 = pd.DataFrame(data=target_arr_0, columns=col)
        print(target_arr_0)

        sns.set_theme(style="whitegrid")
        target_arr_0.hist(bins=9, figsize=(20, 15))
        # sns.histplot(target_arr_0, bins=10).set_title(self.dataset_name)
        # plt.title(self.dataset_name)
        plt.show()

    def save_data(self):
        if not os.path.isdir(config["output_path_gesamtdatensatz"]):
            os.mkdir(config["output_path_gesamtdatensatz"])
        os.chdir(config["output_path_gesamtdatensatz"])
        if not os.path.isdir("Variante_1"):
            os.mkdir("Variante_1")
        os.chdir("Variante_1")

        samples = pd.DataFrame(self.samples_gesamtdatensatz)
        targets = pd.DataFrame(self.targets_gesamtdatensatz)

        dataset = pd.concat([samples, targets], axis=1)

        print("Saving data")
        dataset.to_csv(self.save_name + config["savedataformat"], header=False, index=False)
        print("Saving finished!")


if __name__ == '__main__':
    main()
