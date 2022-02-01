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

"""
###########################################################
#  working directory: eh_masterarbeit
print(os.getcwd())

# Open config file
with open(
    "03_data_preprocessing/03_5_preprocess_MdlPwr_BatPwr_kombi/03_5_Var_2_Zeitschritt/config_preprocess.yml",
    "r",
) as stream:
    config = yaml.safe_load(stream)


def main():
    # Instanz der Klasse
    preprocess = Preprocessing()

    # Lade die .csv Rohdaten und fülle alles nach horizon lenght mit nan
    preprocess.load_csv_data()

    # preprocess.create_hist()

    # Concat sample und label data & speichern
    preprocess.concat_sample_and_target()


class Preprocessing:
    def __init__(self):
        self.scaled_train_samples = []
        self.scaled_train_targets = []
        self.scaled_test_samples = []
        self.scaled_test_targets = []
        self.dataset_name = None

    def load_csv_data(self):
        path = config["data_path"]
        horizon_path = path + "/" + config["horizon_signal"]
        horizon_len = pd.read_csv(horizon_path, delimiter=",")
        event_len = config["array_len"]  # 200
        timestep_len = horizon_len.shape[0]
        horizon_len = horizon_len.iloc[:, 0].values  # numpy array
        # self.dataset_name = path.split("/")[1]
        self.dataset_name = config["dataset_name"]
        print(self.dataset_name)
        print("Timestep Length: ", timestep_len)

        # Listen der sample & target Daten
        label_list = fnmatch.filter(os.listdir(path), "*label*")
        sample_list = fnmatch.filter(os.listdir(path), "*sample*")
        print("Eingangssignale: ", len(sample_list))
        print("Ausgangssignale: ", len(label_list))

        sample_array = np.ones(
            shape=(timestep_len, event_len)
        )  # dummy array für append funktion
        print("Shape (time_step_len, event_len): ", sample_array.shape)

        for sample in sample_list:
            temp = np.array(pd.read_csv(os.path.join(path, sample), delimiter=","))
            # print(temp)
            print(sample)
            # print(np.shape(temp))

            if temp.shape[1] > 1:
                # TODO: Datenbearbeitung nach Horizon Length --> Werte auf 0 setzen!
                rows = temp
                rows = []
                for i, d in enumerate(horizon_len):  # +1
                    row = np.full(event_len, temp[i])
                    row[int(d) :] = np.nan  # int(-1) np.nan
                    rows.append(row)
                rows = np.array(rows)
                # print(rows.shape)
                # print(rows[11:14, :5])
            else:
                rows = temp
            sample_array = np.append(sample_array, rows, axis=1)

        self.sample_array = sample_array[:, 200:]
        print(self.sample_array.shape)
        self.sample_array = pd.DataFrame(self.sample_array)

        target_array = np.ones(
            shape=(timestep_len, event_len)
        )  # dummy array für append funktion
        for label in label_list:
            temp = np.array(pd.read_csv(os.path.join(path, label), delimiter=","))
            # print(temp)
            print(label)
            # print(np.shape(temp))

            rows = temp
            rows = []
            for i, d in enumerate(horizon_len):  # +1
                row = np.full(event_len, temp[i])
                row[
                    int(d) :
                ] = np.nan  #  TODO: targets werden nicht durch horizon_len beschränkt
                rows.append(row)
            rows = np.array(rows)
            print(rows.shape)
            # print(rows[11:14, :5])
            target_array = np.append(target_array, rows, axis=1)

        self.target_array = target_array[:, 200:]
        # print(self.target_array.shape)
        self.target_array = pd.DataFrame(self.target_array)

        print(self.target_array)

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

    def concat_sample_and_target(self):
        data_df = pd.concat([self.target_array, self.sample_array], axis=1)
        print(data_df)
        if not os.path.isdir(config["output_path"]):
            os.mkdir(config["output_path"])
        add = ""
        print("Saving dataset")
        data_df.to_csv(
            config["output_path"] + self.dataset_name + add + config["savedataformat"],
            header=False,
            index=False,
        )
        print("Finished saving!")


if __name__ == "__main__":
    main()
