import numpy as np
import numpy
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml
import matplotlib.pyplot as plt
import fnmatch
import sys
import imblearn
import seaborn as sns

# Open config file
with open(
    "03_data_preprocessing/03_5_preprocess_MdlPwr_BatPwr_kombi/03_5_Var_2_Zeitschritt/config_preprocess.yml",
    "r",
) as stream:
    config = yaml.safe_load(stream)


def main():
    # Instanz der Klasse
    concat = concat_data()

    # Laden der Datensätze
    # concat.load_datasets()

    # Lade bereits zusammengesetzten Datensatz
    concat.lade_Gesamtdatensatz()

    # Untersuchung der Daten auf die Häufigkeitsverteilung
    concat.analyze_data_channels()

    # Gleichverteilungsstrategien für den Datensatz
    # concat.balance_dataset()

    # Untersuche die Signale auf Min- und Maxwerte für die Normalisierung
    # concat.analyze_min_max_values_for_normalization()

    # Wähle Sample und Targetdatensatz und speichere diese als .csv
    # concat.save_dataset()


class concat_data:
    def __init__(self):
        self.array = []
        self.targets = []
        self.samples = []
        self.name = config["gesamt_dataset_name"]

    def lade_Gesamtdatensatz(self):
        path = config["output_path_gesamtdatensatz"]
        dataset_name = self.name + ".csv"
        # col_list = [i for i in range(1400, 2210)]
        self.array = pd.read_csv(
            os.path.join(path, dataset_name)
        )  # , usecols=col_list)
        # Split sample und targets
        self.targets = self.array.iloc[:, :1400]
        self.samples = self.array.iloc[:, 1400:]
        print("Samples shape: ", self.samples.shape)
        print("Targets shape: ", self.targets.shape)

    def load_datasets(self):
        # basis
        path = config["dataset_path"]
        # dataset_list = fnmatch.filter(os.listdir(path), '*.csv*')
        dataset_list = fnmatch.filter(os.listdir(path), "*.csv*")
        print(dataset_list)
        print("Anzahl an Datensätzen: ", len(dataset_list))

        # Iteration durch die Datensätze
        array = np.ones(shape=(10, config["spalten"]))
        # print(array.shape)
        for dataset in dataset_list:
            print(dataset)
            dataset_arr = np.array(
                pd.read_csv(os.path.join(path, dataset), delimiter=",")
            )
            array = np.append(array, dataset_arr, axis=0)
        array = array[10:, :]  # dummy spalten rauslöschen
        print(array.shape)
        self.array = array

        # Filter für die Targets --> 0:1400 Spalten
        target_arr_0 = array
        target_arr_0 = target_arr_0[:, 0:1399:200]
        col = ["CurBat", "StEdrv", "GrSt", "NEng", "VVeh", "TqEng", "T"]
        target_arr_0 = pd.DataFrame(data=target_arr_0, columns=col)
        print(target_arr_0)
        print(target_arr_0.describe())

        sns.set_theme(style="whitegrid")
        target_arr_0.hist(bins=9, figsize=(20, 15))
        # sns.histplot(target_arr_0, bins=10).set_title("Title")
        plt.title(self.name + " Histogram Targets")
        plt.show()

    def balance_dataset(self):
        # 1. Random Undersampling
        undersample = imblearn.under_sampling.RandomUnderSampler(
            sampling_strategy="majority", random_state=0
        )

        # 2. Kann nicht mit nans umgehen, daher fillnan = 0
        self.samples = self.samples.fillna(0)
        self.targets = self.targets.fillna(0)

        # 3. Convert to numpy
        self.samples = np.array(self.samples, dtype="float32")
        self.targets = np.array(self.targets, dtype="float32")
        X_over, y_over = undersample.fit_resample(
            X=self.samples, y=self.targets
        )  # Todo: Problem: Input contains NaN, infinity or a value too large for dtype('float64')
        print("Samples shape after Random Undersampling: ", np.array(X_over).shape)
        print("Targets shape after Random Undersampling: ", np.array(y_over).shape)

    def analyze_min_max_values_for_normalization(self):
        min_array = []
        max_array = []

        # targets
        for i in range(0, 1400, 200):
            target_array = np.array(self.array[:, i : i + 200])
            w_nans = numpy.isnan(
                target_array
            )  # filter die nans raus, vermasseln die Min- und Max Analyse
            target_array[w_nans] = 5
            # target_array[target_array==6553.5] = 0.1
            mini = np.min(target_array).round(4)
            maxi = np.max(target_array).round(4)

            min_array.append(mini)
            max_array.append(maxi)
            print("Shape: ", target_array.shape)
            print("Max: ", maxi)
            print("Min: ", mini)
        print(min_array)
        print(max_array)

        # Split target und sample
        sample_data = self.array[:, 1400:]

        # samples: constants 1 -> ESOC_SOC
        sample_const = np.array(sample_data[:, 0])
        mini = np.min(sample_const).round(4)
        maxi = np.max(sample_const).round(4)

        min_array.append(mini)
        max_array.append(maxi)
        print("ESOC_SOC")
        print("Shape: ", sample_const.shape)
        print("Max: ", maxi)
        print("Min: ", mini)

        # samples: array 2 --> VFinVal
        VFinVal = np.array(sample_data[:, 1:201])
        mini = np.min(VFinVal).round(4)
        maxi = np.max(VFinVal).round(4)

        min_array.append(mini)
        max_array.append(maxi)
        print("VFinVal")
        print("Shape: ", VFinVal.shape)
        print("Max: ", maxi)
        print("Min: ", mini)

        # samples: constants 3:7 --> 5
        print("HornLen bis AirP")
        for i in range(201, 205, 1):
            sample_values = np.array(sample_data[:, i])
            mini = np.min(sample_values).round(4)
            maxi = np.max(sample_values).round(4)
            min_array.append(mini)
            max_array.append(maxi)
            print("Shape: ", sample_values.shape)
            print("Max: ", maxi)
            print("Min: ", mini)

        # samples: array 8 --> SlopFinVal
        SlopFinVal = np.array(sample_data[:, 205:405])
        mini = np.min(SlopFinVal).round(4)
        maxi = np.max(SlopFinVal).round(4)

        min_array.append(mini)
        max_array.append(maxi)
        print("SlopFinVal")
        print("Shape: ", SlopFinVal.shape)
        print("Max: ", maxi)
        print("Min: ", mini)

        # samples: constants 9:12 -> 4
        print("AirTp bis Circumfer")
        for i in range(405, 409, 1):
            sample_values = np.array(sample_data[:, i])
            mini = np.min(sample_values).round(4)
            maxi = np.max(sample_values).round(4)
            min_array.append(mini)
            max_array.append(maxi)
            print("Shape: ", sample_values.shape)
            print("Max: ", maxi)
            print("Min: ", mini)

        # samples: 13-14 --> AFinVal & VSlopAFinPosn
        print("AFinVal & VSlopAFinPosn")
        for i in range(409, 610, 200):
            sample_values = np.array(sample_data[:, i : i + 200])
            mini = np.min(sample_values).round(4)
            maxi = np.max(sample_values).round(4)
            min_array.append(mini)
            max_array.append(maxi)
            print("Shape: ", sample_values.shape)
            print("Max: ", maxi)
            print("Min: ", mini)
        print(min_array)
        print(max_array)

        np.savetxt("norm_min_values_testfahrt.txt", min_array)
        np.savetxt("norm_max_values_testfahrt.txt", max_array)

    def analyze_data_channels(self):

        # 1. build for-loop for analyzing target data (hist)
        # for i in range(0, 1400, 200):
        #     self.array = []
        #     arr = self.targets.iloc[:, i:i+200]
        #     self.array = np.array(arr)
        #     print(self.array.shape)
        #     self.array = np.ravel(self.array)  # Flattening
        #     print(self.array.shape)
        #     fig, ax = plt.subplots(figsize=(20, 15))
        #     # bin_list = [i for i in range(1, 201, 10)]
        #     bin_list = 9
        #     sns.set_theme(style="whitegrid")
        #     sns.histplot(data=self.array, bins=bin_list, multiple="stack", ax=ax)  # discrete=True --> Jeder Wert wird angezeigt
        #     plt.xlabel("Geschwindigkeit (km/h)")
        #     plt.ylabel("Counts (-)")
        #     plt.title("Häufigkeitsverteilung VVeh (km/h) für den Gesamtdatensatz - alle Ereignisse [i]", size=16)
        #     plt.show()

        # 2. Single Target analyze
        # channel = "TAry (s)"
        # i = 1200
        # j = 1400
        # print(channel)
        # arr = self.targets.iloc[:, i:j]
        # arr = np.array(arr)
        # print(arr.shape)
        # arr = np.ravel(arr)
        # print(arr.shape)
        # fig, ax = plt.subplots(figsize=(20, 15))
        # bin_list = [i for i in range(0, 6554, 20)]
        # # bin_list = [-50, -25, 0, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 921]
        # # bin_list = [0, 0.5, 1]
        # # bin_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # # bin_list = 10
        # sns.set_theme(style="whitegrid")
        # sns.histplot(data=arr, bins=bin_list, multiple="stack", ax=ax)  # discrete=True --> Jeder Wert wird angezeigt
        # plt.xlabel(channel)
        # plt.ylabel("Anzahl (-)")
        # plt.title("Häufigkeitsverteilung " + channel + " für den Gesamtdatensatz - alle Ereignisse [i]", size=16)
        # ser = pd.Series(arr)
        # print(ser.describe())
        # plt.show()

        # 3. single samples analyze
        channel = "VSlopAFinPosn (m)"
        i = 609
        j = 809
        print(channel)
        arr = self.samples.iloc[:, i:j]  # Todo: Nur bei Arrays
        arr = np.array(arr)
        print(arr.shape)
        arr = np.ravel(arr)  # TODO: Nur bei Arrays
        print(arr.shape)
        fig, ax = plt.subplots(figsize=(20, 15))
        bin_list = [i for i in range(0, 7200, 100)]
        # bin_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3]
        # bin_list = [0, 0.5, 1]
        # bin_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # bin_list = 10
        sns.set_theme(style="whitegrid")
        sns.histplot(
            data=arr, bins=bin_list, multiple="stack", ax=ax
        )  # discrete=True --> Jeder Wert wird angezeigt
        # sns.set(font_scale=2.0)
        # g.set_yticklabels(g.get_ymajorticklabels(), fontsize=16)
        plt.xlabel(channel)
        plt.ylabel("Anzahl (-)")
        plt.title(
            "Häufigkeitsverteilung "
            + channel
            + " für den Gesamtdatensatz - alle Ereignisse [i]",
            size=20,
        )
        ser = pd.Series(arr)
        print(ser.describe())
        path = "/home/ase/Bilder/hist/hist_gesamtdatensatz_july/"
        plt.savefig(path + channel + ".png", dpi=400)
        plt.show()

        # GrSt = np.array(self.array[:, 400:600]) TODO: Alte (ungenaue) Version
        # print(GrSt.shape)
        # GrSt = GrSt[:, 0]
        # print(GrSt.shape)
        # col = ["GrSt (-)"]
        # df = pd.DataFrame(GrSt, columns=col)
        # df.hist(bins=9, figsize=(20, 15))
        # plt.show()

        # col = ["VVeh (km/h)"]
        # df1 = pd.DataFrame(VVeh, columns=col)
        # print(df1)
        # bin_liste = [i for i in range(1, 10, 201)]
        # df1.hist(bins=bin_liste, figsize=(20, 15))
        # plt.show()

    def save_dataset(self):
        final_df = pd.DataFrame(self.array)

        # create folders if necessary
        path_gesamtdatensatz = config["output_path_gesamtdatensatz"].split(
            "/Variante_2"
        )[0]
        if not os.path.isdir(path_gesamtdatensatz):
            os.mkdir(path_gesamtdatensatz)
        if not os.path.isdir(config["output_path_gesamtdatensatz"]):
            os.mkdir(config["output_path_gesamtdatensatz"])

        print("Saving the concated Dataset")
        final_df.to_csv(
            config["output_path_gesamtdatensatz"]
            + "/"
            + self.name
            + config["savedataformat"],
            header=False,
            index=False,
        )
        print("Finished saving!")


if __name__ == "__main__":
    # print(imblearn.__version__)  -> 0.8.0
    main()
