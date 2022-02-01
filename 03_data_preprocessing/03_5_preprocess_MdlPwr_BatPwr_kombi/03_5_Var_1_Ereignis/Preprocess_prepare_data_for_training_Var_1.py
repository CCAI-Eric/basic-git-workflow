import numpy as np
import numpy
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import yaml
import matplotlib.pyplot as plt
import fnmatch
import sys
import imblearn
import seaborn as sns

"""
1. Variante:
Vorverarbeitung für Eingangsdatenraum: 11 Inputs
Ausgangsdatenraum: 7
Daten hinter Horizontlänge werden zu NaN, dann entfernt und dadurch Matrix zu 1Dim-Vektor geflattet.
"""

# Open config file
with open(
    "03_data_preprocessing/03_5_preprocess_MdlPwr_BatPwr_kombi/03_5_Var_1_Ereignis/config_preprocess_var1.yml",
    "r",
) as stream:
    config = yaml.safe_load(stream)


def normalization_func(X_feature_max, X_feature_min, X_max, X_min, X):
    X_norm = ((X_feature_max - X_feature_min) / (X_max - X_min)) * (
        X - X_min
    ) + X_feature_min
    return X_norm


def main():
    # Instanz der Klasse
    prepare = prepare_data()

    # Lade bereits zusammengesetzten Datensatz
    prepare.lade_Gesamtdatensatz()

    prepare.create_hist()
    # Normalisiere die Datensätze mit eigener Funktion (Min- und Maxwerte analysiert)
    if config["normalization_on"]:
        prepare.normalize_data()

    # Führe OneHotEncodding für kategorische Targets aus (GrSt, StEdrv) TODO: Diese benötigen keine Normierung
    if config["one_hot_encodding_on"]:
        prepare.one_hot_encodding()

    # Splitte Daten in Trainings, Validierungs- und Testdatensatz
    if config["save_splitted_dataset"]:
        prepare.split_data()

    # Wähle Sample und Targetdatensatz und speichere diese als .csv
    # prepare.save_dataset()


class prepare_data:
    def __init__(self):
        self.array = []
        self.targets = []
        self.samples = []
        self.name = config["gesamt_name"]
        self.sample_signals = config["sample_signals"]
        self.targets_signals = config["target_signals"]
        self.target_cl = []
        self.target = []

    def lade_Gesamtdatensatz(self):
        path = config["input_directory"]
        dataset_name = self.name + ".csv"

        self.array = pd.read_csv(os.path.join(path, dataset_name), dtype="float32")
        print("Shape whole dataset: ", self.array.shape)
        a = self.array.shape[0]

        # check gear zeros todo: 1996 rows
        if config["check_gear_zeros"]:
            # print(self.array.iloc[:, self.sample_signals + 2].sample(30))  # Prüfung, ob richtige Spalte
            self.array.iloc[:, self.sample_signals + 2].replace(0, np.nan, inplace=True)
            self.array.dropna(axis=0, inplace=True)
            b = self.array.shape[0]
            print("\nShape difference after checking zeros at gear column: ", a - b)

        # Split sample und targets
        self.samples = self.array.iloc[:, : self.sample_signals]
        self.targets = self.array.iloc[:, self.sample_signals :]
        print("Samples shape: ", self.samples.shape)
        print("Targets shape: ", self.targets.shape)
        print(self.targets.describe())

        # # Ziehe Gang aus Dataframe
        # self.target_cl = self.targets.iloc[:, 2]
        # # print(self.target_cl.sample(10))
        # print(self.target_cl.shape)
        #
        # # Drop Gang aus altem Df
        # self.targets = self.targets.drop("0.0.6", axis=1)
        # print(self.targets.shape)
        # self.targets_signals = self.targets.shape[1]
        # print(self.targets.describe())

    def create_hist(self):
        channel = "Zeitschritte (s)"
        i = 6
        print(channel)
        arr = self.targets.iloc[:, i]  # Todo: Nur bei Arrays
        arr = np.array(arr)
        print(arr.shape)
        # arr = np.ravel(arr)  # TODO: Nur bei Arrays
        print(arr.shape)
        sns.set(font_scale=1.8)
        fig, ax = plt.subplots(figsize=(20, 15))
        # bin_list = [i for i in range(0, 7200, 100)]
        # bin_list = [i for i in range(0, 200, 5)]
        # bin_list = [-0.12, -0.1, -0.08, -0.06, -0.04, -0.04, -0.02, 0,  0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
        # bin_list = [0, 0.5, 1]
        # bin_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # bin_list = [-5, 0, 5, 10, 15, 20, 25]
        bin_list = [
            -50,
            -25,
            0,
            25,
            50,
            75,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            500,
            600,
            700,
            800,
            900,
        ]
        # bin_list = 10
        sns.set_theme(style="whitegrid")
        sns.histplot(
            data=arr,
            multiple="stack",
            ax=ax,
            stat="probability",
            binrange=[0, 6000],
            binwidth=50,
        )  # discrete=True --> Jeder Wert wird angezeigt , binrange=[0, 100]
        # g.set_yticklabels(g.get_ymajorticklabels(), fontsize=16)
        plt.xlabel(channel, fontsize=18)
        plt.ylabel("Anzahl (-)", fontsize=18)
        plt.title(
            "Häufigkeitsverteilung "
            + channel
            + " für den Gesamtdatensatz - alle Ereignisse [i]",
            size=22,
        )
        ser = pd.Series(arr)
        print(ser.describe())
        path = "/home/ase/Bilder/hist/hist_gesamtdatensatz_july/"
        plt.savefig(path + channel + ".png", dpi=400)
        # plt.savefig(path + "fahrzeuggeschwindigkeit" + ".png", dpi=400)
        plt.show()

    def split_data(self):
        # Concat Gang und restliche targets für Splitting
        # self.target = pd.concat([self.target_cl, self.targets], axis=1)
        self.target = np.concatenate((self.target_cl, self.targets), axis=1)
        self.target = pd.DataFrame(self.target)

        print("Shape targets für das Splitting: ", self.target.shape)

        # Splitte Daten in Trainings-, Val- und Testdatensatz
        train_ratio = 0.8
        validation_ratio = 0.1
        test_ratio = 0.1

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.samples,
            self.target,
            random_state=config["seed"],
            test_size=1 - train_ratio,
            shuffle=config["shuffle"],
        )
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_test,
            self.y_test,
            random_state=config["seed"],
            test_size=test_ratio / (test_ratio + validation_ratio),
            shuffle=False,
        )

        print("Train size: 80 %")
        print("Validation size: 10 %")
        print("Test size: 10 %")
        print("X_train shape: ", self.X_train.shape)
        print("X_val shape: ", self.X_val.shape)
        print("X_test shape: ", self.X_test.shape)
        print("y_train shape: ", self.y_train.shape)
        print("y_val shape: ", self.y_val.shape)
        print("y_test shape: ", self.y_test.shape)
        print("\n\nAnalyse der Verteilung der Daten nach dem Splitten: ")
        print(self.X_train.describe())
        print(self.X_val.describe())
        print(self.X_test.describe())

    def normalize_data(self):
        # feature range of normalization
        X_feature_max = config["normalization"]["feature_range"]["max"]
        X_feature_min = config["normalization"]["feature_range"]["min"]

        # Min. und Maxvalues of the targets
        min_list = [
            config["normalization"]["target"]["CurBat"]["Min"],
            config["normalization"]["target"]["StEdrv"]["Min"],
            config["normalization"]["target"]["NEng"]["Min"],
            config["normalization"]["target"]["VVeh"]["Min"],
            config["normalization"]["target"]["TqEng"]["Min"],
            config["normalization"]["target"]["TAry"]["Min"],
        ]

        max_list = [
            config["normalization"]["target"]["CurBat"]["Max"],
            config["normalization"]["target"]["StEdrv"]["Max"],
            config["normalization"]["target"]["NEng"]["Max"],
            config["normalization"]["target"]["VVeh"]["Max"],
            config["normalization"]["target"]["TqEng"]["Max"],
            config["normalization"]["target"]["TAry"]["Max"],
        ]

        targets_name_list = ["CurBat", "StEdrv", "NEng", "VVeh", "TqEng", "TAry"]

        # Normalization for the targets
        for i in range(self.targets_signals):
            print(targets_name_list[i])
            print(self.targets.iloc[:, i].head())
            self.targets.iloc[:, i] = normalization_func(
                X_feature_max,
                X_feature_min,
                max_list[i],
                min_list[i],
                self.targets.iloc[:, i],
            )
            print(self.targets.iloc[:, i].head())

        # Min. und Maxvalues of the samples
        min_list = [
            config["normalization"]["sample"]["ESOC_SOC"]["Min"],
            config["normalization"]["sample"]["VFinVal"]["Min"],
            config["normalization"]["sample"]["VSlopAFinVectLen"]["Min"],
            config["normalization"]["sample"]["CurbWeight"]["Min"],
            config["normalization"]["sample"]["BMS_Batt_Volt"]["Min"],
            config["normalization"]["sample"]["Tm_AmbAirP"]["Min"],
            config["normalization"]["sample"]["SlopFinVal"]["Min"],
            config["normalization"]["sample"]["Tm_AmbAirTp"]["Min"],
            config["normalization"]["sample"]["Cod_Diff_Ratio_Calc"]["Min"],
            config["normalization"]["sample"]["MVeh"]["Min"],
            config["normalization"]["sample"]["WhlPA_Circumfer"]["Min"],
            config["normalization"]["sample"]["AFinVal"]["Min"],
            config["normalization"]["sample"]["VSlopAFinPosn"]["Min"],
        ]

        max_list = [
            config["normalization"]["sample"]["ESOC_SOC"]["Max"],
            config["normalization"]["sample"]["VFinVal"]["Max"],
            config["normalization"]["sample"]["VSlopAFinVectLen"]["Max"],
            config["normalization"]["sample"]["CurbWeight"]["Max"],
            config["normalization"]["sample"]["BMS_Batt_Volt"]["Max"],
            config["normalization"]["sample"]["Tm_AmbAirP"]["Max"],
            config["normalization"]["sample"]["SlopFinVal"]["Max"],
            config["normalization"]["sample"]["Tm_AmbAirTp"]["Max"],
            config["normalization"]["sample"]["Cod_Diff_Ratio_Calc"]["Max"],
            config["normalization"]["sample"]["MVeh"]["Max"],
            config["normalization"]["sample"]["WhlPA_Circumfer"]["Max"],
            config["normalization"]["sample"]["AFinVal"]["Max"],
            config["normalization"]["sample"]["VSlopAFinPosn"]["Max"],
        ]

        # Normalization for the samples
        for i in range(self.sample_signals):
            print(self.samples.iloc[:, i].head())
            self.samples.iloc[:, i] = normalization_func(
                X_feature_max,
                X_feature_min,
                max_list[i],
                min_list[i],
                self.samples.iloc[:, i],
            )
            print(self.samples.iloc[:, i].head())

    def one_hot_encodding(self):
        one = OneHotEncoder(sparse=False)
        self.target_cl = np.array(self.target_cl)
        self.target_cl = self.target_cl.reshape(-1, 1)
        self.target_cl = one.fit_transform(self.target_cl)
        print(self.target_cl.shape)
        print(one.categories_)
        self.target_cl = pd.DataFrame(self.target_cl)
        # self.cl_classes = one.categories_

    def create_coordinates_t_i(self):
        pass
        # Variables for creating coor
        # VSlopAFinVectLen = horzn_len.to_numpy()  # TODO
        # timestep_len = len(VSlopAFinVectLen)
        #
        # # Erzeugung der Spalten für i und t für die Zuordnung (Koordinaten)
        # coordinates_array = []
        # for t in range(timestep_len):
        #     horizon_len = int(VSlopAFinVectLen[t, 0])
        #     if horizon_len >= 2:
        #         for i in range(horizon_len):  # -1
        #             coordinates_array.append([int(i), int(t)])
        #
        # coordinates_array = np.array(coordinates_array)  # TODO: Überprüfe die Shape 3521877
        # print(coordinates_array.shape)
        # print(coordinates_array[:50, :50])
        # self.coords = pd.DataFrame(coordinates_array, columns=["t", "i"])
        # print(self.coords.head())

    def save_dataset(self):
        # Save data in new directory
        os.chdir(config["input_directory"])
        if not os.path.isdir(config["output_directory"]):
            os.mkdir(config["output_directory"])
        os.chdir(config["output_directory"])
        print("Saving data!")
        if config["save_splitted_dataset"]:
            self.X_train.to_csv(
                config["X_train"] + config["savedataformat"], header=False, index=False
            )
            self.X_val.to_csv(
                config["X_val"] + config["savedataformat"], header=False, index=False
            )
            self.X_test.to_csv(
                config["X_test"] + config["savedataformat"], header=False, index=False
            )
            self.y_train.to_csv(
                config["y_train"] + config["savedataformat"], header=False, index=False
            )
            self.y_val.to_csv(
                config["y_val"] + config["savedataformat"], header=False, index=False
            )
            self.y_test.to_csv(
                config["y_test"] + config["savedataformat"], header=False, index=False
            )
        else:
            # save whole dataset without shuffle and splitting
            self.targets = pd.DataFrame(self.targets)
            self.samples = pd.DataFrame(self.samples)
            self.targets.to_csv(
                config["targets"] + config["savedataformat"], header=False, index=False
            )
            self.samples.to_csv(
                config["samples"] + config["savedataformat"], header=False, index=False
            )
        print("Skript vollständig ausgeführt!")


if __name__ == "__main__":
    # print(imblearn.__version__)  -> 0.8.0
    main()
