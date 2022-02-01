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
2. Variante:
Vorverarbeitung für Eingangsdatenraum: 809 Inputs
Ausgangsdatenraum: 1400
Alle Ereignisse [i] eines Zeitschrittes [t] werden gleichzeitig berechnet
"""
# Open config file
with open(
    "03_data_preprocessing/03_5_preprocess_MdlPwr_BatPwr_kombi/03_5_Var_2_Zeitschritt/config_preprocess.yml",
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

    # Überprüfe Datensatz im Bereich des Ganges auf Nullen und entferne diese
    prepare.check_zeros_gear()

    # Normalisiere die Datensätze mit eigener Funktion (Min- und Maxwerte analysiert)
    if config["normalization_on"]:
        prepare.normalize_data()

    # Fülle die verbliebenen NaNs mit Null
    prepare.fill_nans_with_zeros()

    # Splitte Daten in Trainings, Validierungs- und Testdatensatz
    if config["save_splitted_dataset"]:
        prepare.split_data()

    # Führe OneHotEncodding für kategorische Targets aus (GrSt, StEdrv) TODO: Diese benötigen keine Normierung
    # prepare.one_hot_encodding()

    # Wähle Sample und Targetdatensatz und speichere diese als .csv
    prepare.save_dataset()


class prepare_data:
    def __init__(self):
        self.array = []
        self.targets = []
        self.samples = []
        self.name = config["name_gesamtdatensatz"]

    def lade_Gesamtdatensatz(self):
        path = config["input_path_gesamtdatensatz"]
        dataset_name = self.name + ".csv"
        self.array = pd.read_csv(os.path.join(path, dataset_name), dtype="float32")
        self.array = self.array.iloc[:, :]
        # Split sample und targets
        # self.targets = self.array.iloc[:, :1400]
        # self.samples = self.array.iloc[:, 1400:]
        # print("Samples shape: ", self.samples.shape)
        # print("Targets shape: ", self.targets.shape)
        # print(self.targets.describe())

    def check_zeros_gear(self):
        print(self.array.shape)
        a = self.array.shape[0]
        # Ändere nans im gesamten Datensatz zu Ersatzwert 100000
        ersatzwert = 999999
        self.array.fillna(ersatzwert, inplace=True)
        # Die Nullen im Bereich des Ganges werden zu NaN
        print(self.array.iloc[:, 400:600].sample(50))
        self.array.iloc[:, 400:600].replace(0, np.nan, inplace=True)
        # Droppe die Zeilen mit mindestens einem NaN
        self.array.dropna(axis=0, inplace=True)
        self.array = self.array.replace(ersatzwert, np.nan)

        # print(self.array.iloc[:, 400:600] < 1)
        print(self.array.shape)
        b = self.array.shape[0]
        print("Differenz der Zeilen nach Vorverarbeitung 'Check zeros gear': ", a - b)

        self.targets = self.array.iloc[:, :1400]
        self.samples = self.array.iloc[:, 1400:]
        print("Samples shape: ", self.samples.shape)
        print("Targets shape: ", self.targets.shape)

    def fill_nans_with_zeros(self):
        self.samples = self.samples.fillna(0)
        self.targets = self.targets.fillna(0)

    def split_data(self):
        # Splitte Daten in Trainings-, Val- und Testdatensatz
        train_ratio = 0.8
        validation_ratio = 0.1
        test_ratio = 0.1

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.samples,
            self.targets,
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

        # 1. CurBat
        print("CurBat")
        X_max = config["normalization"]["target"]["CurBat"]["Max"]
        X_min = config["normalization"]["target"]["CurBat"]["Min"]
        i = 0
        j = 200
        print(self.targets.iloc[:, i:j].describe())
        self.targets.iloc[:, i:j] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.targets.iloc[:, i:j]
        )
        print(self.targets.iloc[:, i:j].describe())

        # 2. NEng
        print("NEng")
        i = 600
        j = 800
        X_max = config["normalization"]["target"]["NEng"]["Max"]
        X_min = config["normalization"]["target"]["NEng"]["Min"]
        print(self.targets.iloc[:, i:j].describe())
        self.targets.iloc[:, i:j] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.targets.iloc[:, i:j]
        )
        print(self.targets.iloc[:, i:j].describe())

        # 3. VVeh
        print("VVeh")
        i = 800
        j = 1000
        X_max = config["normalization"]["target"]["VVeh"]["Max"]
        X_min = config["normalization"]["target"]["VVeh"]["Min"]
        print(self.targets.iloc[:, i:j].describe())
        self.targets.iloc[:, i:j] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.targets.iloc[:, i:j]
        )
        print(self.targets.iloc[:, i:j].describe())

        # 4. TqEng
        print("TqEng")
        i = 1000
        j = 1200
        X_max = config["normalization"]["target"]["TqEng"]["Max"]
        X_min = config["normalization"]["target"]["TqEng"]["Min"]
        print(self.targets.iloc[:, i:j].describe())
        self.targets.iloc[:, i:j] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.targets.iloc[:, i:j]
        )
        print(self.targets.iloc[:, i:j].describe())

        # 5. TAry
        print("TAry")
        i = 1200
        j = 1400
        X_max = config["normalization"]["target"]["TAry"]["Max"]
        X_min = config["normalization"]["target"]["TAry"]["Min"]
        print(self.targets.iloc[:, i:j].describe())
        self.targets.iloc[:, i:j] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.targets.iloc[:, i:j]
        )
        print(self.targets.iloc[:, i:j].describe())

        # 6. GrSt
        print("GrSt")
        i = 400
        j = 600
        X_max = config["normalization"]["target"]["GrSt"]["Max"]
        X_min = config["normalization"]["target"]["GrSt"]["Min"]
        print(self.targets.iloc[:, i:j].describe())
        self.targets.iloc[:, i:j] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.targets.iloc[:, i:j]
        )
        print(self.targets.iloc[:, i:j].describe())

        """Sample data"""
        # 7. ESOC_SOC
        print("ESOC_SOC")
        i = 0
        X_max = config["normalization"]["sample"]["ESOC_SOC"]["Max"]
        X_min = config["normalization"]["sample"]["ESOC_SOC"]["Min"]
        print(self.samples.iloc[:, i].describe())
        self.samples.iloc[:, i] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i]
        )
        print(self.samples.iloc[:, i].describe())

        # 8. VFinVal
        print("VFinVal")
        i = 1
        j = 201
        X_max = config["normalization"]["sample"]["VFinVal"]["Max"]
        X_min = config["normalization"]["sample"]["VFinVal"]["Min"]
        print(self.samples.iloc[:, i:j].describe())
        self.samples.iloc[:, i:j] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i:j]
        )
        print(self.samples.iloc[:, i:j].describe())

        # 9. VSlopAFinVectLen
        print("VSlopAFinVectLen")
        i = 201
        X_max = config["normalization"]["sample"]["VSlopAFinVectLen"]["Max"]
        X_min = config["normalization"]["sample"]["VSlopAFinVectLen"]["Min"]
        print(self.samples.iloc[:, i].describe())
        self.samples.iloc[:, i] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i]
        )
        print(self.samples.iloc[:, i].describe())

        # 10. CurbWeight
        print("CurbWeight")
        i = 202
        X_max = config["normalization"]["sample"]["CurbWeight"]["Max"]
        X_min = config["normalization"]["sample"]["CurbWeight"]["Min"]
        print(self.samples.iloc[:, i].describe())
        self.samples.iloc[:, i] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i]
        )
        print(self.samples.iloc[:, i].describe())

        # 11. BMS_Batt_Volt
        print("BMS_Batt_Volt")
        i = 203
        X_max = config["normalization"]["sample"]["BMS_Batt_Volt"]["Max"]
        X_min = config["normalization"]["sample"]["BMS_Batt_Volt"]["Min"]
        print(self.samples.iloc[:, i].describe())
        self.samples.iloc[:, i] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i]
        )
        print(self.samples.iloc[:, i].describe())

        # 12. Tm_AmbAirP
        print("Tm_AmbAirP")
        i = 204
        X_max = config["normalization"]["sample"]["Tm_AmbAirP"]["Max"]
        X_min = config["normalization"]["sample"]["Tm_AmbAirP"]["Min"]
        print(self.samples.iloc[:, i].describe())
        self.samples.iloc[:, i] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i]
        )
        print(self.samples.iloc[:, i].describe())

        # 13. SlopFinVal
        print("SlopFinVal")
        i = 205
        j = 405
        X_max = config["normalization"]["sample"]["SlopFinVal"]["Max"]
        X_min = config["normalization"]["sample"]["SlopFinVal"]["Min"]
        print(self.samples.iloc[:, i:j].describe())
        self.samples.iloc[:, i:j] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i:j]
        )
        print(self.samples.iloc[:, i:j].describe())

        # 14. Tm_AmbAirTp
        print("Tm_AmbAirTp")
        i = 405
        X_max = config["normalization"]["sample"]["Tm_AmbAirTp"]["Max"]
        X_min = config["normalization"]["sample"]["Tm_AmbAirTp"]["Min"]
        print(self.samples.iloc[:, i].describe())
        self.samples.iloc[:, i] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i]
        )
        print(self.samples.iloc[:, i].describe())

        # 14. Cod_Diff_Ratio_Calc
        print("Cod_Diff_Ratio_Calc")
        i = 406
        X_max = config["normalization"]["sample"]["Cod_Diff_Ratio_Calc"]["Max"]
        X_min = config["normalization"]["sample"]["Cod_Diff_Ratio_Calc"]["Min"]
        print(self.samples.iloc[:, i].describe())
        self.samples.iloc[:, i] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i]
        )
        print(self.samples.iloc[:, i].describe())

        # 15. MVeh
        print("MVeh")
        i = 407
        X_max = config["normalization"]["sample"]["MVeh"]["Max"]
        X_min = config["normalization"]["sample"]["MVeh"]["Min"]
        print(self.samples.iloc[:, i].describe())
        self.samples.iloc[:, i] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i]
        )
        print(self.samples.iloc[:, i].describe())

        # 16. WhlPA_Circumfer
        print("WhlPA_Circumfer")
        i = 408
        X_max = config["normalization"]["sample"]["WhlPA_Circumfer"]["Max"]
        X_min = config["normalization"]["sample"]["WhlPA_Circumfer"]["Min"]
        print(self.samples.iloc[:, i].describe())
        self.samples.iloc[:, i] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i]
        )
        print(self.samples.iloc[:, i].describe())

        # 17. AFinVal
        print("AFinVal")
        i = 409
        j = 609
        X_max = config["normalization"]["sample"]["AFinVal"]["Max"]
        X_min = config["normalization"]["sample"]["AFinVal"]["Min"]
        print(self.samples.iloc[:, i:j].describe())
        self.samples.iloc[:, i:j] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i:j]
        )
        print(self.samples.iloc[:, i:j].describe())

        # 18. VSlopAFinPosn
        print("VSlopAFinPosn")
        i = 609
        j = 809
        X_max = config["normalization"]["sample"]["VSlopAFinPosn"]["Max"]
        X_min = config["normalization"]["sample"]["VSlopAFinPosn"]["Min"]
        print(self.samples.iloc[:, i:j].describe())
        self.samples.iloc[:, i:j] = normalization_func(
            X_feature_max, X_feature_min, X_max, X_min, self.samples.iloc[:, i:j]
        )
        print(self.samples.iloc[:, i:j].describe())

    def one_hot_encodding(self):
        pass

    def save_dataset(self):
        # Save data in new directory
        os.chdir(config["input_path_gesamtdatensatz"])
        if not os.path.isdir(config["output_directory"]):
            os.mkdir(config["output_directory"])
        os.chdir(config["output_directory"])
        print("Saving data")
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
