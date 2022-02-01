import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml
import matplotlib.pyplot as plt
import fnmatch
import sys

# Open config file
with open(
    "03_data_preprocessing/March21_MdlPwr_MdlBatPwr/config_concat_datasets.yml", "r"
) as stream:
    config = yaml.safe_load(stream)


def main():
    # Instanz der Klasse
    concat = concat_data()

    # Laden der Datensätze
    concat.load_datasets()


class concat_data:
    def __init__(self):
        pass

    def load_datasets(self):
        # basis
        path = config["dataset_path"]
        dataset_list = fnmatch.filter(os.listdir(path), "*Dataset*")
        print(dataset_list)
        print("Anzahl an Datensätzen: ", len(dataset_list))

        # Iteration durch die Datensätze
        for dataset in dataset_list:
            dataset_path = path + dataset
            sample_list = fnmatch.filter(os.listdir(dataset_path), "*sample*")
            label_list = fnmatch.filter(os.listdir(dataset_path), "*label*")
            print("Eingangssignale: ", len(sample_list))
            print("Ausgangssignale: ", len(label_list))
            timestep_len = 200  # TODO: Länge des Datensatzes auslesen
            sample_array = np.ones(shape=(timestep_len, config["array_len"]))
            for sample in sample_list:
                print(sample)
                single_sample_arr = np.array(
                    pd.read_csv(os.path.join(dataset_path, sample), delimiter=",")
                )
                sample_array = np.append(sample_array, single_sample_arr, axis=1)
            self.sample_array = sample_array
            print(self.sample_array.shape)
            self.sample_array = pd.DataFrame(self.sample_array)
            print(self.sample_array)


if __name__ == "__main__":
    main()
