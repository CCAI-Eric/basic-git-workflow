from asammdf import MDF
from asammdf import MDF4
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import tkinter as tk
from tkinter import messagebox
import sys

sys.path.append(os.getcwd())
from python_scripts.visualize_channel import visualize_csv

"""
working directory: /home/ase/Dokumente/eh_basics/masterarbeit_eh
Konfigurationsfile für den Converter: Mdl_Pwr.yml --> liegt im gleichen Verzeichnis
Mdl_Pwr.yml
"""

USE_POOL = False

if USE_POOL:  # Verteilung der Computing Power der CPU
    from multiprocessing import Pool


def main():
    # Dummy tkinter root window
    root = tk.Tk()
    root.withdraw()

    # generate_csv = messagebox.askyesno("Generate CSV?", "Should the csv files be generated?")
    # visualize = messagebox.askyesno("Visualize?", "Visualize channels?")

    converter = Mdf_to_CSV("03_data_preprocessing/03_2_converter/config_Converter.yml")

    for file in os.listdir(converter.input_path):
        path = os.path.join(converter.input_path, file)
        converter.mdf_to_csv(path)  # generate_csv


class Mdf_to_CSV:
    def __init__(self, config_path):
        """

        :param config_path:
        """
        self.load_configs(config_path)

    def load_configs(self, config_path):
        with open(config_path, "r") as y:
            config = yaml.safe_load(y)

        self.input_path = config["input_path"]
        self.output_path = config["output_path"]
        self.array_length = config["array_length"]
        self.values_length = config["values_length"]
        self.replace_nan = config["replace_to_nan"]

        self.signals = config["signals"]

        if not os.path.isdir(self.input_path):
            print("ERROR: Can't find input path: " + self.input_path)

        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)

    def mdf_to_csv(self, file_path, generate_csv=True, visualize=False):
        print("===============================")
        print("FILE: " + file_path)

        mdf0 = MDF(file_path, memory="minimum")
        df0 = mdf0.to_dataframe()

        # mdf = MDF("Testfahrt_Sindelfingen_03_03_21_filtered.mdf", memory='minimum')

        # mdf0.to_dataframe(**)
        # df0 = pd.DataFrame(mdf0.export(fmt='csv'))

        single_value_dfs_samples = [
            self.single_values_to_array_df(df0, sample_value)
            for sample_value in self.signals["samples"]["values"]
        ]

        arrays_dfs_samples = [
            self.array_to_df(df0, sample_array)
            for sample_array in self.signals["samples"]["arrays_200"]
        ]

        single_value_dfs_labels = [
            self.single_values_to_array_df(df0, label_value)
            for label_value in self.signals["labels"]["values"]
        ]

        arrays_dfs_labels = [
            self.array_to_df(df0, sample_array)
            for sample_array in self.signals["labels"]["arrays_200"]
        ]

        self.sample_dfs = single_value_dfs_samples + arrays_dfs_samples
        self.label_dfs = single_value_dfs_labels + arrays_dfs_labels

        csv_folder_name = file_path.split("/")[-1].replace(".mdf", "")  # TODO .mdf
        csv_folder_path = os.path.join(self.output_path, csv_folder_name)

        if not os.path.isdir(csv_folder_path):
            os.mkdir(csv_folder_path)

        if generate_csv:
            self.generate_separate_csvs(csv_folder_path, self.sample_dfs, "sample_")
            self.generate_separate_csvs(csv_folder_path, self.label_dfs, "label_")
            # self.generate_combined_csv(csv_folder_path, self.sample_dfs + self.label_dfs)

        # if visualize:
        #     for file_name in self.get_csv_paths(self.sample_dfs, "sample_"):
        #         if messagebox.askyesno(f"File: {csv_folder_path}", f"Visualize channel {file_name}?"):
        #             visualize_csv(os.path.join(csv_folder_path, file_name))

    def single_values_to_array_df(self, mdf_df, channel_name):
        if channel_name not in mdf_df:
            print("Channel doesn't exist in data frame: " + channel_name)
            return None

        if self.replace_nan:
            # Substitute values beyond the horizon with nan
            channel_indices = [channel_name + "[" + str(i) + "]" for i in range(200)]
            distances = mdf_df[self.signals["horizon_distance"]].values
            values = mdf_df[channel_name].values

            rows = []
            for i, d in enumerate(distances):
                row = np.full(self.array_length, values[i])
                row[int(d) :] = np.nan
                rows.append(row)

            return pd.DataFrame(rows, columns=channel_indices, index=mdf_df.index)
        else:
            df = pd.DataFrame()
            for i in range(self.values_length):
                df[channel_name + "[" + str(i) + "]"] = mdf_df[channel_name]
            return df

    def array_to_df(self, mdf_df, channel_name):
        if channel_name + "[0]" not in mdf_df:
            print("Channel doesn't exist in data frame: " + channel_name)
            return

        if self.replace_nan:
            # Substitute values beyond the horizon with nan
            channel_indices = [channel_name + "[" + str(i) + "]" for i in range(200)]
            distances = mdf_df[self.signals["horizon_distance"]].values
            values = mdf_df[channel_indices].values

            for i, d in enumerate(distances):
                row = values[i]
                row[int(d) :] = np.nan

            return pd.DataFrame(values, columns=channel_indices, index=mdf_df.index)
        else:
            # No substitution of values beyong the horizon length
            df = pd.DataFrame()
            for i in range(self.array_length):
                channel_index = channel_name + "[" + str(i) + "]"
                df[channel_index] = mdf_df[channel_index]
            return df

    def get_csv_paths(self, df_list, folder_path):
        return [
            folder_path + str(df.columns[0].split("[")[0]) + ".csv" for df in df_list
        ]

    def generate_separate_csvs(self, folder_path, df_list, identifier):

        names = self.get_csv_paths(df_list, os.path.join(folder_path, identifier))
        if USE_POOL:
            with Pool(len(df_list)) as p:
                p.starmap(self.write_csv, zip(df_list, names))
        else:
            for i, df in enumerate(df_list):
                self.write_csv(df, names[i])

    # def generate_combined_csv(self, folder_path, dfs):
    #     combined_df = pd.DataFrame()
    #     for df in dfs:
    #         values = df.values
    #         col_vector = values[np.logical_not(np.isnan(values))]
    #         combined_df[df.columns[0].split("[")[0]] = col_vector
    #
    #     # Extract indices (a,t coordiantes)
    #     i_idx = []  # array indices
    #     t_idx = []  # time indices
    #     for ti, row in enumerate(values):
    #         for ii, value in enumerate(row):
    #             if not np.isnan(value):
    #                 i_idx.append(ii)
    #                 t_idx.append(ti)
    #
    #     combined_df["t"] = np.array(t_idx)
    #     combined_df["i"] = np.array(i_idx)
    #
    #     self.write_csv(combined_df, os.path.join(folder_path, 'combined.csv'))

    def write_csv(self, df, file_path):
        if df is not None:
            df.to_csv(file_path, header=False, index=False)
            print("WROTE CSV: " + file_path)


if __name__ == "__main__":
    main()
