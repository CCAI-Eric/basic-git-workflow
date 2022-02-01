import pandas as pd
import os
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import matplotlib.pyplot as plt


def header(msg):  # Formatvorlage für Printbefehle zur besseren Übersicht in der Console
    print("-" * 50)
    print(" [ " + msg + " ]")


class Equal_distribution:
    def __init__(self):
        self.full_dataset = []

    def load_csv_data(self):
        os.chdir("../02_dataset/current")
        self.full_dataset = genfromtxt(
            "Full_concat_dataset_[0]_MdlPwr.csv", delimiter=","
        )
        print(np.shape(self.full_dataset))

    def shuffle_dataset(self):
        self.full_dataset = shuffle(
            self.full_dataset, random_state=0
        )  # shuffle, aber nur die Spalten

    def Tq_Eng(self):
        data_pos = []
        data_neg = []

        # for i in range(len(self.full_dataset)):
        print(self.full_dataset[:, 14])

    def CurbWeight_input(self):

        dataset_class1 = []
        dataset_class2 = []

        print(self.full_dataset[:, 7])
        for i in range(len(self.full_dataset)):
            if self.full_dataset[i, 7] < 2460:
                dataset_class1.append(self.full_dataset[i, :])
            else:
                dataset_class2.append(self.full_dataset[i, :])

        dataset_class1 = np.array(dataset_class1)
        dataset_class2 = np.array(dataset_class2)

        print(np.shape(dataset_class1))
        print(np.shape(dataset_class2))

        curbweight_dataset = []
        for i in range(len(dataset_class1)):
            curbweight_dataset.append(dataset_class1[i])
            curbweight_dataset.append(dataset_class2[i])

        print(np.shape(curbweight_dataset))


if __name__ == "__main__":
    equal = Equal_distribution()

    equal.load_csv_data()
    equal.shuffle_dataset()
    equal.Tq_Eng()


#
# # Lese csv-Datei ein
# os.chdir("csv")
# MdlPwr_input = pd.read_csv('sample_[0]_MdlPwr300_20a_7.1_2020-05-04.csv')
# MdlPwr_output = pd.read_csv('label_[0]_MdlPwr300_20a_7.1_2020-05-04.csv')
#
# MdlPwr_input.columns = ['AitmCmn_MVeh', 'AitmContnsHorzn_AFinValAry200[0] ', 'AitmContnsHorzn_SlopFinValAry200[0]', 'AitmContnsHorzn_VFinValAry200[0]', 'AitmContnsHornz_VSlopAFinPosnAry200[0]', 'AitmContnsHornz_VSlopAFinVectLen', 'Cod_Diff_Ratio_Calc', 'CurbWeight', 'Tm_AmbAirP', 'Tm_AmbAirTp', 'WhlPA_Circumfer']
# MdlPwr_input.hist(bins=30, figsize=(20,15))
# plt.show()
# # n_input_neuronen = MdlPwr_input.shape[1]  # gives number of col count
# # n_output_neuronen = MdlPwr_output.shape[1]
# # print("Aktueller Datensatz für " + str(n_input_neuronen) + " Inputneuronen & " + str(n_output_neuronen) + " Outputneuronen")
# #print(MdlPwr_output.head())
#
# # Columns for Analyse
# #add columns
# # MdlPwr_output.columns = ['GrStAry200[0]', 'NEngAry200[0]', 'TAry200[0]', 'TqEngAry200[0]', 'VTqGrNMaxAVectLen', 'VVehAry200[0]']
# # MdlPwr_output.hist(bins=30, figsize=(20,15))
# # # MdlPwr_output = MdlPwr_output['0.0.3']
# # # MdlPwr_output = MdlPwr_output['0.0.3']
# # # MdlPwr_output.hist(bins=30, figsize=(20,15))
# # plt.show()
#
# # del MdlPwr_output['0.0.4'] # VTqGrNMaxAVectLen
# # del MdlPwr_output['0.0.2'] # TAry
# # del MdlPwr_output['0.0'] # Gang --> GrStAry200
#
#
# MdlPwr = pd.concat([MdlPwr_input, MdlPwr_output], axis=1)
# # print(MdlPwr.shape)
# # print(MdlPwr.head())
# MdlPwr.hist(bins=30, figsize=(20,15))
# # plt.show()
#
# os.chdir('../dataset')
# full_dataset = np.array(MdlPwr)
# print(np.shape(full_dataset))
#
# # print(full_dataset[:, 12])
# # np.savetxt("full_dataset_MdlPwr300_20a_7.1.csv", full_dataset, delimiter=",")
#
# dataset_pos = []
# dataset_neg = []
#
# # Trenne positive und negative Werte des Drehmomentes [i,12] --> TqEng
# # print(full_dataset[:, 14])
# for i in range(len(full_dataset)):
#     if full_dataset[i, 14] > 0:
#         dataset_pos.append(full_dataset[i,:])
#     else:
#         dataset_neg.append(full_dataset[i,:])
#
# dataset_neg = np.array(dataset_neg)
# dataset_pos = np.array(dataset_pos)
#
# # print(np.dataset_pos)
# print(len(dataset_pos))
# print(len(dataset_neg))
#
# # Fülle Array mit 50 % positiven und negativen Werten
#
#
# """
# Man sollte die Werte noch vorher shuffeln!
# """
#
# # Shuffle
# # np.savetxt("full_dataset_neg_MdlPwr300_20a_7.1.csv", dataset_neg, delimiter=",")#
# from sklearn.utils import shuffle
# dataset_neg = shuffle(dataset_neg, random_state=0) # shuffle, aber nur die Spalten!
# # np.savetxt("full_dataset_neg_shuffle_MdlPwr300_20a_7.1.csv", dataset_neg, delimiter=",")
#
# train_samples = []
# train_labels= []
#
# n = 11 # Trennung zwischen Input und Output TODO
#
# for i in range(len(dataset_pos)):
#     train_labels.append(dataset_pos[i, n:]) # 12
#     train_labels.append(dataset_neg[i, n:]) # 12
#     train_samples.append(dataset_pos[i, :n])
#     train_samples.append(dataset_neg[i, :n])
#
# print(len(train_labels))
# print(len(train_samples))
# print(np.shape(train_labels))
# print(np.shape(train_samples))
#
#
#
# os.chdir("..")
#
# # train_labels = []
# # train_samples  = []
# #
# # temp = np.array(MdlPwr_input)
# # train_samples= temp[:,:] # X = Anzahl der Traingswerte , Y = Anzahl der Kanäle /Eingangsneuronen
# # train_labels = np.array(MdlPwr_output)
#
#
# """
# # print(np.shape(train_samples))
# # print(np.shape(train_labels))
# # print(np.shape(scaled_train_samples))
# # print(np.shape(scaled_train_labels))
# """
# # Renormierung Test
# # unscaled_train_samples = scaler.inverse_transform(scaled_train_samples)
# # print(str(unscaled_train_samples)+ " Ausgabe")
#
#
# # Split data into training (training and validation) and test data
# # 0.0021 für 100 bei neuem datensatz
# unscaled_train_samples, unscaled_test_samples, unscaled_train_labels, unscaled_test_labels = train_test_split(train_samples,train_labels, test_size = 0.01331, random_state= 43, shuffle=True)
# # unscaled_train_samples, unscaled_val_samples, unscaled_train_labels, unscaled_val_labels = train_test_split(unscaled_train_samples, unscaled_train_labels, test_size = 0.0022, random_state=43,  shuffle=False)
# # random_state: int or RandomState instance; Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
# # shuffle bool, default=True; Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None
# #print(unscaled_train_samples)
# os.chdir('dataset')
#
# print(len(unscaled_train_labels))
#
#
# # #Save dataset for testing
# # kanal_name = "AitmContnsHorzn_VTqGrNMaxAVectLen"
# # kanal_name = '[0]_6outs_50%_filter'
# # np.savetxt("us_test_samples_" + kanal_name + "_MdlPwr300_20a_7.1.csv", unscaled_test_samples, delimiter=",")
# # np.savetxt("us_train_samples_" + kanal_name + "_MdlPwr300_20a_7.1.csv", unscaled_train_samples, delimiter=",")
# # np.savetxt("us_train_labels_" + kanal_name + "_MdlPwr300_20a_7.1.csv", unscaled_train_labels, delimiter=",")
# # np.savetxt("us_test_labels_" + kanal_name + "_MdlPwr300_20a_7.1.csv", unscaled_test_labels, delimiter=",")
# #
#
# #header("Testdaten [samples, labels] \n 1. Trainingsdaten \n 2. Validierungsdaten \n 3. Testdaten")
# # print(np.shape(unscaled_train_samples))
# # print(np.shape(unscaled_train_labels))
# # # print(np.shape(unscaled_val_samples))
# # # print(np.shape(unscaled_val_labels))
# # print(np.shape(unscaled_test_samples))
# # print(np.shape(unscaled_test_labels))
#
# """
# prinfull_datasett(MdlPwr_output)
# print(MdlPwr_output.shape)
#
# np.TqNeg = []
# np.TqPos = []
# np.TqEng = []
# for index, row in MdlPwr_output.iterrows():
#     #print(row["Speed"])
#     np.TqEng.append(row["0.0.3"])
#
# for i in range(len(np.TqEng)):
#     if np.TqEng[i] > 0:
#         np.TqPos.append(np.TqEng[i])
#     else:
#         np.TqNeg.append(np.TqEng[i])
#
# print(np.TqPos)
# print(len(np.TqPos))
# print(len(np.TqNeg))
#
# proportion = (100/(len(np.TqEng))*len(np.TqPos))
# print('Das Verhältnis der Positiven zu Negative Zahlen bei TqEng liegt bei:', proportion.__round__(2), '%.')
#
# ave = np.mean(np.TqPos[:])
# print('Average', ave)
#
# train_samples = []
# temp = np.array(MdlPwr_input)
# train_samples_1= temp[:,:]
# train_labels = []
# for i in range(len(np.TqPos)):
#     train_labels.append(np.TqPos[i])
#     train_labels.append(np.TqNeg[i])
#     train_samples.append(train_samples_1[i, :])
#     train_samples.append(train_samples_1[i+len(np.TqPos), :])
# print(len(train_labels))
# """
