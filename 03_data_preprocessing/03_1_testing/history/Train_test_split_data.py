from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import os
import numpy as np

class Split_data():
    def __init__(self):
        self.len_train_data = 100
        self.full_dataset = []
        self.sindelfingen_dataset = []
        self.gaertringen_dataset = []
        self.werk_dataset = []
        self.huettlingen_dataset = []
        self.data = []
        self.samples = []
        self.ground_truth = []
        self.train_samples = []
        self.full_dataset_equal = []

    def import_data(self):
        """
        Change into the right directory & read .csv-files of the different datasets
        """
        os.chdir("/home/ase/Dokumente/eh_basics/masterarbeit_eh/02_dataset/current") # TODO
        self.full_dataset = genfromtxt("Full_concat_dataset_[0]_MdlPwr.csv", delimiter=',')
        self.full_dataset_equal = genfromtxt("Full_concat_dataset_[0]_equal_TqEng_MdlPwr.csv", delimiter=',')
        self.sindelfingen_dataset = genfromtxt("AitmContsHorzn_MdlPwr300_[0]_Sindelfingen.csv", delimiter=',')
        self.gaertringen_dataset = genfromtxt("AitmContsHorzn_MdlPwr300_[0]_Gaertringen.csv", delimiter=',')
        self.werk_dataset = genfromtxt("AitmContsHorzn_MdlPwr300_[0]_Werk.csv", delimiter=',')
        self.huettlingen_dataset = genfromtxt("AitmContsHorzn_MdlPwr300_[0]_Huettlingen.csv", delimiter=',')

    def choose_dataset(self, dataset):
        """
        case distinction for which dataset should be used
        :param dataset:
        """
        if dataset == "full_dataset":
            self.data = self.full_dataset
        elif dataset == "sindelfingen_dataset":
            self.data = self.sindelfingen_dataset
        elif dataset == "gaertringen_dataset":
            self.data = self.gaertringen_dataset
        elif dataset == "werk_dataset":
            self.data = self.werk_dataset
        elif dataset == "huettlingen_dataset":
            self.data = self.huettlingen_dataset
        elif dataset == "full_dataset_gleichverteilt_tqeng":
            self.data = self.full_dataset_equal

    def choose_data_channels(self):
        """
        Choose which Inputchannel and Outputchannel you want to take
        & at the moment we are using all 11 Inputchannel & 3 Outputchannel (TqEng, NEng, VVeh)
        """

        # Split sample & ground truth
        self.samples = self.data[:,:11]
        self.ground_truth = self.data[:,11:]

        self.samples = self.samples  # wir nehmen aktuell alle 11 Inputs

        self.ground_truth = np.delete(self.ground_truth, 4, 1)  # del column 4 --> VectLen
        self.ground_truth = np.delete(self.ground_truth, 2, 1)  # del column 2 --> TAry
        self.ground_truth = np.delete(self.ground_truth, 0, 1)  # del column 0 --> GrSt

        print(np.shape(self.samples))
        print(np.shape(self.ground_truth))


    def split_data(self):
        # mit Train test split der sklearn bib --> Validation / Train und Testdata erstellen
        self.train_samples, self.test_samples, self.train_ground_truth, self.test_ground_truth = train_test_split(self.samples, self.ground_truth, test_size=self.len_train_data/len(self.samples),random_state=43, shuffle=True)
        # random state --> for reproducible output

        # return self.train_samples, self.test_samples, self.train_ground_truth, selftest_ground_truth

    def split_data_with_val(self):
        print("len:" + str(len(self.samples)))
        train_samples, test_samples, train_ground_truth, test_ground_truth = train_test_split(self.samples, self.ground_truth,
                                                                                              test_size=self.len_train_data/len(self.samples),
                                                                                              random_state=43, shuffle=True)
        train_samples, val_samples, train_ground_truth, val_ground_truth = train_test_split(train_samples,
                                                                                            train_ground_truth,
                                                                                            test_size=0.1, random_state=43,
                                                                                            shuffle=True)
        print("\nShapes der gesplitteten Daten: Train, Test, Val ")
        print(np.shape(train_samples))
        print(np.shape(train_ground_truth))
        print(np.shape(test_samples))
        print(np.shape(test_ground_truth))
        print(np.shape(val_samples))
        print(np.shape(val_ground_truth))

        return train_samples, test_samples, train_ground_truth, test_ground_truth, val_samples, val_ground_truth

    def save_dataset(self):
        print(os.getcwd())
        os.chdir("gleichverteilt_tqeng_pos_neg")
        print(os.getcwd())
        np.savetxt("train_samples_full_dataset.csv", self.train_samples, delimiter=',')
        np.savetxt("test_samples_full_dataset.csv", self.test_samples, delimiter=',')
        np.savetxt("train_ground_truth_full_dataset.csv", self.train_ground_truth, delimiter=',')
        np.savetxt("test_ground_truth_full_dataset.csv", self.test_ground_truth, delimiter=',')



if __name__ == '__main__':
    dict_dataset = {0: "full_dataset", 1: "sindelfingen_dataset", 2: "gaertringen_dataset", 3: "werk_dataset",
                    4: "huettlingen_dataset", 5: "full_dataset_gleichverteilt_tqeng"}

    dataset = dict_dataset.get(5)  # TODO: Choose a dataset!

    # Instanz der Klasse Split_data
    split_data = Split_data()

    split_data.import_data()
    split_data.choose_dataset(dataset)
    split_data.choose_data_channels()

    # Split data with or without validation dataset
    split_data.split_data()
    # split_data.split_data_with_val()

    # split_data.save_dataset()
