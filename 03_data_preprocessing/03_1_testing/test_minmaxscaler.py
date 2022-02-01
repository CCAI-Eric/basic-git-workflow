from sklearn.preprocessing import MinMaxScaler
import os
from numpy import genfromtxt
import numpy as np
import joblib

os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Sindelfingen_Oversampling")
# NEng = genfromtxt('label_NEng_AitmContnsHorzn_sindelfingen.csv', delimiter=',')
test_samples = genfromtxt("us_test_samples_[0]_6outs_50%_filter_MdlPwr300_20a_7.1.csv", delimiter=',')
test_labels = genfromtxt("us_test_labels_[0]_6outs_50%_filter_MdlPwr300_20a_7.1.csv", delimiter=',')
train_samples = genfromtxt("us_train_samples_[0]_6outs_50%_filter_MdlPwr300_20a_7.1.csv", delimiter=',')
train_labels = genfromtxt("us_train_labels_[0]_6outs_50%_filter_MdlPwr300_20a_7.1.csv", delimiter=',')

train_labels = np.delete(train_labels, 4, 1)  # del column 4 -> VectLen
train_labels = np.delete(train_labels, 2, 1)  # del column 2 -> TAry
train_labels = np.delete(train_labels, 0, 1)  # del column 0 -> Gang
test_labels = np.delete(test_labels, 4, 1)
test_labels= np.delete(test_labels, 2, 1)
test_labels = np.delete(test_labels, 0, 1)

print(test_labels.shape)
# us_test_labels_[0]_6outs_50%_filter_MdlPwr300_20a_7.1.csv
print(train_samples.shape)
input_shape = train_samples.shape[1]
output_shape = train_labels.shape[1]
print(input_shape)
print(output_shape)

# reshapen des Trainingsdatensatz für scaler
train_samples = train_samples.reshape(-1, input_shape) # TODO

os.chdir("/home/ase/Dokumente/eh_basics/masterarbeit_eh/03_data_preprocessing/testing/scaler")
scaler = MinMaxScaler(feature_range=(0.001, 0.9999))
scaler.fit(train_samples)
# scaled_train_labels = scaler.transform(train_labels)
# scaled_test_labels = scaler.transform(test_labels)
print(scaler.data_min_)
print(scaler.data_max_)
# print(scaled_test_samples)
# print(scaled_train_samples)
# print(scaled_test_samples.shape)
""" Save scaled data"""
# np.savetxt('scaled_test_samples.csv', scaled_test_samples, delimiter=',')


"""Save scaler with joblib"""
scaler_filename = "scaler_us_train_samples_[0]_3outs_50%_fil_sindelfingen.save"
joblib.dump(scaler, scaler_filename)

"""Transform with saved scaler"""
# my_scaler = joblib.load(scaler_filename)
# scaled_data = my_scaler.transform(NEng)
# np.savetxt('scaled_data.csv', scaled_data, delimiter=',')
# print(scaled_data)

# # Inverse transform with saved scaler
# re_scaler = joblib.load(scaler_filename)
# rescaled_data = re_scaler.inverse_transform(scaled_data)
# print(rescaled_data)
# print(rescaled_data[1])

