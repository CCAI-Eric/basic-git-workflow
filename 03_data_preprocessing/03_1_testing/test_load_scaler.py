from sklearn.preprocessing import MinMaxScaler
import os
from numpy import genfromtxt
import numpy as np
import joblib
from numpy import genfromtxt

# Inverse transform with saved scaler

scaled_data =genfromtxt("scaled_test_samples.csv", delimiter=',')
scaled_data = scaled_data[:, :3]
print(np.shape(scaled_data))
# scaled_data = scaled_data.reshape(-1, 1)
# print(scaled_data)
os.chdir("scaler")
scaler_filename = "scaler_us_train_labels_[0]_3outs_50%_fil_sindelfingen.save"
re_scaler = joblib.load(scaler_filename)
print("Data Max")
print(re_scaler.data_max_)
print("Data Min")
print(re_scaler.data_min_)
rescaled_data = re_scaler.inverse_transform(scaled_data)
# print(rescaled_data)
print(rescaled_data[1:10, 0])