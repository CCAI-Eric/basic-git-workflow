import pandas as pd

path = "02_dataset/Mdl_Pwr_MdlBatPwr/Gesamtdatensatz/Variante_1/"

# liste = [i for i in range(400, 600)]
# print(liste)
name = "22_04_21_Gesamtdatensatz_flatten_data.csv"
data = pd.read_csv(path+name, delimiter=',', usecols=[15])
data.columns = ["Gang"]
print(data.sample())
print(data.describe())

data_zeros = data.loc[data['Gang'] == 0]
print(data_zeros.describe())
