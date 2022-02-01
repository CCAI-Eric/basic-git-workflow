import pandas as pd


def normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min, X_norm):
    X = ((X_norm - X_feature_min) * (X_max - X_min)) / (X_feature_max - X_feature_min)
    return X


path = "/home/ase/Dokumente/eh_basics/masterarbeit_eh/05_neural_nets/FCNN/Train/predictions/Thu_Apr_15_11_48_22_2021/"
list = [i for i in range(600, 800)]
df = pd.read_csv(
    path + "02_CNN_14_04_21_10e_0.0001_bs256_t4_ground_truth.csv",
    delimiter=",",
    usecols=list,
)
# print(df)
# print(df.describe())
print(df.iloc[:, 0:2].describe())


X_feature_max = 67.5
X_feature_min = 0
X_max = 1
X_min = 0.001

i = 0
j = 3
# df.iloc[:, i:j].where(df.iloc[:, i:j] > 0, normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min, df.iloc[:, i:j]), inplace=True)
df.iloc[:, i:j].where(df.iloc[:, i:j] >= 0, 1, inplace=True)


print(df.iloc[:, 0:2].describe())
