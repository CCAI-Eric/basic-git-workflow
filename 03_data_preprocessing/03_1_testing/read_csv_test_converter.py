import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

os.chdir("/home/ase/Dokumente/eh_basics/masterarbeit_eh/02_dataset/Mdl_Pwr_MdlBatPwr/206-505_210303_1354_simulated")
df = pd.read_csv("label_AitmContnsHorzn_VVehAry200.csv", delimiter=',', sep='.', header=None, encoding='utf-8')
print(df)
print(df.describe())

# Numpy array -> Ravel (flatten)
arr = np.array(df)
print(arr.shape)
arr = np.ravel(arr)
print(arr.shape)
fig, ax = plt.subplots(figsize=(20, 15))
bin_list = [i for i in range(1, 201, 10)]
# print(bin_list)
sns.set_theme(style="whitegrid")
sns.histplot(data=arr, bins=bin_list, multiple="stack", ax=ax)  # discrete=True --> Jeder Wert wird angezeigt
# g.set_xlabels("Geschwindigkeit (km/h)")
# g.set_ylabels("Counts")
plt.xlabel("Geschwindigkeit (km/h)")
plt.ylabel("Counts (-)")
plt.title("Häufigkeitsverteilung VVeh (km/h) für die Testfahrt 03.03.21 - alle Ereignisse [i]", size=16)
plt.show()

# TODO: Histogram für multiple [i]
# df = df.iloc[:, :100]
# sns.displot(data=df, bins=10, multiple="stack")  #discrete=True
# plt.show()

# TODO: Histogram für i=0
# df = df.iloc[:, 0]
# sns.set_theme(style="whitegrid")
# df.hist(bins=10, figsize=(20, 15))
# plt.show()

# import platform
# import sys
# from pprint import pprint
#
# pprint("python=" + sys.version)
# pprint("os=" + platform.platform())
#
# try:
#     import numpy
#     pprint("numpy=" + numpy.__version__)
# except ImportError:
#     pass
#
# try:
#     import asammdf
#     pprint("asammdf=" + asammdf.__version__)
# except ImportError:
#     pass