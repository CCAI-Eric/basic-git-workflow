from tkinter import *
from tkinter import filedialog
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# main for askopenfilename
root = Tk()
root.withdraw()
# root.iconbitmap("hourglass")
root.title("Analysiere Results files mit tkinter!")
root.filename = filedialog.askopenfilename(initialdir="05_neural_nets/FCNN/Train/meta_studie_architektur", title="Select a .csv-file", filetypes=(("csv files", "*.csv"),("all files", "*.*")))

# Read the choosen .csv file into a Pandas DataFrame
results = pd.read_csv(root.filename)
print(results.head())

# Start to analyze the data
results1 = results[["variante", "test_loss", "test_r_squared", "test_accuracy"]]
print(results1.head())

results2= results[["test_loss", "test_r_squared", "test_accuracy"]]

sns.set_theme(style="whitegrid")
sns.scatterplot(data=results2)
plt.show()





# results1 = results[["label", "TqEng (Nm)", "TqEng (Nm).1", "TqEng (Nm).2", "TqEng (Nm).3"]]
# print(results1.head(10))

# df_new = results1[2]
# print(df_new.head())

# print(results.info())
#
# results_metric = results[["variante", "test_r_squared", "test_accuracy"]]
# print(results_metric.head())
# # print(results_metric.describe())


##################
# results.hist(bins=20, figsize=(20,15))
# plt.show()

# df1 = results[["mse", "r2", "Tq Std", "GrSt Std", "Neng Std"]]

# run the main loop
root.mainloop()