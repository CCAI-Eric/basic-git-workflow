import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
results = pd.read_csv("/home/ase/Dokumente/eh_basics/masterarbeit_eh/05_neural_nets/FCNN/Train/meta_studie_architektur/scores/resume_meta_scores.csv")
print(results.head())
# print(results.info())

results_metric = results[["variante", "test_r_squared", "test_accuracy"]]
print(results_metric.head())
# print(results_metric.describe())

sns.set_theme(style="whitegrid")
sns.scatterplot(data=results_metric)
plt.show()


##################
# results.hist(bins=20, figsize=(20,15))
# plt.show()

# df1 = results[["mse", "r2", "Tq Std", "GrSt Std", "Neng Std"]]

# print(df1.head(10))
# df.ix[:, ~df.columns.str.contains('^a')]