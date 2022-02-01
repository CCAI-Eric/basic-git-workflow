import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(color_codes=True)
tips = sns.load_dataset("tips")
tips.head()

# ax = sns.regplot(x="total_bill, y="tip", data=tips)
plt.show()