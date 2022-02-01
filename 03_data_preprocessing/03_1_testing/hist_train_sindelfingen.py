import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(
    "02_dataset/Mdl_Pwr/AitmContsHorzn_sindelfingen/Preprocessed_data/test_ground_truth.csv",
    delimiter=",",
)

data.columns = [
    "Gang (-)",
    "Drehzahl (1/s)",
    "Zeitschritte (s)",
    "Motormoment (Nm)",
    "Geschwindigkeit (km/h)",
]


print(data)
sns.set_theme(style="whitegrid")
data.hist(bins=9)
plt.show()
