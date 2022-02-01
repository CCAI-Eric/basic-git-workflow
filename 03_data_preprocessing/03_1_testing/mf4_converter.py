from asammdf import MDF
import os
from asammdf import MDF4

os.chdir("/home/ase/Dokumente/eh_basics/masterarbeit_eh/01_data_source/mdf_data/Kombination_MdlPwr_MdlBatPwr/test")
mdf = MDF("Testfahrt_Sindelfingen_03_03_21_filtered.mdf", memory='minimum')

# mdf.export(fmt="csv", filename="test_mf4.csv")
# df = mdf.to_dataframe()
# print(df.head())
signallist = ["Veh_Spd", "Eng_Spd"]
# df = mdf.filter(signallist).export(fmt='pandas')
# print(df.sample(20))


df = mdf.filter(signallist).to_dataframe()
print(df.sample(20))