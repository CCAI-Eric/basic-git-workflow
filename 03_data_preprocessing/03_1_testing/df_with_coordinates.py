import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(
    "/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/sindelfingen/filter_VSlopAFinVectLen"
)
# Load data from .csv file
# load sample data
VSlopAFinPosn = pd.read_csv(
    "sample_no_nan_VSlopAFinPosn_AitmContnsHorzn_sindelfingen.csv", delimiter=","
)
SlopFinVal = pd.read_csv(
    "sample_no_nan_SlopFinVal_AitmContnsHorzn_sindelfingen.csv", delimiter=","
)
VFinVal = pd.read_csv(
    "sample_no_nan_VFinVal_AitmContnsHorzn_sindelfingen.csv", delimiter=","
)
AFinVal = pd.read_csv(
    "sample_no_nan_AFinVal_AitmContnsHorzn_sindelfingen.csv", delimiter=","
)

Tm_AmbAirP = pd.read_csv(
    "sample_no_nan_Tm_AmbAirP_AitmContnsHorzn_sindelfingen.csv", delimiter=","
)
VSlopAFinVectLen = pd.read_csv(
    "sample_no_nan_VSlopAFinVectLen_AitmContnsHorzn_sindelfingen.csv", delimiter=","
)
Tm_AmbAirTp = pd.read_csv(
    "sample_no_nan_Tm_AmbAirTp_AitmContnsHorzn_sindelfingen.csv", delimiter=","
)
MVeh = pd.read_csv("sample_no_nan_MVeh_AitmContnsHorzn_sindelfingen.csv", delimiter=",")
Cod_Diff_Ratio_Calc = pd.read_csv(
    "sample_no_nan_Cod_Diff_Ratio_Calc_AitmContnsHorzn_sindelfingen.csv", delimiter=","
)
CurbWeight = pd.read_csv(
    "sample_no_nan_CurbWeight_AitmContnsHorzn_sindelfingen.csv", delimiter=","
)
WhlPA_Circumfer = pd.read_csv(
    "sample_no_nan_WhlPA_Circumfer_AitmContnsHorzn_sindelfingen.csv", delimiter=","
)

# load label data
NEng = pd.read_csv("label_no_nan_NEng_AitmContnsHorzn_sindelfingen.csv", delimiter=",")
T = pd.read_csv("label_no_nan_T_AitmContnsHorzn_sindelfingen.csv", delimiter=",")
TqEng = pd.read_csv(
    "label_no_nan_TqEng_AitmContnsHorzn_sindelfingen.csv", delimiter=","
)
VVeh = pd.read_csv("label_no_nan_VVeh_AitmContnsHorzn_sindelfingen.csv", delimiter=",")
GrSt = pd.read_csv("label_no_nan_GrSt_AitmContnsHorzn_sindelfingen.csv", delimiter=",")

# load coordinate data
coord = pd.read_csv("coordinates_i_t_AitmContnsHorzn_sindelfingen.csv", delimiter=",")
# print(coord.head())


# Concat to sample and label df
sample = pd.concat(
    [
        MVeh,
        AFinVal,
        SlopFinVal,
        VFinVal,
        VSlopAFinPosn,
        VSlopAFinVectLen,
        Cod_Diff_Ratio_Calc,
        CurbWeight,
        Tm_AmbAirP,
        Tm_AmbAirTp,
        WhlPA_Circumfer,
    ],
    axis=1,
)
label = pd.concat([GrSt, NEng, T, TqEng, VVeh], axis=1)

# concat to complete df --> coordinates, sample, label
df = pd.concat([coord, sample, label], axis=1)
df.columns = [
    "i",
    "t",
    "MVeh",
    "AFinVal",
    "SlopFinVal",
    "VFinVal",
    "VSlopAFinPosn",
    "VSlopAFinVectLen",
    "Cod_Diff_Ratio_Calc",
    "CurbWeight",
    "Tm_AmbAirP",
    "Tm_AmbAirTp",
    "WhlPA_Circumfer",
    "GrSt",
    "NEng",
    "T",
    "TqEng",
    "VVeh",
]

# df_target = df[["GrSt", "NEng", "T", "TqEng", "VVeh"]]
df_target = df[["T"]]
df_target.hist(bins=50)
print(df_target.head(100))
print(df_target.tail(100))
print(df_target.describe())
plt.show()
# print(df.head(10))
# print(df.describe())
# print(df)
