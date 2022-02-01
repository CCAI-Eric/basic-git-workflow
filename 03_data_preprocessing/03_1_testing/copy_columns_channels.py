import os
import pandas as pd



os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/werk")

sample_dict = ["sample_MVeh_AitmContnsHorzn_werk","sample_Tm_AmbAirP_AitmContnsHorzn_werk", "sample_VSlopAFinVectLen_AitmContnsHorzn_werk",
               "sample_Tm_AmbAirTp_AitmContnsHorzn_werk", "sample_Cod_Diff_Ratio_Calc_AitmContnsHorzn_werk", "sample_CurbWeight_AitmContnsHorzn_werk",
               "sample_WhlPA_Circumfer_AitmContnsHorzn_werk"]


for i in range(len(sample_dict)):
    sample_name = sample_dict[i]
    df =pd.read_csv(sample_name + ".csv", delimiter=',')
    # print(df.head())
    # print(df.describe())
    df.columns =["0.0"]
    # df["0"] = df
    # print(df["0"].head())


    for i in range(1, 200):
        df[i] = df["0.0"]
    print(df)

    df.to_csv(sample_name + "[i].csv", header=False, index=False)


