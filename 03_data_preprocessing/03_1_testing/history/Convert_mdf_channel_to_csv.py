from asammdf import mdf
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_filter(file_name):
    """
    :param file_name: name of the txt file that stores the target signals to be filtered out
    :return: a list of strings (signal name)
    """

    os.chdir("/home/ase/Dokumente/eh_basics/masterarbeit_eh/01_data_source/mdf_data/Kombination_MdlPwr_MdlBatPwr/test/"
             "txt_files")   # TODO
    s = []
    # print(os.getcwd())
    with open(file_name) as file:
        for line in file:
            s.append(line.rstrip())
        print(len(s))
    return s

def mdf_to_csv(mdf_name, filter):
    """
    :param mdf_name: string
            Name of the MDF file
            filter： list of strings
            list of signals that selected to export to csv
    :return:
    """
    os.chdir("/home/ase/Dokumente/eh_basics/masterarbeit_eh/01_data_source/mdf_data/Kombination_MdlPwr_MdlBatPwr/test")  # TODO
    # print(os.getcwd())

    channel_filter = []
    data_transfer = mdf.MDF(mdf_name)
    for el in filter:
        #if data_transfer.__contains__(el):
        channel_filter.append(el)
    data_filtered = data_transfer.filter(channel_filter)
    data_filtered.export(fmt='csv', filename=name + mdf_name, time_from_zero=False) # sep = ',', decimal = '.',


# Dicts and strings for filter
sample_filter_dict = {0: "Tm_AmbAirP", 1: "VSlopAFinPosn", 2: "SlopFinVal",3: "VFinVal", 4: "AFinVal",5: "VSlopAFinVectLen",
                      6: "Tm_AmbAirTp",7: "MVeh",8: "Cod_Diff_Ratio_Calc", 9: "CurbWeight", 10: "WhlPA_Circumfer"}
# label_filter_dict = {0: "NEng", 1: "T", 2: "TqEng",3: "VVeh", 4: "GrSt"}
label_filter_dict = {0: "VVeh"}
test_filter_dict = {0: "veh_spd"}
# test_filter_dict = {0:"veh_spd_", 1: "Eng_Trq_", 2: "Eng_Spd_", 3: "gear_state_", 4: "Ign_Mode_", 5: "Eng_Run_Mode_"}
Eng_Run_Mode_dict = {0: "Eng_Run_Mode_"}

# route = 'AitmContnsHorzn_huettlingen'  # TODO
mdf_name = "Testfahrt_Sindelfingen_03_03_21" # TODO: ohne .mdf

for i in test_filter_dict: # TODO
    # print(sample_filter_dict.get(i)) # TODO
    # name = "sample_" + str(sample_filter_dict.get(i)) +"_" # string variable input/output  # TODO
    name = "label_" + str(label_filter_dict.get(i)) +"_" # string variable input/output
    # name = "measured_" + str(test_filter_dict.get(i))

    # filter_name = str(sample_filter_dict.get(i)) + "_AitmContnsHorzn.txt"  # TODO
    # filter_name = str(label_filter_dict.get(i)) + "_Ary200_AitmContnsHorzn.txt"
    # filter_name = str(test_filter_dict.get(i)) + "_AitmContnsHorzn.txt"
    filter_name = "VVehTBasAry100_AitmContnsHorzn.txt"
    # print(filter_name)
    fil = create_filter(filter_name)
    print(fil)
    mdf_to_csv(mdf_name + '.mf4', filter=fil)

    # Umwandlung der .csv-Datei über Pandas-Befehle
    df = pd.read_csv(name + mdf_name + "_DataGroup_1.csv", decimal='.')
    # label_[0]_AitmContsHorzn_MdlPwr300_20a_7.1_2019-12-20 11_41_53gaertringen_mit_routenfuehrung_DataGroup_1
    del df['time']
    # df.hist(bins=30, figsize=(20, 15))
    # df.hist()
    # plt.show()
    print(df.sample(50))
    # pd.set_option('display.max_columns', None)
    os.chdir("/home/ase/Dokumente/eh_basics/masterarbeit_eh/01_data_source/mdf_data/Kombination_MdlPwr_MdlBatPwr/test/")
    os.remove(name + mdf_name + "_DataGroup_1.csv")



    # os.chdir("/home/ase/Dokumente/AITM/Daten/02_dataset/Mdl_Pwr/huettlingen")  # TODO
    # df.to_csv(name + ".csv", header=True, index=False)


