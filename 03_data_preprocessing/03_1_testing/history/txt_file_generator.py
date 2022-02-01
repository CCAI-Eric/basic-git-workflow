import os

lenght_v = 200
os.chdir(
    "/home/ase/Dokumente/eh_basics/masterarbeit_eh/01_data_source/mdf_data/Kombination_MdlPwr_MdlBatPwr/test/txt_files"
)

file = open("sample_AitmContnsHorzn.txt", "w")
for i in range(0, lenght_v):
    # file = open("sample_["+ str(i) + "]_MdlPwr300_20a_7.1_2020-05-04.txt", "w")
    # file.write("AitmContnsHorzn_VSlopAFinPosnAry200_[" + str(i) +"] \n")
    # file.write("AitmContnsHorzn_SlopFinValAry200_[" + str(i) +"] \n")
    file.write("AitmContnsHorzn_VFinValAry200_[" + str(i) + "] \n")
    file.write("AitmContnsHorzn_AFinValAry200_[" + str(i) + "] \n")
    # file.write("AitmContnsHorzn_VSlopAFinVectLen \n")
    # file.write("Tm_AmbAirTp\n")
    # file.write("Tm_AmbAirP \n")
    # file.write("AitmCmn_MVeh \n")
    # file.write("Cod_Diff_Ratio_Calc \n")
    # file.write("CurbWeight \n")
    # file.write("WhlPA_Circumfer \n")
    # file.write("ESOC_SOC \n")
    # file.write("BMS_Batt_Volt \n")
    # file.write("HDC_HRC_Run_Mode \n")
    # file.write("Veh_Spd \n")
    # file.write("eng_trq \n")
    # file.write("eng_spd \n")
    # file.write("Gear_State \n")
    # file.write("Ign_mode \n")
    # file.write("Eng_Run_Mode \n")
file.close()


"""
file = open("VVehTBasAry100_AitmContnsHorzn.txt", "w")
for i in range(0,lenght_v):
#     # file.write("AitmContnsHorzn_NEngAry200[" + str(i) +"] \n")
#     # file.write("AitmContnsHorzn_TAry200[" + str(i) +"] \n")
#     # file.write("AitmContnsHorzn_TqEngAry200[" + str(i) +"] \n")
    file.write("AitmContnsHorzn_VVehTBasAry100_[" + str(i) +"] \n")
#     file.write("AitmContnsHorzn_GrStAry200[" + str(i) +"] \n")
#     # file.write("AitmContnsHorzn_VTqGrNMaxAVectLen\n")
file.close()

# output_[1]AitmContsHorzn_MdlPwr300_20a_7.1_2020-05-04_223-291_20A_Abtsgmuend_Fahrt-Sindelfingen-EMode_01_DataGroup_1
"""
