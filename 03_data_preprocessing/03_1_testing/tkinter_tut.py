from tkinter import *
from tkinter import filedialog
import pandas as pd


# main for askopenfilename
root = Tk()
root.withdraw()
# root.iconbitmap("hourglass")
root.title("Analysiere Results files mit tkinter!")
root.filename = filedialog.askopenfilename(initialdir="05_neural_nets/FCNN/Train/meta_studie_architektur", title="Select a .csv-file", filetypes=(("csv files", "*.csv"),("all files", "*.*")))


# Read the choosen .csv file into a Pandas DataFrame
results = pd.read_csv(root.filename)
print(results.head())

# run the main loop
root.mainloop()
