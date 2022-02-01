import matplotlib.pyplot as plt
import numpy as np


x_label = ["GrSt (-)", "NEng (1/s)","T (s)", "TqEng (Nm)", "VVeh (km/h)"]

plt.plot(x_label,[3, 25, 1, 150, 120], "ob")
plt.grid()
plt.show()