import numpy as np
a = np.array([[1, 2], [3, 4]])

print(np.std(a))
1.1180339887498949 # may vary
print(np.std(a, axis=0))
print(np.std(a, axis=1))
