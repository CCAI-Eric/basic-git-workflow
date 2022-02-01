import numpy as np

array = np.random.rand(5, 1)
print(array)
print(array.shape)
array = np.repeat(array, 200, axis=1)
print(array.shape)