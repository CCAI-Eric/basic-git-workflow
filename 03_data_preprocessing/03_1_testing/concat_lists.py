import numpy as np

list1 = np.random.rand(10, 4)
list2 = np.random.rand(10, 1)
print(list1)
print(list2)

list3 = np.concatenate((list2, list1), axis=1)

print(list3)
print(list3.shape)