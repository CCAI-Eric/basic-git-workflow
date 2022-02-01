import numpy as np

# y_pred = np.random.rand(5,1)
y_pred = [4.12, 6.9, 9.6, 3.4]

# print(y_pred.shape)
print(y_pred)

print("\n RUNDEN!")

y_pred = np.around(y_pred)
print(y_pred)
