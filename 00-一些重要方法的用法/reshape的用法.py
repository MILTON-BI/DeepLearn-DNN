import numpy as np

array1 = np.array([i for i in range(64)])
array2 = array1.reshape(8, 8)
array3 = array2.reshape(4, 16)
print(array1)
print(array2)
print(array3)