import numpy as np

a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.ones(7)

a[0:7] = a[0:7]+b

print(a)
