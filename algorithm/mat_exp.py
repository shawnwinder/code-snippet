import numpy as np

exp = 7
base = np.array([[1,2,3], [4,5,6], [7,8,9]])
res = np.array([[1,0,0], [0,1,1], [0,0,1]])
for i in range(exp):
    res = np.dot(res, base)

print(res)

