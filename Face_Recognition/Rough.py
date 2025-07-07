import numpy as np

a = np.load('./data/No.npy')
b = np.load('./data/Yes.npy')
data = []
data.append(a)
data.append(b)
print(data)
