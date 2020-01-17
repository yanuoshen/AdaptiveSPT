import numpy as np
import matplotlib.pyplot as plt

A = np.load('admatrix.npy').squeeze()
A = np.sum(A, axis = 0)
for d in A:
    print(np.argmax(d))
# A = np.softmax(A[1,:,:],dim = 1)
# x = np.random.rand(100).reshape(10,10)
# for k in range(len(A)):
#     A[k, k] = 0
plt.matshow(A, vmin=0, vmax=200)
plt.colorbar()
plt.show()

