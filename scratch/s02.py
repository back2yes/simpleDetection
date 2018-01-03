import numpy as np
import matplotlib.pyplot as plt

# N_Orfactory = 50
N_Orfactory = 50
N_Kenyon = N_Orfactory ** 2

N_BinaryConnection = 6

# each Orfactory has about N_BinaryConnection * N_Orfactory connections

sum = np.zeros(N_Orfactory)
for ii in range(N_Kenyon):
    inds = np.random.choice(N_Orfactory, N_BinaryConnection, replace=False)
    sum[inds] += 1

plt.bar(np.arange(N_Orfactory), sum)
plt.show()
