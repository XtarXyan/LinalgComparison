import numpy as np

np.random.seed(42)
n = 1000
m = n * 2
A = np.random.uniform(-1, 1, (n, n))
b = np.random.uniform(-1, 1, n)
C = np.random.uniform(-1, 1, (m, n))
d = np.random.uniform(-1, 1, n)

# Save the generated data to .npy files
np.save('Data/A.npy', A)
np.save('Data/b.npy', b)
np.save('Data/C.npy', C)
np.save('Data/d.npy', d)
# Save the generated data tp .csv files
np.savetxt('Data/A.csv', A, delimiter=',')
np.savetxt('Data/b.csv', b, delimiter=',')
np.savetxt('Data/C.csv', C, delimiter=',')
np.savetxt('Data/d.csv', d, delimiter=',')