import numpy as np
import os

# Set the size of the matrices and vectors based on console input
n = int(input("Enter the size of the square matrix (N): "))

np.random.seed(42)
m = n * 2
A = np.random.uniform(-1, 1, (n, n))
b = np.random.uniform(-1, 1, n)
C = np.random.uniform(-1, 1, (m, n))
d = np.random.uniform(-1, 1, n)

# Check if Data directory exists, if not create it
if not os.path.exists('Data'):
    os.makedirs('Data')

# Save the generated data to .npy files
np.save('Data/size.npy', np.array([n, m]))
np.save('Data/A.npy', A)
np.save('Data/b.npy', b)
np.save('Data/C.npy', C)
np.save('Data/d.npy', d)
# Save the generated data to .csv files
np.savetxt('Data/size.csv', np.array([n, m]), delimiter=',')
np.savetxt('Data/A.csv', A, delimiter=',')
np.savetxt('Data/b.csv', b, delimiter=',')
np.savetxt('Data/C.csv', C, delimiter=',')
np.savetxt('Data/d.csv', d, delimiter=',')