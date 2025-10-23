# Linalg Comparison

This is a benchmark of how efficiently different programming languages run various linear algebra computations. Currently it tests C++/Eigen, Python/Numpy and MATLAB on the following tasks:
- Solve a linear system Ax = b where A is a square matrix.
- Solve a linear system Ax = b where A is an M x N matrix, where M = 2N.
- Find the Moore-Penrose pseudo-inverse of a square matrix A and solve Ax = b.
- Compute the eigenvector of a square matrix corresponding to its largest eigenvalue.
- Compute the right singular vector of a square matrix corresponding to its largest SV.
- For M = 2N and for a matrix A of size M x N, vector b of size M and vector c of size N, compute the inner product (b^T)Ad.
- Compute the outer product of vectors a and b where b is twice as large as a.

## Requirements to run the code as-is:
### All platforms:
- Python3
- MATLAB
- CMake
- vcpkg (make sure you have environment variables linked)

### Windows:
- MSVC

### Linux:
- G++/Clang++
- Ninja

## To run the code as-is:
### All platforms

To benchmark MATLAB run benchmark.m in a MATLAB environment after generating the data.
All benchmark outputs are generated as "benchmark_results_<language>.txt".

### Windows:

Open a terminal in the root directory of the project.

Generate the data by running:
```Powershell
python data_generation.py
```
When prompted enter the value of N.

To benchmark Python run
```Powershell
python benchmark.py
```

To benchmark C++ run
```Powershell
cmake --preset windows
./build-msvc/LinalgComparison.exe
```

###Linux:
Open bash in the root directory of the project.

Generate the data by running:
```Bash
python data_generation.py
```
When prompted enter the value of N.

To benchmark Python run
```Bash
python benchmark.py
```

To benchmark C++ run
```Bash
cmake --preset default
./build-gcc/LinalgComparison
```

To alternatively benchamrk with Clang run
```Bash
cmake --preset linux-clang
./build-clang/LinalgComparison
```

Of course, you can always make changes to the code and/or cmake presets.
