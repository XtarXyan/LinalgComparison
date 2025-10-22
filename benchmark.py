import numpy as np
import time
import os

# single-threaded execution for fair comparison
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Load data
A = np.load('Data/A.npy')
b = np.load('Data/b.npy')
C = np.load('Data/C.npy')
d = np.load('Data/d.npy')

print(f"NumPy threads controlled via environment variables")
print(f"NumPy version: {np.__version__}")
print(f"BLAS info: {np.show_config()}")

def benchmark_solver(solver, A, b, C, d):
    start_time = time.time()
    x = solver(A, b, C, d)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Solver: {solver.__name__}, Time taken: {elapsed_time:.6f} seconds")
    return elapsed_time

def linear_solver(A, b, C, d):
    """
    Solve Ax = b using least squares (minimizes ||Ax - b||).
    """
    return np.linalg.lstsq(A, b, rcond=None)[0]

def least_squares_solver(A, b, C, d):
    """
    Solve the augmented system [C; A] x = [d; b] in least squares sense.
    """
    CA = np.vstack([C, A])  # Stack C and A vertically
    db = np.concatenate([d, b])  # Create vector of size 2 * n
    return np.linalg.lstsq(CA, db, rcond=None)[0]

def pseudo_inverse_solver(A, b, C, d):
    """
    Solve Ax = b using the Moore-Penrose pseudo-inverse of A.
    """
    return np.linalg.pinv(A) @ b

def eigen_solver(A, b, C, d):
    """
    Compute the eigenvector of A corresponding to its largest eigenvalue.
    """
    eigvals, eigvecs = np.linalg.eig(A)
    return eigvecs[:, np.argmax(eigvals)]

def singular_value_solver(A, b, C, d):
    """
    Compute the right singular vector of A corresponding to its largest singular value.
    """
    U, S, Vt = np.linalg.svd(A)
    return Vt.T[:, 0]

def inner_product_solver(A, b, C, d):
    """
    Compute the inner product b^T A d.
    """
    return b.T @ A @ d

def outer_product_solver(A, b, C, d):
    """
    Compute the outer product of b and d, flattened to a 1D array.
    """
    return A @ np.outer(b, d)


solvers = [
    linear_solver,
    least_squares_solver,
    pseudo_inverse_solver,
    eigen_solver,
    singular_value_solver,
    inner_product_solver,
    outer_product_solver
]

benchmark_runs = 10

def run_benchmarks(solver, count=benchmark_runs):
    results = []
    for _ in range(count):
        elapsed_time = benchmark_solver(solver, A, b, C, d)
        results.append((solver.__name__, elapsed_time))
    return results

if __name__ == "__main__":
    # Warm-up run
    print("Running warm-up")
    for solver in solvers:
        run_benchmarks(solver, count=1)

    print("\n--------------------------------")
    print("Starting benchmark runs")
    print("--------------------------------\n")

    # Aggregate benchmark results over multiple runs
    for solver in solvers:
        results = run_benchmarks(solver)
        avg_time = sum(time for _, time in results) / len(results)
        print(f"Solver: {solver.__name__}, Average Time over {benchmark_runs} runs: {avg_time:.6f} seconds")
        # Write results to a file
        with open('benchmark_results_python.txt', 'a') as f:
            f.write(f"Solver: {solver.__name__}, Average Time over {benchmark_runs} runs: {avg_time:.6f} seconds\n")

