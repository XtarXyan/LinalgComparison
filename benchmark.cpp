#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <vector>
#include <functional>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

// Function pointer type for solvers
using SolverFunc = function<VectorXd(const MatrixXd&, const VectorXd&, const MatrixXd&, const VectorXd&)>;

// Load matrices and vectors from files (placeholder implementation)
MatrixXd loadMatrix(const string& filename, int rows, int cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return MatrixXd();
    }
    MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        stringstream line;
        string value;
        if (!getline(file, value)) {
            cerr << "Error reading line " << i << " from file: " << filename << endl;
            return MatrixXd();
        }
        line.str(value);
        for (int j = 0; j < cols; ++j) {
            string cell;
            if (!getline(line, cell, ',')) {
                cerr << "Error reading column " << j << " from line " << i << " in file: " << filename << endl;
                return MatrixXd();
            }
            mat(i, j) = stod(cell);
        }
    }
    return mat;
}

VectorXd loadVector(const string& filename, int size) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return VectorXd();
    }

    VectorXd vec(size);
    for (int i = 0; i < size; ++i) {
        string value;
        if (!getline(file, value)) {
            cerr << "Error reading line " << i << " from file: " << filename << endl;
            return VectorXd();
        }
        vec(i) = stod(value);
    }
    return vec;
}

// Solver implementations
VectorXd linear_solver(const MatrixXd& A, const VectorXd& b, const MatrixXd& C, const VectorXd& d) {
    /*
    Solve Ax = b using least squares (minimizes ||Ax - b||).
    */
    return A.colPivHouseholderQr().solve(b);
}

VectorXd least_squares_solver(const MatrixXd& A, const VectorXd& b, const MatrixXd& C, const VectorXd& d) {
    /*
    Solve the overdetermined system Cx=d in least squares sense.
    */
    return C.colPivHouseholderQr().solve(d);
}

VectorXd pseudo_inverse_solver(const MatrixXd& A, const VectorXd& b, const MatrixXd& C, const VectorXd& d) {
    /*
    Solve Ax = b using the Moore-Penrose pseudo-inverse of A.
    */
    return A.completeOrthogonalDecomposition().pseudoInverse() * b;
}

VectorXd eigen_solver(const MatrixXd& A, const VectorXd& b, const MatrixXd& C, const VectorXd& d) {
    /*
    Compute the eigenvector of A corresponding to its largest eigenvalue.
    */

    EigenSolver<MatrixXd> es(A);
        if (es.info() != Success) {
            return VectorXd::Zero(A.rows());
        }
        
        auto eigenvalues = es.eigenvalues();
        auto eigenvectors = es.eigenvectors();
        
        // Find the eigenvalue with maximum real part
        int maxIndex = 0;
        double maxReal = eigenvalues(0).real();
        for (int i = 1; i < eigenvalues.size(); ++i) {
            if (eigenvalues(i).real() > maxReal) {
                maxReal = eigenvalues(i).real();
                maxIndex = i;
            }
        }
        
        // Return the real part of the eigenvector
        return eigenvectors.col(maxIndex).real();
}

VectorXd singular_value_solver(const MatrixXd& A, const VectorXd& b, const MatrixXd& C, const VectorXd& d) {
    /*
    Compute the right singular vector of A corresponding to its largest singular value.
    */
    BDCSVD<MatrixXd> svd(A, ComputeFullV);
    if (svd.matrixV().cols() == 0) {
        return VectorXd::Zero(A.cols());
    }
    return svd.matrixV().col(0);
}

VectorXd inner_product_solver(const MatrixXd& A, const VectorXd& b, const MatrixXd& C, const VectorXd& d) {
    /*
    Compute the inner product d^T C b.
    */
    return d.transpose() * C * b;
}

VectorXd outer_product_solver(const MatrixXd& A, const VectorXd& b, const MatrixXd& C, const VectorXd& d) {
    /*
    Compute the outer product of b and d, flattened to a 1D array.
    */
    MatrixXd O = b * d.transpose();
    return Eigen::Map<VectorXd>(O.data(), O.size());
}

double benchmark_solver(SolverFunc solver, const string& name, const MatrixXd& A, const VectorXd& b, const MatrixXd& C, const VectorXd& d) {
    auto start = high_resolution_clock::now();
    VectorXd result = solver(A, b, C, d);
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<microseconds>(end - start);
    double elapsed_time = duration.count() / 1000000.0; // Convert to seconds
    
    cout << "Solver: " << name << ", Time taken: " << fixed << setprecision(6) << elapsed_time << " seconds" << endl;
    return elapsed_time;
}

vector<double> run_benchmarks(SolverFunc solver, const string& name, const MatrixXd& A, const VectorXd& b, const MatrixXd& C, const VectorXd& d, int count = 10) {
    vector<double> results;
    for (int i = 0; i < count; ++i) {
        double elapsed_time = benchmark_solver(solver, name, A, b, C, d);
        results.push_back(elapsed_time);
    }
    return results;
}

int main() {
    // Load sizes
    cout<<"Loading sizes...";
    ifstream sizeFile(DATA_DIR "/size.csv");
    if (!sizeFile.is_open()) {
        cerr << "Error opening Data/size.csv" << endl;
        return 1;
    }
    vector<double> sizes;
    string line;
    while (getline(sizeFile, line)) {
        // split by commas
        stringstream ss(line);
        string token;
        while (getline(ss, token, ',')) {
            if (token.size() == 0) continue;
            try {
                sizes.push_back(stod(token));
            } catch (...) {
                cerr << "Failed to parse number in Data/size.csv: '" << token << "'" << endl;
                return 1;
            }
        }
    }
    if (sizes.size() != 2) {
        cerr << "Data/size.csv should contain exactly two numbers (n and m); found " << sizes.size() << endl;
        return 1;
    }
    const int n = static_cast<int>(std::round(sizes[0]));
    const int m = static_cast<int>(std::round(sizes[1]));

    cout<<"n = "<<n<<", m = "<<m<<endl;

    const int rows_A = n, cols_A = n; // A is (n x n)
    const int rows_C = m, cols_C = n; // C is (m x n)
    const int size_b = n, size_d = m; // b is n x 1, d is m x 1

    // Load data
    cout << "Loading matrices..." << endl;
    MatrixXd A = loadMatrix(DATA_DIR "/A.csv", rows_A, cols_A);
    VectorXd b = loadVector(DATA_DIR "/b.csv", size_b);
    MatrixXd C = loadMatrix(DATA_DIR "/C.csv", rows_C, cols_C);
    VectorXd d = loadVector(DATA_DIR "/d.csv", size_d);

    // Sanity checks to avoid later out-of-bounds/segfaults
    if (A.rows() != rows_A || A.cols() != cols_A) {
        cerr << "Loaded A has wrong dimensions: " << A.rows() << "x" << A.cols() << " (expected " << rows_A << "x" << cols_A << ")" << endl;
        return 1;
    }
    if (b.size() != size_b) {
        cerr << "Loaded b has wrong size: " << b.size() << " (expected " << size_b << ")" << endl;
        return 1;
    }
    if (C.rows() != rows_C || C.cols() != cols_C) {
        cerr << "Loaded C has wrong dimensions: " << C.rows() << "x" << C.cols() << " (expected " << rows_C << "x" << cols_C << ")" << endl;
        return 1;
    }
    if (d.size() != size_d) {
        cerr << "Loaded d has wrong size: " << d.size() << " (expected " << size_d << ")" << endl;
        return 1;
    }

    // Define solvers
    vector<pair<SolverFunc, string>> solvers = {
        {linear_solver, "linear_solver"},
        {least_squares_solver, "least_squares_solver"},
        {pseudo_inverse_solver, "pseudo_inverse_solver"},
        {eigen_solver, "eigen_solver"},
        {singular_value_solver, "singular_value_solver"},
        {inner_product_solver, "inner_product_solver"},
        {outer_product_solver, "outer_product_solver"}
    };

    // Warm-up run
    cout << "Running warm-up" << endl;
    for (const auto& [solver, name] : solvers) {
        run_benchmarks(solver, name, A, b, C, d, 1);
    }
    
    cout << "\n--------------------------------" << endl;
    cout << "Starting benchmark runs" << endl;
    cout << "--------------------------------\n" << endl;
    
    // Main benchmark runs
    ofstream outfile("benchmark_results_cpp.txt", ios::app);
    for (const auto& [solver, name] : solvers) {
        const int benchmark_runs = 10;
        vector<double> results = run_benchmarks(solver, name, A, b, C, d, benchmark_runs);
        
        double avg_time = 0.0;
        for (double time : results) {
            avg_time += time;
        }
        avg_time /= results.size();
        
        cout << "Solver: " << name << ", Average Time over " << benchmark_runs << " runs: " 
             << fixed << setprecision(6) << avg_time << " seconds" << endl;
        
        outfile << "Solver: " << name << ", Average Time over " << benchmark_runs << " runs: " 
                << fixed << setprecision(6) << avg_time << " seconds" << endl;
    }
    outfile.close();
    
    return 0;
}