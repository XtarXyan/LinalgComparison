% Linear Algebra Benchmark - MATLAB Version
% Compares performance of different linear algebra solvers

function benchmark()
    
    % single-threaded execution for fair comparison
    maxNumCompThreads(1);
    fprintf('MATLAB threads set to: %d\n', maxNumCompThreads());

    % Load data
    fprintf('Loading matrices...\n');
    A = readmatrix('Data/A.csv');
    b = readmatrix('Data/b.csv');
    C = readmatrix('Data/C.csv');
    d = readmatrix('Data/d.csv');
    
    fprintf('Matrix sizes: A=%dx%d, b=%d, C=%dx%d, d=%d\n', ...
            size(A,1), size(A,2), length(b), size(C,1), size(C,2), length(d));
    
    % Define solvers
    solvers = {@linear_solver, @least_squares_solver, @pseudo_inverse_solver, ...
               @eigen_solver, @singular_value_solver, @inner_product_solver, ...
               @outer_product_solver};
    solver_names = {'linear_solver', 'least_squares_solver', 'pseudo_inverse_solver', ...
                    'eigen_solver', 'singular_value_solver', 'inner_product_solver', ...
                    'outer_product_solver'};
    
    benchmark_runs = 10;
    
    % Warm-up run
    fprintf('Running warm-up\n');
    for i = 1:length(solvers)
        run_benchmarks(solvers{i}, solver_names{i}, A, b, C, d, 1);
    end
    
    fprintf('\n--------------------------------\n');
    fprintf('Starting benchmark runs\n');
    fprintf('--------------------------------\n\n');
    
    % Main benchmark runs
    fid = fopen('benchmark_results_matlab.txt', 'a');
    for i = 1:length(solvers)
        results = run_benchmarks(solvers{i}, solver_names{i}, A, b, C, d, benchmark_runs);
        avg_time = mean(results);
        
        fprintf('Solver: %s, Average Time over %d runs: %.6f seconds\n', ...
                solver_names{i}, benchmark_runs, avg_time);
        fprintf(fid, 'Solver: %s, Average Time over %d runs: %.6f seconds\n', ...
                solver_names{i}, benchmark_runs, avg_time);
    end
    fclose(fid);
end

function results = run_benchmarks(solver_func, name, A, b, C, d, count)
    results = zeros(count, 1);
    for i = 1:count
        elapsed_time = benchmark_solver(solver_func, name, A, b, C, d);
        results(i) = elapsed_time;
    end
end

function elapsed_time = benchmark_solver(solver_func, name, A, b, C, d)
    tic;
    result = solver_func(A, b, C, d);
    elapsed_time = toc;
    
    fprintf('Solver: %s, Time taken: %.6f seconds\n', name, elapsed_time);
end

% Solver implementations
function x = linear_solver(A, b, C, d)
    % Solve Ax = b using MATLAB's backslash operator (uses LU with partial pivoting)
    x = A \ b;
end

function x = least_squares_solver(A, b, C, d)
    % Solve the augmented system [C; A] x = [d; b] in least squares sense.
    CD = [C; A];
    bd = [b; d];
    x = CD \ bd; 
end

function x = pseudo_inverse_solver(A, b, C, d)
    % Solve Ax = b using the Moore-Penrose pseudo-inverse of A
    x = pinv(A) * b;
end

function x = eigen_solver(A, b, C, d)
    % Compute the eigenvector of A corresponding to its largest eigenvalue
    [V, D] = eig(A);
    eigenvalues = diag(D);
    [~, max_idx] = max(real(eigenvalues));  % Find index of largest real eigenvalue
    x = real(V(:, max_idx));  % Return real part of corresponding eigenvector
end

function x = singular_value_solver(A, b, C, d)
    % Compute the right singular vector of A corresponding to its largest singular value.
    [~, ~, V] = svd(A);
    x = V(:, 1);
end

function result = inner_product_solver(A, b, C, d)
    % Compute the inner product b^T A d
    result = b' * A * d;
end

function result = outer_product_solver(A, b, C, d)
    % Compute the outer product of b and d, flattened to a 1D array
    outer_prod = A * (b * d');
    result = outer_prod(:);  % Flatten to column vector
end
