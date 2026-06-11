function prepare_spmv_input(matrix_path, out_dir, method)
    % Prepare an SpMV test input vector X such that A * X = [1; 2; ...; n].
    %
    % Solves the linear system A*X = target where target = (1:n)', so that
    % a downstream SpMV (y = A*X) produces the easily-verifiable vector
    % [1, 2, 3, ..., n].
    %
    % Inputs:
    %   matrix_path : path to a Matrix Market .mtx file
    %   out_dir     : directory to write X_spmv.txt and B_spmv.txt into
    %   method      : (optional) 'iterative' (default) or 'direct'
    %                 'direct'    = A\b sparse LU. Exact but OOMs on big matrices.
    %                 'iterative' = GMRES + ILU(0) preconditioner. Scales to
    %                               large matrices; output accurate to ~tol.
    %
    % Outputs (header-less, one value per line to match C vector_read):
    %   <out_dir>/X_spmv.txt : the solved input vector X
    %   <out_dir>/B_spmv.txt : the target vector [1..n] (expected SpMV result)
    %
    % Example:
    %   prepare_spmv_input('/matrices/cage14.mtx', 'data/matrices/cage14/in')
    %   prepare_spmv_input('/matrices/bcsstk29.mtx', 'data/.../in', 'direct')

    if nargin < 3
        method = 'iterative';
    end

    printf("Reading %s ...\n", matrix_path);
    A = mmread(matrix_path);

    [m, n] = size(A);
    if m ~= n
        error("Matrix must be square for A\\b solve (got %dx%d)", m, n);
    end

    target = (1:n)';

    if strcmp(method, 'direct')
        printf("Solving A*X = [1..%d] (direct sparse LU)...\n", n);
        X = A \ target;
    else
        printf("Solving A*X = [1..%d] (GMRES + ILU(0))...\n", n);
        tol = 1e-10;
        maxit = 2000;
        restart = 50;

        % ILU(0) preconditioner dramatically speeds convergence. Some matrices
        % have zero pivots; fall back to unpreconditioned GMRES if it fails.
        L = []; U = [];
        try
            [L, U] = ilu(A, struct('type', 'nofill'));
        catch err
            printf("  ILU failed (%s); running GMRES without preconditioner\n", err.message);
        end

        if isempty(L)
            [X, flag, relres, iter] = gmres(A, target, restart, tol, maxit);
        else
            [X, flag, relres, iter] = gmres(A, target, restart, tol, maxit, L, U);
        end

        if flag ~= 0
            warning("GMRES did not fully converge (flag=%d, relres=%.3e). Output X may be approximate.", flag, relres);
        end
        printf("  GMRES finished: relres=%.3e\n", relres);
    end

    % Sanity check: how close is A*X to the target?
    residual = norm(A*X - target) / norm(target);
    printf("Relative residual ||A*X - target|| / ||target|| = %.3e\n", residual);
    if residual > 1e-6
        warning("Solve is inaccurate (residual %.3e). Verification error will be correspondingly large.", residual);
    end

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    x_path = sprintf("%s/X_spmv.txt", out_dir);
    b_path = sprintf("%s/B_spmv.txt", out_dir);

    dlmwrite(x_path, X, 'precision', '%.17g');
    dlmwrite(b_path, target, 'precision', '%.17g');

    printf("Wrote %s\n", x_path);
    printf("Wrote %s\n", b_path);
    printf("Run SpMV with -x %s ; output should equal [1..%d]\n", x_path, n);
end
