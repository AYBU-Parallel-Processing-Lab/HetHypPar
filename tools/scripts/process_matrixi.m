function process_matrixi(matname)
    % Construct input matrix path
    matrix_path = sprintf("/matrices/%s.mtx", matname);

    % Read matrix
    A = mmread(matrix_path);

    matsize = size(A)(1);

    Xt = 1:matsize;
    Xt = Xt';

    B = A*Xt;

    Xi = ones(matsize,1);

    % Base directory for output
    base_dir = sprintf('data/matrices/%s/in', matname);

    % Create output directory structure
    if ~exist(base_dir, 'dir')
        mkdir(base_dir);
    end

    % Write output files
    dlmwrite(sprintf("%s/X_target.txt", base_dir), Xt)
    dlmwrite(sprintf("%s/X_init.txt", base_dir), Xi)
    dlmwrite(sprintf("%s/B.txt", base_dir), B)
end
