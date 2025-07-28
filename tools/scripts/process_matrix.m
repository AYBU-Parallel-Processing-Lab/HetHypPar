function process_matrix(matname)
    % Construct input matrix path
    matrix_path = sprintf("/matrices/%s.mtx", matname);
    
    % Read matrix
    A = mmread(matrix_path);
    
    matsize = size(A)(1);
    
    Xt = 1:matsize;
    Xt = Xt';
    
    B = A*Xt;
    
    Xi = ones(matsize,1);
    
    % Create output directory structure
    output_dir = sprintf("data/%s/in", matname);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Write output files
    dlmwrite(sprintf("data/%s/in/X_target.txt", matname), Xt)
    dlmwrite(sprintf("data/%s/in/X_init.txt", matname), Xi)
    dlmwrite(sprintf("data/%s/in/B.txt", matname), B)
end