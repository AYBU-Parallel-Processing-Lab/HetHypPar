function process_matrix()
    % Base directory containing all matrix subdirectories
    matrices_base = 'data/matrices';

    % Get all subdirectories in data/matrices/
    dir_contents = dir(matrices_base);
    dir_contents = dir_contents([dir_contents.isdir]); % Keep only directories
    dir_contents = dir_contents(~ismember({dir_contents.name}, {'.', '..'})); % Remove . and ..

    % Loop through each matrix directory
    for i = 1:length(dir_contents)
        matname = dir_contents(i).name;

        fprintf('Processing matrix: %s\n', matname);

        % Construct input matrix path
        matrix_path = sprintf("data/matrices/%s.mtx", matname);

        % Check if matrix file exists
        if ~exist(matrix_path, 'file')
            fprintf('Warning: Matrix file not found for %s, skipping...\n', matname);
            continue;
        end

        % Read matrix
        A = mmread(matrix_path);

        matsize = size(A)(1);

        Xt = 1:matsize;
        Xt = Xt';

        B = A*Xt;

        Xi = ones(matsize,1);

        % Base directory for output
        base_dir = sprintf('%s/%s/in', matrices_base, matname);

        % Create output directory structure if it doesn't exist
        if ~exist(base_dir, 'dir')
            mkdir(base_dir);
        end

        % Write output files
        dlmwrite(sprintf("%s/X_target.txt", base_dir), Xt)
        dlmwrite(sprintf("%s/X_init.txt", base_dir), Xi)
        dlmwrite(sprintf("%s/B.txt", base_dir), B)

        fprintf('Completed processing: %s\n\n', matname);
    end

    fprintf('All matrices processed.\n');
end
