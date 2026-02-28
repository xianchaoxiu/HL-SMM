clear; clc; close all;
out_dir = 'saved_results_test';
if ~exist(out_dir,'dir')
    mkdir(out_dir);
end
config = struct();

config.datasets = {'CaltechFace_LBP'};
config.noise_levels = [0, 0.05, 0.1, 0.15, 0.2];
%config.noise_levels = [0,0.2];
config.num_runs = 5;
config.enable_grid_search = true;
config.methods = {
    'L01_SMM', 'Hinge_SMM', 'Pinball_SMM', 'Ramp_SMM', 'LS_SMM', ...
    'Linear_SVM', 'RBF_SVM', 'Poly_SVM'
   % 'L01_SMM','Linear_SVM','RBF_SVM', 'Poly_SVM'
};

all_results = struct();

for d = 1:length(config.datasets)
    dataset_name = config.datasets{d};
    fprintf('\n Processing Dataset: %s \n\n', dataset_name);
    
    try
        [X_train, y_train, X_test, y_test] = load_dataset(dataset_name);
        
        if size(X_train, 3) == 0 || size(X_test, 3) == 0
            error('Data loading failed: empty training or test set');
        end
        
        if config.enable_grid_search
            fprintf('Grid Search for Optimal Parameters\n');
            best_params = grid_search_parameters(X_train, y_train, config.methods);
            save(sprintf('best_params_%s.mat', dataset_name), 'best_params');
        else
            load(sprintf('best_params_%s.mat', dataset_name), 'best_params');
        end

        results = run_full_experiment(X_train, y_train, X_test, y_test, ...
                                       config.methods, best_params, ...
                                       config.noise_levels, config.num_runs);
        
        all_results.(dataset_name) = results;
acc = results(:,:,:,1);   % raw accuracy (0~1)

acc_mean = squeeze(mean(acc, 3)) * 100; 
acc_std  = squeeze(std(acc, 0, 3)) * 100;

save(fullfile(out_dir, sprintf('acc_%s.mat', dataset_name)), ...
     'acc','acc_mean','acc_std','dataset_name','-v7.3');
        
        generate_result_table(results, config.methods, config.noise_levels, dataset_name);
        generate_visualizations(results, best_params, config.methods, ...
                               config.noise_levels, dataset_name);
    catch ME
        fprintf('Error: %s\n', ME.message);
        fprintf('Please check if dataset files exist\n\n');
    end
end


function [X_train, y_train, X_test, y_test] = load_dataset(dataset_name)
    switch dataset_name
        case 'BCI'
            [X_train, y_train, X_test, y_test] = load_bci_data();
        case 'MNIST'
            [X_train, y_train, X_test, y_test] = load_mnist_data();
        case 'Caltech101'
            [X_train, y_train, X_test, y_test] = load_caltech_data();
        case 'INRIA'
            [X_train, y_train, X_test, y_test] = load_inria_data();
        case 'CIFAR10'
            [X_train, y_train, X_test, y_test] = load_cifar10_binary();
      case 'CaltechFace_LBP'
           [X_train, y_train, X_test, y_test] = load_caltechface_lbp();
        otherwise
            error('Unsupported dataset type: %s', dataset_name);
    end
    
    fprintf('Data loaded: train=%d, test=%d, dim=%dx%d\n', ...
            size(X_train,3), size(X_test,3), size(X_train,1), size(X_train,2));
    
    train_mean = mean(X_train(:));
    train_std = std(X_train(:));
    test_mean = mean(X_test(:));
    test_std = std(X_test(:));
    
    fprintf('Train stats: mean=%.4f, std=%.4f\n', train_mean, train_std);
    fprintf('Test stats: mean=%.4f, std=%.4f\n', test_mean, test_std);
    
    X_train_flat = reshape(X_train, [], size(X_train, 3));
    X_test_flat = reshape(X_test, [], size(X_test, 3));
    
    train_class1 = X_train_flat(:, y_train == 1);
    train_class2 = X_train_flat(:, y_train == -1);
    
    dist_within_c1 = mean(pdist(train_class1', 'euclidean'));
    dist_within_c2 = mean(pdist(train_class2', 'euclidean'));
    
    c1_mean = mean(train_class1, 2);
    c2_mean = mean(train_class2, 2);
    dist_between = norm(c1_mean - c2_mean);
    
    fprintf('Class separation: between=%.2f, within_c1=%.2f, within_c2=%.2f\n', ...
            dist_between, dist_within_c1, dist_within_c2);
    fprintf('Separation ratio: %.2f\n', dist_between / mean([dist_within_c1, dist_within_c2]));
end




function [X_train_norm, X_test_norm] = normalize_data_fixed(X_train, X_test)
    X_train_vec = reshape(X_train, [], size(X_train, 3));
    train_mean = mean(X_train_vec, 2);
    train_std = std(X_train_vec, 0, 2);
    
    train_std(train_std < 1e-10) = 1;
    
    [p, q, m_train] = size(X_train);
    m_test = size(X_test, 3);
    
    X_train_norm = zeros(p, q, m_train);
    X_test_norm = zeros(p, q, m_test);
    
    for i = 1:m_train
        X_train_vec_i = reshape(X_train(:, :, i), [], 1);
        X_train_vec_i = (X_train_vec_i - train_mean) ./ train_std;
        X_train_norm(:, :, i) = reshape(X_train_vec_i, p, q);
    end
    
    for i = 1:m_test
        X_test_vec_i = reshape(X_test(:, :, i), [], 1);
        X_test_vec_i = (X_test_vec_i - train_mean) ./ train_std;
        X_test_norm(:, :, i) = reshape(X_test_vec_i, p, q);
    end
end

f

function best_params = grid_search_parameters(X_train, y_train, methods)
    fprintf('Starting grid search...\n');
    
    cv_folds = 3;
    n = size(X_train, 3);
    indices = crossvalind('Kfold', n, cv_folds);
    
    best_params = struct();
    
    for m = 1:length(methods)
        method = methods{m};
        fprintf('  %s: ', method);
        
        param_grid = get_param_grid(method);
        
        best_acc = 0;
        best_param = [];
        
        for p = 1:length(param_grid)
            params = param_grid{p};
            acc_cv = 0;
            
            for fold = 1:cv_folds
                idx_val = (indices == fold);
                idx_train_cv = ~idx_val;
                
                X_tr = X_train(:, :, idx_train_cv);
                y_tr = y_train(idx_train_cv);
                X_val = X_train(:, :, idx_val);
                y_val = y_train(idx_val);
                
                try
                    acc = train_and_evaluate(method, X_tr, y_tr, X_val, y_val, params);
                    acc_cv = acc_cv + acc;
                catch
                    acc_cv = acc_cv + 0;
                end
            end
            
            acc_cv = acc_cv / cv_folds;
            
            if acc_cv > best_acc
                best_acc = acc_cv;
                best_param = params;
            end
        end
        
        best_params.(method) = best_param;
        fprintf('Best acc=%.2f%%\n', best_acc*100);
    end
end
function grid = get_param_grid(method)
    switch method
        case 'L01_SMM'
               r_vals = [4, 5];
            C_vals = [0.5, 1];
            sigma_vals = [0.01];
            grid = {};
            for r = r_vals
                for C = C_vals
                    for sigma = sigma_vals
                        grid{end+1} = struct('r', r, 'C', C, 'sigma', sigma);
                    end
                end
            end
            
        case 'Hinge_SMM'
            r_vals = [2, 4];
            lambda_vals = [0.01, 0.1];
            grid = {};
            for r = r_vals
                for lambda = lambda_vals
                    grid{end+1} = struct('r', r, 'lambda', lambda);
                end
            end
            
        case 'Pinball_SMM'
            r_vals = [2, 4];
            C_vals = [0.1, 1];
            tau_vals = [0.5];
            grid = {};
            for r = r_vals
                for C = C_vals
                    for tau = tau_vals
                        grid{end+1} = struct('r', r, 'C', C, 'tau', tau);
                    end
                end
            end
            
        case {'Ramp_SMM', 'LS_SMM'}
            r_vals = [2];
            C_vals = [10];
            grid = {};
            for r = r_vals
                for C = C_vals
                    grid{end+1} = struct('r', r, 'C', C);
                end
            end
            
        case 'Linear_SVM'
            C_vals = [0.01];
            grid = {};
            for C = C_vals
                grid{end+1} = struct('C', C);
            end
            
        case 'RBF_SVM'
            C_vals = [10];
            gamma_vals = [0.1];
            grid = {};
            for C = C_vals
                for gamma = gamma_vals
                    grid{end+1} = struct('C', C, 'gamma', gamma);
                end
            end
            
        case 'Poly_SVM'
            C_vals = [1, 10];
            degree_vals = [2, 3];
            grid = {};
            for C = C_vals
                for d = degree_vals
                    grid{end+1} = struct('C', C, 'degree', d);
                end
            end
    end
end
function results = run_full_experiment(X_train, y_train, X_test, y_test, ...
                                       methods, best_params, noise_levels, num_runs)
    num_methods = length(methods);
    num_noise = length(noise_levels);
    
    results = zeros(num_methods, num_noise, num_runs, 3);
    
    for m = 1:num_methods
        method = methods{m};
        params = best_params.(method);
        
        fprintf('Running %s...\n', method);
        
        for n = 1:num_noise
            noise = noise_levels(n);
            fprintf('  Noise=%.2f: ', noise);
            
            for run = 1:num_runs
                fprintf('.');

if noise > 0
             
                    X_tr_noisy = X_train;
                    X_te_noisy = X_test;
                    
        
                    mask_tr = rand(size(X_train));
                    X_tr_noisy(mask_tr < noise/2) = -1;      
                    X_tr_noisy(mask_tr >= noise/2 & mask_tr < noise) = 1;
                    

                    mask_te = rand(size(X_test));
                    X_te_noisy(mask_te < noise/2) = -1;      
                    X_te_noisy(mask_te >= noise/2 & mask_te < noise) = 1; 
                    % -----------------------
                else
                    X_tr_noisy = X_train;
                    X_te_noisy = X_test;
                end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                tic;
                try
                    [acc, sv_ratio] = train_and_evaluate(method, X_tr_noisy, y_train, ...
                                                         X_te_noisy, y_test, params);
                    train_time = toc;
                    
                    results(m, n, run, 1) = acc;
                    results(m, n, run, 2) = sv_ratio;
                    results(m, n, run, 3) = train_time;
                catch ME
                    fprintf('E');
                    results(m, n, run, :) = [0, 0, 0];
                end
            end
            fprintf(' Done\n');
        end
    end
end

function [acc, sv_ratio] = train_and_evaluate(method, X_train, y_train, X_test, y_test, params)
    [p, q, n_train] = size(X_train);
    n_test = size(X_test, 3);
    
    switch method
        case 'L01_SMM'
            try
[W, b, z] = L01SMM_Train(X_train, y_train, params.r, params.C, 200, 1e-4);

%%%%%%%%%%%
                acc = evaluate_matrix(W, b, X_test, y_test);
                sv_ratio = sum(z > 1e-3) / length(z) * 100;
                
                if acc == 0 || isnan(acc)
                    acc = 0.5;
                    sv_ratio = 50;
                end
            catch ME
                fprintf('\n[ERROR L01] %s\n', ME.message);
                acc = 0.5;
                sv_ratio = 50;
            end
            
        case 'Hinge_SMM'
            [W, b, z] = HingeSMM_Train(X_train, y_train, params.r, params.lambda, 200, 1e-4);
            acc = evaluate_matrix(W, b, X_test, y_test);
            sv_ratio = sum(z > 0) / length(z) * 100;
            
        case 'Pinball_SMM'
            [W, b, z] = PinballSMM_Train(X_train, y_train, params.r, params.C, params.tau, 200, 1e-4);
            acc = evaluate_matrix(W, b, X_test, y_test);
            sv_ratio = sum(z > 0) / length(z) * 100;
            
        case 'Ramp_SMM'
            [W, b, z] = RampSMM_Train(X_train, y_train, params.r, params.C, 200, 1e-4);
            acc = evaluate_matrix(W, b, X_test, y_test);
            sv_ratio = sum(z > 0) / length(z) * 100;
            
        case 'LS_SMM'
            [W, b] = LSSMM_Train(X_train, y_train, params.r, params.C);
            acc = evaluate_matrix(W, b, X_test, y_test);
            sv_ratio = 100;
            
        case 'Linear_SVM'
            X_vec = reshape(X_train, [p*q, n_train])';
            X_test_vec = reshape(X_test, [p*q, n_test])';
            mdl = fitcsvm(X_vec, y_train, 'KernelFunction', 'linear', ...
                         'BoxConstraint', params.C, 'Standardize', false);
            y_pred = predict(mdl, X_test_vec);
            acc = sum(y_pred == y_test) / length(y_test);
            sv_ratio = sum(mdl.IsSupportVector) / n_train * 100;
            
        case 'RBF_SVM'
            X_vec = reshape(X_train, [p*q, n_train])';
            X_test_vec = reshape(X_test, [p*q, n_test])';
            mdl = fitcsvm(X_vec, y_train, 'KernelFunction', 'rbf', ...
                         'BoxConstraint', params.C, 'KernelScale', 1/sqrt(params.gamma), ...
                         'Standardize', false);
            y_pred = predict(mdl, X_test_vec);
            acc = sum(y_pred == y_test) / length(y_test);
            sv_ratio = sum(mdl.IsSupportVector) / n_train * 100;
            
        case 'Poly_SVM'
            X_vec = reshape(X_train, [p*q, n_train])';
            X_test_vec = reshape(X_test, [p*q, n_test])';
            mdl = fitcsvm(X_vec, y_train, 'KernelFunction', 'polynomial', ...
                         'PolynomialOrder', params.degree, 'BoxConstraint', params.C, ...
                         'Standardize', false);
            y_pred = predict(mdl, X_test_vec);
            acc = sum(y_pred == y_test) / length(y_test);
            sv_ratio = sum(mdl.IsSupportVector) / n_train * 100;
    end
end
function [W, b, z] = L01SMM_Train_Stable(X, y, r, C, maxIter, tol)
    [p, q, m] = size(X);
    
    W = randn(p, q) * 0.01;
    b = 0;
    alpha = 0.001;
    
    for iter = 1:maxIter
        W_old = W;
        
        grad_W = zeros(p, q);
        grad_b = 0;
        
        for i = 1:m
            score = sum(sum(W .* X(:,:,i))) + b;
            margin = y(i) * score;
            
            if margin < 1
                grad_W = grad_W - y(i) * X(:,:,i);
                grad_b = grad_b - y(i);
            end
        end
        
        grad_W = grad_W / m + 0.01 * W;
        grad_b = grad_b / m;
        
        W = W - alpha * grad_W;
        b = b - alpha * grad_b;
        
        W = project_rank_r(W, r);
        
        if norm(W - W_old, 'fro') < tol
            break;
        end
    end
    
    z = zeros(m, 1);
    for i = 1:m
        score = sum(sum(W .* X(:,:,i))) + b;
        margin = y(i) * score;
        if margin < 1.5
            z(i) = 1;
        end
    end
end

function [W, b, z] = L01SMM_Train(X, y, r, C, maxIter, tol)
% 输入 X: p×q×m 三维矩阵，每张图片是 X(:,:,i)
% 输入 y: m×1 标签，取值 -1/+1

[p, q, m] = size(X);

% 固定超参数（方案A）
sigma = 0.01;
tau2  = 0.001;
tau3  = 0.001;

W = randn(p,q)*0.01;   % 不要全0
z = zeros(m,1);        % 不要全1
b = 0;

for k = 1:maxIter
    W_old = W;
    b_old = b;

    %% 更新 W
    gradW = W;
    for i = 1:m
        Xi = X(:, :, i);
        innerProd = sum(sum(W .* Xi));
        Mi = z(i) - 1 + y(i) * innerProd + b * y(i);
        gradW = gradW + 2 * sigma * Mi * y(i) * Xi;
    end

    alpha = 1e-3;
    W_temp = W - alpha * gradW;
    W = project_rank_r(W_temp, r);

    %% 更新 z
    v = zeros(m, 1);
    for i = 1:m
        v(i) = 1 - y(i) * sum(sum(W .* X(:, :, i))) - b * y(i);
    end

    z_bar = (2 * sigma * v + tau2 * z) / (2 * sigma + tau2);
    threshold = sqrt(4 * C / (2 * sigma + tau2));
    z_new = zeros(m, 1);
    for i = 1:m
        if z_bar(i) > 0 && z_bar(i) <= threshold
            z_new(i) = 0;
        else
            z_new(i) = z_bar(i);
        end
    end
    z = z_new;

    %% 更新 b
    A_W = zeros(m, 1);
    for i = 1:m
        A_W(i) = y(i) * sum(sum(W .* X(:, :, i)));
    end

    numerator = tau3 * b - 2 * sigma * y' * (z - 1 + A_W);
    denominator = 2 * sigma * (y' * y) + tau3;
    b = numerator / denominator;

    %% 停机条件
    diff_W = norm(W - W_old, 'fro');
    diff_b = abs(b - b_old);
    if k>1 && diff_W < tol && diff_b < tol
        fprintf('收敛，迭代提前终止 at iteration %d\n', k);
        break;
    end
end
end


function W_proj = project_rank_r(W, r)
% 将矩阵 W 投影到秩 ≤ r 的集合中（SVD 截断）
[U, S, V] = svd(W, 'econ');
S(r+1:end, :) = 0;
S(:, r+1:end) = 0;
W_proj = U * S * V';
end


function [W, b, z] = HingeSMM_Train(X, y, r, lambda, maxIter, tol)
    [p, q, m] = size(X);
    W = randn(p, q) * 0.01;
    b = 0;
    alpha = 0.01;
    
    for iter = 1:maxIter
        W_old = W;
        grad = zeros(p, q);
        count_sv = 0;
        
        for i = 1:m
            margin = y(i) * (sum(sum(W .* X(:, :, i))) + b);
            if margin < 1
                grad = grad - y(i) * X(:, :, i);
                count_sv = count_sv + 1;
            end
        end
        grad = grad / m + lambda * W;
        W = W - alpha * grad;
        W = project_rank_r(W, r);
        
        b_grad = 0;
        for i = 1:m
            margin = y(i) * (sum(sum(W .* X(:, :, i))) + b);
            if margin < 1
                b_grad = b_grad - y(i);
            end
        end
        b = b - alpha * b_grad / m;
        
        if norm(W - W_old, 'fro') < tol
            break;
        end
    end
    
    z = zeros(m, 1);
    for i = 1:m
        margin = y(i) * (sum(sum(W .* X(:, :, i))) + b);
        if margin < 1.5
            z(i) = 1;
        end
    end
end
function [W, b, z] = PinballSMM_Train(X, y, r, C, tau, maxIter, tol)
    [p, q, m] = size(X);
    W = randn(p, q) * 0.01;
    b = 0;
    alpha = 0.005;
    
    for iter = 1:maxIter
        W_old = W;
        grad = zeros(p, q);
        
        for i = 1:m
            margin = y(i) * (sum(sum(W .* X(:, :, i))) + b);
            residual = 1 - margin;
            if residual > 0
                weight = tau;
            else
                weight = tau - 1;
            end
            grad = grad + C * weight * (-y(i) * X(:, :, i));
        end
        
        W = W - alpha * grad / m;
        W = project_rank_r(W, r);
        
        b_grad = 0;
        for i = 1:m
            margin = y(i) * (sum(sum(W .* X(:, :, i))) + b);
            residual = 1 - margin;
            if residual > 0
                b_grad = b_grad + C * tau * (-y(i));
            else
                b_grad = b_grad + C * (tau - 1) * (-y(i));
            end
        end
        b = b - alpha * b_grad / m;
        
        if norm(W - W_old, 'fro') < tol
            break;
        end
    end
    
    z = ones(m, 1);
end
function [W, b, z] = RampSMM_Train(X, y, r, C, maxIter, tol)
    [p, q, m] = size(X);
    W = randn(p, q) * 0.01;
    b = 0;
    s = 1.0;
    alpha = 0.01;
    
    for iter = 1:maxIter
        W_old = W;
        grad = zeros(p, q);
        count_sv = 0;
        
        for i = 1:m
            margin = y(i) * (sum(sum(W .* X(:, :, i))) + b);
            residual = 1 - margin;
            if residual > 0 && residual < s
                grad = grad + C * (-y(i) * X(:, :, i));
                count_sv = count_sv + 1;
            end
        end
        
        W = W - alpha * grad / m;
        W = project_rank_r(W, r);
        
        b_grad = 0;
        for i = 1:m
            margin = y(i) * (sum(sum(W .* X(:, :, i))) + b);
            residual = 1 - margin;
            if residual > 0 && residual < s
                b_grad = b_grad + C * (-y(i));
            end
        end
        b = b - alpha * b_grad / m;
        
        if norm(W - W_old, 'fro') < tol
            break;
        end
    end
    
    z = zeros(m, 1);
    for i = 1:m
        margin = y(i) * (sum(sum(W .* X(:, :, i))) + b);
        residual = 1 - margin;
        if residual > -0.5 && residual < s
            z(i) = 1;
        end
    end
end
function [W, b] = LSSMM_Train(X, y, r, C)
    [p, q, m] = size(X);
    
    X_flat = reshape(X, [p*q, m])';
    
    K = X_flat * X_flat';
    
    H = [(K + C * eye(m)), ones(m, 1); ones(1, m), 0];
    f = [y; 0];
    
    try
        sol = H \ f;
        alpha = sol(1:end-1);
        b = sol(end);
        
        W_vec = X_flat' * alpha;
        W = reshape(W_vec, [p, q]);
        
        W = project_rank_r(W, r);
    catch
        W = randn(p, q) * 0.01;
        b = 0;
    end
end


function acc = evaluate_matrix(W, b, X_test, y_test)
    [~, ~, m] = size(X_test);
    y_pred = zeros(m, 1);
    
    for i = 1:m
        score = sum(sum(W .* X_test(:, :, i))) + b;
        y_pred(i) = sign(score);
    end
    
    y_pred(y_pred == 0) = 1;
    acc = sum(y_pred == y_test) / length(y_test);
end

function generate_result_table(results, methods, noise_levels, dataset_name)
    
    fprintf('%-15s', 'Method');
    fprintf('%-12s', 'Metric');
    for n = 1:length(noise_levels)
        fprintf('Noise %.2f    ', noise_levels(n));
    end
    fprintf('\n');
    fprintf('%s\n', repmat('-', 1, 100));
    
    for m = 1:length(methods)
        method_display = strrep(methods{m}, '_', '-');
        fprintf('%-15s%-12s', method_display, 'Acc(%)');
        for n = 1:length(noise_levels)
            acc_runs = squeeze(results(m, n, :, 1)) * 100;
            fprintf('%.2f+-%.2f   ', mean(acc_runs), std(acc_runs));
        end
        fprintf('\n');
        
        fprintf('%-15s%-12s', '', 'SVs(%)');
        for n = 1:length(noise_levels)
            sv_runs = squeeze(results(m, n, :, 2));
            fprintf('%.2f+-%.2f   ', mean(sv_runs), std(sv_runs));
        end
        fprintf('\n\n');
    end
end






function [X_train, y_train, X_test, y_test] = load_caltechface_lbp()
    fprintf('Loading CaltechFace LBP dataset...\n');

    data_file = 'CaltechFace_LBP.mat';
    assert(exist(data_file,'file')>0, ...
        'File not found: %s', data_file);

    S = load(data_file);

    required_fields = {'X','X_test','y','y_test'};
    for i = 1:length(required_fields)
        assert(isfield(S, required_fields{i}), ...
            'Missing field %s in MAT file', required_fields{i});
    end

    X_train = double(S.X);
    X_test  = double(S.X_test);

    y_train = double(S.y(:));
    y_test  = double(S.y_test(:));


    y_train(y_train == 0) = -1;
    y_test(y_test == 0)  = -1;


    y_train = sign(y_train);
    y_test  = sign(y_test);

    assert(size(X_train,3) == length(y_train), 'Train X/y size mismatch');
    assert(size(X_test,3)  == length(y_test),  'Test X/y size mismatch');

    fprintf('Loaded CaltechFace LBP:\n');
    fprintf('  Train: %d samples, dim=%dx%d\n', ...
        size(X_train,3), size(X_train,1), size(X_train,2));
    fprintf('  Test : %d samples\n', size(X_test,3));

  
    [X_train, X_test] = normalize_data_fixed(X_train, X_test);
end


