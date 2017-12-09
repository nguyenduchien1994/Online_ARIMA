%Find the best In(k) given the dataset
function [fits_train, fits_test, mses, maes, best_k] = find_best_Ink(xtrain, ytrain, xtest, ytest, degree, min_k, max_k)

ek = min_k:1:max_k;
fits_train = zeros(size(ytrain, 1), size(ek, 2));
fits_test = zeros(size(ytest, 1), size(ek, 2));
mses = zeros(size(ek, 2), 2);
maes = zeros(size(ek, 2), 2);
best_k = zeros(2, 1);

for j = 1:1:size(ek, 2);
    %Set k
    k = exp(ek(j));
    
    %Fix the degree to the best degree found in first step
    temp_xtrain = xtrain;
    for d = 1:1:(degree - 1)
        %Change the xtrain
        extra_x = xtrain.^(d + 1);
        temp_xtrain = [temp_xtrain, extra_x];
    end
    
    %Build model
    fit_b = ridge(ytrain, temp_xtrain, k, 0);
    
    %Calculate the fits
    fits_train(:, j) = fit_b(1);
    fits_test(:, j) = fit_b(1);
    for i = 1:1:(degree - 2)
        fits_train(:, j) = fits_train(:, j) + (xtrain .^ i) * fit_b(i + 1);
        fits_test(:, j) = fits_test(:, j) + (xtest .^ i) * fit_b(i + 1);
    end
    
    %Calculate the MSE
    mses(j, 1) = immse(fits_train(:, j), ytrain);
    mses(j, 2) = immse(fits_test(:, j), ytest);
    
    %Calculate the MAE
    maes(d, 1) = mae(ytrain - fits_train(:, j));
    maes(d, 2) = mae(ytest - fits_test(:, j));
end

%Find the k that give minimum MSE for train data
[~, best_k(1)] = min(mses(:, 1));

%Find the k that give minimum MSE for test data
[~, best_k(2)] = min(mses(:, 2));