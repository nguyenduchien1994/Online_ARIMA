%Find the best degree given the dataset
function [fits_train, fits_test, mses_train, mses_test, best_deg_train, best_deg_test] = find_best_degree(xtrain, ytrain, xtest, ytest, max_deg)

%Recreate the train data
temp_xtrain = xtrain;
ps = zeros(max_deg, max_deg + 1);
mses_train = zeros(max_deg, 1);
mses_test = zeros(max_deg, 1);
fits_train = zeros(size(ytrain, 1), max_deg);
fits_test = zeros(size(ytest, 1), max_deg);

for d = 1:1:max_deg
    %Build model
    ps(d, 1:(d+1)) = ridge(ytrain, temp_xtrain, 0, 0);
    
    %Change the xtrain
    extra_x = xtrain.^(d + 1);
    temp_xtrain = [temp_xtrain, extra_x];
end

for d = 1:1:max_deg
    %Prediction results
    fits_train(:, d) = ps(d, 1);
    fits_test(:, d) = ps(d, 1);
    
    for i = 1:1:d
        fits_train(:, d) = fits_train(:, d) + (xtrain .^ i) * ps(d, i + 1);
        fits_test(:, d) = fits_test(:, d) + (xtest .^ i) * ps(d, i + 1);
    end
    
    %Calculate the MSE
    mses_train = immse(fits_train(:, d), ytrain);
    mses_test = immse(fits_test(:, d), ytest); 
end

%Find the degree that give minimum MSE for train data
[~, best_deg_train] = min(mses_train(:));

%Find the degree that give minimum MSE for test data
[~, best_deg_test] = min(mses_test(:));