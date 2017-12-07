clear all


%Linear Regression using method taught in class
%We do not use degree = 1 equation because obviously it is too simple for
%Bitcoin data

%Load data from real Bitcoin price data
data_y = flipud(csvread('bitcoin_daily_usd.csv', 2, 4));
data_x = (1:1:size(data_y, 1))';

%We choose about two thirds as our train data and the rest to be test data
train_size = 1200;
xtrain = data_x(1:train_size, 1);
ytrain = data_y(1:train_size, 1);
xtest = data_x(train_size + 1:end, 1);
ytest = data_y(train_size + 1:end, 1);

%First step, determine the poly degree in our model
max_deg = 30;

%Recreate the train data
temp_xtrain = xtrain;
ps = zeros(max_deg, max_deg + 1);
mses = zeros(max_deg, 2);
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
    mses(d, 1) = immse(fits_train(:, d), ytrain);
    mses(d, 2) = immse(fits_test(:, d), ytest); 
end

%Find the degree that give minimum MSE for test data
[~, best_deg] = min(mses(:, 2))

regress_d2 = fits_train(:, 2);
regress_d4 = fits_train(:, 4);
regress_best = fits_train(:, best_deg);
regress_d10 = fits_train(:, 10);
regress_d14 = fits_train(:, 14);

%Plot the predictio and origin data
figure(1);
scatter(xtrain, ytrain);
hold on
plot(xtrain, regress_d2, xtrain, regress_d4, xtrain, regress_best, xtrain, regress_d10, xtrain, regress_d14);
hold off
title('Results for polynomial fits with degree 2, 4, 10, 14 in train data');
legend('origin data', 'degree = 2', 'degree = 4', 'best degree', 'degree = 10', 'degree = 14');
xlabel('date');
ylabel('Bitcoin price(dollars)');

%Plot the MSE for train data and test data with degreee from 1 to
%Max_degree
figure(2);
plot((1:1:max_deg)', mses(:, 1), (1:1:max_deg)', mses(:, 2));
title(['MSE for training data and test data with polynomial degree from 1 to ', num2str(max_deg)]);
legend('MSE for training data', 'MSE for testing data');
xlabel('degree');
ylabel('MSE');

%Second step, since we found the best degree, then we need to find the
%In(namida)
ek = -25:1:25;
fits_train_b = zeros(size(ytrain, 1), size(ek, 2));
fits_test_b = zeros(size(ytest, 1), size(ek, 2));
mses_b = zeros(size(ek, 2), 2);
for j = 1:1:size(ek, 2);
    %Set k
    k = exp(ek(j));
    
    %Fix the degree to the best degree found in first step
    temp_xtrain = xtrain;
    for d = 1:1:(best_deg - 1)
        %Change the xtrain
        extra_x = xtrain.^(d + 1);
        temp_xtrain = [temp_xtrain, extra_x];
    end
    
    %Build model
    fit_b = ridge(ytrain, temp_xtrain, k, 0);
    
    %Calculate the fits
    fits_train_b(:, j) = fit_b(1);
    fits_test_b(:, j) = fit_b(1);
    for i = 1:1:(best_deg - 2)
        fits_train_b(:, j) = fits_train_b(:, j) + (xtrain .^ i) * fit_b(i + 1);
        fits_test_b(:, j) = fits_test_b(:, j) + (xtest .^ i) * fit_b(i + 1);
    end
    
    mses_b(j, 1) = immse(fits_train_b(:, j), ytrain);
    mses_b(j, 2) = immse(fits_test_b(:, j), ytest);
end

%Find the k that give minimum MSE for test data
[~, best_k] = min(mses(:, 2));
best_k_value = exp(best_k)

%(Plot MSE for train data and test data with best degree and different values of In(namida)
figure(3);
plot((ek)', mses_b(:, 1), (ek)', mses_b(:, 2));
title(['MSE for training data and test data with polynomial degree = ', num2str(best_deg) ,' In(k) from ', num2str(ek(1)),' to ', num2str(ek(end))]);
legend('MSE for training data', 'MSE for testing data');
xlabel('In(k)');
ylabel('MSE');

%Compare the result for non-regularized and L2-regularized fit
fit_L2_min = fits_test_b(:, best_k);
figure(4);
scatter(xtest, ytest);
hold on
plot(xtest, fits_test(:, best_deg), xtest, fit_L2_min);
hold off
title(['Original degree poly degree = ', num2str(best_deg), 'fit vs L2-regularized degree = ', num2str(best_k_value), ' fit with smallest MSE']);
legend('test data', 'original poly fit', 'L2-regularized fit with smallest MSE');
xlabel('date');
ylabel('Bitcoin price (dollars)');


%Plot the result for test data L2-regularized fit with best degree and best In(k) 
figure(5);
scatter(xtest, ytest);
hold on
plot(xtest, fit_L2_min);
hold off
title(['L2-regularized degree = ', num2str(best_deg), ' In(k) = ', num2str(best_k), ' fit with smallest MSE']);
legend('test data', 'L2-regularized fit with smallest MSE');
xlabel('date');
ylabel('Bitcoin price (dollars)');



