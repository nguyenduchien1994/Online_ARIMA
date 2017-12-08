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
[fits_train, fits_test, mses_train, mses_test, best_deg_train, best_deg_test] = find_best_degree(xtrain, ytrain, xtest, ytest, max_deg);
best_deg_test

regress_d2 = fits_train(:, 2);
regress_d4 = fits_train(:, 4);
regress_best = fits_train(:, best_deg_test);
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
plot((1:1:max_deg)', mses_train(:), (1:1:max_deg)', mses_test(:));
title(['MSE for training data and test data with polynomial degree from 1 to ', num2str(max_deg)]);
legend('MSE for training data', 'MSE for testing data');
xlabel('degree');
ylabel('MSE');

%Second step, since we found the best degree, then we need to find the
%In(namida)
min_k = -25;
max_k = 25;
ek = min_k:1:max_k;
[fits_train, fits_test, mses_train, mses_test, best_k_train, best_k_test] = find_best_Ink(xtrain, ytrain, xtest, ytest, best_deg_test, min_k, max_k);
best_k_value = exp(best_k_test)

%(Plot MSE for train data and test data with best degree and different values of In(namida)
figure(3);
plot((ek)', mses_train(:), (ek)', mses_test(:));
title(['MSE for training data and test data with polynomial degree = ', num2str(best_deg_test) ,' In(k) from ', num2str(ek(1)),' to ', num2str(ek(end))]);
legend('MSE for training data', 'MSE for testing data');
xlabel('In(k)');
ylabel('MSE');

%Compare the result for non-regularized and L2-regularized fit
fit_L2_min = fits_test(:, best_k_test);
figure(4);
scatter(xtest, ytest);
hold on
plot(xtest, fits_test(:, best_deg_test), xtest, fit_L2_min);
hold off
title(['Original degree poly degree = ', num2str(best_deg_test), 'fit vs L2-regularized degree = ', num2str(best_k_value), ' fit with smallest MSE']);
legend('test data', 'original poly fit', 'L2-regularized fit with smallest MSE');
xlabel('date');
ylabel('Bitcoin price (dollars)');


%Plot the result for test data L2-regularized fit with best degree and best In(k) 
figure(5);
scatter(xtest, ytest);
hold on
plot(xtest, fit_L2_min);
hold off
title(['L2-regularized degree = ', num2str(best_deg_test), ' In(k) = ', num2str(best_k_test), ' fit with smallest MSE']);
legend('test data', 'L2-regularized fit with smallest MSE');
xlabel('date');
ylabel('Bitcoin price (dollars)');



