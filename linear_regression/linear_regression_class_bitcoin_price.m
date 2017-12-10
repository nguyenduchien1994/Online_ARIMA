%Linear Regression using method taught in class for Bitcoin data

%Load data from real Bitcoin price data
data_y = flipud(csvread('bitcoin_daily_usd.csv', 2, 4));
data_y = data_y(500:999);
data_x = (1:1:size(data_y, 1))';

%We choose about two thirds as our train data and the rest to be test data
train_size = round(size(data_y, 1) * 0.7);
xtrain = data_x(1:train_size, 1);
ytrain = data_y(1:train_size, 1);
xtest = data_x(train_size + 1:end, 1);
ytest = data_y(train_size + 1:end, 1);

%First step, determine the poly degree in our model
max_deg = 30;
[fits_train, fits_test, mses, maes, best_deg] = find_best_degree(xtrain, ytrain, xtest, ytest, max_deg);

mses_train = mses(:, 1);
mses_test = mses(:, 2);
maes_train = maes(:, 1);
maes_test = maes(:, 2);

best_deg_train = best_deg(1)
best_deg_test = best_deg(2)

regress_d2 = fits_train(:, 2);
regress_d4 = fits_train(:, max_deg);
regress_d10 = fits_train(:, round(max_deg/2));
regress_d14 = fits_train(:, round(max_deg/3));
regress_best_train = fits_train(:, best_deg_train);
regress_best_test = fits_train(:, best_deg_test);

%Plot the prediction and origin data
figure(1);
scatter(xtrain, ytrain);
hold on
plot(xtrain, regress_d2, xtrain, regress_d4, xtrain, regress_d10, xtrain, regress_d14, xtrain, regress_best_train, xtrain, regress_best_test);
hold off
title('Results for polynomial fits with various degrees in train data');
legend('origin data', 'degree = 2', ['degree = ', num2str(max_deg)], ['degree = ', num2str(round(max_deg/2))], ['degree = ', num2str(round(max_deg/3))], 'best train degree',  'best test degree');
xlabel('date');
ylabel('Artificial data');

%Plot the MSE for train data and test data with degreee from 1 to
%Max_degree
figure(2);
plot((1:1:max_deg)', mses_train(:), (1:1:max_deg)', mses_test(:));
title(['MSE for training data and test data with polynomial degree from 1 to ', num2str(max_deg)]);
legend('MSE for training data', 'MSE for testing data');
xlabel('degree');
ylabel('MSE');

%Plot the MAE for train data and test data with degreee from 1 to
%Max_degree
figure(3);
plot((1:1:max_deg)', maes_train(:), (1:1:max_deg)', maes_test(:));
title(['MAE for training data and test data with polynomial degree from 1 to ', num2str(max_deg)]);
legend('MAE for training data', 'MAE for testing data');
xlabel('degree');
ylabel('MAE');

%Second step, since we found the best degree for train data and test data, then we need to find the
%In(namida) with L2 regularization for the best deg train and best deg test
best_deg = [best_deg_train; best_deg_test];
min_k = -25;
max_k = 25;
ek = min_k:1:max_k;

for i = 1:1:2 
    [fits_train, fits_test, mses, maes, best_k] = find_best_Ink(xtrain, ytrain, xtest, ytest, best_deg(i), min_k, max_k);
    
    mses_train = mses(:, 1);
    mses_test = mses(:, 2);
    maes_train = maes(:, 1);
    maes_test = maes(:, 2);
    
    best_k_value = exp(best_k(i))
    

    %Plot MSE for train data and test data with best degree and different values of In(namida)
    figure(4);
    subplot(2, 1, i);
    plot((ek)', mses_train(:), (ek)', mses_test(:));
    title(['MSE for training data and test data with polynomial degree = ', num2str(best_deg(i)) ,' In(k) from ', num2str(ek(1)),' to ', num2str(ek(end))]);
    legend('MSE for training data', 'MSE for testing data');
    xlabel('In(k)');
    ylabel('MSE');
    
    %Plot MAE for train data and test data with best degree and different values of In(namida)
    figure(5);
    subplot(2, 1, i);
    plot((ek)', maes_train(:), (ek)', maes_test(:));
    title(['MAE for training data and test data with polynomial degree = ', num2str(best_deg(i)) ,' In(k) from ', num2str(ek(1)),' to ', num2str(ek(end))]);
    legend('MAE for training data', 'MSE for testing data');
    xlabel('In(k)');
    ylabel('MAE');

    %Compare the result for non-regularized and L2-regularized fit
    fit_L2_min = fits_test(:, best_k(i));
    figure(5);
    subplot(2, 1, i);
    scatter(xtest, ytest);
    hold on
    plot(xtest, fits_test(:, best_deg(i)), xtest, fit_L2_min);
    hold off
    title(['Original degree poly degree = ', num2str(best_deg(i)), 'fit vs L2-regularized degree = ', num2str(best_k_value), ' fit with smallest MSE']);
    legend('test data', 'original poly fit', 'L2-regularized fit with smallest MSE');
    xlabel('date');
    ylabel('Artificial data');


    %Plot the result for test data L2-regularized fit with best degree and best In(k) 
    figure(6);
    subplot(2, 1, i);
    scatter(xtest, ytest);
    hold on
    plot(xtest, fit_L2_min);
    hold off
    title(['L2-regularized degree = ', num2str(best_deg(i)), ' In(k) = ', num2str(best_k(i)), ' fit with smallest MSE']);
    legend('test data', 'L2-regularized fit with smallest MSE');
    xlabel('date');
    ylabel('Artificial data');
end