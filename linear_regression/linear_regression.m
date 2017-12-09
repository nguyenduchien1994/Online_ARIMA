%Linear Regression
%Load data from real Bitcoin price data
data_y = flipud(csvread('bitcoin_daily_usd.csv', 2, 4));
data_x = (1:1:size(data_y, 1))';

%We choose about two thirds as our train data and the rest to be test data
train_size = 1200;
xtrain = data_x(1:train_size, 1);
ytrain = data_y(1:train_size, 1);
xtest = data_x(train_size + 1:end, 1);
ytest = data_y(train_size + 1:end, 1);

%Use fitlm() to build the linear model
mdl = fitlm(xtrain, ytrain, 'poly9', 'RobustOpts', 'on')
%Get coefficient from the fitlm model
%estimate_1 = 1 + (-0.0011825) * xtrain + (2.4641e-09) * xtrain .^2;
estimate_1_train = 1 + (9.5588e-15) * xtrain .^6 + (-1.8643e-17) * xtrain .^7 + (1.2087e-20) * xtrain .^8 + (-2.5514e-24) * xtrain .^9;
estimate_1_test = 1 + (9.5588e-15) * xtest .^6 + (-1.8643e-17) * xtest .^7 + (1.2087e-20) * xtest .^8 + (-2.5514e-24) * xtest .^9;
MSE_fitlm_train = (sum((estimate_1_train - ytrain) .^ 2))^(0.5)/size(ytrain, 1)
MSE_fitlm_test = (sum((estimate_1_test - ytest) .^ 2))^(0.5)/size(ytest, 1)

%Use robustfit() to build the linear model
%Get coefficient from the robust fit 
rob_coeff = robustfit(xtrain, ytrain);
estimate_2_train = rob_coeff(1) + rob_coeff(2) * xtrain;
estimate_2_test = rob_coeff(1) + rob_coeff(2) * xtest;
MSE_robustfit_train = (sum((estimate_2_train - ytrain) .^ 2))^(0.5)/size(ytrain, 1)
MSE_robustfit_test = (sum((estimate_2_test - ytest) .^ 2))^(0.5)/size(ytest, 1)

%Use regress() to build the linear model
b = regress(ytrain, xtrain);
estimate_3_train = b * xtrain;
estimate_3_test = b * xtest;
MSE_regress_train = (sum((estimate_3_train - ytrain) .^ 2))^(0.5)/size(ytrain, 1)
MSE_regress_test = (sum((estimate_3_test - ytest) .^ 2))^(0.5)/size(ytest, 1)

%Plot the train dand test ata and estimates
figure(1);
scatter(xtrain, ytrain);
hold on;
plot(xtrain, estimate_1_train, xtrain, estimate_2_train , xtrain, estimate_3_train);
hold off;
title('Bitcoin Daily Price Time Series Linear Regression Plot on train data');
leg1 = legend('Bitcoin price', 'fitlm() prediction', 'robustfit() prediction', 'regress() prediction');
xlabel('Time from 28-April-2013 to 31-July-2017');
ylabel('Bitcoin Daily Price(in USD)');

figure(2);
scatter(xtest, ytest);
hold on;
plot(xtest, estimate_1_test, xtest, estimate_2_test , xtest, estimate_3_test);
hold off;
title('Bitcoin Daily Price Time Series Linear Regression Plot on test data');
leg1 = legend('Bitcoin price', 'fitlm() prediction', 'robustfit() prediction', 'regress() prediction');
xlabel('Time from 28-April-2013 to 31-July-2017');
ylabel('Bitcoin Daily Price(in USD)');