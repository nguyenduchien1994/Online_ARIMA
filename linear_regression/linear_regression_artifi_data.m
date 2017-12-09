%Linear Regression
%Load data from real artifical data
data_y = load('artificial_data.mat');
data_y = (data_y.data_d0)';
data_x = (1:1:size(data_y, 1))';

%We choose about two thirds as our train data and the rest to be test data
train_size = round(size(data_y, 1) * 0.7);
xtrain = data_x(1:train_size, 1);
ytrain = data_y(1:train_size, 1);
xtest = data_x(train_size + 1:end, 1);
ytest = data_y(train_size + 1:end, 1);

%Use fitlm() to build the linear model
mdl = fitlm(xtrain, ytrain, 'poly9', 'RobustOpts', 'on')
%Get coefficient from the fitlm model
%estimate_1 = 1 + (-0.0011825) * xtrain + (2.4641e-09) * xtrain .^2;
estimate_1_train = 1 + (-1.461e-08) * xtrain .^5 + (3.3831e-10) * xtrain .^6 + (-2.9154e-12) * xtrain .^7 + (1.1117e-14) * xtrain .^8 + (-1.5844e-17) * xtrain .^9;
estimate_1_test = 1 + (-1.461e-08) * xtest .^5 + (3.3831e-10) * xtest .^6 + (-2.9154e-12) * xtest .^7 + (1.1117e-14) * xtest .^8 + (-1.5844e-17) * xtest .^9;
MSE_fitlm_train = immse(estimate_1_train, ytrain)
MSE_fitlm_test = immse(estimate_1_test, ytest)

%Use robustfit() to build the linear model
%Get coefficient from the robust fit 
rob_coeff = robustfit(xtrain, ytrain);
estimate_2_train = rob_coeff(1) + rob_coeff(2) * xtrain;
estimate_2_test = rob_coeff(1) + rob_coeff(2) * xtest;
MSE_robustfit_train = immse(estimate_2_train, ytrain)
MSE_robustfit_test = immse(estimate_2_test, ytest)

%Use regress() to build the linear model
b = regress(ytrain, xtrain);
estimate_3_train = b * xtrain;
estimate_3_test = b * xtest;
MSE_regress_train = immse(estimate_3_train, ytrain)
MSE_regress_test = immse(estimate_3_test, ytest)

%Plot the train dand test ata and estimates
figure(1);
scatter(xtrain, ytrain);
hold on;
plot(xtrain, estimate_1_train, xtrain, estimate_2_train , xtrain, estimate_3_train);
hold off;
title('Artificial train data Time Series Linear Regression Plot on train data');
leg1 = legend('Artificial train data', 'fitlm() prediction', 'robustfit() prediction', 'regress() prediction');
xlabel('Index');
ylabel('Artificial data');

figure(2);
scatter(xtest, ytest);
hold on;
plot(xtest, estimate_1_test, xtest, estimate_2_test , xtest, estimate_3_test);
hold off;
title('Artificial test data Time Series Linear Regression Plot on test data');
leg1 = legend('Artificial test data', 'fitlm() prediction', 'robustfit() prediction', 'regress() prediction');
xlabel('Index');
ylabel('Artificial data');