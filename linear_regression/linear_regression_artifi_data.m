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
regress_fitlm_train = 1 + (-1.461e-08) * xtrain .^5 + (3.3831e-10) * xtrain .^6 + (-2.9154e-12) * xtrain .^7 + (1.1117e-14) * xtrain .^8 + (-1.5844e-17) * xtrain .^9;
regress_fitlm_test = 1 + (-1.461e-08) * xtest .^5 + (3.3831e-10) * xtest .^6 + (-2.9154e-12) * xtest .^7 + (1.1117e-14) * xtest .^8 + (-1.5844e-17) * xtest .^9;
MSE_fitlm_train = immse(regress_fitlm_train, ytrain)
MAE_fitlm_train = mae(ytrain - regress_fitlm_train)
MSE_fitlm_test = immse(regress_fitlm_test, ytest)
MAE_fitlm_test = mae(ytest -regress_fitlm_test)

%OLS
xmean = mean(xtrain);
ymean = mean(ytrain);
n = size(xtrain, 1);
coeff1 = (n * xmean * ymean - sum(xtrain .* ytrain)) / (n * (xmean^2) - sum(xtrain.^2));
coeff2 = ymean - coeff1 * xmean;
regress_ols_train = coeff1 * xtrain + coeff2;
regress_ols_test = coeff1 * xtest + coeff2;

w_ols = coeff1
b_ols = coeff2
MSE_ols_train = immse(regress_ols_train, ytrain)
MAE_ols_train = mae(ytrain - regress_ols_train)
MSE_ols_test = immse(regress_ols_test, ytest)
MAE_ols_test = mae(ytest - regress_ols_test)

%CAUCHY, FAIR, HUBER, TALWAR
coeff_cauchy = robustfit(xtrain, ytrain, 'cauchy', 2.385, 'on');
coeff_fair = robustfit(xtrain, ytrain, 'fair', 1.4, 'on');
coeff_huber = robustfit(xtrain, ytrain, 'huber', 1.345, 'on');
coeff_talwar = robustfit(xtrain, ytrain, 'talwar', 2.795, 'on');

regress_cauchy_train = coeff_cauchy(1) + coeff_cauchy(2) * xtrain;
regress_cauchy_test = coeff_cauchy(1) + coeff_cauchy(2) * xtest;
MSE_cauchy_train = immse(regress_cauchy_train, ytrain)
MAE_cauchy_train = mae(ytrain - regress_cauchy_train)
MSE_cauchy_test = immse(regress_cauchy_test, ytest)
MAE_cauchy_test = mae(ytest - regress_cauchy_test)

regress_fair_train = coeff_fair(1) + coeff_fair(2) * xtrain;
regress_fair_test = coeff_fair(1) + coeff_fair(2) * xtest;
MSE_fair_train = immse(regress_fair_train, ytrain)
MAE_fair_train = mae(ytrain - regress_fair_train)
MSE_fair_test = immse(regress_fair_test, ytest)
MAE_fair_test = mae(ytest - regress_fair_test)

regress_huber_train = coeff_huber(1) + coeff_huber(2) * xtrain;
regress_huber_test = coeff_huber(1) + coeff_huber(2) * xtest;
MSE_huber_train = immse(regress_huber_train, ytrain)
MAE_huber_train = mae(ytrain - regress_huber_train)
MSE_huber_test = immse(regress_huber_test, ytest)
MAE_huber_test = mae(ytest - regress_huber_test)

regress_talwar_train = coeff_talwar(1) + coeff_talwar(2) * xtrain;
regress_talwar_test = coeff_talwar(1) + coeff_talwar(2) * xtest;
MSE_talwar_train = immse(regress_talwar_train, ytrain)
MAE_talwar_train = mae(ytrain - regress_talwar_train)
MSE_talwar_test = immse(regress_talwar_test, ytest)
MAE_talwar_test = mae(ytest - regress_talwar_test)

w_huber = coeff_huber(2)
b_huber = coeff_huber(1)

%Plot the train dand test ata and estimates
figure(1);
scatter(xtrain, ytrain);
hold on;
plot(xtrain, regress_fitlm_train, xtrain, regress_ols_train, xtrain, regress_cauchy_train);
hold off;
title('Artificial train data Time Series Linear Regression Plot on train data');
leg1 = legend('Artificial train data', 'fitlm() prediction', 'OLS prediction', 'cauchy prediction');
xlabel('Index');
ylabel('Artificial data');

figure(2);
scatter(xtest, ytest);
hold on;
plot(xtest, regress_fitlm_test, xtest, regress_ols_test, xtest, regress_cauchy_test);
hold off;
title('Artificial test data Time Series Linear Regression Plot on test data');
leg1 = legend('Artificial test data', 'fitlm() prediction', 'OLS prediction', 'cauchy prediction');
xlabel('Index');
ylabel('Artificial data');