%Linear Regression
%Load data from real Bitcoin price data
data_y = flipud(csvread('bitcoin_daily_usd.csv', 2, 4));
data_x = (1:1:size(data_y, 1))';

%Use fitlm() to build the linear model
mdl = fitlm(data_x, data_y, 'poly9', 'RobustOpts', 'on')
%Get coefficient from the fitlm model
%estimate_1 = 1 + (-0.0011825) * data_x + (2.4641e-09) * data_x .^2;
estimate_1 = 1 + (9.5588e-15) * data_x .^6 + (-1.8643e-17) * data_x .^7 + (1.2087e-20) * data_x .^8 + (-2.5514e-24) * data_x .^9;
MSE_fitlm = (sum((estimate_1 - data_y) .^ 2))^(0.5)/size(data_y, 1)

%Use robustfit() to build the linear model
%Get coefficient from the robust fit 
rob_coeff = robustfit(data_x, data_y);
estimate_2 = rob_coeff(1) + rob_coeff(2) * data_x;
MSE_robustfit = (sum((estimate_2 - data_y) .^ 2))^(0.5)/size(data_y, 1)

%Use regress() to build the linear model
b = regress(data_y, data_x);
estimate_3 = b * data_x;
MSE_regress = (sum((estimate_3 - data_y) .^ 2))^(0.5)/size(data_y, 1)

%Use the method in the HW7


%Plot the origin data and estimates
figure(1);
scatter(data_x, data_y);
hold on;
plot(data_x, estimate_1, data_x, estimate_2 , data_x, estimate_3);
hold off;
title('Bitcoin Daily Price Time Series Linear Regression Plot');
leg1 = legend('Bitcoin price', 'fitlm() prediction', 'robustfit() prediction', 'regress() prediction');
xlabel('Time from 28-April-2013 to 31-July-2017');
ylabel('Bitcoin Daily Price (in USD)');