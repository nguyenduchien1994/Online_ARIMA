clear all
clc

%Load data from the artifical data
data_y = load('artificial_data.mat');
data_y = (data_y.data_d0)';
data_x = (1:1:size(data_y, 1))';

%We choose about two thirds as our train data and the rest to be test data
train_size = round(size(data_y, 1) * 0.7);
ytrain = data_y(1:train_size, 1);
xtrain_index = data_x(1:train_size, 1);
ytest = data_y(train_size:end, 1);
xtest_index = data_x(train_size:end, 1);

%Determine how many previous days we are going to use to predict the next
%dlen value
max_day_len = 50;
mses_train = zeros(max_day_len, 1);
maes_train = zeros(max_day_len, 1);
mses_test = zeros(max_day_len, 1);
maes_test = zeros(max_day_len, 1);

for dlen = 1:1:max_day_len
    %Intialize the xtrain and xtest value
    xtrain = zeros((size(ytrain, 1) - dlen), dlen);
    xtest = zeros(size(ytest, 1), dlen);

    %We choose to not build xtrain data from index 1 to dlen data because before index 1, there
    %is previous y data exist
    for i = (dlen + 1):1:size(ytrain, 1)
        temp_xtrain_row = [];
        for j = (i - 1):-1:(i - dlen)
            temp_xtrain_row = [temp_xtrain_row, ytrain(j)];
        end
        
        xtrain(i - dlen, :) = temp_xtrain_row;
    end
    
    %Build the xtest data
    temp_ytest = ytest;
    for i = 1:1:dlen
       temp_ytest = [ytrain(end - (i - 1)); temp_ytest]; 
    end
    
    for i = 1:1:dlen
        temp_xtest_row = [];
        for j = i:1:(dlen + (i - 1))
            temp_xtest_row = [temp_xtest_row, temp_ytest(j)];
        end
        
        xtest(i, :) = temp_xtest_row;
    end
    
    for i = (dlen + 1):1:size(ytest, 1)
        temp_xtest_row_2 = [];
        for j = (i - 1):-1:(i - dlen)
            temp_xtest_row_2 = [temp_xtest_row_2, ytest(j)];
        end
        
        xtest(i, :) = temp_xtest_row_2;
    end

    %Get the coefficients using ridge()
    ps_train = ridge(ytrain((dlen + 1):end), xtrain, 0, 0);

    %Predict the train result
    fit_train = zeros(size(xtrain, 1), 1);
    for i = 1:1:size(xtrain, 1)
        temp = [1; (xtrain(i, :))'];
        fit_train(i) = temp' * ps_train;
    end

    %Predict the test result
    fit_test = zeros(size(ytest, 1), 1);
    for i = 1:1:size(ytest, 1)
        temp = [1; (xtest(i, :))'];
        fit_test(i) = temp' * ps_train;
    end

    %Calculate the mses and maes
    mses_train(dlen) = immse(fit_train, ytrain((dlen + 1):end));
    maes_train(dlen) = mae(ytrain((dlen + 1):end) - fit_train);
    mses_test(dlen) = immse(fit_test, ytest);
    maes_test(dlen) = mae(ytest - fit_test);
end

figure(1)
plot(1:1:max_day_len, mses_train, 1:1:max_day_len, mses_test, 1:1:max_day_len, maes_train, 1:1:max_day_len, maes_test);
title('Different previous day length vs MSEs and MAEs');
leg1 = legend('Train MSE', 'Test MSE', 'Train MAE', 'Test MAE');
xlabel('previous day length');
ylabel('MSE and MAE');

[~, best_day_length_test] = min(mses_test)

dlen = best_day_length_test;

%Intialize the xtrain and xtest value
xtrain = zeros((size(ytrain, 1) - dlen), dlen);
xtest = zeros(size(ytest, 1), dlen);

%We choose to not build xtrain data from index 1 to dlen data because before index 1, there
%is previous y data exist
for i = (dlen + 1):1:size(ytrain, 1)
    temp_xtrain_row = [];
    for j = (i - 1):-1:(i - dlen)
        temp_xtrain_row = [temp_xtrain_row, ytrain(j)];
    end

    xtrain(i - dlen, :) = temp_xtrain_row;
end

%Build the xtest data
temp_ytest = ytest;
for i = 1:1:dlen
   temp_ytest = [ytrain(end - (i - 1)); temp_ytest]; 
end

for i = 1:1:dlen
    temp_xtest_row = [];
    for j = i:1:(dlen + (i - 1))
        temp_xtest_row = [temp_xtest_row, temp_ytest(j)];
    end

    xtest(i, :) = temp_xtest_row;
end

for i = (dlen + 1):1:size(ytest, 1)
    temp_xtest_row_2 = [];
    for j = (i - 1):-1:(i - dlen)
        temp_xtest_row_2 = [temp_xtest_row_2, ytest(j)];
    end

    xtest(i, :) = temp_xtest_row_2;
end

%Get the coefficients using ridge()
ps_train = ridge(ytrain((dlen + 1):end), xtrain, 0, 0);

%Predict the train result
fit_train = zeros(size(xtrain, 1), 1);
for i = 1:1:size(xtrain, 1)
    temp = [1; (xtrain(i, :))'];
    fit_train(i) = temp' * ps_train;
end

%Predict the test result
fit_test = zeros(size(ytest, 1), 1);
for i = 1:1:size(ytest, 1)
    temp = [1; (xtest(i, :))'];
    fit_test(i) = temp' * ps_train;
end

figure(2)
plot(data_x, data_y, xtrain_index(dlen + 1:end), fit_train, xtest_index, fit_test);
title(['Regression Artificial data result (give minimum test MSE) for previous day length = ',  num2str(dlen)]);
leg1 = legend('True value', 'Fitted value', 'Forecast value');
xlabel('Index (Date)');
ylabel('Artificial data');

best_train_MSE = mses_train(best_day_length_test)
best_test_MSE = mses_test(best_day_length_test)
best_train_MAE = maes_train(best_day_length_test)
best_train_MAE = maes_test(best_day_length_test)