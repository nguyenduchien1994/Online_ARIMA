% Generation of artificial data
% ------------
%
% The purpose of this script is to generate a small-scale artificial
% dataset on which we can quickly test and debug our code.

% Clear workspace and close windows
clear all
close all

% Define number of points to generate
n = 300;

% Define AR and MA model coefficients
alpha = [0.6 -0.5 0.4 -0.4 0.3];
beta = [0.3 -0.2];

% Define model noise standard deviation
sigma = 0.3;

% Define the AR and MA model lags
p = length(alpha);
q = length(beta);

% Initialize variables for generated data
data_d1 = zeros(1, p);
noises = zeros(1, q);

% Generate observations
for i = 1:n
    noise = normrnd(0, sigma);
    observation = sum(data_d1(end:-1:end-p+1) .* alpha);
    observation = observation + sum(noises(end:-1:end-q+1) .* beta);
    observation = observation + noise;
    data_d1 = [data_d1 observation];
    noises = [noises noise];
end
data_d1 = data_d1(p+1:end);

% Obtain the original, undifferenced data
data_d0 = 0;
for i = 1:length(data_d1)
    data_d0(i+1) = data_d1(i) + data_d0(i);
end
data_d0 = data_d0(2:end);
    
% Save variables
save('artificial_data.mat', 'data_d1', 'data_d0', 'n')