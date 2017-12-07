import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA

# Load data
artificial_data = scipy.io.loadmat('artificial_data.mat')
X = np.array(artificial_data['data_d0'][0])
X_diff = np.array(artificial_data['data_d1'][0])

# Visualize original data
plt.figure()
plt.plot(X)
plt.title('Artificial data')
plt.xlabel('Index')
plt.ylabel('Observed value')

# Check for stationarity of the original data using the KPSS and
# Augmented Dickey-Fuller tests
print('Augmented Dickey-Fuller test: p=%f' % sm.tsa.stattools.adfuller(X)[1])
print('KPSS test: p=%f' % sm.tsa.stattools.kpss(X)[1])

# Check for stationarity of the differenced data using the KPSS and
# Augmented Dickey-Fuller tests
print('Augmented Dickey-Fuller test: p=%f' % sm.tsa.stattools.adfuller(X_diff)[1])
print('KPSS test: p=%f' % sm.tsa.stattools.kpss(X_diff)[1])

# Visualize the differenced data
plt.figure()
plt.plot(X_diff)
plt.title('Artificial data after first-differencing')
plt.xlabel('Index')
plt.ylabel('Observed value')

# Plot Autocorrelation Function and Partial Autocorrelation Function of
# the differenced data
sm.graphics.tsa.plot_acf(X_diff, lags=20)
plt.title('Autocorrelation of first-differenced data')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
sm.graphics.tsa.plot_pacf(X_diff, lags=20)
plt.title('Partial autocorrelation of first-differenced data')
plt.xlabel('Lag')
plt.ylabel('Partial autocorrelation')

# Compute AIC for different choices of p, q, and d=1 to choose ARIMA order
p_max = 5
d = 1
q_max = 5
aic = np.zeros((p_max + 1, q_max + 1))
for p in range(p_max + 1):
    for q in range(q_max + 1):
        try:        
            model = ARIMA(X, [p, d, q])
            fitted_model = model.fit(disp=-1)
            aic[p][q] = fitted_model.aic
        except:
            aic[p][q] = np.inf

# Choose values of p and q that gives the smallest AIC
p, q = np.unravel_index(aic.argmin(), aic.shape)

# Use the best values of p and q to model data (in-sample)
model = ARIMA(X, [p, d, q])
model_fit = model.fit(disp=-1)
X_est = model_fit.predict(start=1, end=len(X)-1, typ='levels')

# Plot predicted data along with original data
plt.figure()
plt.plot(X[1:])
plt.plot(X_est)
plt.title('Artificial data')
plt.xlabel('Index')
plt.ylabel('Observed value')
plt.legend(['True value', 'Predicted value'], loc=4)

# The following section is borrowed from IBM Data Science Experience
# https://datascience.ibm.com/exchange/public/entry/view/815137c868b916821dec777bdc23013c

# Use 70% of data to begin with, predict the remaining 30% using rolling forecast
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
history = [x for x in X_train]
X_pred = list()

# Perform rolling forecast
for t in range(len(X_test)):
    model = ARIMA(history, [p, d, q])
    model_fit = model.fit(disp=-1)
    X_pred.append(model_fit.forecast()[0])
    history.append(X_test[t])
    
# Plot the true remaining 30% data with the predicted values
plt.figure()
plt.plot(X_test)
plt.plot(X_pred)
plt.title('Last 30% of artificial data')
plt.xlabel('Index')
plt.ylabel('Observed value')
plt.legend(['True value', 'Predicted value'], loc=4)