#!/usr/bin/env python
# coding: utf-8

# # Import

# In[20]:


#pip install pmdarima


# In[40]:


from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tools.eval_measures import rmse


# In[41]:


# Load the dataset
file_path =  'C:/Users/kishore/Desktop/forcasting/K54Ddata_34812636.xlsx'
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Plot the original data using a line plot
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['K54D'], label='K54D')
plt.title('Original Earning Data')
plt.xlabel('Date')
plt.ylabel('Earning')
plt.legend()
plt.grid(True)
plt.show()


# ## Summary

# In[42]:


# Compute descriptive statistics
statistics = data['K54D'].describe()

# Print the statistical summary
print("Statistical Summary for Average Weekly Earning Data:")
print(statistics)


# ## ACF and Partial ACF

# In[43]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

# Load the dataset
file_path = 'C:/Users/kishore/Desktop/forcasting/K54Ddata_34812636.xlsx'
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Plot ACF and PACF plots to identify model parameters
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data['K54D'], ax=ax[0], lags=40)
plot_pacf(data['K54D'], ax=ax[1], lags=40)
plt.show()


# # Exponential smoothing model

# In[44]:


# Load the data
data = pd.read_excel('K54Ddata_34812636.xlsx')

# Convert 'Date' column to datetime format and set as index
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')
data.set_index('Date', inplace=True)

# Fit Holt-Winters model with additive seasonality
fit1 = ExponentialSmoothing(data['K54D'], seasonal='add').fit()

# Fit Holt-Winters model with multiplicative seasonality
fit2 = ExponentialSmoothing(data['K54D'], seasonal='mul').fit()

# Get alpha, beta, and gamma values for additive model
alpha_additive = fit1.params['smoothing_level']
beta_additive = fit1.params['smoothing_trend']
gamma_additive = fit1.params['smoothing_seasonal']

# Get alpha, beta, and gamma values for multiplicative model
alpha_multiplicative = fit2.params['smoothing_level']
gamma_multiplicative = fit2.params['smoothing_seasonal']

print("Additive Model Parameters:")
print("Alpha:", alpha_additive)
print("Gamma:", gamma_additive)

print("\nMultiplicative Model Parameters:")
print("Alpha:", alpha_multiplicative)
print("Gamma:", gamma_multiplicative)

# Forecasting for the next 1 year (assuming monthly data)
forecast_additive = fit1.forecast(12)
forecast_multiplicative = fit2.forecast(12)

# Plotting the forecast
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['K54D'], label='Actual', color='black')
plt.plot(pd.date_range(start=data.index[-1], periods=13, freq='M')[1:], forecast_additive, label='HW Additive Forecast (Next 1 Year)', color='Orange')
plt.plot(pd.date_range(start=data.index[-1], periods=13, freq='M')[1:], forecast_multiplicative, label='HW Multiplicative Forecast (Next 1 Year)', linestyle='-.', color='Green')
plt.xlabel('Year')
plt.ylabel('Average Weekly Earning')
plt.title("Holt-Winters Exponential Smoothing Forecasting" )
plt.legend()
plt.show()


# In[45]:


# Calculate RMSE for the additive model
rmse_additive = rmse(data['K54D'], fit1.fittedvalues)

# Calculate RMSE for the multiplicative model
rmse_multiplicative = rmse(data['K54D'], fit2.fittedvalues)

print("RMSE for Additive Model:", rmse_additive)
print("RMSE for Multiplicative Model:", rmse_multiplicative)


# # SARIMA Forecasting

# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
file_path = 'C:/Users/kishore/Desktop/forcasting/K54Ddata_34812636.xlsx'
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Find the best parameters using auto_arima
auto_model = auto_arima(data['K54D'], seasonal=True, m=12, trace=True)

# Get the best model parameters
order = auto_model.order
seasonal_order = auto_model.seasonal_order

print("Best model parameters (p, d, q):", order)
print("Best seasonal parameters (P, D, Q, S):", seasonal_order)

# Fit SARIMAX model with the best parameters
model = SARIMAX(data['K54D'], order=order, seasonal_order=seasonal_order)
fit_model = model.fit()

# Generate forecasts until December 2024 (48 months)
forecast_index = pd.date_range(start=data.index[-1], periods=13, freq='M')[1:]  # Forecast index for the next 48 months
forecast = fit_model.forecast(steps=len(forecast_index))  # Forecast for the next 48 months

# Plot original data and forecasts
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['K54D'], label='Actual')
plt.plot(forecast_index, forecast, label='Forecast', linestyle='--')
plt.title('SARIMA Forecasting until December 2024')
plt.xlabel('Date')
plt.ylabel('Earning')
plt.legend()
plt.grid(True)
plt.show()


# ### RMSE

# In[48]:


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(data['K54D'].iloc[:len(forecast)], forecast))
print("Root Mean Squared Error (RMSE):", rmse)


# ## Print Forecsted values

# In[49]:


# Extract forecasted values from December 2023 to December 2024
forecast_dec_2023_to_dec_2024 = forecast[(forecast_index >= '2023-12-01') & (forecast_index <= '2025-01-01')]

# Display the forecasted values
print(forecast_dec_2023_to_dec_2024)

