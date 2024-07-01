#!/usr/bin/env python
# coding: utf-8

# # Import

# In[49]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tools.eval_measures import rmse


# In[50]:


file_path = 'K226data_34812636.xlsx'
data = pd.read_excel(file_path)
print(data)


# # Preliminary Analysis

# In[51]:


file_path = 'K226data_34812636.xlsx'
data = pd.read_excel(file_path)
# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['K226'], marker='o', color='black')
plt.title('Extraction of Crude Petroleum and Natural Gas Over Time')
plt.xlabel('Year')
plt.ylabel('Extraction of Crude Petroleum and Natural Gas')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[52]:


# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Aggregate data by month
monthly_data = data.resample('M', on='Date').mean()

# Calculate yearly averages
yearly_averages = monthly_data.resample('Y').mean()

# Display the monthly data and yearly averages in a tabular format
print("Monthly Averages:")
print(monthly_data)
print("\nYearly Averages:")
print(yearly_averages)


# ### Decomposition

# In[53]:


# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(data['K226'], model='additive')

# Plot the decomposed components
plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(data.index, decomposition.trend, color='blue', label='Trend')
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(data.index, decomposition.seasonal, color='green', label='Seasonal')
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(data.index, decomposition.resid, color='red', label='Residual')
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(data.index, data['K226'], color='black', label='Original')
plt.legend()
plt.tight_layout()
plt.show()

# Generate forecasts using the trend and seasonal components
forecast_model = ExponentialSmoothing(decomposition.trend, seasonal='additive', seasonal_periods=12)
forecast = forecast_model.fit().forecast(len(data))


# ### Coorelation Matrix

# In[55]:


# Compute the correlation matrix
correlation_matrix = data.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


# In[56]:


# Read the data
data = pd.read_excel('K226data_34812636.xlsx')

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Calculate correlation matrix
correlation_matrix = data.corr()

# Plot 
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')
plt.title('Correlation Matrix')
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.tight_layout()
plt.show()

# Print the correlation matrix
print(correlation_matrix)


# ### Scatter Plot

# In[57]:


data = pd.read_excel( 'K226data_34812636.xlsx')

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')
data['Year'] = data['Date'].dt.year

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Year'], data['K226'], color='black', alpha=0.5) 
plt.title('Scatter Plot for Extraction of crude petroleum and natural gas')
plt.xlabel('Year')
plt.ylabel('Extraction of crude petroleum and natural gas')
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Autocoorelation

# In[58]:


data = pd.read_excel('K226data_34812636.xlsx')

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Plot ACF
plt.figure(figsize=(10, 6))
plot_acf(data['K226'], lags=50, ax=plt.gca(), color='black')
plt.title('Autocorrelation Function for Extraction of crude petroleum and natural gas')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()


# ### Moving Average

# In[59]:


sns.set_style("whitegrid")

# Define file path
file = "K226data_34812636.xlsx"
series = pd.read_excel(file, header=0, index_col=0, parse_dates=True).squeeze()

# Create empty Series for MA7
MA7 = pd.Series(index=series.index)

# Fill series with MA7
for i in np.arange(3, len(series) - 3):
    MA7[i] = np.mean(series[(i-3):(i+4)])

# Create empty Series for MA2x12
MA2x12 = pd.Series(index=series.index)

# Fill series with MA2x12
for i in np.arange(6, len(series) - 6):
    MA2x12[i] = np.sum(series[(i-6):(i+7)] * np.concatenate([[1/24], np.repeat(1/12, 11), [1/24]]))

# Plot original time series
series.plot()
MA7.plot()
MA2x12.plot()
plt.show()


# # Exponential Smoothing

# In[60]:


# Define file path
file_path =  "K226data_34812636.xlsx"

# Read data
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Fit Holt's linear exponential smoothing model
fit_linear = ExponentialSmoothing(data['K226'], trend='add', seasonal_periods=12).fit()

# Fit Holt-Winters model with additive seasonality
fit_additive = ExponentialSmoothing(data['K226'], seasonal_periods=12, seasonal='add').fit()

# Fit Holt-Winters model with multiplicative seasonality
fit_multiplicative = ExponentialSmoothing(data['K226'], seasonal_periods=12, seasonal='mul').fit()

forecast_linear = fit_linear.forecast(12)
forecast_additive = fit_additive.forecast(12)
forecast_multiplicative = fit_multiplicative.forecast(12)

# Plot forecasting
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['K226'], label='Actual', color='black')
plt.plot(forecast_linear.index, forecast_linear, label="Holt's Linear Exponential Smoothing Forecast", linestyle='--', color='red')
plt.plot(forecast_additive.index, forecast_additive, label='Additive Forecast', linestyle='--', color='orange')
plt.plot(forecast_multiplicative.index, forecast_multiplicative, label='Multiplicative Forecast', linestyle='--', color='green')
plt.xlabel('Date', color='white')
plt.ylabel('Extraction', color='white')
plt.title('Extraction of Crude Petroleum and Natural Gas', color='white')
plt.legend()
plt.show()


# ### Forecasted Values

# In[61]:


print("Holt's Linear Exponential Smoothing Forecast")
print(forecast_linear)

print("\nAdditive Holt-Winters Forecast:")
print(forecast_additive)

print("\nMultiplicative Holt-Winters Forecast:")
print(forecast_multiplicative)


# ### RMSE

# In[62]:


# Calculate RMSE for the Holt's Linear Exponential Smoothing (LES) model
rmse_linear = rmse(data['K226'], fit_linear.fittedvalues)

# Calculate RMSE for the additive model
rmse_additive = rmse(data['K226'], fit_additive.fittedvalues)

# Calculate RMSE for the multiplicative model
rmse_multiplicative = rmse(data['K226'], fit_multiplicative.fittedvalues)

print("RMSE for Holt's Linear Exponential Smoothing Model:", rmse_linear)
print("RMSE for Additive Model:", rmse_additive)
print("RMSE for Multiplicative Model:", rmse_multiplicative)

