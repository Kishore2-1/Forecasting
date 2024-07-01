#!/usr/bin/env python
# coding: utf-8

# # Import

# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tools.eval_measures import rmse


# In[24]:


file_path = 'JQ2Jdata_34812636.xlsx'
data = pd.read_excel(file_path)
print(data)


# # Preliminary Analysis

# In[25]:


file_path = 'JQ2Jdata_34812636.xlsx'
data = pd.read_excel(file_path)
# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['JQ2J'], marker='o', color='black')
plt.title('Total turnover and orders Over Time')
plt.xlabel('Year')
plt.ylabel('Total turnover and orders')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[26]:


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

# In[27]:


# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(data['JQ2J'], model='additive')
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
plt.plot(data.index, data['JQ2J'], color='black', label='Original')
plt.legend()
plt.tight_layout()
plt.show()

# Generate forecasts using the trend and seasonal components
forecast_model = ExponentialSmoothing(decomposition.trend, seasonal='additive', seasonal_periods=12)
forecast = forecast_model.fit().forecast(len(data))


# ### Coorelation Matrix

# In[28]:


# Compute the correlation matrix
correlation_matrix = data.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


# In[29]:


# Read the data
data = pd.read_excel('JQ2Jdata_34812636.xlsx')

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

# In[30]:


data = pd.read_excel( 'JQ2Jdata_34812636.xlsx')

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')
data['Year'] = data['Date'].dt.year

# Plot 
plt.figure(figsize=(10, 6))
plt.scatter(data['Year'], data['JQ2J'], color='black', alpha=0.5) 
plt.title('Scatter Plot for Total turnover and orders')
plt.xlabel('Year')
plt.ylabel('Total turnover and orders')
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Autocoorelation

# In[31]:


data = pd.read_excel('JQ2Jdata_34812636.xlsx')

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Plot ACF
plt.figure(figsize=(10, 6))
plot_acf(data['JQ2J'], lags=50, ax=plt.gca(), color='black')
plt.title('Autocorrelation Function for Total turnover and orders')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()


# ### Moving Average

# In[32]:


sns.set_style("whitegrid")

# Define file path
file = "JQ2Jdata_34812636.xlsx"
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

# In[33]:


# Load the data
data = pd.read_excel('JQ2Jdata_34812636.xlsx')

# Convert 'Date' column to datetime format and set as index
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')
data.set_index('Date', inplace=True)

# Fit Holt-Winters model with additive seasonality
fit1 = ExponentialSmoothing(data['JQ2J'], seasonal='add').fit()

# Fit Holt-Winters model with multiplicative seasonality
fit2 = ExponentialSmoothing(data['JQ2J'], seasonal='mul').fit()

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
plt.plot(data.index, data['JQ2J'], label='Actual', color='black')
plt.plot(pd.date_range(start=data.index[-1], periods=13, freq='M')[1:], forecast_additive, label='HW Additive Forecast (Next 1 Year)', color='Orange')
plt.plot(pd.date_range(start=data.index[-1], periods=13, freq='M')[1:], forecast_multiplicative, label='HW Multiplicative Forecast (Next 1 Year)', linestyle='-.', color='Green')
plt.xlabel('Year')
plt.ylabel('Total turnover and orders')
plt.title("Holt-Winters Exponential Smoothing Forecasting" )
plt.legend()
plt.show()


# ### Forecasted Values

# In[34]:


print("Additive Holt-Winters Forecast:")
print(forecast_additive)

print("\nMultiplicative Holt-Winters Forecast:")
print(forecast_multiplicative)


# ### RMSE

# In[35]:


# Calculate RMSE for the additive model
rmse_additive = rmse(data['JQ2J'], fit1.fittedvalues)

# Calculate RMSE for the multiplicative model
rmse_multiplicative = rmse(data['JQ2J'], fit2.fittedvalues)

print("RMSE for Additive Model:", rmse_additive)
print("RMSE for Multiplicative Model:", rmse_multiplicative)

