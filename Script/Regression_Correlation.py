#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[16]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.tsa.api import Holt
# Setting plotting style
sns.set_style("whitegrid")
from sklearn.metrics import mean_squared_error


# In[17]:


# Load data
file = "FTSEdata_34812636.xlsx"
series = pd.read_excel(file, sheet_name='Sheet1', header=0).squeeze()
Openfull0 = pd.read_excel(file, sheet_name='Sheet2', header=0)


# In[18]:


# Extracting individual variables
Open = series.Open
K54D = series.K54D
EAFV = series.EAFV
K226 = series.K226
JQ2J = series.JQ2J


# ## Holt's Linear Model Forecasting

# In[19]:


fit1 = Holt(K54D).fit(optimized=True)
fcast1 = fit1.forecast(12).rename("Additive 2 damped trend")

fit2 = Holt(EAFV).fit(optimized=True)
fcast2 = fit2.forecast(12).rename("Additive 2 damped trend")

fit3 = Holt(K226).fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
fcast3 = fit3.forecast(12).rename("Additive 2 damped trend")

fit4 = Holt(JQ2J).fit(optimized=True)
fcast4 = fit4.forecast(12).rename("Additive 2 damped trend")


# In[20]:


# Fitted values and forecast values arrays
a1 = np.array(fit1.fittedvalues)
a2 = np.array(fit2.fittedvalues)
a3 = np.array(fit3.fittedvalues)
a4 = np.array(fit4.fittedvalues)

v1 = np.array(fcast1)
v2 = np.array(fcast2)
v3 = np.array(fcast3)
v4 = np.array(fcast4)


# In[21]:


# Ordinary Least Squares (OLS) Regression
formula = 'Open ~ K54D + EAFV + K226 + JQ2J'
results = ols(formula, data=series).fit()
results.summary()


# In[22]:


# Extracting regression coefficients
b0 = results.params.Intercept
b1 = results.params.K54D
b2 = results.params.EAFV
b3 = results.params.K226
b4 = results.params.JQ2J

# Fitted part of the Open forecast
F = a1
for i in range(288):
    F[i] = b0 + a1[i]*b1 + a2[i]*b2 + a3[i]*b3 + a4[i]*b4

# Forecast values of Open
E = v1
for i in range(12):
    E[i] = b0 + v1[i]*b1 + v2[i]*b2 + v3[i]*b3 + v4[i]*b4
    
    
# Combining fitted values and forecast values
K = np.append(F, E)

# Forecasting Error
values = Openfull0.Openfull[0:288]
Error = values - F
MSE = sum(Error ** 2) * 1.0 / len(F)
LowerE = E - 1.282*np.sqrt(MSE)
UpperE = E + 1.282*np.sqrt(MSE)


# # Plotting

# In[23]:


# Plotting
fig, ax = plt.subplots(5, 1, figsize=(10, 15))

# Plotting forecasts for individual variables
for i, (fit, forecast, data, title) in enumerate(zip([fit1, fit2, fit3, fit4], [fcast1, fcast2, fcast3, fcast4], [K54D, EAFV, K226, JQ2J], ['K54D', 'EAFV', 'K226', 'JQ2J'])):
    fit.fittedvalues.plot(color='red', ax=ax[i])
    forecast.plot(color='red', legend=True, ax=ax[i])
    data.plot(color='black', legend=True, ax=ax[i])
    ax[i].set_title(f'Forecast of {title} with Holt linear method')

# Plotting Open forecast
ax[4].plot(K, color='red', label='Forecast values')
ax[4].plot(Open, color='black', label='Original data')
ax[4].fill_between(range(len(F), len(K)), LowerE, UpperE, color='b', alpha=.1, label="Confidence Interval")
ax[4].legend()
ax[4].set_title('Open regression forecast with confidence interval')

plt.tight_layout()
plt.show()


# # Predicted values

# In[24]:


# Print predicted values
print("Predicted values for Open:")
for i, value in enumerate(E, 1):
    print(f"Forecast {i}: {value}")


# In[25]:


# Define actual values for the forecast period (assuming they start from index 288)
actual_values = Open[-12:]  # Adjust as needed

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_values, E))
print("Root Mean Squared Error (RMSE) for the predicted values:", rmse)

