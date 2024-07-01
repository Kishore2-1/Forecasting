# Forecasting

## Overview

This repository presents a comprehensive technical exploration of predictive analytical methods applied to diverse datasets sourced from the UK Office for National Statistics (ONS) and financial markets. The study focuses on forecasting economic indicators such as average weekly earnings, retail sales, production indices, and turnover orders using three main techniques: exponential smoothing (including Holt-Winters methods), ARIMA (Seasonal Autoregressive Integrated Moving Average), and regression analysis. Each method is meticulously applied and evaluated through Python scripts, highlighting their effectiveness in capturing trends, seasonality, and predicting future values. The repository includes detailed datasets, Python scripts for analysis, visual results, and comprehensive documentation, providing insights and methodologies crucial for data-driven decision-making in economic and financial contexts.

## Datasets

1.**K54D:** Average Weekly Earnings in the private sector.
  - Data Source: Extracted from the Average Weekly Earnings Dataset (ONS).
  - Cleaning Process: The dataset was imported into Excel (K54D.xlsx) and formatted to ensure consistent date columns for chronological analysis. Outliers and irrelevant data were removed to balance the dataset.

2. **EAFV:** Retail Sales Index for household goods.
  - Data Source: Extracted from the Retail Sales Dataset (ONS).
  - Cleaning Process: Imported into Excel (EAFV.xlsx), formatted to maintain date integrity. The dataset was balanced by removing outliers and irrelevant data, ensuring reliability in trend analysis.

3. **K226:** Index of Production focusing on crude petroleum and natural gas.
  - Data Source: Extracted from the Index of Production Dataset (ONS).
  - Cleaning Process: Imported into Excel (K226.xlsx), formatted to maintain chronological integrity. Outliers and unnecessary data were removed to enhance data consistency and accuracy.

4. **JQ2J:** Turnover and orders within manufacturing and business sectors.
  - Data Source: Extracted from the Turnover and Orders Dataset (ONS).
  - Cleaning Process: Imported into Excel (JQ2J.xlsx), formatted for consistent date handling. Irrelevant data and outliers were removed to ensure a balanced dataset suitable for accurate analysis.

During the cleaning process, each dataset was meticulously prepared in Excel to facilitate subsequent statistical modeling and forecasting techniques such as exponential smoothing, ARIMA, and regression. The cleaned datasets ensured that only relevant and accurate data points were retained, optimizing the reliability and effectiveness of the analytical models employed in this study.

## Methodology

### Exponential Smoothing

- **Preliminary Analysis:** Initial analysis involved decomposition plots, scatter plots, correlation matrices, and Autocorrelation Function (ACF) plots to identify underlying trends, seasonality patterns, and interdependencies among variables within each dataset (K54D, EAFV, K226, JQ2J).
  
- **Model Selection:** Based on the observed seasonal and trend components, the Holt-Winters method was chosen for its ability to effectively capture and forecast these patterns. Both additive and multiplicative variants of the Holt-Winters method were explored and compared for forecasting accuracy.

### ARIMA

- **Preliminary Analysis:** ACF and Partial Autocorrelation Function (PACF) plots were utilized to analyze the presence of seasonality and trends in the Average Weekly Earnings dataset (K54D).
  
- **Model Selection:** SARIMA (Seasonal Autoregressive Integrated Moving Average) was selected to model the seasonal and trend components observed in the data. The parameters for SARIMA were determined iteratively to ensure accurate forecasting. Comparison with Holt's Winter method highlighted the method's forecasting strengths and weaknesses.

### Regression

- **Preliminary Analysis:** Time series plots were employed to visualize trends in variables (K54D, EAFV, K226, JQ2J) and understand their relationships.
  
- **Model Selection:** The Holt linear method was used for individual variable forecasting, while Ordinary Least Squares (OLS) regression was applied to predict the FTSE 100 index 'Open' values. This approach facilitated the analysis of how economic indicators influence stock market performance over time.

## Conclusions

1. **Exponential Smoothing Effectiveness:** The application of Holt-Winters exponential smoothing proved effective in capturing and forecasting seasonal and trend components across various economic indicators (K54D, EAFV, K226, JQ2J). The multiplicative model generally outperformed the additive model, showcasing its ability to handle data with varying magnitudes and seasonality patterns.

2. **ARIMA vs. Exponential Smoothing:** While SARIMA demonstrated proficiency in modeling seasonal variations, Holt's Winter method consistently provided more accurate forecasts for the Average Weekly Earnings dataset (K54D). This suggests that, despite the complexity of seasonal adjustments, exponential smoothing methods can offer competitive forecasting performance compared to ARIMA models.

3. **Regression for Market Forecasting:** Leveraging the Holt linear method for individual economic indicators (K54D, EAFV, K226, JQ2J) and applying OLS regression to predict FTSE 100 index 'Open' values highlighted the significance of economic indicators in explaining stock market movements. The regression models provided insights into how changes in economic activity influence financial markets, thereby aiding in informed decision-making.

4. **Data Preparation and Cleaning:** Rigorous data preparation, including formatting datasets in Excel sheets and ensuring uniformity across timeframes, was crucial for accurate modeling. Cleaning procedures focused on removing outliers and balancing datasets, enhancing the reliability of forecasting models and ensuring robust results.

5. **Recommendations for Future Use:** For future analyses, exploring more advanced machine learning models and integrating external factors such as geopolitical events or global economic trends could further enhance forecasting accuracy. Continuous validation and refinement of models against real-world data will be essential to maintain predictive reliability amidst evolving economic conditions.
