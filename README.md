# Forecasting

## Overview

This repository presents a comprehensive technical exploration of predictive analytical methods applied to diverse datasets sourced from the UK Office for National Statistics (ONS) and financial markets. The study focuses on forecasting economic indicators such as average weekly earnings, retail sales, production indices, and turnover orders using three main techniques: exponential smoothing (including Holt-Winters methods), ARIMA (Seasonal Autoregressive Integrated Moving Average), and regression analysis. Each method is meticulously applied and evaluated through Python scripts, highlighting their effectiveness in capturing trends, seasonality, and predicting future values. The repository includes detailed datasets, Python scripts for analysis, visual results, and comprehensive documentation, providing insights and methodologies crucial for data-driven decision-making in economic and financial contexts.

## Datasets

1. K54D: Average Weekly Earnings in the private sector.
  - Data Source: Extracted from the Average Weekly Earnings Dataset (ONS).
  - Cleaning Process: The dataset was imported into Excel (K54D.xlsx) and formatted to ensure consistent date columns for chronological analysis. Outliers and irrelevant data were removed to balance the dataset.

2. EAFV: Retail Sales Index for household goods.
  - Data Source: Extracted from the Retail Sales Dataset (ONS).
  - Cleaning Process: Imported into Excel (EAFV.xlsx), formatted to maintain date integrity. The dataset was balanced by removing outliers and irrelevant data, ensuring reliability in trend analysis.

3. K226: Index of Production focusing on crude petroleum and natural gas.
  - Data Source: Extracted from the Index of Production Dataset (ONS).
  - Cleaning Process: Imported into Excel (K226.xlsx), formatted to maintain chronological integrity. Outliers and unnecessary data were removed to enhance data consistency and accuracy.

4. JQ2J: Turnover and orders within manufacturing and business sectors.
  - Data Source: Extracted from the Turnover and Orders Dataset (ONS).
  - Cleaning Process: Imported into Excel (JQ2J.xlsx), formatted for consistent date handling. Irrelevant data and outliers were removed to ensure a balanced dataset suitable for accurate analysis.

During the cleaning process, each dataset was meticulously prepared in Excel to facilitate subsequent statistical modeling and forecasting techniques such as exponential smoothing, ARIMA, and regression. The cleaned datasets ensured that only relevant and accurate data points were retained, optimizing the reliability and effectiveness of the analytical models employed in this study.
