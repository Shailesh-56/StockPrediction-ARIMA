# StockPrediction-ARIMA
https://bit.ly/stock-prediction-arima
**Overview**
=============
This project focuses on predicting stock prices using time series analysis techniques. The aim is to provide insights into stock market trends and assist in decision-making by leveraging historical stock price data.

**Objectives**
==============
Analyze and visualize historical stock price data.

Build predictive models using techniques like ARIMA and LSTM.

Evaluate the accuracy of the models and optimize them for better performance.

**Features**
============
Time series decomposition and visualization.

Building predictive models using ARIMA and LSTM.

Performance evaluation of models using metrics like RMSE.

Forecasting future stock prices.

**Dataset**
===========
Source: (https://finance.yahoo.com)

Description: The dataset contains historical stock prices, including Open, High, Low, Close, and Volume.

Data Access: The data was retrieved using the **yfinance** library, which allows programmatic access to Yahoo Finance for downloading stock price data efficiently.

**Tools and Technologies**
===========================
Programming Language: Python

Libraries and Frameworks:

Data Manipulation:numpy,pandas

Visualization:matplotlib,seaborn,mplfinance

Time Series Analysis:statsmodels (ARIMA, SARIMAX, seasonal decomposition),pmdarima (auto ARIMA),prophet (Time series forecasting)
statsmodels.tsa.holtwinters (Exponential Smoothing)

Deep Learning:tensorflow.keras (Sequential, LSTM, GRU, Dense, Dropout)

Preprocessing:MinMaxScaler

Evaluation Metrics:root_mean_squared_error

Model Diagnostics:plot_acf, plot_pacf

Other Tools: Jupyter Notebook or Spyder for Deployement

**Methodology**
===============
Data Collection:
-----------------
Historical stock price data is sourced from [https://finance.yahoo.com].

Exploratory Data Analysis (EDA):
--------------------------------
Data Loading and Overview
Load the dataset and display basic information such as rows, columns, and data types.

Handling Missing Data
Check for missing values and describe how they were handled (e.g., dropped, imputed).

Data Cleaning
Address any duplicate rows or outliers.

Univariate Analysis
Analyze individual features using statistical summaries and visualizations (e.g., histograms, box plots).

Bivariate Analysis
Explore relationships between pairs of features using plots (e.g., scatter plots, correlation matrices).

Multivariate Analysis
Investigate interactions between multiple features using advanced visualizations like heatmaps or pair plots.

Feature Engineering
Create new features or transform existing ones to improve model performance.

Data Transformation and Scaling
Apply necessary transformations (e.g., normalization, standardization) for better model accuracy.


Model Building:
---------------
ARIMA: Autoregressive Integrated Moving Average model.

LSTM: Long Short-Term Memory neural network.

Evaluation:
-----------
Evaluate models using metrics like RMSE, MSE, and MAE.

![image](https://github.com/user-attachments/assets/7b1738cd-3862-4a82-976a-5e4fb8dae1b3)


Forecasting:
-------------
Predict future stock prices based on trained models.

**Streamlit App**:
===================
This project includes a Streamlit app that provides an interactive interface for stock price prediction using time series analysis.

Features:
---------
Input stock details: Starting date, ending date, and the company ticker.

Fetch and display historical stock price data dynamically.

Visualize stock price trends with interactive charts.

Perform time series analysis using ARIMA, SARIMA, LSTM, and other models.

Display predictions and compare model performance (based on RMSE values).

Forecast future stock prices for the selected company.

Input Parameters:
-----------------
Starting Date: The date from which historical stock data should be retrieved.

Ending Date: The date until which historical stock data should be retrieved.

Company Ticker: The stock ticker symbol of the company (e.g., AAPL for Apple, GOOG for Google).

**Results**
============
ARIMA and SARIMA models achieved the lowest RMSE values, indicating that they were the most accurate in predicting stock prices in this analysis.

**Conclusion**
==============
The ARIMA and SARIMA models are the best choices for predicting stock prices in this project. While deep learning models (LSTM and GRU) offer potential, further tuning or more data might be required to match the performance of ARIMA-based methods.
