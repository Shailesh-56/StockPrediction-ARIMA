import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import yfinance as yf
import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page title
st.set_page_config(
    page_title="Stock Price Forecasting App",
    page_icon="📈",
    layout="wide",
)

# Header Section
st.title("📈 Stock Price Forecasting App")
st.markdown("""
    Analyze historical stock prices and forecast future trends using time series models.
    Choose your stock ticker and time period to get started.
""")

# Sidebar Inputs: Start Date, End Date, Company Name
st.sidebar.header("Input Parameters")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("1980-01-01"))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
company_name = st.sidebar.text_input("Enter Company Ticker")

# Fetch and Display Data Using Sidebar Menu
menu_option = st.sidebar.selectbox("Select Option", ["Show Data", "Show Plots"])

# Sidebar to apply various TimeSeries Models
menu_model = st.sidebar.selectbox("Select Model", ["ARIMA", 'SARIMA', 'ETS', 'GRU', 'LSTM', 'Prophet'])

# Fetch stock data if a company ticker is provided
if company_name:
    try:
        # Download stock data
        stock_data = yf.download(company_name, start=start_date, end=end_date)

        if menu_option == "Show Data":
            # Display the first 10 records
            st.subheader(f"Stock Data for {company_name} - First 10 Records")
            st.write(stock_data.head(10))

        elif menu_option == "Show Plots":
            # Plotting
            st.subheader("Closing Price Plot")
            # Plot Close price
            plt.figure(figsize=(10, 5))
            plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
            plt.title('Close Price vs Date')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            st.pyplot(plt)
            
            
            # Plot For Moving Avverage
            stock_data['SMA_50']=stock_data['Close'].rolling(window=50).mean()
            stock_data['SMA_200']=stock_data['Close'].rolling(window=200).mean()
            plt.figure(figsize=(10,6))
            plt.plot(stock_data['Close'],label='Closing Price',color='b')
            plt.plot(stock_data['SMA_50'],label='50-Day SMA',color='g')
            plt.plot(stock_data['SMA_200'],label='200-Day SMA',color='r')
            plt.title('Closing Price with Moving Average')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend(loc='upper left')
            st.pyplot(plt)
            
            # Plotting the decomposed components
            result = seasonal_decompose(stock_data['Close'], model='additive', period=12)
            plt.figure(figsize=(10,8))

            # Plot the original series
            plt.subplot(411)
            plt.plot(stock_data['Close'], label='Original')
            plt.title('Original Series')
            plt.legend(loc='best')

            # Plot the trend component
            plt.subplot(412)
            plt.plot(result.trend, label='Trend', color='orange')
            plt.title('Trend Component')
            plt.legend(loc='best')

            # Plot the seasonal component
            plt.subplot(413)
            plt.plot(result.seasonal, label='Seasonal', color='green')
            plt.title('Seasonal Component')
            plt.legend(loc='best')

            # Plot the residual component
            plt.subplot(414)
            plt.plot(result.resid, label='Residual', color='red')
            plt.title('Residual Component')
            plt.legend(loc='best')
            plt.tight_layout()
            st.pyplot(plt)
            
            # Lag Plot
            plt.figure(figsize=(10,8))
            pd.plotting.lag_plot(stock_data['Close'])
            plt.title('Lag Plot of Ford Stock Closing Prices')
            plt.xlabel('y(t)')
            plt.ylabel('y(t+1)')
            st.pyplot(plt)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please enter a valid company ticker to fetch data.")

if menu_model == "ARIMA" and company_name:
    stock_data = stock_data[['Close']].dropna()
    
    # Stationarity Check
    adf = adfuller(stock_data['Close'])
    st.write(f"ADF Test P-Value (Original Series): {adf[1]}")
    if adf[1] < 0.05:
        st.success("The series is stationary (p-value < 0.05). Proceeding with ARIMA modeling.")
        processed_data = stock_data
        differencing_needed = False
    else:
        st.warning("The series is not stationary (p-value >= 0.05). Applying differencing.")
        processed_data = stock_data.diff().dropna()
        differencing_needed = True
        adf_diff = adfuller(processed_data['Close'])
        st.write(f"ADF Test P-Value (Differenced Series): {adf_diff[1]}")

    # Plot Original and Processed Series
    st.subheader("Time Series Plots")
    fig, ax = plt.subplots(2, 1, figsize=(14, 7))
    ax[0].plot(stock_data, label='Original Series')
    ax[0].set_title('Original Series')
    ax[0].legend()

    if differencing_needed:
        ax[1].plot(processed_data, label='Differenced Series', color='orange')
        ax[1].set_title('Differenced Series')
    else:
        ax[1].plot(processed_data, label='Stationary Series', color='green')
        ax[1].set_title('Stationary Series')
    ax[1].legend()
    st.pyplot(fig)
        
    # Splitting Data into Train and Test Sets
    split_index = int(len(processed_data) * 0.80)
    train_data = processed_data[:split_index]
    test_data = processed_data[split_index:]
    
    st.write(train_data.shape)
    st.write(test_data.shape)
    
    # Train-Test Split Plot
    st.subheader("Train-Test Split Plot")
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label='Train Data')
    plt.plot(test_data, label='Test Data', color='orange')
    plt.title('Train-Test Split')
    plt.legend()
    st.pyplot(plt)
    
    
    # ARIMA Model Order Selection Using Auto ARIMA
    st.markdown("### ARIMA Model Selection")
    autoarima_model = auto_arima(train_data, trace=True, suppress_warnings=True)
    st.write(autoarima_model.summary())
    
    # Fit ARIMA Model
    st.markdown("### Fitting ARIMA Model")
    arima_order = autoarima_model.order
    arima_model = ARIMA(train_data, order=arima_order)
    arima_model = arima_model.fit()
    st.text(arima_model.summary())
    
    # Input number of days for forecasting
    st.sidebar.header("Forecast Settings")
    forecast_days = st.sidebar.number_input("Enter number of days to forecast:", min_value=1, value=30)
    
    # Forecast future values
    st.subheader("Forecasting Future Values")
    if differencing_needed:
    # Reverse differencing to convert to original scale
        arima_forecast_diff = arima_model.forecast(steps=forecast_days)
        last_original_value = stock_data['Close'].iloc[-1]  # Last value of the original series
        arima_forecast_original = np.cumsum(np.insert(arima_forecast_diff, 0, last_original_value))
        arima_forecast_original = arima_forecast_original[1:]  # Remove the duplicated last original value
    else:
    # Use the forecast as is since no differencing was applied
        arima_forecast_original = arima_model.forecast(steps=forecast_days)
        # arima_forecast = arima_model.forecast(steps=forecast_days)

    # Display Actual and Forecasted Values
    st.write("Forecasted Values:")
    forecast_index = pd.date_range(start=stock_data.index[-1], periods=forecast_days + 1, freq="D")[1:]
    forecast_df = pd.DataFrame({
        "Date": forecast_index,
        "Forecasted Values": arima_forecast_original
    })
    st.write(forecast_df)

    # Plot Original Closing Prices and Forecasted Values
    st.subheader("Forecast Plot: Original Closing Prices vs Forecasted Values")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the original closing prices
    ax.plot(stock_data.index, stock_data['Close'], label='Original Closing Prices', color='blue')
    
    # Ensure stock_data index is in datetime format
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data.index = pd.to_datetime(stock_data.index)

    # Plot the forecasted values
    forecast_start_date = stock_data.index[-1] + pd.Timedelta(days=1)
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_days, freq="D")
    ax.plot(forecast_index, arima_forecast_original, label='Forecasted Values', color='red')
    
    # Add title and labels
    ax.set_title('Original Closing Prices and Forecasted Values')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    
    
    
    
    
    
    
    