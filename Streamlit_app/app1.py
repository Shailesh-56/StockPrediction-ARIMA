
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import datetime
import yfinance as yf

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Price Forecasting App",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Forecasting App")

st.markdown("""
Analyze historical stock prices, explore time-series patterns,
test stationarity, and forecast future prices using ARIMA.
""")


# ============================================================
# 2. SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Input Parameters")

start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime.date(2020, 1, 1)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.date.today()
)

company_name = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="AAPL",
    help="Examples: AAPL, MSFT, GOOGL, TSLA, RELIANCE.NS"
).strip().upper()

menu_option = st.sidebar.selectbox(
    "Select Option",
    ["Show Data", "Show Plots"]
)

menu_model = st.sidebar.selectbox(
    "Select Model",
    ["ARIMA", "SARIMA", "ETS", "GRU", "LSTM", "Prophet"]
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_yfinance_columns(data):
    """Normalize Yahoo Finance columns, including MultiIndex columns."""

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)

        if "Close" in level0:
            data.columns = level0
        elif "Close" in level1:
            data.columns = level1

    # Avoid duplicate columns after normalization
    data = data.loc[:, ~data.columns.duplicated()]

    return data


def get_close_prices(data):
    """Return a clean, numeric closing-price Series."""

    if data is None or data.empty:
        return pd.Series(dtype=float)

    if "Close" not in data.columns:
        return pd.Series(dtype=float)

    close = data["Close"]

    # Handle cases where Close is still a DataFrame
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            return pd.Series(dtype=float)
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close = close.sort_index()

    # Remove duplicate dates
    close = close.loc[~close.index.duplicated(keep="last")]

    return close


# ============================================================
# 4. DOWNLOAD STOCK DATA
# ============================================================

stock_data = pd.DataFrame()
close_prices = pd.Series(dtype=float)

if not company_name:
    st.info("Enter a stock ticker in the sidebar to get started.")

elif start_date >= end_date:
    st.error("The start date must be earlier than the end date.")

else:
    try:
        with st.spinner("Downloading stock data from Yahoo Finance..."):

            stock_data = yf.download(
                company_name,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                threads=False
            )

        if stock_data.empty:
            st.error(
                "Yahoo Finance returned no data. Check the ticker "
                "symbol, date range, and data availability."
            )

        else:
            stock_data = normalize_yfinance_columns(stock_data)
            close_prices = get_close_prices(stock_data)

            if close_prices.empty:
                st.error(
                    "No valid closing prices were found in the "
                    "downloaded data."
                )

            else:
                st.success(
                    f"Loaded {len(close_prices):,} observations "
                    f"for {company_name}."
                )

                st.caption(
                    f"Data period: {close_prices.index.min().date()} "
                    f"to {close_prices.index.max().date()}"
                )

                # ------------------------------------------------
                # 4A. SHOW DATA
                # ------------------------------------------------

                if menu_option == "Show Data":
                    st.subheader(
                        f"{company_name} — Historical Stock Data"
                    )

                    st.dataframe(
                        stock_data.head(10),
                        use_container_width=True
                    )

                    st.subheader("Closing Price Summary")

                    st.dataframe(
                        close_prices.describe().to_frame(
                            name="Close Price"
                        ),
                        use_container_width=True
                    )

                # ------------------------------------------------
                # 4B. SHOW PLOTS
                # ------------------------------------------------

                elif menu_option == "Show Plots":

                    st.subheader("Closing Price History")

                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(
                        close_prices.index,
                        close_prices.values,
                        label="Closing Price"
                    )
                    ax.set_title(
                        f"{company_name} Closing Price vs Date"
                    )
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    ax.legend()
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Moving averages
                    st.subheader("Moving Averages")

                    fig, ax = plt.subplots(figsize=(12, 5))

                    ax.plot(
                        close_prices.index,
                        close_prices.values,
                        label="Closing Price",
                        alpha=0.8
                    )

                    ax.plot(
                        close_prices.rolling(50).mean(),
                        label="50-Day SMA"
                    )

                    ax.plot(
                        close_prices.rolling(200).mean(),
                        label="200-Day SMA"
                    )

                    ax.set_title("Closing Price with Moving Averages")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    ax.legend()
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Seasonal decomposition
                    st.subheader("Seasonal Decomposition")

                    if len(close_prices) >= 24:
                        try:
                            result = seasonal_decompose(
                                close_prices,
                                model="additive",
                                period=12,
                                extrapolate_trend="freq"
                            )

                            fig = result.plot()
                            fig.set_size_inches(12, 9)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        except Exception as e:
                            st.warning(
                                f"Seasonal decomposition unavailable: {e}"
                            )
                    else:
                        st.info(
                            "At least 24 observations are recommended "
                            "for seasonal decomposition."
                        )

                    # Lag plot
                    st.subheader("Lag Plot")

                    if len(close_prices) >= 3:
                        fig, ax = plt.subplots(figsize=(8, 6))

                        pd.plotting.lag_plot(
                            close_prices,
                            ax=ax
                        )

                        ax.set_title(
                            f"{company_name} Lag Plot"
                        )
                        ax.set_xlabel("Price at time t")
                        ax.set_ylabel("Price at time t+1")
                        ax.grid(alpha=0.25)

                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

    except Exception as e:
        st.error(
            f"An error occurred while downloading or displaying data: {e}"
        )


# ============================================================
# 5. ARIMA FORECASTING
# ============================================================

if menu_model != "ARIMA":
    st.info(
        f"{menu_model} is not implemented in this version. "
        "Select ARIMA to run the implemented forecasting pipeline."
    )

elif company_name and not close_prices.empty:

    # --------------------------------------------------------
    # 5A. VALIDATE THE SERIES
    # --------------------------------------------------------

    if len(close_prices) < 30:
        st.warning(
            "At least 30 valid closing-price observations are "
            "recommended. Select a longer date range."
        )

    elif close_prices.nunique() <= 1:
        st.error(
            "Closing prices are constant. A meaningful ARIMA "
            "forecast cannot be produced from this series."
        )

    else:
        st.header(f"ARIMA Forecasting — {company_name}")

        # Forecast settings
        st.sidebar.header("Forecast Settings")

        forecast_days = st.sidebar.number_input(
            "Number of trading days to forecast",
            min_value=1,
            max_value=365,
            value=30,
            step=1
        )

        # ----------------------------------------------------
        # 5B. STATIONARITY TEST
        # ----------------------------------------------------

        st.subheader("Augmented Dickey-Fuller (ADF) Test")

        # Reserve the final 20% for testing before model fitting
        split_index = int(len(close_prices) * 0.80)

        train_data = close_prices.iloc[:split_index]
        test_data = close_prices.iloc[split_index:]

        if len(train_data) < 20 or len(test_data) < 1:
            st.error(
                "Insufficient training or testing observations."
            )

        else:
            try:
                adf_result = adfuller(train_data)

                adf_statistic = adf_result[0]
                adf_pvalue = adf_result[1]

                col1, col2 = st.columns(2)

                col1.metric(
                    "ADF Statistic",
                    f"{adf_statistic:.4f}"
                )

                col2.metric(
                    "ADF p-value",
                    f"{adf_pvalue:.6f}"
                )

                if adf_pvalue < 0.05:
                    st.success(
                        "The training series is stationary at the "
                        "5% significance level."
                    )
                else:
                    st.warning(
                        "The training series appears non-stationary. "
                        "Auto ARIMA will select differencing when needed."
                    )

            except Exception as e:
                st.warning(f"ADF test could not be completed: {e}")

            # ------------------------------------------------
            # 5C. TRAIN / TEST PLOT
            # ------------------------------------------------

            st.subheader("Training and Testing Split")

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(
                train_data.index,
                train_data.values,
                label="Training Data"
            )

            ax.plot(
                test_data.index,
                test_data.values,
                label="Testing Data"
            )

            ax.set_title("Chronological Train-Test Split")
            ax.set_xlabel("Date")
            ax.set_ylabel("Closing Price")
            ax.legend()
            ax.grid(alpha=0.25)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.write(f"Training observations: {len(train_data)}")
            st.write(f"Testing observations: {len(test_data)}")

            # ------------------------------------------------
            # 5D. AUTO ARIMA MODEL SELECTION
            # ------------------------------------------------

            st.subheader("Automatic ARIMA Order Selection")

            try:
                with st.spinner(
                    "Selecting ARIMA parameters. This may take a while..."
                ):
                    auto_model = auto_arima(
                        train_data,
                        seasonal=False,
                        start_p=0,
                        start_q=0,
                        max_p=3,
                        max_q=3,
                        max_d=2,
                        stepwise=True,
                        error_action="ignore",
                        suppress_warnings=True,
                        trace=False
                    )

                st.write("Selected ARIMA order:", auto_model.order)
                st.text(str(auto_model.summary()))

                # ------------------------------------------------
                # 5E. FIT AND EVALUATE ON HELD-OUT TEST DATA
                # ------------------------------------------------

                st.subheader("Model Evaluation")

                with st.spinner("Fitting ARIMA model..."):

                    evaluation_model = ARIMA(
                        train_data,
                        order=auto_model.order
                    ).fit()

                    test_forecast = evaluation_model.forecast(
                        steps=len(test_data)
                    )

                test_forecast = np.asarray(
                    test_forecast
                ).reshape(-1)

                actual_test = test_data.to_numpy()

                rmse = np.sqrt(
                    mean_squared_error(
                        actual_test,
                        test_forecast
                    )
                )

                st.metric(
                    "Test RMSE",
                    f"{rmse:.4f}"
                )

                st.caption(
                    "RMSE measures the typical forecast error in "
                    "the units of the stock price. Lower is better "
                    "when comparing forecasts on the same data."
                )

                # Actual versus predicted
                comparison_df = pd.DataFrame(
                    {
                        "Actual Close": actual_test,
                        "Predicted Close": test_forecast
                    },
                    index=test_data.index
                )

                st.dataframe(
                    comparison_df,
                    use_container_width=True
                )

                fig, ax = plt.subplots(figsize=(12, 5))

                ax.plot(
                    test_data.index,
                    actual_test,
                    label="Actual Test Prices"
                )

                ax.plot(
                    test_data.index,
                    test_forecast,
                    label="ARIMA Test Forecast"
                )

                ax.set_title("Actual vs Predicted Test Prices")
                ax.set_xlabel("Date")
                ax.set_ylabel("Closing Price")
                ax.legend()
                ax.grid(alpha=0.25)

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # ------------------------------------------------
                # 5F. REFIT ON ALL AVAILABLE DATA
                # ------------------------------------------------

                st.subheader("Future Stock Price Forecast")

                with st.spinner(
                    "Refitting model on all available observations..."
                ):
                    final_model = ARIMA(
                        close_prices,
                        order=auto_model.order
                    ).fit()

                    future_forecast = final_model.forecast(
                        steps=int(forecast_days)
                    )

                future_forecast = np.asarray(
                    future_forecast
                ).reshape(-1)

                # Use business days as approximate trading dates
                future_dates = pd.bdate_range(
                    start=close_prices.index[-1]
                    + pd.offsets.BDay(1),
                    periods=int(forecast_days)
                )

                forecast_df = pd.DataFrame(
                    {
                        "Date": future_dates,
                        "Forecasted Close": future_forecast
                    }
                )

                st.dataframe(
                    forecast_df,
                    use_container_width=True
                )

                # ------------------------------------------------
                # 5G. FUTURE FORECAST PLOT
                # ------------------------------------------------

                st.subheader("Historical Prices and Future Forecast")

                fig, ax = plt.subplots(figsize=(12, 6))

                # Show the recent history for readability
                history_length = min(250, len(close_prices))
                recent_history = close_prices.iloc[-history_length:]

                ax.plot(
                    recent_history.index,
                    recent_history.values,
                    label="Historical Closing Price"
                )

                ax.plot(
                    future_dates,
                    future_forecast,
                    label="ARIMA Forecast"
                )

                ax.axvline(
                    close_prices.index[-1],
                    linestyle="--",
                    alpha=0.6,
                    label="Forecast Start"
                )

                ax.set_title(
                    f"{company_name}: ARIMA Future Forecast"
                )
                ax.set_xlabel("Date")
                ax.set_ylabel("Closing Price")
                ax.legend()
                ax.grid(alpha=0.25)

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "Forecasts are statistical estimates, not guaranteed "
                    "future prices. Business-day dates are approximate "
                    "and do not account for exchange holidays."
                )

            except Exception as e:
                st.error(
                    f"ARIMA modeling or forecasting failed: {e}"
                )

elif menu_model == "ARIMA" and company_name:
    st.warning(
        "ARIMA cannot run because valid closing-price data "
        "was not downloaded. Check the ticker and date range."
    )
