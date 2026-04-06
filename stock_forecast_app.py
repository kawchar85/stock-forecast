import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Price Forecasting", page_icon="📈", layout="wide")

st.title("📈 Stock Price Forecasting")
st.caption("Time Series Analysis using ARIMA, SARIMA, and Prophet")
st.markdown("---")

st.sidebar.header("Settings")

ticker = st.sidebar.selectbox(
    "Select Stock",
    options=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NFLX", "NVDA"],
    format_func=lambda x: {
        "AAPL" : "Apple (AAPL)",
        "GOOGL": "Google (GOOGL)",
        "MSFT" : "Microsoft (MSFT)",
        "TSLA" : "Tesla (TSLA)",
        "AMZN" : "Amazon (AMZN)",
        "META" : "Meta (META)",
        "NFLX" : "Netflix (NFLX)",
        "NVDA" : "NVIDIA (NVDA)"
    }[x]
)
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=5*365))
end_date   = st.sidebar.date_input("End Date",   value=date.today())
model_name = st.sidebar.selectbox("Forecasting Model", ["Prophet", "ARIMA", "SARIMA"])
horizon    = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30, step=7)

run = st.sidebar.button("Run Forecast")

if not run:
    st.info("Configure settings in the sidebar and click **Run Forecast** to begin.")
    st.stop()

with st.spinner(f"Downloading {ticker} data..."):
    try:
        raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if raw.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check the symbol and try again.")
            st.stop()
        df = raw[['Close']].copy()
        df.columns = ['Price']
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

if len(df) < 60:
    st.error("Not enough data. Please select a longer date range (at least 60 trading days).")
    st.stop()

st.header("Stock Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price",  f"${df['Price'].iloc[-1]:.2f}")
col2.metric("52-Week High",   f"${df['Price'].max():.2f}")
col3.metric("52-Week Low",    f"${df['Price'].min():.2f}")
col4.metric("Total Trading Days", f"{len(df)}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df['Price'], color='steelblue', linewidth=1.5)
ax.set_title(f"{ticker} Closing Price", fontsize=14)
ax.set_ylabel("Price (USD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.header("Stationarity Check")

from statsmodels.tsa.stattools import adfuller

rolling_mean = df['Price'].rolling(window=30).mean()
rolling_std  = df['Price'].rolling(window=30).std()

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(df.index, df['Price'],    label='Price',        color='steelblue', linewidth=1.2)
ax2.plot(df.index, rolling_mean,   label='Rolling Mean', color='orange',    linewidth=2)
ax2.plot(df.index, rolling_std,    label='Rolling Std',  color='red',       linewidth=1.5)
ax2.set_title("Rolling Mean and Standard Deviation (30-day window)", fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)

adf_result = adfuller(df['Price'].dropna())
p_val = adf_result[1]

col_a, col_b = st.columns(2)
col_a.metric("ADF Statistic", f"{adf_result[0]:.4f}")
col_b.metric("p-value",       f"{p_val:.4f}")

if p_val < 0.05:
    st.success("Series is stationary (p < 0.05). Ready for modeling.")
else:
    st.warning("Series is non-stationary (p >= 0.05). The model will apply differencing internally.")

test_size  = horizon
train_data = df['Price'][:-test_size]
test_data  = df['Price'][-test_size:]

st.markdown("---")
st.header(f"Forecast — {model_name}")

forecast_values = None
lower_bound     = None
upper_bound     = None

if model_name == "Prophet":
    try:
        from prophet import Prophet

        df_prophet = df['Price'].reset_index()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

        train_prophet = df_prophet.iloc[:-test_size]
        test_prophet  = df_prophet.iloc[-test_size:]

        with st.spinner("Fitting Prophet model..."):
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m.fit(train_prophet)
            future   = test_prophet[['ds']]
            forecast = m.predict(future)

        forecast_values = forecast['yhat'].values
        lower_bound     = forecast['yhat_lower'].values
        upper_bound     = forecast['yhat_upper'].values
        forecast_index  = test_data.index

    except ImportError:
        st.error("Prophet is not installed. Run: pip install prophet")
        st.stop()

elif model_name in ["ARIMA", "SARIMA"]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    log_train = np.log(train_data)

    with st.spinner(f"Fitting {model_name} model..."):
        if model_name == "ARIMA":
            mod = SARIMAX(log_train, order=(1, 1, 1), trend='n')
        else:
            mod = SARIMAX(log_train, order=(1, 1, 1),
                          seasonal_order=(1, 1, 1, 5), trend='n')

        fit = mod.fit(disp=False)
        pred = fit.forecast(steps=test_size)

    forecast_values = np.exp(pred.values)
    forecast_index  = test_data.index

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(train_data.index, train_data, label='Train',  color='steelblue', linewidth=1.5)
ax3.plot(test_data.index,  test_data,  label='Actual', color='orange',    linewidth=2)
ax3.plot(forecast_index, forecast_values,
         label=f'{model_name} Forecast', color='seagreen', linewidth=1.8, linestyle='--')

if lower_bound is not None and upper_bound is not None:
    ax3.fill_between(forecast_index, lower_bound, upper_bound,
                     alpha=0.2, color='seagreen', label='95% Uncertainty Interval')

ax3.set_title(f"{ticker} — {model_name} Forecast ({horizon} days)", fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)

st.markdown("---")
st.header("Model Performance")
st.caption("Evaluated on the held-out test period. Lower values are better.")

actual = test_data.values
pred   = forecast_values

mae  = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
mape = np.mean(np.abs((actual - pred) / actual)) * 100

c1, c2, c3 = st.columns(3)
c1.metric("MAE",  f"{mae:.2f}",  help="Mean Absolute Error")
c2.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error")
c3.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")

st.markdown("---")
result_df = pd.DataFrame({
    'Date'    : forecast_index,
    'Actual'  : actual,
    'Forecast': pred.round(2)
})
csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Forecast CSV", csv, f"{ticker}_forecast.csv", "text/csv")
