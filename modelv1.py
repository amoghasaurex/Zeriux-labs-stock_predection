import streamlit as st
import numpy as np  # Import numpy
np.float_ = np.float64  # Override np.float_ with np.float64

from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2020-01-01"  # Shorten the historical data to avoid outdated trends
TODAY = date.today().strftime("%Y-%m-%d")

st.title("ZerixLabs - Stock Prediction 📈📉 😃")

selected_stocks = st.text_input("Enter The Stock Symbol For Prediction", "AAPL").upper()

# Focus on very short-term prediction
n_days = st.slider("Days Of Prediction", 1, 7)  
period = n_days  # Predict for 1 to 7 days only

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Are loading re baba pls wait haan :D")
data = load_data(selected_stocks)
data_load_state.text("YOOO data loaded :D")

st.subheader("RAWRRRRR DATA :)")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="STOCK OPENNN"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="STOCK CLOSEEE"))
    fig.layout.update(title_text="Time dataa", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Forecasting
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Highly Tuned Model for Short-Term Accuracy
m = Prophet(
    seasonality_mode='additive',  # Short-term, linear seasonality
    changepoint_prior_scale=0.01,  # Very low to avoid overfitting
    seasonality_prior_scale=10.0,  # Increased to capture strong seasonality patterns
    daily_seasonality=True,  # Enable daily seasonality for short-term predictions
    weekly_seasonality=False,  # Weekly seasonality can be disabled
    yearly_seasonality=False  # Disable yearly seasonality, not relevant for a few days
)

m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("FORECASTTTTT 🪅")
st.write(forecast.tail())

st.write("Forecastt dataaa :D")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast componentss")
fig2 = m.plot_components(forecast)
st.write(fig2)
