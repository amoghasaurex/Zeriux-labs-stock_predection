import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("ZeriuxLabs - Stock Prediction 📈📉 😃")

stock_input = st.text_input("Enter the Stock Symbol for Prediction", "AAPL").upper()

n_years = st.slider("Years Of Prediction", 1, 10)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Are loading re baba pls wait haan :D")
data = load_data(stock_input)
data_load_state.text("YOOO data loaded :D")

if data.empty or data['Close'].isna().sum() > 0:
    st.error("Oops! The data could not be loaded or has insufficient valid entries. Please check the stock symbol and try again.")
else:
    st.subheader("RAWRRRRR DATA :)")
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Stock Open"))
        fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Stock Close"))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    df_train = data[["Date", "Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    if df_train.shape[0] < 2 or df_train['y'].isna().sum() > 0:
        st.error("Not enough valid data points to fit the model. Please try a different stock.")
    else:
        m = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_range=0.9,
            seasonality_prior_scale=10,
            changepoint_prior_scale=0.1,
        )

        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        m.fit(df_train)

        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Inspect the forecast data
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        st.subheader("Forecast Data 🪅")
        st.write(forecast.tail())

        st.write("Forecasttttt dataaa :D")
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write("Forecast Componentsss :D")
        fig2 = m.plot_components(forecast)
        st.write(fig2)
