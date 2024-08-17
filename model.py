import streamlit as st 
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("ZerixLabs - Stock Prediction")

stocks = ("AAPL (Apple) ","GOOG (Google)", "NVDA (Nvidia)", "TSLA (Tesla)", "RKLB (Rocket Lab)", "ASTS (AST)","INTC (Intel)", "WBD (Warner Bros)", "AMD (Advanced Micro Devices)", "AMZN (Amazon)", "MSFT (Microsoft)")
selected_stocks = st.selectbox("Select The Stock For Predection", stocks)

n_years = st.slider("Years Of Prediction", 1, 20)
period = n_years * 365
