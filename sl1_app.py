import streamlit as st
import pandas as pd
import yfinance as yf

st.write('New app demo')

max_date = '2023-12-01'
new_data = yf.download('AAPL', start=pd.to_datetime(max_date))

st.write(new_data)

