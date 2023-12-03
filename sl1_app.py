import streamlit as st
import pandas as pd
import yfinance as yf

st.write('New app demo')

max_date = '2023-11-01'
df = yf.download('AAPL', start=pd.to_datetime(max_date))

st.write(df)

csv = df.to_csv().encode()
b64 = base64.b64encode(csv).decode()
href = f'Download CSV File'
st.markdown(href, unsafe_allow_html=True)

