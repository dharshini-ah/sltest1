import streamlit as st
import pandas as pd
import yfinance as yf
import base64
import sqlite3



st.write('New app demo')

max_date = '2023-11-01'
df = yf.download('AAPL', start=pd.to_datetime(max_date))

st.write(df)

# Code to download file
def get_table_download_link_csv(df):
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
    return href

st.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)

