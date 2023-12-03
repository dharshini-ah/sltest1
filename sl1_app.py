import streamlit as st
import pandas as pd
import yfinance as yf
import base64


st.write('New app demo')

max_date = '2023-11-01'
df = yf.download('AAPL', start=pd.to_datetime(max_date))

st.write(df)

#csv = df.to_csv().encode()
#b64 = base64.b64encode(csv).decode()
#href = f'Download CSV File'
#st.markdown(href, unsafe_allow_html=True)


def get_table_download_link_csv(df):
    #csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
    return href

st.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)

