import streamlit as st
import pandas as pd
import yfinance as yf
import base64
import sqlite3
import os



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

def create_new_database(engine_path):
    try:
        if os.path.exists(engine_path):
            # Delete the file
            os.remove(engine_path)
            print(f"{engine_path} has been deleted.")
        conn = sqlite3.connect(engine_path)
        conn.close()
    except Exception as e:
        print(f"An error occurred while deleting {engine_path}: {str(e)}")

#Boardgames/auto-setup/gamessqlite.db

dir_path = f'https://github.com/dharshini-ah/sltest1'
market = 'nasdaq'
engine_path = f'{dir_path}/{market}.db'

create_new_database(engine_path) #This is peformed when you try to get fresh copy of the data

conn = sqlite3.connect(engine_path)
cursor = conn.cursor()

if not df.empty:
    df.to_sql(name='AAPL', con=conn, if_exists='append', index_label='Date')

