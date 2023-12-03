import streamlit as st
import pandas as pd
import yfinance as yf
import base64
import sqlite3
import os
import ta
import numpy as np
from datetime import datetime



st.write('New app demo')


#delete last row in the table
def delete_last_row(stock,cursor):
    try:
        cursor.execute(f'SELECT MAX(Date) FROM "{stock}"')
        max_date = cursor.fetchone()[0]
        if max_date is not None:
            cursor.execute(f'DELETE FROM "{stock}" WHERE Date="{max_date}"')
    except Exception as e:
        print(f"An error in delete_last_row for {stock}: {str(e)}")


def add_latest_row_data(stock, conn, cursor):
    try:
        cursor.execute(f'SELECT MAX(Date) FROM "{stock}"')
        max_date = cursor.fetchone()[0]
        if max_date is not None:
            new_data = yf.download(stock, start=pd.to_datetime(max_date))
            new_rows = new_data[new_data.index > max_date]
            if not new_rows.empty:
                new_rows.to_sql(name=stock, con=conn, if_exists='append', index_label='Date')

            print(str(len(new_rows)) + ' new rows imported to DB')

    except Exception as e:
        print(f"An error in add_latest_row_data for {stock}: {str(e)}")


def get_base_stock_data(conn, stock, start_date, end_date):
    try:
        df = yf.download(stock,start=start_date, end=end_date)
        df.to_sql(name=stock, con=conn, if_exists='replace', index=True)
    except Exception as e:
        print(f"An error in get_base_data for {stock}: {str(e)}")


def get_stock_list(market):
    stocks = []
    if (market == 'nasdaq'):
        stocks = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4].Ticker
        stocks = stocks.to_list()

    if (market == 'nifty50'):
        stocks = pd.read_html('https://en.wikipedia.org/wiki/NIFTY_50')[2].Symbol
        stocks = stocks + '.NS'
        stocks = stocks.to_list()

    if (market == 'nifty100'):
        stocks_2 = pd.read_html('https://en.wikipedia.org/wiki/NIFTY_Next_50')[3].Symbol
        stocks_2 = stocks_2 + '.NS'
        stocks_2 = stocks_2.to_list()

        stocks = pd.read_html('https://en.wikipedia.org/wiki/NIFTY_50')[2].Symbol
        stocks = stocks + '.NS'
        stocks = stocks.to_list() + stocks_2

    if market == 'snp500':
        stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].Symbol
        stocks = stocks.to_list()

    if (market == 'nasdaq_ETF'):
        stocks = ['^GSPC', 'QQQ', 'SPY', 'DIA', 'XLK', 'SMH','IWM','IBB','FXI']


    if market == 'USForex':
        stocks = ['USDJPY=X', 'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'INR=X', 'CNY=X','SGD=X','GBPJPY=X','EURGBP=X']

    if len(stocks) == 0:
        stocks=[market]

    return stocks

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

def preprocess_data(market,engine_path,start_date,end_date,create_new_db,update_latest_data):
    try:
        stock_list = get_stock_list(market)
        print(stock_list)
        if create_new_db:
            create_new_database(engine_path) #This is peformed when you try to get fresh copy of the data
            conn = sqlite3.connect(engine_path)
            for stock in stock_list:
                get_base_stock_data(conn, stock, start_date, end_date)
            conn.close()

        if update_latest_data:
          conn = sqlite3.connect(engine_path)
          cursor = conn.cursor()
          for stock in stock_list:
              delete_last_row(stock,cursor)
              add_latest_row_data(stock, conn, cursor)
          cursor.close()
          conn.close()
    except Exception as e:
        #cursor.close()
        #conn.close()
        print(f"An error in preprocess_data: {str(e)}")

def get_table_data(cursor,table_name):
  cursor.execute(f'SELECT * FROM "{table_name}"')
  data = cursor.fetchall()
  columns = [desc[0] for desc in cursor.description]
  df = pd.DataFrame(data, columns=columns)
  return df

def get_weekly_ohlc_data(cursor,table_name):
  cursor.execute(f'SELECT * FROM "{table_name}"')
  data = cursor.fetchall()
  columns = [desc[0] for desc in cursor.description]
  df = pd.DataFrame(data, columns=columns)
  df.set_index('Date', inplace=True)
  df.index = df.index.astype('datetime64[ns]')
  #weekly_ohlc = df.resample('W-Wed').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
  #return weekly_ohlc
  return df

def get_stock_ohlc(stock,cursor):

    df = pd.DataFrame()
    try:
        df = get_weekly_ohlc_data(cursor,stock)
        df.dropna(inplace=True)
    except Exception as exp:
        print(exp)

    df = df.round(3)
    return df

def add_indicators(df, approach, sma_duration):
    df['RSI'] = ta.momentum.rsi(df['Close'], window=2)
    df['SMA_200'] = df['Close'].rolling(window=sma_duration).mean()
    df['IBS'] = (df.Close - df.Low) / (df.High - df.Low) # IBS = (Close - Low) / (High - Low)
    df['Month'] = df.index.month
    df.dropna(inplace=True)
    return df

def add_trade_conditions(df, approach, min_ibs, max_ibs):

    #df['Buy_Start'] = np.where((df['RSI'] < 40) & (df['IBS'] < 0.1), 1, 0) #0.685; 73.168; -68.27
    #df['Buy_End'] = np.where((df['RSI'] > 96) & (df['IBS'] > 0.92), 1, 0)

    #df['Buy_Start'] = np.where((df['IBS'] < 0.2), 1, 0) # Nasdaq 0.711; 255; -48.76
    #df['Buy_End'] = np.where((df['IBS'] > 0.96), 1, 0) # Nasdaq

    #df['Buy_Start'] = np.where((df['IBS'] < 0.2), 1, 0) # Nifty 0.694; 332; -25.12
    #df['Buy_End'] = np.where((df['IBS'] > 0.9), 1, 0) # Nifty

    if approach == 2:
        df['Buy_Start'] = np.where((df['IBS'] < min_ibs), 1, 0)
        df['Buy_End'] = np.where((df['IBS'] > max_ibs), 1, 0) # Nifty

    if approach == 3:
        df['Buy_Start'] = np.where((df['IBS'] < min_ibs) & (df['Close'] > df['SMA_200']), 1, 0) #0.2
        df['Buy_End'] = np.where((df['IBS'] > max_ibs), 1, 0) # 0.87

    return df

def trade(df, cur_stock, cur_approach, slpercentage):
    buy_date = []
    sell_date = []
    buy_price = []
    sell_price = []
    trade_status = []
    max_dd = []
    activity = []
    activity_date = []
    ind_RSI = []
    ind_IBS = []

    in_trade = False
    low_in_trade = 100000 #To find draw down
    dd_per = 0
    for i, (curIndex, row) in enumerate(df.iterrows()):
        #print(curIndex)
        if not in_trade and row.Buy_Start == 1:
            in_trade = True
            buy_date.append(curIndex)
            buy_price.append(row.Close)
            activity_date.append(curIndex)
            activity.append(f'Buy {cur_stock} on {curIndex} for {row.Close}')
            ind_RSI.append(row.RSI)
            ind_IBS.append(row.IBS)

        #to calculate draw down
        if in_trade and row.Low < low_in_trade:
            low_in_trade = row.Low
            dd_per = round((row.Low - buy_price[-1])*100 / buy_price[-1],2)

        #if in trade, check for sl
        slreached = False
        if in_trade:
            pl = (row.Close - buy_price[-1]) * 100 /buy_price[-1]
            if dd_per < slpercentage:
                slreached = True

        if in_trade and (row.Buy_End == 1 or slreached) :
            in_trade = False
            sell_date.append(curIndex)
            sell_price.append(row.Close)
            trade_status.append('Completed')
            activity_date.append(curIndex)
            activity.append(f'Sell {cur_stock} on {curIndex} for {row.Close}')
            max_dd.append(dd_per)
            low_in_trade = 100000 #To find draw down

        if (i == len(df) - 1) and in_trade:
            sell_date.append(curIndex)
            sell_price.append(row.Close)
            trade_status.append('Ongoing')
            activity_date.append(curIndex)
            activity.append(f'Buy Open for {cur_stock} on {curIndex} for {row.Close}')
            max_dd.append(dd_per)
            dd_per = 0

    data = {
        'approach': cur_approach,
        'stock': cur_stock,
        'buy_date': buy_date,
        'sell_date': sell_date,
        'buy_price': buy_price,
        'sell_price': sell_price,
        'max_dd': max_dd,
        'trade_status':trade_status,
        'ind_RSI':ind_RSI,
        'ind_IBS':ind_IBS
    }

    dfSum = pd.DataFrame(data)
    #print(dfSum)
    if len(dfSum) > 0:
        dfSum['Days_In_Trade'] = (dfSum['sell_date'] - dfSum['buy_date']).dt.days
        dfSum['Days_From_Start'] = dfSum['sell_date'] - dfSum['buy_date'].min()
        dfSum['Profit'] = dfSum['sell_price'] - dfSum['buy_price']
        dfSum['ProfitPer'] = round((dfSum['Profit'] / dfSum['buy_price']) + 1,2)

        success = len(dfSum[dfSum.Profit>0])
        total = len(dfSum)
        dfSum['Final_Success_Rate'] = round((success*100/total),2)
        dfSum['Final_Profit_Sum'] = round((dfSum.Profit.sum()),2)

        dfSum['Cum_Profit'] = round(dfSum['ProfitPer'].cumprod(),2)
        dfSum['Cum_Trade_Days'] = round(dfSum['Days_In_Trade'].cumsum(),2)
        dfSum['Month'] = dfSum['buy_date'].dt.month
        dfSum['Year'] = dfSum['buy_date'].dt.year
        dfSum['Invest_Amount'] = 10000
        dfSum['Gross_Profit'] = round((dfSum['Invest_Amount'] / dfSum['buy_price'] ) * dfSum['Profit'],2)
        dfSum['Gross_Profit_Per'] = round((dfSum.Gross_Profit/10000)+1,2)
        dfSum['Cum_GPP'] = round(dfSum['Gross_Profit_Per'].cumprod(),2)

    data_activity = {
        'activity_date':activity_date,
        'activity':activity
    }
    dfActivity = pd.DataFrame(data_activity)
    return dfSum, dfActivity

def update_df(conn, table_name, df):
  df.to_sql(name=table_name, con=conn, if_exists='replace', index=True)

def process_data(market,engine_path,dir_path, slpercentage, cur_approach, min_ibs, max_ibs, start_date, end_date,sma_duration):
    try:
        stock_list = get_stock_list(market)
        #print(stock_list)
        dfTradeSum = pd.DataFrame()
        dfActivitySum = pd.DataFrame()

        conn = sqlite3.connect(engine_path)
        cursor = conn.cursor()

        for stock in stock_list:
            dfRaw = get_stock_ohlc(stock,cursor)
            #cur_approach = 3
            df = dfRaw.copy()
            df = df[df.index >= start_date]
            df = df[df.index <= end_date]
            df = add_indicators(df, cur_approach,sma_duration)
            df = add_trade_conditions(df, cur_approach, min_ibs, max_ibs)
            #df.to_csv(f'{dir_path}/Data/ProcessedData/dfInd_{stock}.csv')
            dfCSum, dfActivity = trade(df, stock, cur_approach, slpercentage)
            dfTradeSum = pd.concat([dfTradeSum,dfCSum])
            dfActivitySum = pd.concat([dfActivitySum,dfActivity])

        dfTradeSum.to_csv(f'{dir_path}/Data_BLSH/dfTradeSum_{market}.csv')
        dfActivitySum.to_csv(f'{dir_path}/Data_BLSH/dfActivitySum_{market}.csv')
        update_df(conn, f'1_dfTradeSum_{market}', dfTradeSum)
        update_df(conn, f'1_dfActivitySum_{market}', dfActivitySum)

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"An error in process_data: {str(e)}")
        #cursor.close()
        #conn.close()


def get_favorable_trades(market,dir_path,ref_stock, engine_path):

    try:

        conn = sqlite3.connect(engine_path)
        cursor = conn.cursor()

        #get list of dates
        dfRefStock = get_stock_ohlc(ref_stock,cursor)
        dfTradeSum = get_table_data(cursor,f'1_dfTradeSum_{market}')
        #dfTradeSum = pd.read_csv(f'{dir_path}/Data/ProcessedData/dfTradeSum_{market}.csv')
        #dfTradeSum = dfTradeSum[dfTradeSum.Cum_GPP>1.5]
        dfActTrade = pd.DataFrame(columns=dfTradeSum.columns)
        dfTradeSum = dfTradeSum.reset_index(drop=True)

        active_stock_end_date = pd.to_datetime('1999-01-01')
        days_in_trade = 100
        for cur_date in dfRefStock.index:
            #week_day = pd.to_datetime(cur_date).dayofweek
            month = pd.to_datetime(cur_date).month

            #if week_day in [0,1,2,3,4]:
            #if (days_in_trade == 1 and cur_date <= active_stock_end_date):
            #    continue

            #don't trade in 2,6,9
            #if market == 'nasdaq' and month in [9]:
            #  continue

            if cur_date < active_stock_end_date:
              continue

            dfTradeSum.buy_date = dfTradeSum.buy_date.astype('datetime64[ns]')
            dfTriggers = dfTradeSum[dfTradeSum.buy_date == cur_date]

            if len(dfTriggers) > 0 and len(dfTriggers.stock.to_list()) > 0:
                target_row = dfTriggers.sort_values(by='Cum_GPP', ascending=False).head(1)
                target_row = target_row.tail(1)
                #nasdaq
                #1 - 3753.81; -20.9
                #2 - 26.71;-17.11
                #3 - 158.87; -16.3
                #3 - without SL; 66.17; -27.62
                #4 - 21.62; -20.65

                #nifty50
                #1 - 72.35; -24.28


                #if cur_date == dfRefStock.iloc[-1].name:
                #    print(dfTriggers.sort_values(by='Cum_GPP', ascending=False).head(5))
                #print(f'Trade on {cur_date}; Stock List = {target_row.stock}')
                dfActTrade = pd.concat([dfActTrade,target_row])
                active_stock_end_date = pd.to_datetime(target_row.sell_date.values[0])
                days_in_trade = target_row.Days_In_Trade

        dfActTrade = dfActTrade.reset_index(drop=True)
        #print(dfActTrade)
        dfActTrade['Cum_GPP_Act'] = round(dfActTrade['Gross_Profit_Per'].cumprod(),2)
        dfActTrade.to_csv(f'{dir_path}/Data_BLSH/dfFavorableTrades_{market}.csv')
        update_df(conn, f'1_dfFavorableTrades_{market}', dfActTrade)

        total_trades = dfActTrade.Gross_Profit.count()
        profits_cntr = len(dfActTrade[dfActTrade.Gross_Profit>0])
        success_rate = profits_cntr / total_trades

        print(f'Total Profit = {dfActTrade.Gross_Profit.sum()}')
        print(f'Total Trades = {total_trades}')
        print(f'Success rate = {success_rate}')

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"An error in get_favorable_trades : {str(e)}")
        #cursor.close()
        #conn.close()

def get_max_invested_amount(market,dir_path,engine_path,step_up_amount):

  try:
    conn = sqlite3.connect(engine_path)
    cursor = conn.cursor()

    df = get_table_data(cursor,f'1_dfFavorableTrades_{market}')
    #df = pd.read_csv(f'{dir_path}/Data/ProcessedData/dfFavorableTrades_{market}.csv')
    df.buy_date = df.buy_date.astype('datetime64[ns]')
    df.sell_date = df.sell_date.astype('datetime64[ns]')

    #print(df)
    min_start_date = df['buy_date'].min()
    max_end_date = df['sell_date'].max()

    # Define a series of dates
    date_series = pd.date_range(start=min_start_date, end=max_end_date, freq='D')

    # Create a list to store the results
    results = []

    # Iterate over the series of dates and find the maximum investment for each date
    for date in date_series:
        mask = (df['buy_date'] < date) & (df['sell_date'] > date)
        mask_start = (df['buy_date'] == date)
        max_start = df.loc[mask_start, 'Invest_Amount'].sum()
        mask_completed = (df['sell_date'] == date) & (df['trade_status'] == 'Completed')
        max_investment = df.loc[mask, 'Invest_Amount'].sum()
        profit_today = round(df.loc[mask_completed, 'Gross_Profit'].sum(),2)
        if len(df[df.buy_date==date].stock.values):
            active_stocks = df[df.buy_date==date].stock.values[0]
        else:
            active_stocks = None
        dd_per = df.max_dd.min()

        stocks_count = df.loc[mask, 'Invest_Amount'].count()
        results.append({'date': date, 'max_start': max_start,'max_investment': max_investment,'profit_today':profit_today,'stocks_count': stocks_count,'active_stocks': active_stocks,'dd_per':dd_per})

    # Create a new DataFrame from the list of results
    results_df = pd.DataFrame(results)
    init_invest_amount = 10000
    amount_in_bank = init_invest_amount

    for i, (curIndex, row) in enumerate(results_df.iterrows()):
        cur_profit = 0
        if row.profit_today != 0:
            cur_profit = (amount_in_bank / init_invest_amount) * row.profit_today
            amount_in_bank += cur_profit
            amount_in_bank += step_up_amount #step up the savings
        results_df.at[i, 'amount_in_bank'] = amount_in_bank

    if len(results_df) > 0:
        #print(results_df)
        print(f'Max Invested=  {results_df.max_investment.max()}')
        print(f'Start = {results_df.iloc[0].date}; End = {results_df.iloc[-1].date}')
        print(f'Multiplied by {results_df.iloc[-1].amount_in_bank/init_invest_amount}')
        print(f'Max DD = {results_df.dd_per.min()}')
        results_df.to_csv(f'{dir_path}/Data_BLSH/dfInvestmentProfile_{market}.csv')
        update_df(conn, f'1_dfInvestmentProfile_{market}', results_df)

    cursor.close()
    conn.close()

  except Exception as e:
      print(f"An error in get_max_invested_amount : {str(e)}")


def find_action_on_date(market,dir_path,engine_path,ref_stock):
  dfReturn = pd.DataFrame()
  df = pd.DataFrame()
  try:
    conn = sqlite3.connect(engine_path)
    cursor = conn.cursor()

    dfRefStock = get_stock_ohlc(ref_stock,cursor)
    target_date = dfRefStock.iloc[-1].name
    print(target_date)

    df = get_table_data(cursor,f'1_dfFavorableTrades_{market}')
    #df = pd.read_csv(f'{dir_path}/Data/ProcessedData/dfFavorableTrades_{market}.csv')
    df.buy_date = df.buy_date.astype('datetime64[ns]')
    df.sell_date = df.sell_date.astype('datetime64[ns]')

    # Exit list
    dfTmp = df[(df.trade_status=='Completed')&(df.sell_date==pd.to_datetime(target_date))]
    print(f'\nExit stocks on {target_date} = {len(dfTmp)}   **********************\n')
    if len(dfTmp) > 0:
        print(dfTmp[['stock','buy_date','buy_price','sell_date','sell_price','trade_status','Days_In_Trade','ProfitPer']])
        dfTmp_T = dfTmp[['stock','buy_date','buy_price','sell_date','sell_price','trade_status','Days_In_Trade','ProfitPer']]
        dfReturn = pd.concat([dfReturn.reset_index(drop=True),dfTmp_T.reset_index(drop=True)], axis=0)

    # Buy List
    #dfTmp = df[(df.buy_date==pd.to_datetime(target_date))]
    dfTmp = df[(df.trade_status == 'Ongoing') & (df.sell_date == pd.to_datetime(target_date))]
    dfTmp_buyBefore = df[(df.trade_status == 'Ongoing') & (df.buy_date == pd.to_datetime(target_date))]
    if len(dfTmp) > 0 and len(dfTmp) == len(dfTmp_buyBefore):
        print(f'\nBuy stocks  on {target_date} = {len(dfTmp)}   **********************\n')
        if len(dfTmp) > 0:
            print(dfTmp[['stock','buy_date','buy_price','sell_date','sell_price','trade_status','Days_In_Trade','ProfitPer']])
            dfTmp_T = dfTmp[['stock','buy_date','buy_price','sell_date','sell_price','trade_status','Days_In_Trade','ProfitPer']]
            dfReturn = pd.concat([dfReturn.reset_index(drop=True),dfTmp_T.reset_index(drop=True)], axis=0)
    else:
        print(f'\nBuy stocks  on {target_date} = 0   **********************\n')


    dfTmp = df[(df.trade_status=='Ongoing')].head(1)
    print(f'\nTotal ongoing stocks  on {target_date} = {len(dfTmp)}   *********************\n')
    if len(dfTmp) > 0:
        print(dfTmp[['stock','buy_date','buy_price','sell_date','sell_price','trade_status','Days_In_Trade','ProfitPer']])
        dfTmp_T = dfTmp[['stock','buy_date','buy_price','sell_date','sell_price','trade_status','Days_In_Trade','ProfitPer']]
        dfReturn = pd.concat([dfReturn.reset_index(drop=True),dfTmp_T.reset_index(drop=True)], axis=0)

    #if len(dfReturn) > 0:
      #update_df(conn, f'1_dfForDayAction_{market}', dfReturn)
    #  dfReturn.to_csv(f'{dir_path}/Data/dfForDayAction_{market}.csv')

    #bbhc
    dfYearlyRet = pd.DataFrame(df.groupby('Year')['Gross_Profit_Per'].prod())
    update_df(conn, f'1_dfYearlyReturn_{market}', dfYearlyRet)
    print(dfYearlyRet)

    cursor.close()
    conn.close()


  except Exception as e:
      print(f"An error in find_action_on_date : {str(e)}")
  return dfReturn, df.tail(5)


def main_process(market, ref_stock, create_new_db, update_latest_data, dir_path, slpercentage,cur_approach, min_ibs, max_ibs, start_date, end_date,sma_duration):
  #market = 'nasdaq' # 2011 to 7-Nov-2023 3063.8 -31.04
  #market = 'snp500' # 2011 to 7-Nov-2023 1087.6 -39.82
  #market = 'nifty50' # 2011 to 7-Nov-2023 117 -29.29
  #market = 'nifty100' # 2011 to 7-Nov-2023 22 -61.41
  #market = 'USDJPY=X' #0.8426; 21; -3.75
  #market = 'nasdaq_ETF'
  #market = 'QQQ'
  #ref_stock = 'AAPL'
  #ref_stock = 'TCS.NS'
  #ref_stock = 'QQQ' #3.743 2011
  #ref_stock = 'SPY' #1.765 2014
  #ref_stock = '^GSPC'
  #ref_stock = market

  #start_date = '2011-01-01'
  #end_date = '2023-11-22'
  #create_new_db = True
  #update_latest_data = True
  step_up_amount = 0
  engine_path = f'{dir_path}/Data_BLSH/{market}.db'

  preprocess_data(market,engine_path,start_date,end_date,create_new_db,update_latest_data)
  process_data(market,engine_path,dir_path,slpercentage,cur_approach, min_ibs, max_ibs,start_date,end_date,sma_duration)
  get_favorable_trades(market,dir_path,ref_stock, engine_path)
  get_max_invested_amount(market,dir_path,engine_path,step_up_amount)
  dfRet, dfDetail_Ret = find_action_on_date(market,dir_path,engine_path,ref_stock)
  return dfRet, dfDetail_Ret


dfSum=pd.DataFrame()
dfSum_Detail=pd.DataFrame()
# dir_path = f'D:\\ACL\Personal\Hem\Prog\BLSH_QS'
# dir_path = f'/content/drive/MyDrive/Fin/Prod/BLSH'
dir_path = f'.'
market = 'nasdaq'
ref_stock = 'AAPL'
slpercentage = -9 #nasdaq
cur_approach = 3 #3&14=4010 2&14=1038
min_ibs = 0.18 #0.18=5669;0.17=2300;0.2=4010; -20.9;0.4=876;0.25=1372;0.4=296
max_ibs = 0.87 #0.87=4010; 0.8=963;0.9=730;0.85=82;0.7=292;0.6=479;0.86=1258
sma_duration = 13 #3&13=8784;3&14=5669;3&50=214 2&50=414

#market = 'nifty50'
#ref_stock = 'TCS.NS'
#slpercentage = -8 #nifty50
#cur_approach = 3
#min_ibs = 0.41 #0.4=346;0.786;-18.66....0.3=99.3...0.5=181...0.35=53.85....0.45=234;0.39=237;0.41=450.93;0.42=321
#min_ibs = 0.25
#max_ibs = 0.87 #0.87=346;0.8=48;0.85=29;0.86=54


all_new_db = False
update_latest_data = True
start_date = '2001-01-01' #IBS 10291 vs Mom 8568.43
end_date = '2023-11-22'
start_date = '2011-01-01' #IBS 8784 Vs Mom 3266.77
end_date = '2024-01-01'

start_date = '2023-10-01' #To process today data

print('Entering main process')
dfRet, dfDetail_Ret = main_process(market, ref_stock, all_new_db, update_latest_data, dir_path, slpercentage,cur_approach,min_ibs, max_ibs, start_date, end_date, sma_duration)
dfSum = pd.concat([dfSum,dfRet])
dfSum_Detail = pd.concat([dfSum_Detail,dfDetail_Ret])
print('Completed main process')

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
#dfSum.to_csv(f'{dir_path}/Data_BLSH/dfSum_{formatted_datetime}.csv')
#dfSum_Detail.to_csv(f'{dir_path}/Data_BLSH/dfSum_Detail_{formatted_datetime}.csv')
#print(dfSum_Detail)
#print(dfSum)
