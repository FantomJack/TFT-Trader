import pandas as pd
import numpy as np
import ta

folder = '../data/'
export_folder = '../csv_data/'

tickers = ['AAPL', 'ACN', 'ADBE', 'ADI', 'ADP', 'ALTR', 'AMAT', 'AMD', 'AMZN', 'AVGO', 'BA', 'BIDU', 'DE', 'DELL',
                 'DIS', 'EA', 'EBAY', 'GOOG', 'GOOGL', 'HPE', 'HPQ', 'IBM', 'INTC', 'MSFT', 'NVDA']

for ticker in tickers:
    df = pd.read_csv(folder + ticker + '.txt', header=None)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    df['Date'] = pd.to_datetime(df["date"])
    df["day"] = pd.to_datetime(df["date"]).dt.day
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.weekday
    df.drop(columns=['date'], inplace=True)

    df['growth'] = ((df['Close'] - df['Open']) / df['Open'] * 100).round(4)
    df["sma_5"] = (df["growth"].rolling(window=5).mean()).round(4)
    df["sma_14"] = (df["growth"].rolling(window=14).mean()).round(4)

    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_return'].rolling(window=10).std()
    df["rsi"] = ta.momentum.RSIIndicator(df['growth'], window=14).rsi()

    atr_indicator = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['atr'] = atr_indicator.average_true_range()
    df['atr_rel'] = df['atr'] / df['Close']

    cmf_indicator = ta.volume.ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'],
                                                        volume=df['Volume'], window=20)
    df['cmf'] = cmf_indicator.chaikin_money_flow()

    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'growth', 'sma_5', 'sma_14',
            'volatility', 'rsi', 'atr_rel', 'cmf', 'day', 'month', 'day_of_week']
    df = df[cols]
    df = df[19:]

    test_df = df[df['Date'] >= pd.to_datetime('2023-11-01')]
    test_df.to_csv(export_folder + 'test/' + ticker + '.csv', index=False)



    start_dates = [
        {'start': '2000-01-01', 'folder': '1/'},
        {'start': '2000-01-01', 'folder': '2/'},
        {'start': '2010-01-01', 'folder': '3/'},
        {'start': '2020-01-01', 'folder': '4/'}
    ]
    for start in start_dates:
        train_df = df[(df['Date'] >= pd.to_datetime(start['start']))
                      & (df['Date'] < pd.to_datetime('2024-01-01'))]

        train_df.to_csv(export_folder + start['folder'] + ticker + '.csv', index=False)