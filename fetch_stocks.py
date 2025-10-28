import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(tickers, days=5):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=days)

    all_data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        for date, row in hist.iterrows():
            all_data.append({
                "ticker": ticker,
                "date": date.date(),
                "open": row["Open"],
                "close": row["Close"],
                "high": row["High"],
                "low": row["Low"],
                "volume": row["Volume"]
            })

    return pd.DataFrame(all_data)
