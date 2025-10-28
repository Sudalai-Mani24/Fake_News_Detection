from fetch_yahoo import fetch_yahoo_finance
from fetch_newsapi import fetch_newsapi
from save_to_postgres import save_to_postgres
from fetch_stocks import fetch_stock_data
from save_stocks import save_stocks_to_postgres
from fake_news_model import classify_dataframe_with_baseline, load_baseline
from textblob import TextBlob
import pandas as pd
from datetime import datetime
import psycopg2

# ---------------------------
# Helper: Check for existing news in DB
# ---------------------------
def get_existing_titles():
    conn = psycopg2.connect(
        dbname="finance_news_db",
        user="postgres",
        password="FENIX_24",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM finance_news")
    existing = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return set(existing)

# ---------------------------
# Helper: Add sentiment
# ---------------------------
def add_sentiment(df, text_col="title"):
    sentiments = []
    for txt in df[text_col]:
        polarity = TextBlob(txt).sentiment.polarity
        if polarity > 0.1:
            sentiments.append("positive")
        elif polarity < -0.1:
            sentiments.append("negative")
        else:
            sentiments.append("neutral")
    df["sentiment"] = sentiments
    return df

# ---------------------------
# Helper: Link news and stock impact
# ---------------------------
def calculate_stock_impact(news_df, stock_df):
    # Ensure dates are datetime.date
    news_df['date'] = pd.to_datetime(news_df['published_at']).dt.date
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date

    records = []
    for _, news in news_df.iterrows():
        news_date = news['date']
        for ticker in stock_df['ticker'].unique():
            stock_day = stock_df[(stock_df['ticker'] == ticker) & (stock_df['date'] == news_date)]
            if not stock_day.empty:
                open_price = stock_day['open'].values[0]
                close_price = stock_day['close'].values[0]
                pct_change = ((close_price - open_price) / open_price) * 100
                records.append({
                    "news_title": news['title'],
                    "published_at": news['published_at'],
                    "predicted_label": news['predicted_label'],
                    "sentiment": news['sentiment'],
                    "ticker": ticker,
                    "open_price": open_price,
                    "close_price": close_price,
                    "pct_change": pct_change
                })
    return pd.DataFrame(records)

# ---------------------------
# Helper: Save stock impact to Postgres
# ---------------------------
def save_stock_impact(df):
    conn = psycopg2.connect(
        dbname="finance_news_db",
        user="postgres",
        password="FENIX_24",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO news_stock_impact (news_title, published_at, predicted_label, sentiment, ticker, open_price, close_price, pct_change)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, (
            row['news_title'], row['published_at'], row['predicted_label'], row['sentiment'],
            row['ticker'], row['open_price'], row['close_price'], row['pct_change']
        ))
    conn.commit()
    cursor.close()
    conn.close()

# ---------------------------
# Main Pipeline
# ---------------------------
if __name__ == "__main__":
    # Step 1: Fetch news
    print("Fetching news...")
    yahoo_df = fetch_yahoo_finance()
    newsapi_df = fetch_newsapi()
    all_news = pd.concat([yahoo_df, newsapi_df]).drop_duplicates(subset=["title"]).reset_index(drop=True)
    all_news['fetched_at'] = datetime.now()
    print(f" Fetched {len(all_news)} unique news articles")

    # Step 2: Filter out already existing news
    existing_titles = get_existing_titles()
    new_news = all_news[~all_news['title'].isin(existing_titles)].reset_index(drop=True)
    if not new_news.empty:
        # Step 3: Classify news (REAL/FAKE)
        print(f"Classifying {len(new_news)} new articles...")
        model_loaded = load_baseline()
        news_with_preds = classify_dataframe_with_baseline(new_news, text_col="title", model_loaded=model_loaded)
        news_with_preds['predicted_at'] = datetime.now()

        # Step 4: Add sentiment
        news_with_preds = add_sentiment(news_with_preds, text_col="title")

        # Step 5: Save to Postgres
        save_to_postgres(news_with_preds)
        print(f"Classified & saved {len(news_with_preds)} news articles with predictions & sentiment")
    else:
        print("No new news to classify.")
        news_with_preds = pd.DataFrame()  # for later merging

    # Step 6: Fetch stock prices
    print("Fetching stock prices...")
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "SPY"]
    stock_df = fetch_stock_data(tickers, days=2)
    stock_df['fetched_at'] = datetime.now()
    save_stocks_to_postgres(stock_df)
    print(f"Saved {len(stock_df)} stock price records")

    # Step 7: Calculate stock impact for new news
    if not news_with_preds.empty:
        impact_df = calculate_stock_impact(news_with_preds, stock_df)
        save_stock_impact(impact_df)
        print(f"Calculated & saved stock impact for {len(impact_df)} news-stock records")

    print(" Pipeline with stock impact completed successfully!")
