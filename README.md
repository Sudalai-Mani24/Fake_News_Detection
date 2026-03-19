🚀 Finance News & Stock Market Impact Analysis Pipeline

An end-to-end AI-powered data pipeline that fetches real-time financial news, detects fake news, analyzes sentiment, and links it with stock market movements to uncover actionable insights.

📌 Project Overview

This project automates the process of:

📡 Fetching financial news from APIs

🧠 Classifying news as REAL or FAKE using NLP

😊 Performing sentiment analysis (positive/neutral/negative)

📈 Fetching stock price data for major tickers

🔗 Linking news with stock price changes (% impact)

🗄️ Storing everything in PostgreSQL for analysis

👉 Goal: Understand how news authenticity & sentiment influence stock market behavior

⚙️ Tech Stack

Python (Pandas, NumPy)

Machine Learning (Scikit-learn, TF-IDF, Logistic Regression)

NLP (TextBlob, text processing)

APIs (Yahoo Finance, NewsAPI)

Database (PostgreSQL)

Automation (Custom pipeline scripts)

🏗️ Project Structure
finance_pipeline/
│
├── fetch_yahoo.py          # Fetch finance news (Yahoo)
├── fetch_newsapi.py        # Fetch news from NewsAPI
├── fetch_stocks.py         # Fetch stock price data
├── save_to_postgres.py     # Save news data
├── save_stocks.py          # Save stock data
├── fake_news_model.py      # ML model (train + predict)
├── run_pipeline.py         # Main pipeline script 🚀
├── requirements.txt        # Dependencies

🔄 Pipeline Workflow
News APIs → Deduplication → Fake News Detection → Sentiment Analysis
        ↓
   PostgreSQL (finance_news table)
        ↓
Stock Data Fetch → PostgreSQL (stock_prices table)
        ↓
News + Stock Merge → % Price Change Calculation
        ↓
news_stock_impact table

🧠 Machine Learning

Model: TF-IDF + Logistic Regression

Task: Fake News Classification (REAL vs FAKE)

Additional: Sentiment Analysis using TextBlob

Output:

predicted_label

prob_real

sentiment

🗄️ Database Tables
📄 finance_news

source

title

description

url

published_at

predicted_label (optional enhancement)

sentiment (optional enhancement)

📊 stock_prices

ticker

date

open

close

volume

🔗 news_stock_impact

news_title

ticker

predicted_label

sentiment

open_price

close_price

pct_change

▶️ How to Run
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Train model (first time only)
python fake_news_model.py --mode train --input labeled_news.csv --text_col text --label_col label
3️⃣ Run full pipeline
python run_pipeline.py
🧪 Example Output

News classified as:

✅ REAL / ❌ FAKE

😊 Sentiment (positive/neutral/negative)

Stock impact:

📈 % price change per news event

💡 Key Features

✔️ End-to-end automated pipeline
✔️ Fake news detection + sentiment analysis
✔️ Real-time data integration (news + stocks)
✔️ PostgreSQL storage for scalable analysis
✔️ Modular & production-ready architecture

📊 Future Enhancements

📉 Advanced sentiment models (BERT / FinBERT)
📊 Interactive dashboard (Streamlit / Power BI)
⏱️ Real-time streaming pipeline
📈 Time-series forecasting of stock trends
🧠 Improved fake news dataset for better accuracy

🎯 Use Cases

Financial market analysis
Algorithmic trading insights
News credibility tracking
Business intelligence dashboards

🧑‍💻 Author

Sudalai Mani S
Aspiring Data Analyst | Data Science & AI Enthusiast
