
# 🚀 Finance News & Stock Impact Pipeline

An end-to-end **AI-powered pipeline** that fetches financial news, detects fake news, analyzes sentiment, and links it with stock price movements to understand market impact.


## 📌 Features

* Fetches real-time news (Yahoo Finance, NewsAPI)
* Classifies news as **REAL / FAKE** using NLP
* Performs **sentiment analysis** (positive/neutral/negative)
* Fetches stock prices (AAPL, TSLA, etc.)
* Calculates **% stock price change per news**
* Stores all data in **PostgreSQL**


## ⚙️ Tech Stack

* Python (Pandas, NumPy)
* Machine Learning (TF-IDF, Logistic Regression)
* NLP (TextBlob)
* PostgreSQL
* APIs (Yahoo Finance, NewsAPI)


## 🏗️ Structure

```
finance_pipeline/
├── fetch_yahoo.py
├── fetch_newsapi.py
├── fetch_stocks.py
├── save_to_postgres.py
├── save_stocks.py
├── fake_news_model.py
├── run_pipeline.py
```


## ▶️ Run

```bash
pip install -r requirements.txt
python fake_news_model.py --mode train --input labeled_news.csv --text_col text --label_col label
python run_pipeline.py
```


## 🎯 Output

* News → REAL/FAKE + Sentiment
* Stock → Price data
* Combined → **News vs Stock Impact (%)**


💡 Key Features

✔️ End-to-end automated data pipeline

✔️ Fake news detection + sentiment analysis

✔️ Real-time data integration (news + stock data)

✔️ PostgreSQL storage for scalable data handling

✔️ Modular and production-ready architecture

📊 Future Enhancements

📉 Advanced sentiment models (BERT / FinBERT)

📊 Interactive dashboard (Streamlit / Power BI)

⏱️ Real-time streaming pipeline

📈 Time-series forecasting for stock trends

🧠 Enhanced fake news dataset for improved accuracy

🎯 Use Cases

Financial market analysis

Algorithmic trading insights

News credibility tracking

Business intelligence dashboards

🧑‍💻 Author

Sudalai Mani S
B.E Computer Science and Engineering (2025)
AI & Data Science Enthusiast
