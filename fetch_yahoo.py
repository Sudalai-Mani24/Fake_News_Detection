import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def fetch_yahoo_finance():
    url = "https://finance.yahoo.com/topic/stock-market-news/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    articles = soup.find_all("h3", class_="Mb(5px)")
    data = []
    for article in articles:
        title = article.get_text()
        link = "https://finance.yahoo.com" + article.find("a")["href"]
        data.append({
            "source": "Yahoo Finance",
            "title": title,
            "description": None,
            "url": link,
            "published_at": datetime.utcnow()  # since Yahoo often doesn't give exact timestamp
        })
    
    return pd.DataFrame(data)
