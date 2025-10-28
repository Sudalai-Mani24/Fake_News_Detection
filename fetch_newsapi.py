from newsapi import NewsApiClient
import pandas as pd

def fetch_newsapi():
    newsapi = NewsApiClient(api_key="4b86b497fe4c408cbf6d9334c44c253e")  # replace with your key
    articles = newsapi.get_everything(
        q="finance OR stock OR market",
        sources="reuters,bloomberg,yahoo-finance",
        language="en",
        sort_by="publishedAt",
        page_size=50
    )
    
    data = []
    for article in articles["articles"]:
        data.append({
            "source": article["source"]["name"],
            "title": article["title"],
            "description": article["description"],
            "url": article["url"],
            "published_at": article["publishedAt"]
        })
    
    return pd.DataFrame(data)
