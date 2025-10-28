import pandas as pd
import random

# Real finance news examples
real_news = [
    "Tesla shares surge after positive earnings report",
    "Apple announces record quarterly revenue",
    "Microsoft acquires startup to expand AI capabilities",
    "Amazon launches new service to deliver groceries in 30 minutes",
    "Google releases new AI model for medical imaging",
    "Netflix reports subscription growth exceeds expectations",
    "Facebook announces metaverse expansion",
    "JP Morgan reports strong Q2 earnings",
    "Oil prices rise after global demand forecast",
    "New IPO from fintech startup attracts investors"
]

# Fake/suspicious finance news examples
fake_news = [
    "Government secretly plans to ban all electric cars",
    "Secret study proves chocolate cures heart disease",
    "Elon Musk arrested for insider trading in space project",
    "Fake news claims central bank will print unlimited money",
    "Bitcoin price fixed by secret cartel, claims insider",
    "Apple secretly bribing regulators to avoid taxes",
    "Amazon warehouse workers secretly replaced by robots",
    "Stock market manipulated by extraterrestrial technology",
    "New study shows banks creating money out of thin air",
    "Government to make cryptocurrency illegal next week"
]

# Expand to ~100 each
def expand_news(news_list, n):
    expanded = []
    for i in range(n):
        text = random.choice(news_list)
        expanded.append(text)
    return expanded

real_expanded = expand_news(real_news, 100)
fake_expanded = expand_news(fake_news, 100)

# Create DataFrame
data = []
for i, text in enumerate(real_expanded):
    data.append([i+1, text, "REAL"])
for i, text in enumerate(fake_expanded):
    data.append([i+101, text, "FAKE"])

df = pd.DataFrame(data, columns=["id", "text", "label"])

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save CSV
df.to_csv("C:/Users/GOD/Documents/finance_pipeline/labeled_news_large.csv", index=False)
print("labeled_news_large.csv created with", len(df), "rows.")
