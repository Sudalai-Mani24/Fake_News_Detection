import psycopg2

def save_stocks_to_postgres(df):
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
            INSERT INTO stock_prices (ticker, date, open, close, high, low, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, (
            row["ticker"],
            row["date"],
            row["open"],
            row["close"],
            row["high"],
            row["low"],
            row["volume"]
        ))

    conn.commit()
    cursor.close()
    conn.close()
