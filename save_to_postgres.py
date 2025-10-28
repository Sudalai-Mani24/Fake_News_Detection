import psycopg2

def save_to_postgres(df):
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
            INSERT INTO finance_news (source, title, description, url, published_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, (
            row["source"],
            row["title"],
            row["description"],
            row["url"],
            row["published_at"]
        ))

    conn.commit()
    cursor.close()
    conn.close()
