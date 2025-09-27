# -*- coding: utf-8 -*-
"""
Real-Time News Sentiment Dashboard with Streamlit + TextBlob
"""

import os
import uuid
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from textblob import TextBlob

# Optional fallback news
try:
    from gnews import GNews
except ImportError:
    os.system("pip install gnews")
    from gnews import GNews

# ===================== CONFIG =====================
NEWSAPI_KEY = "1790ee5393dda1919633f337f9bdfbae"  # <-- Replace with your NewsAPI key
PRED_DIR = "predictions_parquet"
os.makedirs(PRED_DIR, exist_ok=True)

# ===================== NEWS FETCHERS =====================
def fetch_news_newsapi(limit=20):
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWSAPI_KEY,
        "language": "en",
        "pageSize": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        rows = []
        for a in articles:
            published = a.get("publishedAt")
            try:
                published = pd.to_datetime(published)
            except Exception:
                published = pd.Timestamp.now()
            rows.append({
                "id": a.get("url") or str(uuid.uuid4()),
                "source": (a.get("source") or {}).get("name"),
                "title": a.get("title"),
                "publishedAt": published
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"NewsAPI failed ({e}). Using GNews fallback.")
        return fetch_news_gnews(limit)

def fetch_news_gnews(limit=20):
    g = GNews(language="en", country="US")
    articles = g.get_top_news()[:limit]
    rows = []
    for a in articles:
        published = a.get("publishedAt")
        try:
            published = pd.to_datetime(published)
        except Exception:
            published = pd.Timestamp.now()
        rows.append({
            "id": a.get("url") or str(uuid.uuid4()),
            "source": a.get("source"),
            "title": a.get("title"),
            "publishedAt": published
        })
    return pd.DataFrame(rows)

# ===================== SENTIMENT PREDICTION =====================
def classify_sentiment(df):
    if df.empty:
        return None
    df['sentiment'] = df['title'].fillna("").apply(
        lambda t: "Positive" if TextBlob(t).sentiment.polarity > 0 else "Negative"
    )
    df['prob_pos'] = df['title'].fillna("").apply(
        lambda t: max(TextBlob(t).sentiment.polarity, 0) if t else 0
    )
    fname = os.path.join(PRED_DIR, f"pred_{uuid.uuid4().hex}.parquet")
    df.to_parquet(fname, index=False)
    return df

# ===================== STREAMLIT DASHBOARD =====================
st.set_page_config(page_title="Real-Time News Sentiment", layout="wide")
st.title("📰 Real-Time News Sentiment Dashboard")

refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 10, 120, 30)

if st.button("Fetch & Classify Latest News"):
    df_new = fetch_news_newsapi(20)
    out = classify_sentiment(df_new)
    st.success(f"Processed {len(out)} headlines") if out is not None else st.warning("No headlines fetched or classified.")

def load_recent(n=200):
    import glob
    files = sorted(glob.glob(os.path.join(PRED_DIR, "*.parquet")), key=os.path.getmtime, reverse=True)[:50]
    if not files:
        return pd.DataFrame(columns=["id", "source", "title", "publishedAt", "sentiment", "prob_pos"])
    df_list = []
    for f in files:
        try:
            dfx = pd.read_parquet(f)
            df_list.append(dfx)
        except Exception as e:
            st.warning(f"Error reading file {f}: {e}")
    if not df_list:
        return pd.DataFrame(columns=["id", "source", "title", "publishedAt", "sentiment", "prob_pos"])
    df = pd.concat(df_list, ignore_index=True)
    df = df.drop_duplicates(subset=['id'])
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df = df.sort_values('publishedAt', ascending=False).head(n)
    return df

df = load_recent()
if df.empty:
    st.warning("No headlines to display. Fetch latest news above.")

st.subheader("Latest Headlines")
st.dataframe(df[['publishedAt', 'source', 'title', 'sentiment', 'prob_pos']].rename(columns={'title': 'headline'}), height=400)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Sentiment Distribution")
    if not df.empty:
        counts = df['sentiment'].value_counts().rename_axis('sentiment').reset_index(name='count')
        fig = px.bar(counts, x='sentiment', y='count', color='sentiment', title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment data available.")

with col2:
    st.subheader("Positive Sentiment Trend")
    if not df.empty and 'publishedAt' in df and 'prob_pos' in df:
        df_sorted = df.sort_values('publishedAt')
        if df_sorted['prob_pos'].any():
            st.line_chart(df_sorted.set_index('publishedAt')['prob_pos'].fillna(0))
        else:
            st.line_chart(pd.Series([0], index=[pd.Timestamp.now()]))
            st.info("No positive sentiment detected in latest headlines.")
    else:
        st.info("No data for positive sentiment trend.")

st.markdown(f"**Last updated:** {pd.Timestamp.now()}")

st.write("Debugging Data Preview:")
st.write(df.head())
