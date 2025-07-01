import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk

# Download required corpora
nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

st.set_page_config(page_title="Instagram Comment Sentiment Analyzer", layout="centered")

st.title("📊 Instagram Comment Sentiment Analyzer")
st.write("Upload a text or CSV file of Instagram comments, and we’ll analyze the sentiment.")

# Upload file
uploaded_file = st.file_uploader("Upload a .csv or .txt file", type=["csv", "txt"])

def load_comments(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        comments = df.iloc[:, 0].dropna().astype(str).tolist()
    elif file.name.endswith(".txt"):
        comments = file.read().decode("utf-8").splitlines()
        comments = [c.strip() for c in comments if c.strip()]
    else:
        comments = []
    return comments

def analyze_sentiments(comments):
    results = {"positive": 0, "neutral": 0, "negative": 0}
    for comment in comments:
        polarity = TextBlob(comment).sentiment.polarity
        if polarity > 0.1:
            results["positive"] += 1
        elif polarity < -0.1:
            results["negative"] += 1
        else:
            results["neutral"] += 1
    total = len(comments)
    percentages = {k: round((v / total) * 100, 2) for k, v in results.items()}
    return results, percentages

if uploaded_file:
    with st.spinner("Analyzing comments..."):
        comments = load_comments(uploaded_file)
        if not comments:
            st.error("No valid comments found in the file.")
        else:
            raw_counts, sentiment_pct = analyze_sentiments(comments)
            
            # Display results
            st.subheader("Sentiment Breakdown")
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"{sentiment_pct['positive']}%")
            col2.metric("Neutral", f"{sentiment_pct['neutral']}%")
            col3.metric("Negative", f"{sentiment_pct['negative']}%")

            # Pie chart
            fig, ax = plt.subplots()
            ax.pie(
                raw_counts.values(),
                labels=raw_counts.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=["#00cc96", "#636efa", "#ef553b"]
            )
            ax.axis("equal")
            st.pyplot(fig)
