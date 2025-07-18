import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing import clean_text  # Your own stemming/text cleaner

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit page setup
st.set_page_config(page_title="Twitter Sentiment Dashboard", layout="centered")
st.title("🐦 Twitter Sentiment Analysis Dashboard")

# --- Single Tweet Input ---
st.header("🔍 Predict Sentiment for a Single Tweet")
tweet_input = st.text_area("Enter a tweet here:", value="I love the way this product works! Totally worth the money.")

if st.button("Predict"):
    if tweet_input.strip() == "":
        st.warning("Please enter something.")
    else:
        cleaned = clean_text(tweet_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"
        st.success(f"Sentiment: **{sentiment}**")

# --- CSV Upload for Batch Analysis ---
st.header("📁 Upload CSV File for Bulk Sentiment Analysis")
uploaded_file = st.file_uploader("Upload a CSV file with a `text` column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'text' not in df.columns:
        st.error("The CSV must contain a `text` column.")
    else:
        with st.spinner("Processing tweets..."):
            df['cleaned'] = df['text'].astype(str).apply(clean_text)
            df['prediction'] = model.predict(vectorizer.transform(df['cleaned']))

        # Map predictions to labels
        df['Sentiment'] = df['prediction'].map({0: 'Negative', 1: 'Positive'})
        st.success("Analysis Complete! Here's a preview:")
        st.dataframe(df[['text', 'Sentiment']].head(10))

        # --- Visualization Section ---
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts()

        # PIE CHART
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff4d4d', '#33cc33'])
        ax1.axis('equal')
        st.pyplot(fig1)

        # BAR CHART
        st.subheader("📈 Bar Chart")
        st.bar_chart(sentiment_counts)

        # WORD CLOUD
        st.subheader("☁️ Word Cloud of Tweets")
        all_words = ' '.join(df['cleaned'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
        st.image(wordcloud.to_array(), use_column_width=True)

        # Downloadable results
        st.download_button(
            label="📥 Download Results as CSV",
            data=df[['text', 'Sentiment']].to_csv(index=False).encode('utf-8'),
            file_name='sentiment_results.csv',
            mime='text/csv'
        )
