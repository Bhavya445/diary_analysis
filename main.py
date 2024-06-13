import streamlit as st
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

st.title("Diary Tone")

neg = []
pos = []
dates = [f"2023-10-{i}" for i in range(21, 28)]

# Read and analyze each diary entry
for date in dates:
    try:
        with open(f"diary/{date}.txt", "r", encoding="utf-8") as file:
            entry = file.read()
            s = analyzer.polarity_scores(entry)
            neg.append(s["neg"])
            pos.append(s["pos"])
    except FileNotFoundError:
        st.error(f"File {date}.txt not found. Please make sure the file exists.")
        neg.append(0)  # Assuming no negative sentiment if file not found
        pos.append(0)  # Assuming no positive sentiment if file not found

# Display the results
st.subheader("Positivity")
figure1 = px.line(x=dates, y=pos, labels={"x": "Dates", "y": "Positivity Score"})
st.plotly_chart(figure1)

st.subheader("Negativity")
figure2 = px.line(x=dates, y=neg, labels={"x": "Dates", "y": "Negativity Score"})
st.plotly_chart(figure2)
