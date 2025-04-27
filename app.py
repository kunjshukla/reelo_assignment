import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import pinecone
from dotenv import load_dotenv
import os
from datetime import datetime
from bertopic import BERTopic
import time

# Load environment variables
load_dotenv()

# Set page config as the first Streamlit command
st.set_page_config(layout="wide", page_title="Review Sentiment Dashboard")

# Initialize session state
if 'reviews' not in st.session_state:
    st.session_state.reviews = []
if 'sentiments' not in st.session_state:
    st.session_state.sentiments = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
if 'avg_rating' not in st.session_state:
    st.session_state.avg_rating = 0

# Hugging Face API function
def analyze_sentiment(text):
    url = 'https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english'
    headers = {
        'Authorization': 'Bearer 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',
        'Content-Type': 'application/json'
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json={'inputs': text}, timeout=10)
            st.write(f"Attempt {attempt + 1} - Status Code: {response.status_code}")
            st.write(f"Raw Response: {response.text}")
            
            if response.status_code == 503:
                if attempt < max_retries - 1:
                    st.warning(f"Service unavailable (503). Retrying in 5 seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(5)
                    continue
                else:
                    st.error(f"Service unavailable after {max_retries} attempts.")
                    return "Neutral"
            elif response.status_code != 200:
                st.error(f"API request failed with status {response.status_code}: {response.text}")
                return "Neutral"
            
            result = response.json()
            sentiment = result[0][0]['label'] == 'POSITIVE' and 'Positive' or 'Negative'
            score = result[0][0]['score']
            if sentiment == 'Positive' and score < 0.6:
                sentiment = 'Neutral'
            elif sentiment == 'Negative' and score < 0.6:
                sentiment = 'Neutral'
            return sentiment
        except requests.exceptions.JSONDecodeError as e:
            st.error(f"JSON Decode Error: {str(e)}")
            st.write(f"Raw Response: {response.text}")
            return "Neutral"
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")
            return "Neutral"
    return "Neutral"



# Zapier Webhook function


def send_to_zapier(review, sentiment):
    webhook_url = 'https://hooks.zapier.com/hooks/catch/22654014/2pu4oqa/'
    rating = {'Positive': 5, 'Neutral': 3, 'Negative': 1}[sentiment]
    
    # Generate embedding
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding = model.encode(review).tolist()

    # Initialize Pinecone client
    pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index("reviews-index")

    # Use timestamp for unique ID
    new_id = str(int(datetime.now().timestamp() * 1000))

    # Upsert to Pinecone
    index.upsert([(new_id, embedding, {'review': review, 'sentiment': sentiment, 'rating': float(rating)})])

    # Send only non-embedding data to Zapier
    data = {
        'review': review,
        'sentiment': sentiment,
        'rating': rating
    }

    # Send to Zapier
    response = requests.post(webhook_url, json=data)
    return response.status_code == 200


def classify_review(review):
    review = review.lower()
    if any(word in review for word in ['service', 'staff', 'wait']):
        return 'Service'
    elif any(word in review for word in ['location', 'place', 'area']):
        return 'Location'
    elif any(word in review for word in ['coffee', 'product', 'item']):
        return 'Product'
    return 'Other'

# Update the review submission logic
review = st.text_input("Enter your review (e.g., 'Amazing coffee!')")
if st.button("Analyze Sentiment", key="analyze"):
    if review:
        sentiment = analyze_sentiment(review)
        rating = {'Positive': 5, 'Neutral': 3, 'Negative': 1}[sentiment]
        category = classify_review(review)
        new_review = {
            'review': review,
            'sentiment': sentiment,
            'rating': rating,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'category': category
        }
        st.session_state.reviews.append(new_review)
        st.session_state.sentiments[sentiment] = st.session_state.sentiments.get(sentiment, 0) + 1
        st.session_state.avg_rating = sum(r['rating'] for r in st.session_state.reviews) / len(st.session_state.reviews) if st.session_state.reviews else 0
        send_to_zapier(review, sentiment)
        st.success(f"Sentiment: {sentiment}")
        st.write(f"Category: {category}")  # Display category



# Charts Section
col1, col2, col3 = st.columns(3)

with col1:
    total = sum(st.session_state.sentiments.values())
    fig_pie = px.pie(
        names=list(st.session_state.sentiments.keys()),
        values=list(st.session_state.sentiments.values()),
        title="Sentiment Distribution",
        color_discrete_map={'Positive': '#00FF00', 'Neutral': '#FFFF00', 'Negative': '#FF0000'}
    )
    fig_pie.update_layout(width=300, height=300)
    st.plotly_chart(fig_pie)

with col2:
    total = sum(st.session_state.sentiments.values())
    percentages = {k: (v / total * 100 if total else 0) for k, v in st.session_state.sentiments.items()}
    fig_bar = px.bar(
        x=list(percentages.keys()),
        y=list(percentages.values()),
        title="Sentiment Breakdown",
        color=list(percentages.keys()),
        color_discrete_map={'Positive': '#00FF00', 'Neutral': '#FFFF00', 'Negative': '#FF0000'}
    )
    fig_bar.update_layout(width=300, height=300, yaxis_range=[0, 100])
    st.plotly_chart(fig_bar)

with col3:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=st.session_state.avg_rating,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 5]},
            'bar': {'color': '#00FF00' if st.session_state.avg_rating >= 4 else '#FFFF00' if st.session_state.avg_rating >= 2 else '#FF0000'},
            'steps': [
                {'range': [0, 1], 'color': '#FF0000'},
                {'range': [1, 2], 'color': '#FF4500'},
                {'range': [2, 3], 'color': '#FFA500'},
                {'range': [3, 4], 'color': '#FFFF00'},
                {'range': [4, 5], 'color': '#00FF00'}
            ]
        },
        title={'text': "Average Rating"}
    ))
    fig_gauge.update_layout(width=300, height=300)
    st.plotly_chart(fig_gauge)

# Submitted Reviews Section
st.subheader("Submitted Reviews")
if not st.session_state.reviews:
    st.write("No reviews yet. Submit a review to start!")
else:
    for r in st.session_state.reviews:
        st.write(f"- {r['review']} - <span style='color: {'green' if r['sentiment'] == 'Positive' else 'yellow' if r['sentiment'] == 'Neutral' else 'red'}; font-weight: bold;'>{r['sentiment']}</span>", unsafe_allow_html=True)

# Apply custom CSS
st.markdown(
    """
    <style>
    .stApp { background-color: #000000; }
    .stTextInput > div > div > input { border-color: #E5E7EB; }
    .stButton>button { background-color: #BFDBFE; color: #1E40AF; border: none; padding: 0.5rem 1rem; border-radius: 0.375rem; }
    .stButton>button:hover { background-color: #93C5FD; }
    </style>
    """,
    unsafe_allow_html=True
)