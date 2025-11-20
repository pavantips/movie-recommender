import streamlit as st
import pandas as pd
import joblib

# Load data
df = pd.read_csv('movies_clean.csv', sep='|')
cosine_sim = joblib.load('cosine_similarity.pkl')

st.title("🎬 Movie Recommender System")

# Movie selection
movie = st.selectbox("Select a movie:", df['title'].values)

if st.button("Get Recommendations"):
    # Get recommendations
    recommendations = get_recommendations(movie)
    
    st.subheader("Recommended Movies:")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")