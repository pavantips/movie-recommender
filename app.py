import streamlit as st
import pandas as pd
import joblib

# Load data
df = pd.read_parquet('movies_clean.parquet')
cosine_sim = joblib.load('cosine_similarity.pkl')

st.title("🎬 Movie Recommender System")

# Define recommendation function ← ADD THIS!
def get_recommendations(title, top_n=10):
    # Find movie index
    idx = df[df['title'] == title].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N (skip first - it's the movie itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return movie titles
    return df['title'].iloc[movie_indices]

# Movie selection
movie = st.selectbox("Select a movie:", df['title'].values)

if st.button("Get Recommendations"):
    # Get recommendations
    recommendations = get_recommendations(movie)
    
    st.subheader("Recommended Movies:")
    for i, rec_movie in enumerate(recommendations, 1):
        st.write(f"{i}. {rec_movie}")