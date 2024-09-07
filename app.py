import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommendation System")

# Use st.cache_data to cache the movie data
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

movies_data = load_data()

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

movie_name = st.text_input("Enter a movie name:")
if movie_name:
    list_of_titles = movies_data['title'].tolist()
    close_match = difflib.get_close_matches(movie_name, list_of_titles, n=1)
    
    if close_match:
        movie_index = movies_data[movies_data.title == close_match[0]].index[0]
        similarity_scores = list(enumerate(similarity[movie_index]))
        sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        st.write(f"Top 5 movies similar to '{close_match[0]}':")
        for i, (index, score) in enumerate(sorted_similar_movies[1:6]):
            st.write(f"{i+1}. {movies_data.iloc[index].title}")
    else:
        st.write("Movie not found.")
