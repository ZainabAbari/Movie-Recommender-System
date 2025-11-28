import streamlit as st
import pickle
import pandas as pd

# Load Pickled Files
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity_model = pickle.load(open('model.pkl', 'rb'))
genre_matrix = pickle.load(open('genre_matrix.pkl', 'rb'))

# Streamlit UI Setup
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommender System")
st.write("A simple **content-based movie recommender** using genres and cosine similarity.")

# Recommendation Function
def recommend(movie_title):
    """Returns a list of similar movies based on genre similarity."""
    
    # Check if movie exists
    if movie_title not in movies['title'].values:
        return ["❌ Movie not found in the dataset"]
    
    # Get movie index
    idx = movies[movies['title'] == movie_title].index[0]
    
    # Get nearest neighbors
    distances, indices = similarity_model.kneighbors(
        genre_matrix[idx], n_neighbors=6
    )
    
    # Collect recommended titles (skip itself)
    recommendations = []
    for movie_index in indices[0]:
        title = movies.iloc[movie_index].title
        
        # Prevent recommending the same movie
        if title != movie_title:
            recommendations.append(title)
    return recommendations 

# Movie Selection Box
movie_list = sorted(movies['title'].unique())
selected_movie = st.selectbox("Choose a movie:", movie_list)

# Recommend Button
if st.button("🔍 Recommend"):
    st.subheader(f"🎥 Movies similar to **{selected_movie}**:")
    
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write("- " + movie)

# Footer
st.markdown("""
---
Developed by **Zainab Abari**  
""")
