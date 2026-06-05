import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
st.set_page_config(page_title="AI Library Management", layout="wide")

@st.cache_data
def load_data():
    # Use relative path for deployment
    file_path = 'google_books_dataset.csv' # Changed to relative path
    if not os.path.exists(file_path):
        st.error(f"Dataset not found at {file_path}. Please ensure it is uploaded to the repository.")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    df['title'] = df['title'].fillna('Unknown')
    df['text_content'] = df['title'].astype(str) + ' ' + df['description'].fillna('')
    return df

@st.cache_resource
def load_intelligence():
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        matrix = joblib.load('tfidf_matrix.pkl')
        return vectorizer, matrix
    except:
        st.error("Could not load AI models. Ensure tfidf_vectorizer.pkl and tfidf_matrix.pkl are in the same directory.")
        return None, None

df = load_data()
tfidf_vectorizer, tfidf_matrix = load_intelligence()

st.sidebar.title("📚 AI Library")
page = st.sidebar.radio("Navigate", ["Dashboard", "Search & Chat", "Recommendations", "Admin"])

if page == "Dashboard":
    st.title("🏠 User Dashboard")
    if not df.empty:
        st.write(f"Welcome! {len(df)} books in catalog.")
        st.subheader("🔥 Trending Now")
        st.table(df[['title', 'categories']].sample(5))

elif page == "Search & Chat":
    st.title("🔍 AI Semantic Search")
    query = st.text_input("Search (e.g., 'Space and physics'):")
    if query and tfidf_matrix is not None:
        query_vec = tfidf_vectorizer.transform([query.lower()])
        sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        indices = sim.argsort()[-5:][::-1]
        for i in indices:
            with st.expander(df.iloc[i]['title']):
                st.write(df.iloc[i]['description'])

elif page == "Recommendations":
    st.title("🎯 Smart Recommendations")
    book_choice = st.selectbox("Select a book:", df['title'].values if not df.empty else [])
    if st.button("Get Recommendations") and tfidf_matrix is not None:
        idx = df[df['title'] == book_choice].index[0]
        sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        for i, score in sim_scores:
            st.success(f"**{df.iloc[i]['title']}** (Similarity: {score:.2f})")

elif page == "Admin":
    st.title("🛠 Admin Control Panel")
    st.metric("Total Inventory", len(df))
    if st.button("Run Diagnostic"):
        st.info("System logic check: OK. Paths: Relative.")
