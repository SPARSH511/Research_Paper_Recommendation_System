import streamlit as st
import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize

# Path of the current directory 
path = os.path.dirname(os.path.realpath(__file__))
print(path)

# Load preprocessed data
def load_data():
    if os.path.exists(path+'\\preprocessed_data_embeddings.pkl'):
        # Load preprocessed data
        with open('preprocessed_data_embeddings.pkl', 'rb') as f:
            preprocessed_data = pickle.load(f)
        return preprocessed_data
    else:
        FILE = path+'\\arxiv-metadata-oai-snapshot.json'

        def get_data():
            with open(FILE) as f:
                for line in f:
                    yield line

        dataframe = {
            "id": [],
            "submitter": [],
            "authors": [],
            "title": [],
            "doi": [],
            "categories": [],
            "abstract": [],
            "update_date": []
        }

        data = get_data()
        for i, paper in enumerate(data):
            paper = json.loads(paper)
            try:
                date = int(paper['update_date'].split('-')[0])
                if date > 2019:
                    dataframe['id'].append(
                        r'https://arxiv.org/pdf/'+paper['id'])
                    dataframe['submitter'].append(paper['submitter'])
                    dataframe['authors'].append(paper['authors'])
                    dataframe['title'].append(paper['title'])
                    dataframe['doi'].append(paper['doi'])
                    dataframe['categories'].append(paper['categories'])
                    dataframe['abstract'].append(paper['abstract'])
                    dataframe['update_date'].append(paper['update_date'])
            except:
                pass

        df = pd.DataFrame(dataframe)
        del dataframe
        df['length'] = df['abstract'].str.len()

        def word_count(x):
            return len(x.split())

        df['word_count'] = df['abstract'].apply(word_count)

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['abstract'])
        preprocessed_data = {
            'df': df, 'tfidf_vectorizer': tfidf_vectorizer, 'tfidf_matrix': tfidf_matrix}
        with open('preprocessed_data_embeddings.pkl', 'wb') as f:
            pickle.dump(preprocessed_data, f)

        return preprocessed_data


def recommend_papers(topic_of_interest, df, tfidf_vectorizer, tfidf_matrix, resultsno):
    processed_topic = word_tokenize(topic_of_interest.lower())
    processed_topic_str = " ".join(processed_topic)
    user_tfidf = tfidf_vectorizer.transform([processed_topic_str])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-resultsno:][::-1]
    recommendations = []
    for idx in top_indices:
        recommendations.append(df['title'][idx])
    return recommendations


# Streamlit UI
st.title('Research Paper Recommender')

# Sidebar for user input
topic_of_interest = st.text_input('Enter your topic of interest:', '')
resultsno = st.text_input('Enter the number of results you want:', '')

# Load data
preprocessed_data = load_data()
df = preprocessed_data['df']
tfidf_vectorizer = preprocessed_data['tfidf_vectorizer']
tfidf_matrix = preprocessed_data['tfidf_matrix']

# Button to trigger recommendation
if st.button('Get Recommendations'):
    if topic_of_interest:
        recommendations = recommend_papers(
            topic_of_interest, df, tfidf_vectorizer, tfidf_matrix, int(resultsno))
        st.subheader('Top Recommendations:')
        for i, recommendation in enumerate(recommendations, start=1):
            # Display recommendation as a clickable link
            with st.expander(f"{recommendation}"):
                # Get the details of the selected paper
                selected_paper = df[df['title'] == recommendation].iloc[0]
                # Display details of the selected paper using expander
                st.write("Submitter:", selected_paper['submitter'])
                st.write("Authors:", selected_paper['authors'])
                st.write("Categories:", selected_paper['categories'])
                st.write("Abstract:", selected_paper['abstract'])
                st.write("Update Date:", selected_paper['update_date'])
                st.write("Link:", selected_paper['id'], unsafe_allow_html=True)
    else:
        st.warning('Please enter a topic of interest!')
