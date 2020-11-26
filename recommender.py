
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


df = pd.read_csv("data/books.csv")

### PREPROCESSING
# check uniqu values
df.nunique()

# check missing values
len(df) - df.count() # 3x faster than df.isna().sum()
df.dropna(subset=['description'], inplace=True)

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer()

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['description'])
tfidf_matrix.shape

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and book titles. Convert the index into series
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

### Function that takes in book title as input and outputs most similar books
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the book that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return df['title'].iloc[book_indices].drop_duplicates()

get_recommendations('The Four Loves')
get_recommendations('Blink-182')
get_recommendations('The Rule of Four')
get_recommendations('Cypress Gardens')
