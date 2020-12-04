
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


df = pd.read_csv("data/books.csv")

### 0 ###
### general preprocessing

# check uniqu values
df.nunique()

# check missing values
len(df) - df.count() # 3x faster than df.isna().sum()
df.dropna(subset=['description'], inplace=True)
df = df.reset_index(drop=True)

### 1 ### 
### recommendations based on description

# clean data (need further check!)
# 



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



### 2 ### 
### recommendations based on authors and categories (NER removed as it took too long time)

'''
###### NER tagging (tried it but took very too long)
import nltk
from nltk.tag import StanfordNERTagger as snt
from nltk.parse.corenlp import CoreNLPParser

stg = snt('C:/Users/moda1/AppData/Roaming/nltk_data/stanfordNLP/stanford-ner-4.0.0/classifiers/english.conll.4class.distsim.crf.ser.gz','C:/Users/moda1/AppData/Roaming/nltk_data/stanfordNLP/stanford-ner-4.0.0/stanford-ner.jar', encoding='utf-8')
parser = CoreNLPParser(url='http://localhost:64000')

df2 = df.sample(100)

def nertagging(description):
    description = str(description)
    nertagged = stg.tag(description.split())
    
    nertag = []
    for x, y in nertagged:
        if y != "O":
            nertag.append(x)
    
    nertag = list(dict.fromkeys(nertag))
    return nertag

df2['nertag'] = df2['description'].apply(lambda x : nertagging(x))
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# covert authors and categories from str to lists
df['authors'] = df['authors'].str.split(';')
df['categories'] = df['categories'].str.split(';')

# select features
features = ['authors', 'categories']

# clean data (need further check!)
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if value exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

for feature in features:
    df[feature] = df[feature].apply(clean_data)


# create a "metadata soup"
def create_soup(x):
    return ' '.join(x['authors']) + ' ' + ' '.join(x['categories'])

df['soup'] = df.apply(create_soup, axis=1)

# Define a count vectorizer
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])
count_matrix.shape

# Compute the Cosine Similarity matrix based on the count_matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of our main DataFrame and construct reverse mapping as before
df = df.reset_index()
indices = pd.Series(df.index, index=df['title'])

get_recommendations('The Four Loves', cosine_sim2)
get_recommendations('Blink-182', cosine_sim2)
get_recommendations('The Rule of Four', cosine_sim2)
get_recommendations('Cypress Gardens', cosine_sim2)



### 3 ### 
### recommendations based on topics (using LDA) + authors and categories

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import spacy
import re

# prepare NLTK Stop words
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'book']) # add as needed

# clean data (need further check!)
data = df.description.values.tolist()

def clean_data(sentences):
    for text in sentences:
        text = re.sub('\S*@\S*\s?', '', text)  # remove emails
        text = re.sub('\s+', ' ', text)  # remove newline chars
        text = re.sub("\'", "", text)  # remove single quotes
        text = gensim.utils.simple_preprocess(str(text), deacc=True) # lowercase & tokenize + deacc=True removes punctuations
        yield(text)

data_words = list(clean_data(data))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Remove Stopwords, Form Bigrams, Trigrams and Lemmatization
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts] # remove stop words
    texts = [bigram_mod[doc] for doc in texts] # make bigrams
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts] # make trigrams
    texts_out = [] #lemmatize
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

data_ready = process_words(data_words)  # processed Text Data!

## Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]

# Build LDA model (need further check of parameters!)
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=30, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=50,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

doc_lda = lda_model[corpus]

# Compute Perplexity (the lower, the better)
print('\nPerplexity: ', lda_model.log_perplexity(corpus))

# Compute Coherence Score (the higher, the better)
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_ready, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Finding the dominant topic in each sentence
topics_df1 = pd.DataFrame()
topics_df2 = pd.DataFrame()
topics_df3 = pd.DataFrame()

for i, row_list in enumerate(lda_model[corpus]):
    row = row_list[0] if lda_model.per_word_topics else row_list            
    row = sorted(row, key=lambda x: (x[1]), reverse=True)
    for j, (topic_num, prop_topic) in enumerate(row):
        if len(row) >= 3:        
            if j ==0:
                topics_df1 = topics_df1.append(pd.Series(int(topic_num)), ignore_index=True)
            elif j ==1:
                topics_df2 = topics_df2.append(pd.Series(int(topic_num)), ignore_index=True)
            elif j ==2:
                topics_df3 = topics_df3.append(pd.Series(int(topic_num)), ignore_index=True)
            else:
                break
        elif len(row) == 2:
            if j ==0:
                topics_df1 = topics_df1.append(pd.Series(int(topic_num)), ignore_index=True)
            elif j ==1:
                topics_df2 = topics_df2.append(pd.Series(int(topic_num)), ignore_index=True)
                topics_df3 = topics_df3.append(pd.Series('',), ignore_index=True)
        elif len(row) == 1:
                topics_df1 = topics_df1.append(pd.Series(int(topic_num)), ignore_index=True)
                topics_df2 = topics_df2.append(pd.Series(''), ignore_index=True)  
                topics_df3 = topics_df3.append(pd.Series(''), ignore_index=True)        

topics_df1.rename(columns={0:'1st_Topic'}, inplace=True)
topics_df2.rename(columns={0:'2nd_Topic'}, inplace=True)
topics_df3.rename(columns={0:'3rd_Topic'}, inplace=True)
topics_df1 = topics_df1.astype(int).astype(str)

# combine topics dataframe to original df
df = pd.concat([df, topics_df1, topics_df2, topics_df3], axis=1, sort=False)


# create a "metadata soup 2"
def create_soup2(x):
    return ''.join(str(x['1st_Topic'])) + ' ' + ''.join(str(x['2nd_Topic'])) + ' ' + ''.join(str(x['3rd_Topic'])) + ' ' + ' '.join(x['authors']) + ' ' + ' '.join(x['categories'])

df['soup2'] = df.apply(create_soup2, axis=1)

# Define a count vectorizer
count = CountVectorizer(stop_words='english')
count_matrix2 = count.fit_transform(df['soup2'])
count_matrix2.shape

# Compute the Cosine Similarity matrix based on the count_matrix
cosine_sim3 = cosine_similarity(count_matrix2, count_matrix2)

# Reset index of our main DataFrame and construct reverse mapping as before
df = df.reset_index()
indices = pd.Series(df.index, index=df['title'])

get_recommendations('The Four Loves', cosine_sim3)
get_recommendations('Blink-182', cosine_sim3)
get_recommendations('The Rule of Four', cosine_sim3)
get_recommendations('Cypress Gardens', cosine_sim3)




### finding the optimal number of topics for LDA
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_ready, start=50, limit=500, step=50)
# Show graph
import matplotlib.pyplot as plt
limit=500; start=50; step=50;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_ready, start=100, limit=1000, step=100)
# Show graph
limit=1000; start=100; step=100;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_ready, start=500, limit=2000, step=500)
# Show graph
limit=2000; start=500; step=500;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_ready, start=2, limit=20, step=2)
limit=20; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()



