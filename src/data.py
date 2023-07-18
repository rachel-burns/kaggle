#data import and preparation
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split

def get_data(filepath):
    return pd.read_csv(filepath)

#add keyword and location to doc2vec model
def clean_docs(docs):
    #real tokenization
    docs = [x.lower().split() for x in docs]
    #pre-process text (punctuation, emoji, tagging, links, etc)

    return docs

def get_tagged_docs(df):
    docs = list(df['text'])
    docs_clean = clean_docs(docs)
    tagged_docs = [TaggedDocument(docs_clean[i], [i]) for i, doc in enumerate(docs_clean)]
    return tagged_docs

def get_vectors(tagged_docs, fname):
    model = Doc2Vec.load('../models/'+fname)
    doc_vectors = np.array([model.infer_vector(tagged_docs[i].words) for i in range(len(tagged_docs))])
    return doc_vectors

def preprocess_documents(df, fname='doc2vec.model'):
    tagged_docs = get_tagged_docs(df)
    doc_vectors = get_vectors(tagged_docs, fname)
    return doc_vectors

def split_traning_x_y(df, test_perc = 0.25):
    ## split up train datset and make the inputs and output sepreate as well
    y = df['target']
    x = df.drop(columns=['target'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_perc, random_state=11, shuffle = True)
    return x_train, x_test, y_train, y_test

def split_traning(df, test_perc = 0.25):
    train_df, test_df = train_test_split(df, test_size=test_perc, random_state=11, shuffle = True)
    return train_df, test_df