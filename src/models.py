# train and predict (break into two separate modules?)
import data
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def train_doc2vec(df, fname='doc2vec.model',vector_size=5, dm=1, window=2, min_count=1, workers=4, epochs=10):
    tagged_docs = data.get_tagged_docs(df)
    model_doc2vec = Doc2Vec(tagged_docs, vector_size=vector_size, dm=dm, window=window,  min_count=min_count, workers=workers, epochs=epochs)
    model_doc2vec.save('../models/'+fname)
    return model_doc2vec

