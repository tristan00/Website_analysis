import pandas as pd
import pickle
import glob
import gensim
import numpy as np
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import ldamodel
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector, GRU, Input
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from keras import callbacks, layers
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import f1_score, accuracy_score
from keras.models import load_model
import sqlite3
from common import tokenize, DataManager, dir_loc, clean_text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import uuid
import os
from sklearn.ensemble import RandomForestClassifier
import time
import random
from sklearn.svm import SVC
import gc
import fasttext
import functools
import operator
import pickle
from sklearn.decomposition import PCA
import copy

path = '/home/td/Documents/web_models'

replacement_value = 'replacement_value'


def vectorize_topic_models(topic_tuples, num_of_topics):
    vector = [0 for _ in range(num_of_topics)]
    for i in topic_tuples:
        vector[i[0]] = i[1]
    return vector


class Model():
    valid_types = ['dnn',
                   'logistic',
                   'rf',
                   'svm',
                   'lr',
                   'elasticnet']

    def __init__(self, model_type,
                 num_of_hidden_layers=2,
                 width_of_hidden_layers=256,
                 batch_size=32,
                 nb_epoch=100,
                 patience=10,
                 rf_n_estimators=100,
                 rf_criterion='gini',
                 rf_max_depth=8,
                 rf_min_samples_split=2,
                 kernel='rbf',
                 engine_id=None
                 ):

        assert model_type in self.valid_types

        self.id = str(uuid.uuid4())
        self.save_file_loc = f'{dir_loc}/text_analysis_results/models/{engine_id}_{self.id}_{model_type}_model'
        self.model_type = model_type
        self.num_of_hidden_layers = num_of_hidden_layers
        self.width_of_hidden_layers = width_of_hidden_layers
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.patience = patience
        self.rf_n_estimators = rf_n_estimators
        self.rf_criterion = rf_criterion
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.kernel = kernel
        self.metric = None

    def fit(self, x_train, x_val, y_train, y_val):
        print('Model fit', x_train.shape, x_val.shape, y_train.shape, y_val.shape)

        if self.model_type == 'dnn':

            self.model = Sequential()
            self.model.add(Dense(self.width_of_hidden_layers, input_dim=x_train.shape[1], activation='relu'))
            for i in range(self.num_of_hidden_layers):
                self.model.add(Dense(self.width_of_hidden_layers, activation='relu'))
            self.model.add(Dense(1, activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

            cb1 = callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=self.patience,
                                          verbose=0,
                                          mode='auto')
            cb2 = callbacks.ModelCheckpoint(self.save_file_loc,
                                            monitor='val_loss',
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)
            self.model.fit(x_train, y_train,
                           validation_data=(x_val, y_val),
                           callbacks=[cb1, cb2],
                           batch_size=self.batch_size,
                           nb_epoch=self.nb_epoch)
            self.model = load_model(self.save_file_loc)

        if self.model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=self.rf_n_estimators,
                                                criterion=self.rf_criterion,
                                                max_depth=self.rf_max_depth,
                                                min_samples_split=self.rf_min_samples_split)
            self.model.fit(x_train, y_train)
        if self.model_type == 'svm':
            self.model = SVC(kernel=self.kernel)
            self.model.fit(x_train, y_train)
        if self.model_type == 'lr':
            self.model = LogisticRegression()
            self.model.fit(x_train, y_train)
        if self.model_type == 'elasticnet':
            self.model = ElasticNet()
            self.model.fit(x_train, y_train)

        if self.model_type in ['lr', 'rf', 'elasticnet', 'svm', 'logistic']:
            with open(self.save_file_loc, 'wb') as f:
                pickle.dump(self.model, f)

        self.metric = self.evaluate(x_val, y_val)

    def predict(self, x):
        if hasattr(self.model, 'predict_proba'):
            preds = self.model.predict(x)[:,1]
        else:
            preds = self.model.predict(x)
        print(f'model prediction shape: {preds.shape}')
        return preds

    def evaluate(self, x_val, y_val):
        preds = np.rint(self.model.predict(x_val)).astype(int)
        metric = accuracy_score(preds, y_val)
        return metric


class TextEncoder():
    valid_types = ['tfidf',
                   'count',
                   'binary',
                   'lda',
                   'doc2vec',
                   'bert_avg',
                   'fasttext']

    def __init__(self, encoding_type, engine_id, max_vocab_size=10000, min_n_gram=1, max_n_gram=2, num_of_topics=64,
                 encoding_size=64, vectorizer_epochs=3, max_page_size=10000, fasttext_algorithm='skipgram',
                 tokenizer_level='word', max_df = 1.0):

        assert encoding_type in self.valid_types

        self.engine_id = engine_id
        self.id = str(uuid.uuid4())
        self.save_file_loc = f'{dir_loc}/text_analysis_results/models/{engine_id}_{self.id}_{encoding_type}_encoder'
        self.fasttext_training_file_location = f'{dir_loc}/text_analysis_results/fasttext/{engine_id}_{self.id}_{encoding_type}'

        self.encoding_type = encoding_type
        self.max_vocab_size = max_vocab_size
        self.min_n_gram = min_n_gram
        self.max_n_gram = max_n_gram
        self.tokenizer_level = tokenizer_level
        self.max_df = max_df
        self.num_of_topics = num_of_topics
        self.encoding_size = encoding_size
        self.common_dictionary = None
        self.vectorizer_epochs = vectorizer_epochs
        self.fasttext_algorithm = fasttext_algorithm
        self.max_page_size = max_page_size

    def fit(self, documents):
        documents = [tokenize(d) for d in documents]
        documents = [d[:self.max_page_size] for d in documents]
        documents = [' '.join(d) for d in documents]

        if self.encoding_type in ['tfidf', 'count', 'binary']:

            if self.encoding_type == 'tfidf':
                self.vectorizer = CountVectorizer(ngram_range=(self.min_n_gram, self.max_n_gram),
                                                  max_features=self.max_vocab_size, binary=False, max_df=self.max_df,
                                                  analyzer=self.tokenizer_level)
                self.vectorizer.fit(documents)
            if self.encoding_type == 'count':
                self.vectorizer = CountVectorizer(ngram_range=(self.min_n_gram, self.max_n_gram),
                                                  max_features=self.max_vocab_size, binary=False, max_df=self.max_df,
                                                  analyzer=self.tokenizer_level)
                self.vectorizer.fit(documents)
            if self.encoding_type == 'binary':
                self.vectorizer = CountVectorizer(ngram_range=(self.min_n_gram, self.max_n_gram),
                                                  max_features=self.max_vocab_size, binary=False, max_df=self.max_df,
                                                  analyzer=self.tokenizer_level)
                self.vectorizer.fit(documents)
            with open(self.save_file_loc, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        if self.encoding_type == 'lda':
            documents_tokenized = [tokenize(i) for i in documents]
            self.common_dictionary = Dictionary(documents_tokenized)
            common_corpus = [self.common_dictionary.doc2bow(text) for text in documents_tokenized]
            self.vectorizer = ldamodel.LdaModel(common_corpus, id2word=self.common_dictionary,
                                                num_topics=self.num_of_topics, passes=self.vectorizer_epochs)
            self.vectorizer.save(self.save_file_loc)
        if self.encoding_type == 'doc2vec':
            tagged_documents = [TaggedDocument(tokenize(doc), [i]) for i, doc in enumerate(documents)]
            self.vectorizer = Doc2Vec(tagged_documents, vector_size=self.encoding_size, window=2, min_count=1,
                                      workers=4, epochs=self.vectorizer_epochs, max_vocab_size=100000)
            self.vectorizer.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
            self.vectorizer.save(self.save_file_loc)
        if self.encoding_type == 'fasttext':
            with open(self.fasttext_training_file_location, 'w') as f:
                for i in documents:
                    f.write(clean_text(i) + '\n')
            self.vectorizer = fasttext.train_unsupervised(self.fasttext_training_file_location, model=self.fasttext_algorithm,
                                                          dim=self.encoding_size)
            self.vectorizer.save_model(self.save_file_loc)

    def transform(self, documents):

        documents = [tokenize(d) for d in documents]
        documents = [d[:self.max_page_size] for d in documents]
        documents = [' '.join(d) for d in documents]

        if self.encoding_type in ['tfidf', 'count', 'binary']:
            return self.vectorizer.transform(documents).toarray()
        if self.encoding_type == 'lda':
            documents_tokenized = [tokenize(i) for i in documents]
            other_corpus = [self.common_dictionary.doc2bow(i) for i in documents_tokenized]
            results = []
            for i in other_corpus:
                result = self.vectorizer[i]
                result = vectorize_topic_models(result, self.num_of_topics)
                results.append(result)

            return np.array(results)
        if self.encoding_type in ['doc2vec']:
            documents_tokenized = [tokenize(i) for i in documents]

            results = []
            for i in documents_tokenized:
                if i:
                    try:
                        results.append(self.vectorizer[i][0])
                    except KeyError:
                        results.append([0 for _ in range(self.encoding_size)])
                else:
                    results.append([0 for _ in range(self.encoding_size)])

            return np.array(results)

        if self.encoding_type in ['fasttext']:
            documents_clean = [clean_text(i) for i in documents]

            results = []
            for i in documents_clean:
                if i:
                    results.append(self.vectorizer.get_sentence_vector(i))
                    # results.append(self.vectorizer[i])
                else:
                    results.append(np.array([0 for _ in range(self.encoding_size)]))

            return np.array(results)


class Pipeline():
    valid_text_combination_methods = ['hstack',
                                      'pca']
    def __init__(self, text_combination_method, pca_components):
        self.text_combination_method = text_combination_method
        self.pca = PCA(n_components=pca_components)


    def transform(self, x_list):
        result = x_list[0]

        if len(x_list) > 1 and self.text_combination_method == 'hstack':
            result = np.hstack(x_list)
        if self.text_combination_method == 'pca':
            result = self.pca.transform(np.hstack(x_list))

        print('combine_variable_size_text_columns', [i.shape for i in x_list], result.shape)
        return result

    def fit(self, x_list):
        if len(x_list) > 1 and self.text_combination_method == 'pca':
            self.pca.fit(x_list)


def get_random_param_grid():
    vectorization_types = ['tfidf',
                         'count',
                         'binary',
                         'fasttext',
                         'doc2vec']

    params_grid = {
                  'text_types': [['meta'], ['text'], ['html'], ['meta', 'text'], ['meta', 'html'], ['text', 'html'], ['meta', 'text', 'html']],
              'query_encoding_type': copy.copy(vectorization_types),
            'meta_encoding_type': copy.copy(vectorization_types),
            'text_encoding_type': copy.copy(vectorization_types),
                        'html_encoding_type': copy.copy(vectorization_types),
                  'model_type': ['dnn',
                             'rf',
                             'svm',
                             'lr',
                             'elasticnet'],
                    'text_combination_method':['hstack'],
                    'pca_components':list(range(4, 12)),
        'use_pca':[True, False],
                  'max_vocab_size': list(range(100, 2000)),
                  'min_n_gram': [1],
                  'max_n_gram': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'num_of_topics': list(range(32, 128)),
                  'encoding_size': list(range(32, 256)),
                  'num_of_hidden_layers': list(range(1, 4)),
                  'width_of_hidden_layers': list(range(4, 260, 4)),
                  'batch_size': [32],
                  'rf_criterion': ['gini', 'entropy'],
                  'rf_max_depth': list(range(3, 16)),
                  'rf_min_samples_split': list(range(2, 16)),
                  'vectorizer_epochs': list(range(5, 25)),
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'max_page_size': [100, 500, 1000, 5000, 10000],
                  'fasttext_algorithm': ['skipgram', 'cbow'],
                  'tokenizer_level': ['word', 'char', 'char_wb'],
                    'max_df': [i/100 for i in range(1, 101)]
                  }
    next_params = dict()
    for p in params_grid:
        next_params[p] = random.choice(params_grid[p])
    return next_params



if __name__ == '__main__':
    pass
