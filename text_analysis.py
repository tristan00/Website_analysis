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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from keras.models import Model, load_model
import sqlite3
from common import tokenize, DataManager, dir_loc
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import uuid
import os
from sklearn.ensemble import RandomForestClassifier
import time
import random


path = '/home/td/Documents/web_models'

from sklearn.preprocessing import OneHotEncoder
import functools
import operator

replacement_value = 'replacement_value'


class OHE(OneHotEncoder):
    def __init__(self, n_values=None, categorical_features=None,
                 categories=None, sparse=True, dtype=np.float64,
                 handle_unknown='error', min_perc=.001, col_name=''):
        super().__init__(n_values=n_values, categorical_features=categorical_features,
                         categories=categories, sparse=sparse, dtype=dtype,
                         handle_unknown=handle_unknown)
        self.min_perc = min_perc
        self.col_name = col_name
        self.valid_values = []
        self.col_names = []
        self.nan_replacement_value = None

    def fit(self, X, y=None):
        input_series = self.process_input(X)
        super().fit(input_series)
        self.col_names = ['{col_base_name}_{value_name}'.format(col_base_name=self.col_name, value_name=i) for i in
                          self.categories_[0]]

    def transform(self, X):
        input_series = self.process_input(X)
        output = super().transform(input_series)
        return self.process_output(output)

    def process_input(self, s):
        if not self.nan_replacement_value:
            self.nan_replacement_value = s.mode()[0]
        s = s.fillna(s.mode())
        s = s.astype(str)

        if not self.valid_values:
            self.valid_values = [i for i, j in dict(s.value_counts(normalize=True)).items() if j >= self.min_perc]

        prediction_values_to_replace = [i for i in s.unique() if i not in self.valid_values]
        replace_dict = {i: replacement_value for i in prediction_values_to_replace}
        replace_dict.update({i: i for i in self.valid_values})
        s = s.map(replace_dict.get)
        return s.values.reshape(-1, 1)

    def process_output(self, output):
        output_df = pd.DataFrame(data=output.toarray(),
                                 columns=self.col_names)
        return output_df

    def length(self):
        return len(self.categories_)


def vectorize_topic_models(topic_tuples, num_of_topics):
    vector = [0 for _ in range(num_of_topics)]
    for i in topic_tuples:
        vector[i[0]] = i[1]
    return vector


def get_simple_dnn(input_dim, num_of_hidden_layers=2, width_of_hidden_layers=256):

    return dnn


class TextModel():
    valid_types = ['dnn',
                   'logistic',
                   'rf']

    def __init__(self, m_type,
                 num_of_hidden_layers=2,
                 width_of_hidden_layers=256,
                 batch_size=32,
                 nb_epoch=100,
                 patience=10,
                 rf_n_estimators=100,
                 rf_criterion='gini',
                 rf_max_depth=8,
                 rf_min_samples_split=2):

        assert m_type in self.valid_types

        self.m_type = m_type
        self.num_of_hidden_layers = num_of_hidden_layers
        self.width_of_hidden_layers = width_of_hidden_layers
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.patience = patience
        self.rf_n_estimators = rf_n_estimators
        self.rf_criterion = rf_criterion
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split

    def fit(self, x_train, x_val, y_train, y_val):
        if self.m_type == 'dnn':

            self.model = Sequential()
            self.model.add(Dense(self.width_of_hidden_layers, input_dim=x_train.shape[1], activation='relu'))
            for i in range(self.num_of_hidden_layers):
                self.model.add(Dense(self.width_of_hidden_layers, activation='relu'))
            self.model.add(Dense(1, activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_squared_error'])

            file_name = str(uuid.uuid4())
            cb1 = callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=self.patience,
                                          verbose=0,
                                          mode='auto')
            cb2 = callbacks.ModelCheckpoint(f'/tmp/{file_name}',
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
            self.model = load_model(f'/tmp/{file_name}')
            os.remove(f'/tmp/{file_name}')

        if self.m_type == 'logistic':
            self.model = LogisticRegression()

            print(x_train.shape, y_train.shape)
            self.model.fit(x_train, y_train)

        if self.m_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=self.rf_n_estimators,
                                                criterion=self.rf_criterion,
                                                max_depth=self.rf_max_depth,
                                                min_samples_split=self.rf_min_samples_split)
            self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x_val, y_val):
        preds = np.rint(self.model.predict(x_val)).astype(int)
        metric = accuracy_score(preds, y_val)
        return metric


class TextEncoding():
    valid_types = ['tfidf',
                   'count',
                   'binary',
                   'lda',
                   'doc2vec']

    def __init__(self, e_type, max_vocab_size=10000, min_n_gram=1, max_n_gram=1, max_df=.1, num_of_topics=64,
                 encoding_size=64, gensim_passes = 1):

        assert e_type in self.valid_types

        self.e_type = e_type
        self.max_vocab_size = max_vocab_size
        self.min_n_gram = min_n_gram
        self.max_n_gram = max_n_gram
        self.max_df = max_df
        self.num_of_topics = num_of_topics
        self.encoding_size = encoding_size
        self.common_dictionary = None
        self.gensim_passes = gensim_passes

    def fit(self, documents):
        if self.e_type == 'tfidf':
            self.vectorizer = CountVectorizer(ngram_range=(self.min_n_gram, self.max_n_gram),
                                              max_features=self.max_vocab_size, binary=False, max_df=self.max_df)
            self.vectorizer.fit(documents)
        if self.e_type == 'count':
            self.vectorizer = CountVectorizer(ngram_range=(self.min_n_gram, self.max_n_gram),
                                              max_features=self.max_vocab_size, binary=False, max_df=self.max_df)
            self.vectorizer.fit(documents)
        if self.e_type == 'binary':
            self.vectorizer = CountVectorizer(ngram_range=(self.min_n_gram, self.max_n_gram),
                                              max_features=self.max_vocab_size, binary=True, max_df=self.max_df)
            self.vectorizer.fit(documents)
        if self.e_type == 'lda':
            documents_tokenized = [tokenize(i) for i in documents]
            self.common_dictionary = Dictionary(documents_tokenized)
            common_corpus = [self.common_dictionary.doc2bow(text) for text in documents_tokenized]
            self.vectorizer = ldamodel.LdaModel(common_corpus, id2word=self.common_dictionary, num_topics=self.num_of_topics, passes=self.gensim_passes)
        if self.e_type == 'doc2vec':
            tagged_documents = [TaggedDocument(tokenize(doc), [i]) for i, doc in enumerate(documents)]
            self.vectorizer = Doc2Vec(tagged_documents, vector_size=self.encoding_size, window=2, min_count=1,
                                      workers=4, epochs=self.gensim_passes)
            self.vectorizer.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


    def transform(self, documents):

        if self.e_type in ['tfidf', 'count', 'binary']:
            return self.vectorizer.transform(documents).toarray()
        if self.e_type == 'lda':
            documents_tokenized = [tokenize(i) for i in documents]
            other_corpus = [self.common_dictionary.doc2bow(i) for i in documents_tokenized]
            results = []
            for i in other_corpus:
                result = self.vectorizer[i]
                result = vectorize_topic_models(result, self.num_of_topics)
                results.append(result)

            return np.array(results)
        if self.e_type == 'doc2vec':
            documents_tokenized = [tokenize(i) for i in documents]

            results = []
            for i in documents_tokenized:
                if i:
                    results.append(self.vectorizer[i][0])
                else:
                    results.append([0 for _ in range(self.encoding_size)])

            return np.array(results)


class ModelPipeline():
    valid_text_combination_methods = ['side-by-side',
                                      'subtraction']

    def __init__(self, e_type='binary',
                 max_vocab_size=1000,
                 min_n_gram=1,
                 max_n_gram=1,
                 max_df=.1,
                 num_of_topics=64,
                 encoding_size=64,
                 m_type='logistic',
                 num_of_hidden_layers=2,
                 width_of_hidden_layers=256,
                 batch_size=32,
                 nb_epoch=100,
                 patience=10,
                 rf_n_estimators=100,
                 rf_criterion='gini',
                 rf_max_depth=8,
                 rf_min_samples_split=2,
                 text_combination_method='side-by-side',
                 gensim_passes = 1):

        assert text_combination_method in self.valid_text_combination_methods

        self.e_type = e_type
        self.max_vocab_size = max_vocab_size
        self.min_n_gram = min_n_gram
        self.max_n_gram = max_n_gram
        self.max_df = max_df
        self.num_of_topics = num_of_topics
        self.encoding_size = encoding_size
        self.m_type = m_type
        self.num_of_hidden_layers = num_of_hidden_layers
        self.width_of_hidden_layers = width_of_hidden_layers
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.patience = patience
        self.rf_n_estimators = rf_n_estimators
        self.rf_criterion = rf_criterion
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.text_combination_method = text_combination_method
        self.gensim_passes = gensim_passes

        self.encoding_training_time = 0
        self.pre_model_training_transformations_time = 0
        self.model_training_time = 0
        self.prediction_transformation_time = 0
        self.prediction_model_time = 0
        self.eval_metric = 0

    def run_pipeline(self, df):
        training_time_start = time.time()
        self.text_encoding = TextEncoding(e_type=self.e_type,
                                          max_vocab_size=self.max_vocab_size,
                                          min_n_gram=self.min_n_gram,
                                          max_n_gram=self.max_n_gram,
                                          max_df=self.max_df,
                                          num_of_topics=self.num_of_topics,
                                          encoding_size=self.encoding_size,
                                          gensim_passes = self.gensim_passes)
        self.text_model = TextModel(m_type=self.m_type,
                                    num_of_hidden_layers=self.num_of_hidden_layers,
                                    width_of_hidden_layers=self.width_of_hidden_layers,
                                    batch_size=self.batch_size,
                                    nb_epoch=self.nb_epoch,
                                    patience=self.patience,
                                    rf_n_estimators=self.rf_n_estimators,
                                    rf_criterion=self.rf_criterion,
                                    rf_max_depth=self.rf_max_depth,
                                    rf_min_samples_split=self.rf_min_samples_split)

        meta_documents = df['meta'].tolist()
        text_documents = df['text'].tolist()
        target = df['meta_matches_text']

        meta_documents_train, meta_documents_val, text_documents_train, text_documents_val, y_train, y_val = train_test_split(
            meta_documents, text_documents, target)

        #not thorough to train vectorizer on val data but adding val data handles oov case easily.
        self.text_encoding.fit(meta_documents_train + meta_documents_val + text_documents_train + text_documents_val)

        self.encoding_training_time = time.time() - training_time_start

        meta_documents_train_enc = self.text_encoding.transform(meta_documents_train)
        meta_documents_val_enc = self.text_encoding.transform(meta_documents_val)
        text_documents_train_enc = self.text_encoding.transform(text_documents_train)
        text_documents_val_enc = self.text_encoding.transform(text_documents_val)

        x_train = self.combine_text_columns(meta_documents_train_enc, text_documents_train_enc)
        x_val = self.combine_text_columns(meta_documents_val_enc, text_documents_val_enc)

        self.pre_model_training_transformations_time = time.time() - training_time_start - self.encoding_training_time
        self.text_model.fit(x_train, x_val, y_train, y_val)

        self.model_training_time = time.time() - training_time_start - self.encoding_training_time - self.pre_model_training_transformations_time

        meta_documents_val_enc = self.text_encoding.transform(meta_documents_val)
        text_documents_val_enc = self.text_encoding.transform(text_documents_val)
        x_val = self.combine_text_columns(meta_documents_val_enc, text_documents_val_enc)

        self.prediction_transformation_time = time.time() - training_time_start - self.encoding_training_time - self.pre_model_training_transformations_time - self.model_training_time

        self.eval_metric = self.text_model.evaluate(x_val, y_val)
        self.prediction_model_time = time.time() - training_time_start - self.encoding_training_time - self.pre_model_training_transformations_time - self.model_training_time - self.prediction_transformation_time

        return {'encoding_training_time': self.encoding_training_time,
                'pre_model_training_transformations_time': self.pre_model_training_transformations_time,
                'model_training_time': self.model_training_time,
                'prediction_transformation_time': self.prediction_transformation_time,
                'prediction_model_time': self.prediction_model_time,
                'eval_metric': self.eval_metric}

    def combine_text_columns(self, x1, x2):
        if self.text_combination_method == 'side-by-side':
            return np.hstack([x1, x2])
        if self.text_combination_method == 'subtraction':
            return x1 - x2


params = {'e_type': ['tfidf',
                     'count',
                     'binary',
                     'lda',
                     'doc2vec'],
          'm_type': ['dnn',
                     'logistic',
                     'rf'],
          'text_combination_method': ['side-by-side',
                                             'subtraction'],
          'max_vocab_size': [i for i in range(10, 1000)],
          'max_n_gram': [i for i in range(1, 4)],
          'num_of_topics': [i for i in range(4, 128)],
          'encoding_size': [i for i in range(4, 128)],
          'num_of_hidden_layers': [i for i in range(1, 5)],
          'width_of_hidden_layers': [i for i in range(4, 2056, 4)],
          'batch_size': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
          'rf_criterion': ['gini', 'entropy'],
          'rf_max_depth': [i for i in range(3, 100)],
          'rf_min_samples_split': [i for i in range(2, 50)],
          'gensim_passes':[i for i in range(1, 100)]
          }

dm = DataManager()
df1 = dm.get_dataset_of_meta_matching_page_text(max_dataset_size=10000)
df2 = dm.get_dataset_of_meta_not_matching_page_text(max_dataset_size=10000)

df = pd.concat([df1, df2])
print(df.describe())

num_of_param_searches = 1000
results = []

for _ in range(num_of_param_searches):

    next_params = dict()
    for p in params:
        next_params[p] = random.choice(params[p])

    # next_params['m_type'] = 'logistic'
    # next_params['e_type'] = 'doc2vec'
    # next_params['text_combination_method'] = 'subtraction'

    print()
    print(f'Next set of params: {next_params}')
    pipeline = ModelPipeline(**next_params)
    result = pipeline.run_pipeline(df)

    next_params.update(result)
    results.append(next_params)
    print(f'result: {result}')

    sorted_results = sorted(results, key = lambda x: x['eval_metric'], reverse=True)
    print(sorted_results)

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(f'{dir_loc}/text_analysis.csv', sep = '|', index = False)


