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
from keras.models import Model, load_model
import sqlite3
from common import tokenize, DataManager, dir_loc, clean_text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import uuid
import os
from sklearn.ensemble import RandomForestClassifier
import time
import random
import tensorflow_hub as hub
import tensorflow as tf
from bert_embedding import BertEmbedding
from sklearn.svm import SVC
import mxnet as mx
import gc
import fasttext
from sklearn.preprocessing import OneHotEncoder
import functools
import operator
import pickle

path = '/home/td/Documents/web_models'

replacement_value = 'replacement_value'


def vectorize_topic_models(topic_tuples, num_of_topics):
    vector = [0 for _ in range(num_of_topics)]
    for i in topic_tuples:
        vector[i[0]] = i[1]
    return vector


class TextModel():
    valid_types = ['dnn',
                   'logistic',
                   'rf',
                   'svm',
                   'lr',
                   'elasticnet']

    def __init__(self, m_type,
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
                 pipeline_id = None
                 ):

        assert m_type in self.valid_types

        self.file_loc = f'{dir_loc}/text_analysis_results/models/{pipeline_id}_{m_type}_model'
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
        self.kernel = kernel

    def fit(self, x_train, x_val, y_train, y_val):
        if self.m_type == 'dnn':

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
            cb2 = callbacks.ModelCheckpoint(self.file_loc,
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
            self.model = load_model(self.file_loc)

        if self.m_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=self.rf_n_estimators,
                                                criterion=self.rf_criterion,
                                                max_depth=self.rf_max_depth,
                                                min_samples_split=self.rf_min_samples_split)
            self.model.fit(x_train, y_train)
        if self.m_type == 'svm':
            self.model = SVC(kernel=self.kernel)
            print(x_train.shape, y_train.shape)
            self.model.fit(x_train, y_train)
        if self.m_type == 'lr':
            self.model = LogisticRegression()
            print(x_train.shape, y_train.shape)
            self.model.fit(x_train, y_train)
        if self.m_type == 'elasticnet':
            self.model = ElasticNet()
            print(x_train.shape, y_train.shape)
            self.model.fit(x_train, y_train)

        if self.m_type in ['lr', 'rf', 'elasticnet', 'svm', 'logistic']:
            with open(self.file_loc, 'wb') as f:
                pickle.dump(self.model, f)

    def predict(self, x):
        preds = self.model.predict(x)
        print(f'model prediction shape: {preds.shape}')
        return preds

    def evaluate(self, x_val, y_val):
        preds = np.rint(self.model.predict(x_val)).astype(int)
        metric = accuracy_score(preds, y_val)
        return metric


class TextEncoding():
    valid_types = ['tfidf',
                   'count',
                   'binary',
                   'lda',
                   'doc2vec',
                   'bert_avg',
                   'fasttext']

    def __init__(self, e_type, max_vocab_size=10000, min_word_n_gram=1, max_word_n_gram=2, min_char_n_gram=1,
                 max_char_n_gram=2, max_df=.01, num_of_topics=64,
                 encoding_size=64, epochs=3, max_page_size=10000, fasttext_algorithm='skipgram',
                 tokenizer_level='word', pipeline_id = None):

        assert e_type in self.valid_types

        self.file_loc = f'{dir_loc}/text_analysis_results/models/{pipeline_id}_{e_type}_encoder'
        self.e_type = e_type
        self.max_vocab_size = max_vocab_size
        self.min_word_n_gram = min_word_n_gram
        self.max_word_n_gram = max_word_n_gram
        self.min_char_n_gram = min_char_n_gram
        self.max_char_n_gram = max_char_n_gram
        self.tokenizer_level = tokenizer_level
        self.max_df = max_df
        self.num_of_topics = num_of_topics
        self.encoding_size = encoding_size
        self.common_dictionary = None
        self.epochs = epochs
        self.fasttext_algorithm = fasttext_algorithm
        self.max_page_size = max_page_size
        # self.glove_model = gensim.models.KeyedVectors.load_word2vec_format(f'{dir_loc}/glove.840B.300d.txt')

    def fit(self, documents):
        documents = [tokenize(d) for d in documents]
        documents = [d[:self.max_page_size] for d in documents]
        documents = [' '.join(d) for d in documents]

        if self.e_type in ['tfidf', 'count', 'binary']:
            if self.tokenizer_level == 'word':
                min_n_gram = self.min_word_n_gram
                max_n_gram = self.max_word_n_gram
            else:
                min_n_gram = self.min_char_n_gram
                max_n_gram = self.max_char_n_gram

            if self.e_type == 'tfidf':
                self.vectorizer = CountVectorizer(ngram_range=(min_n_gram, max_n_gram),
                                                  max_features=self.max_vocab_size, binary=False, max_df=self.max_df,
                                                  analyzer=self.tokenizer_level)
                self.vectorizer.fit(documents)
            if self.e_type == 'count':
                self.vectorizer = CountVectorizer(ngram_range=(min_n_gram, max_n_gram),
                                                  max_features=self.max_vocab_size, binary=False, max_df=self.max_df,
                                                  analyzer=self.tokenizer_level)
                self.vectorizer.fit(documents)
            if self.e_type == 'binary':
                self.vectorizer = CountVectorizer(ngram_range=(min_n_gram, max_n_gram),
                                                  max_features=self.max_vocab_size, binary=False, max_df=self.max_df,
                                                  analyzer=self.tokenizer_level)
                self.vectorizer.fit(documents)
            with open(self.file_loc, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        if self.e_type == 'lda':
            documents_tokenized = [tokenize(i) for i in documents]
            self.common_dictionary = Dictionary(documents_tokenized)
            common_corpus = [self.common_dictionary.doc2bow(text) for text in documents_tokenized]
            self.vectorizer = ldamodel.LdaModel(common_corpus, id2word=self.common_dictionary,
                                                num_topics=self.num_of_topics, passes=self.epochs)
            self.vectorizer.save(self.file_loc)
        if self.e_type == 'doc2vec':
            tagged_documents = [TaggedDocument(tokenize(doc), [i]) for i, doc in enumerate(documents)]
            self.vectorizer = Doc2Vec(tagged_documents, vector_size=self.encoding_size, window=2, min_count=1,
                                      workers=4, epochs=self.epochs, max_vocab_size=100000)
            self.vectorizer.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
            self.vectorizer.save(self.file_loc)
        if self.e_type == 'fasttext':
            with open('/tmp/fasttext_file', 'w') as f:
                for i in documents:
                    f.write(clean_text(i) + '\n')
            self.vectorizer = fasttext.train_unsupervised('/tmp/fasttext_file', model=self.fasttext_algorithm,
                                                          dim=self.encoding_size)
            self.vectorizer.save_model(self.file_loc)

    def transform(self, documents):

        documents = [tokenize(d) for d in documents]
        documents = [d[:self.max_page_size] for d in documents]
        documents = [' '.join(d) for d in documents]

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
        if self.e_type in ['doc2vec']:
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

        if self.e_type in ['fasttext']:
            documents_clean = [clean_text(i) for i in documents]

            results = []
            for i in documents_clean:
                if i:
                    results.append(self.vectorizer.get_sentence_vector(i))
                    # results.append(self.vectorizer[i])
                else:
                    results.append(np.array([0 for _ in range(self.encoding_size)]))

            return np.array(results)


class ModelPipeline():
    valid_text_combination_methods = ['hstack',
                                      'subtraction',
                                      'multiplication',
                                      'addition']

    def __init__(self, e_type='binary',
                 max_vocab_size=1000,
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
                 text_combination_method='hstack',
                 epochs=1,
                 kernel='rbf',
                 max_page_size=1000,
                 text_types=None,
                 fasttext_algorithm='skipgram', min_word_n_gram=1, max_word_n_gram=2, min_char_n_gram=1,
                 max_char_n_gram=2, tokenizer_level='word', pipeline_id=None):

        assert text_combination_method in self.valid_text_combination_methods

        self.file_loc = f'{dir_loc}/text_analysis_results/models/{pipeline_id}_model'
        self.text_combination_method = text_combination_method
        self.encoding_training_time = 0
        self.pre_model_training_transformations_time = 0
        self.model_training_time = 0
        self.prediction_transformation_time = 0
        self.prediction_model_time = 0
        self.eval_metric = 0
        self.text_types = text_types
        self.pipeline_id = pipeline_id
        self.text_encoding = TextEncoding(e_type=e_type,
                                          max_vocab_size=max_vocab_size,
                                          min_word_n_gram=min_word_n_gram,
                                          max_word_n_gram=max_word_n_gram,
                                          min_char_n_gram=min_char_n_gram,
                                          max_char_n_gram=max_char_n_gram,
                                          max_df=max_df,
                                          num_of_topics=num_of_topics,
                                          encoding_size=encoding_size,
                                          epochs=epochs,
                                          max_page_size=max_page_size,
                                          fasttext_algorithm=fasttext_algorithm,
                                          tokenizer_level=tokenizer_level,
                                          pipeline_id=pipeline_id)
        self.text_model = TextModel(m_type=m_type,
                                    num_of_hidden_layers=num_of_hidden_layers,
                                    width_of_hidden_layers=width_of_hidden_layers,
                                    batch_size=batch_size,
                                    nb_epoch=nb_epoch,
                                    patience=patience,
                                    rf_n_estimators=rf_n_estimators,
                                    rf_criterion=rf_criterion,
                                    rf_max_depth=rf_max_depth,
                                    rf_min_samples_split=rf_min_samples_split,
                                    kernel=kernel,
                                    pipeline_id=pipeline_id)

    def run_pipeline_for_query_text_association(self, df):
        '''
        Learns to predict if query matches website text

        :param df: Df with cols: all the text types, target
        :return:
        '''
        training_time_start = time.time()
        df_train, df_val = train_test_split(df, random_state=1)

        train_x1 = []
        val_x1 = []
        all_documents_list = list()

        for text_type in self.text_types:
            train_x1.append(list(df_train[text_type]))
            val_x1.append(list(df_val[text_type]))

            all_documents_list.extend(list(df_train[text_type]))
            all_documents_list.extend(list(df_val[text_type]))

        train_x2 = list(df_train['query'])
        val_x2 = list(df_val['query'])
        all_documents_list.extend(list(df_train['query']))
        all_documents_list.extend(list(df_val['query']))

        self.text_encoding.fit(all_documents_list)
        self.encoding_training_time = time.time() - training_time_start

        train_x1_encoded = [self.text_encoding.transform(train_x1[c]) for c, _ in enumerate(self.text_types)]
        val_x1_encoded = [self.text_encoding.transform(val_x1[c]) for c, _ in enumerate(self.text_types)]

        train_x1_combined = self.combine_variable_size_text_columns(train_x1_encoded)
        val_x1_combined = self.combine_variable_size_text_columns(val_x1_encoded)

        train_x2_encoded = self.text_encoding.transform(train_x2)
        val_x2_encoded = self.text_encoding.transform(val_x2)

        train_x_combined = self.combine_text_columns(train_x1_combined, train_x2_encoded)
        val_x_combined = self.combine_text_columns(val_x1_combined, val_x2_encoded)

        self.pre_model_training_transformations_time = time.time() - training_time_start - self.encoding_training_time

        print(type(train_x_combined), type(val_x_combined), type(df_train['target']), type(df_val['target']))
        print(train_x_combined.shape, val_x_combined.shape, df_train['target'].shape, df_val['target'].shape)
        self.text_model.fit(train_x_combined, val_x_combined, df_train['target'], df_val['target'])

        self.model_training_time = time.time() - training_time_start - self.encoding_training_time - self.pre_model_training_transformations_time
        self.text_model.predict(val_x_combined)

        self.prediction_transformation_time = time.time() - training_time_start - self.encoding_training_time - self.pre_model_training_transformations_time - self.model_training_time
        self.eval_metric = self.text_model.evaluate(val_x_combined, df_val['target'])

        self.prediction_model_time = time.time() - training_time_start - self.encoding_training_time - self.pre_model_training_transformations_time - self.model_training_time - self.prediction_transformation_time
        del self.text_encoding.vectorizer, self.text_model.model
        return {'encoding_training_time': self.encoding_training_time,
                'pre_model_training_transformations_time': self.pre_model_training_transformations_time,
                'model_training_time': self.model_training_time,
                'prediction_transformation_time': self.prediction_transformation_time,
                'prediction_model_time': self.prediction_model_time,
                'eval_metric': self.eval_metric,
                'pipeline_id': self.pipeline_id}

    def run_pipeline_for_text_meta_association(self, df):
        '''
        Learns to predict if meta matches the provided webpage.

        :param df: Df with cols:  meta, text, meta_matches_text
        :return:
        '''
        training_time_start = time.time()

        meta_documents = df['meta'].tolist()
        text_documents = df['text'].tolist()
        target = df['meta_matches_text']

        meta_documents_train, meta_documents_val, text_documents_train, text_documents_val, y_train, y_val = train_test_split(
            meta_documents, text_documents, target, random_state=1)

        # not thorough to train vectorizer on val data but adding val data handles oov case easily.
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
        print(x1.shape, x2.shape)
        if self.text_combination_method == 'hstack':
            return np.hstack([x1, x2])
        if self.text_combination_method == 'addition':
            return x1 + x2
        if self.text_combination_method == 'subtraction':
            return x1 - x2
        if self.text_combination_method == 'multiplication':
            return x1 * x2

    def combine_variable_size_text_columns(self, x_list):
        result = x_list[0]

        if len(x_list) > 1 and self.text_combination_method == 'hstack':
            result = np.hstack(x_list)
        if len(x_list) > 1 and self.text_combination_method == 'addition':
            result = functools.reduce(operator.add, x_list)
        if len(x_list) > 1 and self.text_combination_method == 'subtraction':
            result = functools.reduce(operator.sub, x_list)
        if len(x_list) > 1 and self.text_combination_method == 'multiplication':
            result = functools.reduce(operator.mul, x_list)

        print('combine_variable_size_text_columns', [i.shape for i in x_list], result.shape)
        return result


def run_meta_text_association_param_search():
    params = {'e_type': ['tfidf',
                         'count',
                         'binary',
                         'doc2vec',
                         'fasttext'],
              'm_type': ['dnn',
                         'rf',
                         'svm'],
              'text_combination_method': ['hstack',
                                          'subtraction',
                                          'multiplication'],
              'max_vocab_size': [i for i in range(10, 500)],
              'max_n_gram': [1],
              'num_of_topics': [i for i in range(4, 32)],
              'encoding_size': [i for i in range(4, 32)],
              'num_of_hidden_layers': [i for i in range(1, 4)],
              'width_of_hidden_layers': [i for i in range(4, 132, 4)],
              'batch_size': [32],
              'rf_criterion': ['gini', 'entropy'],
              'rf_max_depth': [i for i in range(3, 16)],
              'rf_min_samples_split': [i for i in range(2, 10)],
              'epochs': [i for i in range(1, 3)],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'max_page_size': [100, 500, 1000]
              }

    if not os.path.exists(f'{dir_loc}/meta_text_association_results'):
        os.makedirs(f'{dir_loc}/meta_text_association_results')

    dm = DataManager()
    df1 = dm.get_dataset_of_meta_matching_page_text(max_dataset_size=5000)
    df2 = dm.get_dataset_of_meta_not_matching_page_text(max_dataset_size=5000)
    df = pd.concat([df1, df2])
    print(df.describe())

    num_of_param_searches = 10000
    results = []

    run_id = str(uuid.uuid4())

    for _ in range(num_of_param_searches):
        next_params = dict()
        for p in params:
            next_params[p] = random.choice(params[p])

        # next_params['m_type'] = 'svm'
        # next_params['e_type'] = 'fasttext'
        # next_params['text_combination_method'] = 'subtraction'

        print()
        print(f'Next set of params: {next_params}')
        pipeline = ModelPipeline(**next_params)
        result = pipeline.run_pipeline_for_text_meta_association(df)

        del pipeline
        gc.collect()

        next_params.update(result)
        results.append(next_params)
        print(f'result: {result}')

        sorted_results = sorted(results, key=lambda x: x['eval_metric'], reverse=True)
        print(sorted_results)

        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv(f'{dir_loc}/text_analysis_results/text_analysis_{run_id}.csv', sep='|', index=False)


def run_meta_query_association_param_search():
    params = {'text_types': [['html']],
            # 'text_types': [['meta'], ['text'], ['html'], ['meta', 'text'], ['meta', 'html'], ['text', 'html'],
            #                  ['meta', 'text', 'html']],
              'e_type': ['tfidf',
                         'count',
                         'binary',
                         'fasttext',
                         'doc2vec'],
              'm_type': ['dnn',
                         'rf',
                         'svm',
                         'lr',
                         'elasticnet'],
              'text_combination_method': ['hstack', 'addition', 'multiplication', 'subtraction'],
              'max_vocab_size': [i for i in range(10, 2000)],
              'min_word_n_gram': [1],
              'max_word_n_gram': [1, 2],
              'min_char_n_gram': [1],
              'max_char_n_gram': [1, 2, 3],
              'num_of_topics': [i for i in range(4, 128)],
              'encoding_size': [i for i in range(4, 128)],
              'num_of_hidden_layers': [i for i in range(1, 4)],
              'width_of_hidden_layers': [i for i in range(4, 132, 4)],
              'batch_size': [32],
              'rf_criterion': ['gini', 'entropy'],
              'rf_max_depth': [i for i in range(3, 16)],
              'rf_min_samples_split': [i for i in range(2, 10)],
              'epochs': [i for i in range(1, 5)],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'max_page_size': [100, 500, 1000, 5000, 10000],
              'fasttext_algorithm': ['skipgram', 'cbow'],
              'tokenizer_level': ['word', 'char']
              }

    if not os.path.exists(f'{dir_loc}/text_query_analysis_results'):
        os.makedirs(f'{dir_loc}/text_query_analysis_results')

    dm = DataManager()
    df = dm.get_query_dataset(max_dataset_size=50000, balance=True, text_types=('html',))
    print(f'dataset shape: {df.shape}')

    num_of_param_searches = 10000
    results = []
    run_id = str(uuid.uuid4())

    for _ in range(num_of_param_searches):
        pipeline_id = str(uuid.uuid4())
        next_params = dict()
        for p in params:
            next_params[p] = random.choice(params[p])
        next_params['pipeline_id'] = pipeline_id

        print()
        print(f'Next set of params: {next_params}')
        pipeline = ModelPipeline(**next_params)
        result = pipeline.run_pipeline_for_query_text_association(df)

        file_loc = f'text_analysis_results/pickled_pipelines/{run_id}_{pipeline_id}'
        result['file_loc'] = file_loc
        with open(f'{dir_loc}/{file_loc}', 'wb') as f:
            pickle.dump(pipeline, f)
        del pipeline
        gc.collect()

        next_params.update(result)
        results.append(next_params)
        print(f'result: {result}')

        sorted_results = sorted(results, key=lambda x: x['eval_metric'], reverse=True)
        print(sorted_results)

        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv(f'{dir_loc}/text_analysis_results/result_metrics/query_text_analysis_{run_id}.csv', sep='|', index=False)


if __name__ == '__main__':
    run_meta_query_association_param_search()
