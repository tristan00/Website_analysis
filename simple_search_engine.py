import pandas as pd
import sqlite3
import pickle
from common import get_initial_website_list, dir_loc, url_record_file_name, db_name, url_record_backup_file_name, clean_text, index1_db_name, get_distance, initial_website_file_name, tokenize, DataManager
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from scipy.spatial.distance import cosine
import numpy as np
import uuid
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
from keras import models, layers, optimizers, callbacks
from sklearn.model_selection import train_test_split

class SimpleSearchEngine:
    def __init__(self, doc2vec_num_of_dim = 16, min_word_len = 4, max_training_documents = 100000):
        self.doc2vec_num_of_dim = doc2vec_num_of_dim
        self.min_word_len = min_word_len
        self.max_training_documents = max_training_documents
        self.model_title = Doc2Vec(size=self.doc2vec_num_of_dim,
                workers = 6)
        self.model_description = Doc2Vec(size=self.doc2vec_num_of_dim,
                workers = 6)
        self.model_keywords = Doc2Vec(size=self.doc2vec_num_of_dim,
                workers = 6)
        self.model_meta = Doc2Vec(size=self.doc2vec_num_of_dim,
                                   workers=6)
        self.model_query = Doc2Vec(size=self.doc2vec_num_of_dim,
                                   workers=6)
        self.main_model = self.get_main_model()
        self.dm = DataManager()

    def get_results(self, query):
        query_clean = tokenize(query)
        query_clean = [i for i in query_clean if len(i) >= self.min_word_len]

        where_clause1 = ' and '.join(["meta like '% {w} %'".format(w=w) for w in query_clean])
        where_clause2 = ' or '.join(["title like '% {w} %'".format(w=w) for w in query_clean])

        query_string1 = '''
        Select url, title, meta 
        from websites
        where ({where_clause1}) or ({where_clause2})  
        '''.format(where_clause1 =where_clause1, where_clause2 =where_clause2)

        query_string2 = '''
                Select url, title, meta 
                from websites
                where ({where_clause2})  
                '''.format(where_clause1=where_clause1, where_clause2=where_clause2)

        print(query_string1)

        with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name)) as conn_disk:
            res = conn_disk.execute(query_string1)

            counter = 0
            for i in res:
                counter += 1
                print(i[0])
            print(counter)

    def fit_all_models(self):
        self.fit_title_doc2vec()
        self.fit_meta_doc2vec()
        self.fit_query_doc2vec()
        self.fit_keyword_doc2vec()
        self.fit_description_doc2vec()
        data_df = self.dm.get_labeled_data_sample(self.max_training_documents)
        data_df = data_df.sample(frac = 1)
        data_df = data_df.reset_index()
        meta_vec = data_df.apply(lambda x: self.model_meta.infer_vector(tokenize(x['meta'])), axis = 1, result_type = 'expand')
        title_vec = data_df.apply(lambda x: self.model_meta.infer_vector(tokenize(x['title'])), axis = 1, result_type = 'expand')
        query_vec = data_df.apply(lambda x: self.model_meta.infer_vector(tokenize(x['query'])), axis = 1, result_type = 'expand')
        keyword_vec = data_df.apply(lambda x: self.model_meta.infer_vector(tokenize(x['query'])), axis = 1, result_type = 'expand')
        description_vec = data_df.apply(lambda x: self.model_meta.infer_vector(tokenize(x['query'])), axis = 1, result_type = 'expand')

        target = data_df[['target']]

        x = pd.concat([title_vec, meta_vec, query_vec, keyword_vec, description_vec], axis = 1).values
        y = target.values
        x_train, x_val, y_train, y_val = train_test_split(x, y)

        cb = callbacks.EarlyStopping(monitor='val_loss',
                                     min_delta=0,
                                     patience=5,
                                     verbose=0, mode='auto')
        mcp_save = callbacks.ModelCheckpoint('{}/main_model.h5'.format(dir_loc), save_best_only=True, monitor='val_loss',
                                             verbose=1)

        self.main_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=[cb, mcp_save])
        self.main_model = models.load_model('{}/main_model.h5'.format(dir_loc))


    def get_main_model(self):
        model = models.Sequential()
        model.add(layers.Dense(32, input_dim=self.doc2vec_num_of_dim*5, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit_title_doc2vec(self):
        print('starting title doc2vec training')
        start_time = time.time()
        titles = self.dm.sample_titles(self.max_training_documents)
        print('got data: {0}'.format(time.time() - start_time))
        documents = [TaggedDocument(tokenize(doc), [i]) for i, doc in enumerate(titles)]
        self.model_title.build_vocab(documents)
        print('vocab built: {0}'.format(time.time() - start_time))
        self.model_title.train(documents, total_examples=self.model_title.corpus_count, epochs=10)
        self.model_title.save('{dir}/{model_name}'.format(dir=dir_loc, model_name='doc2vec_title'))
        print('title model fit {}'.format(time.time() - start_time))
        print('')


    def fit_meta_doc2vec(self):
        print('starting meta doc2vec training')
        start_time = time.time()

        titles = self.dm.sample_metas(self.max_training_documents)
        documents = [TaggedDocument(tokenize(doc), [i]) for i, doc in enumerate(titles)]
        print('got data: {0}'.format(time.time() - start_time))

        self.model_meta.build_vocab(documents)
        print('vocab built: {0}'.format(time.time() - start_time))

        self.model_meta.train(documents, total_examples=self.model_meta.corpus_count, epochs=10)
        self.model_meta.save('{dir}/{model_name}'.format(dir=dir_loc, model_name='doc2vec_meta'))
        print('meta model fit {}'.format(time.time() - start_time))
        print('')

    def fit_query_doc2vec(self):
        print('starting query doc2vec training')
        start_time = time.time()
        queries = self.dm.sample_queries(self.max_training_documents)
        print('got data: {0}'.format(time.time() - start_time))

        documents = [TaggedDocument(tokenize(doc), [i]) for i, doc in enumerate(queries)]
        self.model_query.build_vocab(documents)
        print('vocab built: {0}'.format(time.time() - start_time))

        self.model_query.train(documents, total_examples=self.model_query.corpus_count, epochs=10)

        self.model_query.save('{dir}/{model_name}'.format(dir=dir_loc, model_name='doc2vec_query'))
        print('query model fit {}'.format(time.time() - start_time))

        print('')


    def fit_keyword_doc2vec(self):
        print('starting keyword doc2vec training')
        start_time = time.time()

        titles = self.dm.sample_metas(self.max_training_documents)
        documents = [TaggedDocument(tokenize(doc), [i]) for i, doc in enumerate(titles)]
        print('got data: {0}'.format(time.time() - start_time))

        self.model_keywords.build_vocab(documents)
        print('vocab built: {0}'.format(time.time() - start_time))

        self.model_keywords.train(documents, total_examples=self.model_keywords.corpus_count, epochs=10)
        self.model_keywords.save('{dir}/{model_name}'.format(dir=dir_loc, model_name='doc2vec_keyword'))
        print('keyword model fit {}'.format(time.time() - start_time))
        print('')


    def fit_description_doc2vec(self):
        print('starting description doc2vec training')
        start_time = time.time()
        queries = self.dm.sample_queries(self.max_training_documents)
        print('got data: {0}'.format(time.time() - start_time))

        documents = [TaggedDocument(tokenize(doc), [i]) for i, doc in enumerate(queries)]
        self.model_description.build_vocab(documents)
        print('vocab built: {0}'.format(time.time() - start_time))

        self.model_description.train(documents, total_examples=self.model_description.corpus_count, epochs=10)

        self.model_description.save('{dir}/{model_name}'.format(dir=dir_loc, model_name='doc2vec_query'))
        print('description model fit {}'.format(time.time() - start_time))

        print('')


if __name__ == '__main__':
    s = SimpleSearchEngine()
    s.fit_all_models()
    # tests = ['python sqlite']
    #
    # for i in tests:
    #     print(i, s.get_results(i))




