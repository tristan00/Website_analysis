from common import tokenize, DataManager, dir_loc, clean_text
import fasttext
# from keras.models import Sequential, load_model
from keras import layers, callbacks, models
import uuid
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from text_methods import TextEncoder, get_random_param_grid, Pipeline, Model




class SimpleSearchEngine():
    '''
    Will use aol training set and a
    '''

    all_text_types = ['html', 'text', 'meta', 'query']

    def __init__(self, text_types,
                 query_encoding_type,
                 meta_encoding_type,
                 text_encoding_type,
                 html_encoding_type,
                 model_type,
                 text_combination_method,
                 pca_components,
                 max_vocab_size,
                 min_n_gram,
                 max_n_gram,
                 num_of_topics,
                 encoding_size,
                 num_of_hidden_layers,
                 width_of_hidden_layers,
                 batch_size,
                 rf_criterion,
                 rf_max_depth,
                 rf_min_samples_split,
                 vectorizer_epochs,
                 kernel,
                 max_page_size,
                 fasttext_algorithm,
                 tokenizer_level,
                 max_df,
                 use_pca,
                 dm):

        self.dm = dm
        self.engine_id = str(uuid.uuid4())

        self.text_types = text_types

        self.vectorizers = dict()
        self.vectorizers['html'] = TextEncoder(html_encoding_type, self.engine_id, max_vocab_size=max_vocab_size, min_n_gram=min_n_gram, max_n_gram=max_n_gram, num_of_topics=num_of_topics,
                 encoding_size=encoding_size, vectorizer_epochs=vectorizer_epochs, max_page_size=max_page_size, fasttext_algorithm=fasttext_algorithm,
                 tokenizer_level=tokenizer_level, max_df = max_df)
        self.vectorizers['text'] = TextEncoder(text_encoding_type, self.engine_id, max_vocab_size=max_vocab_size, min_n_gram=min_n_gram, max_n_gram=max_n_gram, num_of_topics=num_of_topics,
                 encoding_size=encoding_size, vectorizer_epochs=vectorizer_epochs, max_page_size=max_page_size, fasttext_algorithm=fasttext_algorithm,
                 tokenizer_level=tokenizer_level, max_df = max_df)
        self.vectorizers['meta'] = TextEncoder(meta_encoding_type, self.engine_id, max_vocab_size=max_vocab_size, min_n_gram=min_n_gram, max_n_gram=max_n_gram, num_of_topics=num_of_topics,
                 encoding_size=encoding_size, vectorizer_epochs=vectorizer_epochs, max_page_size=max_page_size, fasttext_algorithm=fasttext_algorithm,
                 tokenizer_level=tokenizer_level, max_df = max_df)
        self.vectorizers['query'] = TextEncoder(query_encoding_type, self.engine_id, max_vocab_size=max_vocab_size, min_n_gram=min_n_gram, max_n_gram=max_n_gram, num_of_topics=num_of_topics,
                 encoding_size=encoding_size, vectorizer_epochs=vectorizer_epochs, max_page_size=max_page_size, fasttext_algorithm=fasttext_algorithm,
                 tokenizer_level=tokenizer_level, max_df = max_df)

        self.pipeline_1 = Pipeline(text_combination_method, pca_components)
        if use_pca:
            self.pipeline_2 = Pipeline('pca', pca_components)
        else:
            self.pipeline_2 = Pipeline('hstack', pca_components)


        self.model = Model(model_type,
                 num_of_hidden_layers=num_of_hidden_layers,
                 width_of_hidden_layers=width_of_hidden_layers,
                 batch_size=batch_size,
                 nb_epoch=100,
                 patience=10,
                 rf_n_estimators=100,
                 rf_criterion=rf_criterion,
                 rf_max_depth=rf_max_depth,
                 rf_min_samples_split=rf_min_samples_split,
                 kernel=kernel,
                 engine_id=self.engine_id
                 )

        self.index_file_location = f'/{dir_loc}/indexes/index_{self.engine_id}/'
        self.index_size = None


    def train(self, n):
        df = self.dm.get_query_dataset(max_dataset_size=n, balance=True, text_types=self.text_types)
        df['query'] = df['query'].apply(clean_text)
        self.fit_text_vectorizers(df)
        vectorized_texts = self.transform_df_with_text_vectorizers(df)
        self.pipeline_1.fit(vectorized_texts)
        self.pipeline_2.fit(self.pipeline_1.transform(vectorized_texts))

        df_train, df_val = train_test_split(df)

        x_train = self.transform_df_with_text_vectorizers(df_train)
        x_train = self.pipeline_1.transform(x_train)
        x_train = self.pipeline_2.transform([x_train])

        x_val = self.transform_df_with_text_vectorizers(df_val)
        x_val = self.pipeline_1.transform(x_val)
        x_val = self.pipeline_2.transform([x_val])

        y_train = df_train['target']
        y_val = df_val['target']

        self.model.fit(x_train, x_val, y_train, y_val)
        print(f'Model accuracy score: {self.model.metric}')


    def index_websites(self, n = None):
        gen = self.dm.data_generator(self.text_types, n)

        encoded_inputs = []
        for i in gen:
            encoded_inputs.append(i)

        self.index_df = pd.DataFrame.from_dict(encoded_inputs)
        output_dict = dict()
        for text_type in self.text_types:
            output_dict[text_type] = self.vectorizers[text_type].transform(list(self.index_df[text_type]))

        self.df_indexes = dict()
        for text_type in self.text_types:
            self.df_indexes[text_type] = pd.DataFrame(data=output_dict[text_type], index=self.index_df.index)
            self.df_indexes[text_type].to_csv(f'{self.index_file_location}/{text_type}', sep = '|', index = False)
            self.index_size = self.df_indexes[text_type].shape[0]

    def get_query(self, q_str, n_results):
        df_indexes_copy = self.index_df.copy()

        x = []
        query_rep = self.vectorizers['query'].transform([q_str])[0]
        x.append(np.array([query_rep for _ in range(self.index_size)]))

        for text_type in self.text_types:
            x.append(self.df_indexes[text_type].values)

        x = self.pipeline_1.transform(x)
        x = self.pipeline_2.transform(x)

        df_indexes_copy['preds'] = self.model.predict(x)
        df_encoded_copy = df_indexes_copy.sort_values(by = ['preds'], ascending=False)
        return df_encoded_copy[:n_results]


    def fit_text_vectorizers(self, df):
        self.vectorizers['query'].fit(set(df['query']))
        for text_type in self.text_types:
            self.vectorizers[text_type].fit(set(df[text_type]))

    def transform_df_with_text_vectorizers(self, df, query = True):
        results = []
        if query:
            results.append(self.vectorizers['query'].transform(list(df['query'])))
        for text_type in self.text_types:
            results.append(self.vectorizers[text_type].transform(list(df[text_type])))
        return results


def save_engine(engine, loc):
    del engine.model, engine.vectorizer, engine.dm
    with open(loc, 'wb') as f:
        pickle.dump(engine, f)


def load_engine(loc):
    with open(loc, 'rb') as f:
        engine = pickle.load(f)
    engine.dm = DataManager()
    return engine


def test_params():
    data_manager = DataManager()
    training_size = 10000

    results = []
    run_id = str(uuid.uuid4())

    while True:
        next_params = get_random_param_grid()
        print(f'next_params: {next_params}')

        next_params['dm'] = data_manager

        s = SimpleSearchEngine(**next_params)
        s.train(training_size)
        next_params['accuracy'] = s.model.metric
        next_params['engine_id'] = s.engine_id
        next_params['training_size'] = training_size
        next_params['dm'] = 0
        print(f'next_params: {next_params}')

        results.append(next_params)
        results = sorted(results, key = lambda x: x['accuracy'], reverse=True)
        print(f'results: {results[:10]}')

        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv(f'{dir_loc}/text_analysis_results/result_metrics/query_text_analysis_{run_id}.csv', sep='|', index=False)


if __name__ == '__main__':
    test_params()
    # test_queries = ['python search test', 'microsoft sql', 'weather today']
    # data_size = 100000
    #
    # s = SimpleSearchEngine(vectorizer_type = 'fasttext', text_type = 'meta', num_of_dim = 300)
    # s.train(n = data_size)
    # s.index_websites(n = data_size)
    #
    # for query in test_queries:
    #     start_time = time.time()
    #     df_preds = s.get_query('python search test', 10)
    #     query_time = time.time() - start_time
    #     urls = df_preds['url'].tolist()
    #     scores = df_preds['preds'].tolist()
    #     print(f'query_time: {query_time}, query: {query}, urls: {urls}, scores: {scores}')





