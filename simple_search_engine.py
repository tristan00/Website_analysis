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


def clean_and_filter_text(s, min_word_len = 5):
    return ' '.join(i for i in tokenize(s) if len(i) >= min_word_len)



class SimpleSearchEngine():
    '''
    Will use aol training set and a
    '''

    def __init__(self, num_of_dim = 256,
                 min_word_len = 4,
                 dnn_architecture = (512, 512, 512, 512, 512),
                 fasttext_training_file_location = None,
                 model_file_location = None,
                 index_file_location = None,
                 vectorizer_file_location = None,
                 fasttext_algorithm = 'skipgram'):

        self.dm = DataManager()
        self.id = str(uuid.uuid4())
        self.fasttext_algorithm = fasttext_algorithm
        self.encoding_size = num_of_dim
        self.min_word_len = min_word_len
        self.dnn_architecture = dnn_architecture

        if not model_file_location:
            self.model_file_location = f'/{dir_loc}/models/model_{self.id}'
        if not vectorizer_file_location:
            self.vectorizer_file_location = f'/{dir_loc}/models/vec_{self.id}'
        if not fasttext_training_file_location:
            self.fasttext_training_file_location = f'/{dir_loc}/models/fasttext_training_file_{self.id}'
        if not index_file_location:
            self.index_file_location = f'/{dir_loc}/indexes/index_{self.id}'

    def train(self, n):
        df = self.dm.get_query_dataset(max_dataset_size=n, balance=True, text_types=('html', ))
        df['html'] = df['html'].apply(lambda x: clean_and_filter_text(x, self.min_word_len))
        df['query'] = df['query'].apply(clean_text)
        documents = set(df['html']) | set(df['query'])

        df_train, df_val = train_test_split(df)

        with open(self.fasttext_training_file_location, 'w') as f:
            for i in documents:
                f.write(i + '\n')

        self.vectorizer = fasttext.train_unsupervised(self.fasttext_training_file_location, model=self.fasttext_algorithm,
                                                          dim = self.encoding_size, epoch = 10)
        self.vectorizer.save_model(self.vectorizer_file_location)
        html_encoded_train = np.array([self.vectorizer[i] for i in df_train['html'].tolist()])
        query_encoded_train = np.array([self.vectorizer[i] for i in df_train['query'].tolist()])
        html_encoded_val = np.array([self.vectorizer[i] for i in df_val['html'].tolist()])
        query_encoded_val = np.array([self.vectorizer[i] for i in df_val['query'].tolist()])

        x_train = np.hstack([html_encoded_train, query_encoded_train])
        x_val = np.hstack([html_encoded_val, query_encoded_val])
        y_train = df_train['target']
        y_val = df_val['target']

        self.model = models.Sequential()
        self.model = models.Sequential()
        self.model.add(layers.Dense(self.dnn_architecture[0], input_dim=x_train.shape[1], activation='relu'))

        for i in self.dnn_architecture[1:]:
            self.model.add(layers.Dense(i, activation='relu'))

        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        cb1 = callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=10,
                                      verbose=0,
                                      mode='auto')
        cb2 = callbacks.ModelCheckpoint(self.model_file_location,
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True,
                                        save_weights_only=False,
                                        mode='auto',
                                        period=1)
        self.model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       callbacks=[cb1, cb2],
                       batch_size=32,
                       nb_epoch=100)
        self.model = models.load_model(self.model_file_location)


    def index_websites(self, n = None):
        gen = self.dm.data_generator(['html'], n)

        encoded_inputs = []
        for i in gen:
            output_dict = dict()
            html = i['html']
            html_encoded = self.vectorizer.get_sentence_vector(clean_and_filter_text(html))

            output_dict['url'] = i['url']
            for c, i in enumerate(html_encoded):
                output_dict[str(c).zfill(5)] = i
            encoded_inputs.append(output_dict)
        self.df_encoded = pd.DataFrame(encoded_inputs)
        self.df_encoded.to_csv(self.index_file_location, sep = '|', index = False)

    def get_query(self, q_str, n_results):
        df_encoded_copy = self.df_encoded.copy()
        fastext_columns = [i for i in df_encoded_copy.columns if i != 'url']
        html_np_arrays = df_encoded_copy[fastext_columns].values

        query_rep = self.vectorizer.get_sentence_vector(clean_text(q_str))
        query_np = np.array([query_rep for _ in range(self.df_encoded.shape[0])])
        print(query_np.shape, html_np_arrays.shape)
        x = np.hstack([html_np_arrays, query_np])
        df_encoded_copy['pred'] = self.model.predict(x)
        df_encoded_copy = df_encoded_copy[['url', 'pred']]
        df_encoded_copy = df_encoded_copy.sort_values(by = ['pred'], ascending=False)
        return df_encoded_copy[:n_results]


def save_engine(engine, loc):
    del engine.model, engine.vectorizer, engine.dm
    with open(loc, 'wb') as f:
        pickle.dump(engine, f)


def load_engine(loc):
    with open(loc, 'rb') as f:
        engine = pickle.load(f)
    engine.dm = DataManager()
    return engine


if __name__ == '__main__':
    test_queries = ['python search test', 'microsoft sql', 'weather today']
    data_size = 100000

    s = SimpleSearchEngine()
    s.train(n = 200000)
    s.index_websites()

    for i in test_queries:
        start_time = time.time()
        df_preds = s.get_query('python search test', 10)
        query_time = time.time() - start_time
        urls = df_preds['url'].tolist()
        scores = df_preds['pred'].tolist()
        print(f'query_time: {query_time}, urls: {urls}, scores: {scores}')





