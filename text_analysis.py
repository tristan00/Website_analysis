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
from sklearn.metrics import f1_score
from keras.models import Model
import sqlite3
from common import tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
path = '/home/td/Documents/web_models'

from sklearn.preprocessing import OneHotEncoder
import functools
import operator
from common import get_data

replacement_value = 'replacement_value'


class OHE(OneHotEncoder):
    def __init__(self, n_values=None, categorical_features=None,
                 categories=None, sparse=True, dtype=np.float64,
                 handle_unknown='error', min_perc = .001, col_name = ''):
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
        super().fit( input_series)
        self.col_names = ['{col_base_name}_{value_name}'.format(col_base_name=self.col_name, value_name = i) for i in self.categories_[0]]

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
            self.valid_values = [i for  i, j in dict(s.value_counts(normalize = True)).items() if j >= self.min_perc]

        prediction_values_to_replace = [i for i in s.unique() if i not in self.valid_values]
        replace_dict = {i: replacement_value for i in prediction_values_to_replace}
        replace_dict.update({i:i for i in self.valid_values})
        s = s.map(replace_dict.get)
        return s.values.reshape(-1, 1)

    def process_output(self, output):
        output_df = pd.DataFrame(data = output.toarray(),
                                columns = self.col_names)
        return output_df

    def length(self):
        return len(self.categories_)


def vectorize_topic_models(topic_tuples, num_of_topics):
    vector = [0 for _ in range(num_of_topics)]
    for i in topic_tuples:
        vector[i[0]] = i[1]
    return vector


class WebsiteModels():
    def __init__(self, max_vocab_size = 10000, min_n_gram = 1, max_n_gram = 1):
        self.max_vocab_size = max_vocab_size
        self.bow_vectorizer = CountVectorizer(ngram_range = (min_n_gram, max_n_gram),max_features = self.max_vocab_size, binary=True, max_df=.1)

    def fit_baseline(self, documents):
        self.bow_vectorizer.fit(documents)

    def evaluate_simple_lr(self, documents, target):
        lr_model = LogisticRegression()
        x = self.bow_vectorizer.transform(documents).todense()
        x_train, x_val, y_train, y_val = train_test_split(x, target, test_size=.1, random_state=1)
        lr_model.fit(x_train, y_train)
        return f1_score(lr_model.predict(x_val), y_val)

    def evaluate_dnn_autoencoder(self, documents, target, encoding_size = 16):
        autoencoder = Sequential()
        autoencoder.add(Dense(1024, input_dim=self.max_vocab_size, activation='relu'))
        autoencoder.add(Dense(256, activation='relu'))
        autoencoder.add(Dense(encoding_size, activation='relu', name= 'encoder_output'))
        autoencoder.add(Dense(256, activation='relu'))
        autoencoder.add(Dense(1024, activation='relu'))
        autoencoder.add(Dense(self.max_vocab_size, activation='sigmoid'))
        autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_squared_error'])

        self.bow_vectorizer.fit(documents)
        x = self.bow_vectorizer.transform(documents).todense()
        x_train, x_val, y_train, y_val = train_test_split(x, target, test_size=.1, random_state=1)

        cb1 = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        cb2 = callbacks.ModelCheckpoint('{dir}/{model_name}.h5'.format(dir = path, model_name = 'dnn_autoencoder'),
                                        monitor='val_loss', verbose=0, save_best_only=True,
                                        save_weights_only=False, mode='auto', period=1)
        autoencoder.fit(x_train, x_train, validation_data=(x_val, x_val),callbacks=[cb1, cb2], batch_size = 32, nb_epoch=100)
        encoder = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer('encoder_output').output)
        train_encoded = encoder.predict(x_train)
        val_encoded = encoder.predict(x_val)

        lr_model = LogisticRegression()
        lr_model.fit(train_encoded, y_train)
        return f1_score(lr_model.predict(val_encoded), y_val)


    def evaluate_lda_topic_models(self, documents, target, num_topics = 16):
        x_train, x_val, y_train, y_val = train_test_split(documents, target, test_size=.1, random_state=1)
        docs = [tokenize(i) for i in x_train]
        common_dictionary = Dictionary(docs)
        common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
        lda = ldamodel.LdaModel(common_corpus, id2word=common_dictionary, num_topics=num_topics)
        print(lda.print_topics())

        x_train_topics = []
        for i in x_train:
            doc_tokenized = tokenize(i)
            doc_tokenized = common_dictionary.doc2bow(doc_tokenized)
            topic_tuples = lda[doc_tokenized]
            topics_vector = vectorize_topic_models(topic_tuples, num_topics)
            x_train_topics.append(topics_vector)

        x_val_topics = []
        for i in x_val:
            doc_tokenized = tokenize(i)
            doc_tokenized = common_dictionary.doc2bow(doc_tokenized)
            topic_tuples = lda[doc_tokenized]
            topics_vector = vectorize_topic_models(topic_tuples, num_topics)
            x_val_topics.append(topics_vector)

        lr_model = LogisticRegression()
        lr_model.fit(x_train_topics, y_train)
        return f1_score(lr_model.predict(x_val_topics), y_val)

    def evaluate_doc2vec(self, documents, target, encoding_size = 16):
        x_train, x_val, y_train, y_val = train_test_split(documents, target, test_size=.1, random_state=1)
        tagged_documents = [TaggedDocument(tokenize(doc), [i]) for i, doc in enumerate(x_train)]
        model = Doc2Vec(tagged_documents, vector_size=encoding_size, window=2, min_count=1, workers=4)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

        x_train_vec = []
        x_val_vec = []

        for i in x_train:
            x_train_vec.append(model.infer_vector(tokenize(i)))

        for i in x_val:
            x_val_vec.append(model.infer_vector(tokenize(i)))

        lr_model = LogisticRegression()
        lr_model.fit(x_train_vec, y_train)
        return f1_score(lr_model.predict(x_val_vec), y_val)



    def get_autoencoder_model(self, timesteps, input_dim , latent_dim):
        inputs = Input(shape=(timesteps, input_dim))
        encoded = LSTM(latent_dim)(inputs)

        decoded = RepeatVector(timesteps, )(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)

        sequence_autoencoder = Model(inputs, decoded)
        sequence_autoencoder.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])
        return sequence_autoencoder


    def seq2seq_autoencoder(self, documents, max_words=1000, max_text_len = 1000):
        enc = OHE()
        documents_tokenized = [tokenize(i) for i in documents]

        all_docs = functools.reduce(operator.concat, documents_tokenized)
        all_docs = pd.Series(all_docs)
        enc.fit(all_docs)
        documents_enc = []

        for i in documents_tokenized:
            next_doc = enc.transform(np.array(i).reshape(-1, 1))
            print(next_doc.max())
            documents_enc.append(next_doc)

        documents_np = np.array(documents_enc)
        # documents_np = documents_np.reshape(documents_np.shape[0], max_text_len, 1)
        documents_np = documents_np.reshape((documents_np.shape[0], max_text_len, len(enc.categories_.tolist())))
        autoencoder = self.get_autoencoder_model(max_text_len, enc.length(), 128)
        autoencoder.fit(documents_np, documents_np, batch_size=1)


df = get_data(1000)
score_dict = dict()
models = WebsiteModels(max_vocab_size = 5000)


models.seq2seq_autoencoder(df['page_text'])

models.fit_baseline(df['page_text'].dropna().tolist())


score_dict['lda_score_2'] = models.evaluate_lda_topic_models(df['page_text'], df['is_https'], num_topics = 2)
score_dict['lda_score_4'] = models.evaluate_lda_topic_models(df['page_text'], df['is_https'], num_topics = 4)
score_dict['lda_score_8'] = models.evaluate_lda_topic_models(df['page_text'], df['is_https'], num_topics = 8)
score_dict['lda_score_16'] = models.evaluate_lda_topic_models(df['page_text'], df['is_https'], num_topics = 16)
score_dict['lda_score_32'] = models.evaluate_lda_topic_models(df['page_text'], df['is_https'], num_topics = 32)
score_dict['lda_score_64'] = models.evaluate_lda_topic_models(df['page_text'], df['is_https'], num_topics = 64)
score_dict['dnn_score_8'] = models.evaluate_dnn_autoencoder(df['page_text'], df['is_https'], encoding_size = 8)
score_dict['dnn_score_16'] = models.evaluate_dnn_autoencoder(df['page_text'], df['is_https'], encoding_size = 16)
score_dict['dnn_score_32'] = models.evaluate_dnn_autoencoder(df['page_text'], df['is_https'], encoding_size = 32)
score_dict['dnn_score_64'] = models.evaluate_dnn_autoencoder(df['page_text'], df['is_https'], encoding_size = 64)
score_dict['dnn_score_128'] = models.evaluate_dnn_autoencoder(df['page_text'], df['is_https'], encoding_size = 128)
score_dict['doc2vec_score_16'] = models.evaluate_doc2vec(df['page_text'], df['is_https'], encoding_size = 16)
score_dict['doc2vec_score_32'] = models.evaluate_doc2vec(df['page_text'], df['is_https'], encoding_size = 32)
score_dict['doc2vec_score_64'] = models.evaluate_doc2vec(df['page_text'], df['is_https'], encoding_size = 64)
score_dict['doc2vec_score_128'] = models.evaluate_doc2vec(df['page_text'], df['is_https'], encoding_size = 128)
score_dict['doc2vec_score_256'] = models.evaluate_doc2vec(df['page_text'], df['is_https'], encoding_size = 256)

score_dict['lr_score'] = models.evaluate_simple_lr(df['response'], df['is_https'])


print('''simple logistic regression predicting scheme f1 score: {lr_score}
        dnn autoencoder with size 8 encoding predicting scheme f1 score: {dnn_score_8}
        dnn autoencoder with size 16 encoding predicting scheme f1 score: {dnn_score_16}
        dnn autoencoder with size 32 encoding predicting scheme f1 score: {dnn_score_32}
        dnn autoencoder with size 64 encoding predicting scheme f1 score: {dnn_score_64}
        dnn autoencoder with size 128 encoding predicting scheme f1 score: {dnn_score_128}
        lda topic model with 2 topics predicting scheme f1 score: {lda_score_2}
        lda topic model with 4 topics predicting scheme f1 score: {lda_score_4}
        lda topic model with 8 topics predicting scheme f1 score: {lda_score_8}
        lda topic model with 16 topics predicting scheme f1 score: {lda_score_16}
        lda topic model with 32 topics predicting scheme f1 score: {lda_score_32}
        lda topic model with 64 topics predicting scheme f1 score: {lda_score_64}
        doc2vec model with 16 embeddings predicting scheme f1 score: {doc2vec_score_16}
        doc2vec model with 32 embeddings predicting scheme f1 score: {doc2vec_score_32}
        doc2vec model with 64 embeddings predicting scheme f1 score: {doc2vec_score_64}
        doc2vec model with 128 embeddings predicting scheme f1 score: {doc2vec_score_128}
        doc2vec model with 256 embeddings predicting scheme f1 score: {doc2vec_score_256}
        '''.format(**score_dict))

# dnn_autoencoder = DNN_Encoder(max_vocab_size = 5000)
# dnn_autoencoder.fit(df['response'].dropna().tolist())
