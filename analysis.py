import pandas as pd
import pickle
import glob
import gensim
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import ldamodel
import string
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from keras import callbacks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from keras.models import Model
import sqlite3


translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
path = '/home/td/Documents/web_models'


def is_home_page(r):
    if len(r['path']) < 2 and len(r['params']) < 2 and len(r['query']) < 2 and len(r['fragment']) < 2:
        return 1
    return 0

def is_https(r):
    if r['scheme'] == 'https':
        return 1
    return 0


def get_data2():
    url_record_loc = '/home/td/Documents/web_data/links.pkl'
    db_loc = '/home/td/Documents/web_data/website.db'

    with open(url_record_loc, 'rb') as f:
        links = pickle.load(f)

    for i in links:
        links[i]['num_of_links_to'] = len(links[i]['urls_linking_to_page'])

    df = pd.DataFrame.from_dict(list(links.values()))
    df = df[df['scraped'] == 1]

    with sqlite3.connect(db_loc) as conn_disk:
        query = '''Select * from websites where url = ?'''

        for k, v in df.iterrows():
            res = conn_disk.execute(query, (v['url'],))
            res_list = list(res)
            if res_list:
                df.loc[df['url'] == v['url'], 'page_text'] = res_list[0][1]
    df = df.dropna(subset=['page_text'])
    df = df.drop_duplicates(subset = ['url'])
    df = df.reset_index(drop = True)

    df['is_home_page'] = df.apply(is_home_page, axis = 1)
    df['is_https'] = df.apply(is_https, axis = 1)
    print(df.shape)
    print(df.describe())




def get_data():
    files = glob.glob('/home/td/Documents/web_data/data*')
    dfs = []
    for i in files:
        with open(i, 'rb') as f:
            dfs.append(pd.DataFrame.from_dict(pickle.load(f)))

    df = pd.concat(dfs)
    df = df.drop_duplicates(subset = ['url'])
    df = df.reset_index(drop = True)

    df['is_home_page'] = df.apply(is_home_page, axis = 1)
    df['is_https'] = df.apply(is_https, axis = 1)
    print(df.describe())
    print(df.shape)
    return df


def tokenize(s):
    return str(s).lower().translate(translator).split()


def vectorize_topic_models(topic_tuples, num_of_topics):
    vector = [0 for _ in range(num_of_topics)]
    for i in topic_tuples:
        vector[i[0]] = i[1]
    return vector


def create_topic_models(df):
    docs  = df['response'].dropna().tolist()
    docs = [tokenize(i) for i in docs]
    common_dictionary = Dictionary(docs)
    common_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    lda = ldamodel.LdaModel(common_corpus, id2word=common_dictionary, num_topics=10)
    print(lda.print_topics())

    # for k, v in df.iterrows():
    #     if not pd.isna(v['response']):
    #         doc_tokenized = tokenize(v['response'])
    #         doc_tokenized = common_dictionary.doc2bow(doc_tokenized)
    #         a = lda[doc_tokenized]
    #         print(a)



class WebsiteModels():
    def __init__(self, max_vocab_size = 5000, min_n_gram = 1, max_n_gram = 1):
        self.max_vocab_size = max_vocab_size
        self.bow_vectorizer = CountVectorizer(ngram_range = (min_n_gram, max_n_gram),max_features = self.max_vocab_size, binary=True)

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

        cb1 = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
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






df = get_data2()
# create_topic_models(df)
# df = df.sample(n = 500)
#
# score_dict = dict()
#
# models = WebsiteModels(max_vocab_size = 5000)
# models.fit_baseline(df['response'].dropna().tolist())
# score_dict['lda_score_2']  = models.evaluate_lda_topic_models(df['response'], df['is_https'])
# score_dict['lda_score_4']  = models.evaluate_lda_topic_models(df['response'], df['is_https'])
# score_dict['lda_score_8']  = models.evaluate_lda_topic_models(df['response'], df['is_https'])
# score_dict['lda_score_16']  = models.evaluate_lda_topic_models(df['response'], df['is_https'])
# score_dict['lda_score_32']  = models.evaluate_lda_topic_models(df['response'], df['is_https'])
# score_dict['lda_score_64']  = models.evaluate_lda_topic_models(df['response'], df['is_https'])
#
# score_dict['dnn_score_8'] = models.evaluate_dnn_autoencoder(df['response'], df['is_https'], encoding_size = 8)
# score_dict['dnn_score_16'] = models.evaluate_dnn_autoencoder(df['response'], df['is_https'], encoding_size = 16)
# score_dict['dnn_score_32'] = models.evaluate_dnn_autoencoder(df['response'], df['is_https'], encoding_size = 32)
# score_dict['dnn_score_64'] = models.evaluate_dnn_autoencoder(df['response'], df['is_https'], encoding_size = 64)
# score_dict['dnn_score_128'] = models.evaluate_dnn_autoencoder(df['response'], df['is_https'], encoding_size = 128)
#
# score_dict['lr_score'] = models.evaluate_simple_lr(df['response'], df['is_https'])
#
#
# print('''simple logistic regression predicting scheme f1 score: {lr_score}
#         dnn autoencoder with size 8 encoding predicting scheme f1 score: {dnn_score_8}
#         dnn autoencoder with size 16 encoding predicting scheme f1 score: {dnn_score_16}
#         dnn autoencoder with size 32 encoding predicting scheme f1 score: {dnn_score_32}
#         dnn autoencoder with size 64 encoding predicting scheme f1 score: {dnn_score_64}
#         dnn autoencoder with size 128 encoding predicting scheme f1 score: {dnn_score_128}
#         lda topic model with 2 topics predicting scheme f1 score: {lda_score_2}
#         lda topic model with 4 topics predicting scheme f1 score: {lda_score_4}
#         lda topic model with 8 topics predicting scheme f1 score: {lda_score_8}
#         lda topic model with 16 topics predicting scheme f1 score: {lda_score_16}
#         lda topic model with 32 topics predicting scheme f1 score: {lda_score_32}
#         lda topic model with 64 topics predicting scheme f1 score: {lda_score_64}
#         '''.format(**score_dict))
#
# # dnn_autoencoder = DNN_Encoder(max_vocab_size = 5000)
# # dnn_autoencoder.fit(df['response'].dropna().tolist())
