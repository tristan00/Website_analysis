import pandas as pd
import sqlite3
import pickle
from common import get_initial_website_list, dir_loc, url_record_file_name, db_name, url_record_backup_file_name, clean_text, index1_db_name, get_distance
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from scipy.spatial.distance import cosine
import numpy as np
import uuid

distance_col1 = 'ranking_column_1'
distance_col2 = 'ranking_column_2'
distance_col3 = 'ranking_column_3'
distance_col4 = 'ranking_column_4'


class SimpleSearchEngine():

    def __init__(self, max_features = 5000, max_df = .2, max_records_to_fit = 5000):
        self.bow_vectorizer1 = CountVectorizer(max_features = max_features, binary=True, max_df=max_df)
        self.max_records_to_fit = max_records_to_fit
        self.decomp = PCA(n_components=100)


    def get_results(self, query):
        query_clean = clean_text(query)
        query_vec = self.bow_vectorizer1.transform([query_clean]).toarray()[0]

        a = self.kw_df.iloc[0][self.bow_vectorizer1.vocabulary_].values

        self.kw_df[distance_col3] = self.kw_df.apply(lambda x: get_distance(x[self.bow_vectorizer1.vocabulary_].values, query_vec), axis = 1)
        self.kw_df[distance_col3] = QuantileTransformer().fit_transform(self.kw_df[distance_col3].values.reshape((-1, 1)))
        self.kw_df[distance_col3] = 1 - self.kw_df[distance_col3]

        self.kw_df[distance_col4] = self.kw_df[distance_col3] * self.kw_df[distance_col2]
        self.kw_df = self.kw_df.sort_values(distance_col4, ascending=False)
        rec = self.kw_df.sort_values(distance_col4, ascending=False).iloc[0]
        return rec.name


    def fit_engine(self):
        self.generate_keyword_index()
        self.generate_ranking_index()


    def generate_keyword_index(self):
        #url_record_file_name
        with open('{dir}/{url_record_file_name}'.format(dir=dir_loc, url_record_file_name=url_record_backup_file_name), 'rb') as f:
            links = pickle.load(f)

        for i in links:
            links[i]['num_of_links_to'] = len(links[i]['urls_linking_to_page'])

        df = pd.DataFrame.from_dict(list(links.values()))
        df = df[df['scraped'] == 1]
        df = df.sort_values('num_of_links_to', ascending=False)
        df_top = df[:self.max_records_to_fit]

        with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name)) as conn_disk:
            query = '''Select * from websites where url = ?'''
            for k, v in df_top.iterrows():
                res = conn_disk.execute(query, (v['url'],))
                res_list = list(res)
                if res_list:
                    df_top.loc[df['url'] == v['url'], 'response'] = res_list[0][1]

            df_top['response'] = df_top['response'].apply(clean_text)
            self.bow_vectorizer1.fit(df_top['response'])
            self.kw_df = pd.DataFrame(columns = self.bow_vectorizer1.vocabulary_ )

            for k, v in df_top.iterrows():
                res = conn_disk.execute(query, (v['url'],))
                res_list = list(res)
                if res_list:
                    text = clean_text(res_list[0][1])
                    d1 = self.bow_vectorizer1.transform([text]).toarray()
                    new_series = pd.Series(d1[0].astype(np.int8), index = self.bow_vectorizer1.vocabulary_)
                    self.kw_df.loc[v['url']] = new_series
                    print(self.kw_df.shape)


    def generate_ranking_index(self):
        with open('{dir}/{url_record_file_name}'.format(dir=dir_loc, url_record_file_name=url_record_file_name), 'rb') as f:
            links = pickle.load(f)

        links = {i: links[i] for i in links if links[i]['scraped'] == 1}

        for i in links:
            links[i]['score1'] = 1
            links[i]['score2'] = 0
            links[i]['score3'] = 0
            links[i]['score4'] = 0
            links[i][distance_col2] = 0

        valid_urls = set(links.keys())

        for i in links:
            for j in links[i]['page_links']:
                if j in valid_urls:
                    links[j]['score2'] += links[i]['score1']

        for i in links:
            for j in links[i]['page_links']:
                if j in valid_urls:
                    links[j]['score3'] += links[i]['score2']

        for i in links:
            for j in links[i]['page_links']:
                if j in valid_urls:
                    links[j]['score4'] += links[i]['score3']


        links_list = list(links.values())
        self.ranking_df = pd.DataFrame.from_dict(links_list)
        scaler = QuantileTransformer()
        self.ranking_df[distance_col2] = scaler.fit_transform(self.ranking_df['score4'].values.reshape((-1, 1)))
        self.ranking_df = self.ranking_df[['url', distance_col2]]
        self.ranking_df = self.ranking_df.set_index(['url'])
        self.kw_df = self.kw_df.join(self.ranking_df)

        # self.ranking_df.to_csv('res.csv')



if __name__ == '__main__':
    s = SimpleSearchEngine()
    s.fit_engine()

    tests = ['test query testing',
             'world of warcraft',
             'data science']

    for i in tests:
        print(i, s.get_results(i))




