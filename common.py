
import pickle
import pandas as pd
import sqlite3
import numpy as np
import string
from urllib import parse
import math
from bs4 import BeautifulSoup
import random
import tqdm
import glob

# dir_loc = '/home/td/Documents/web_data'
dir_loc = '/media/td/Samsung_T5/web_data'
# dir_loc = 'E:/web_data'
url_record_file_name = 'links.pkl'
url_record_backup_file_name = 'links_backup.pkl'
db_name = 'website.db'
db_name2 =  'website2.db'
index1_db_name = 'index1.db'
initial_website_file_name = 'web_crawler_input.txt'

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))


class DataManager():
    def __init__(self, minimum_query_words = 2, maximum_query_words = 5):
        query = '''Select url, title, meta, keywords, description from websites'''
        with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name)) as conn_disk:
            self.urls = set()
            self.titles = set()
            self.metas = set()
            self.keywords = set()
            self.descriptions = set()

            res = conn_disk.execute(query)
            for i in res:
                self.urls.add(i[0])
                self.titles.add(i[1])
                self.metas.add(i[2])
                self.keywords.add(i[3])
                self.descriptions.add(i[4])
            self.queries = set()
        self.implemented_query_generation_options = ['random_title_extract', 'random_metas_extract']
        self.minimum_query_words = minimum_query_words
        self.maximum_query_words = maximum_query_words
        self.search_query_dataset = dict()
        self.load_query_datasets()


    def load_query_datasets(self):
        try:
            with open('{dir_loc}/search_query_dataset_dict.pkl'.format(dir_loc=dir_loc), 'rb') as f:
                 self.search_query_dataset = pickle.load(f)
            with open('{dir_loc}/queries_list.pkl'.format(dir_loc=dir_loc), 'rb') as f:
                self.queries = pickle.load(f)

        except:
            files = glob.glob('{dir_loc}/AOL-user-ct-collection/*'.format(dir_loc=dir_loc))

            dfs = []
            for i in files:
                with open(i, 'r') as f:
                    lines = f.readlines()
                    for l in lines:
                        split_line = l.split('\t')
                        url = str(split_line[-1]).strip()
                        query = str(split_line[1]).strip()
                        self.search_query_dataset.setdefault(url, list())
                        self.search_query_dataset[url].append(query)
                        self.queries.add(query)
            self.queries = list(self.queries)
            with open('{dir_loc}/search_query_dataset_dict.pkl'.format(dir_loc=dir_loc), 'wb') as f:
                pickle.dump(self.search_query_dataset, f)
            with open('{dir_loc}/queries_list.pkl'.format(dir_loc=dir_loc), 'wb') as f:
                pickle.dump(self.queries, f)



    def extract_query_using_aol_dataset(self, url):
        query = random.choice(self.search_query_dataset[url])
        return query


    def sample_queries(self, n):
        return random.sample(self.queries, k = min(n, len(self.queries)))


    def sample_titles(self, n):
        return random.sample(self.titles, k = min(n, len(self.titles)))


    def sample_metas(self, n):
        return random.sample(self.metas, k = min(n, len(self.metas)))


    def sample_keywords(self, n):
        return random.sample(self.titles, k = min(n, len(self.keywords)))


    def sample_descriptions(self, n):
        return random.sample(self.metas, k = min(n, len(self.descriptions)))


    def get_positive_data_point(self, url, conn):
        query = self.extract_query_using_aol_dataset(url)
        _, meta, title = [i for i in conn.execute(
            "select url, meta, title from websites where url = '{}' order by request_timestamp DESC".format(url))][0]
        return query, url, meta, title

    def get_negative_data_point(self, conn, url1, url2 ):
        _, meta1, title1 = [i for i in conn.execute(
            "select url, meta, title from websites where url = '{}' order by request_timestamp DESC".format(url1))][0]
        _, meta2, title2 = [i for i in conn.execute(
            "select url, meta, title from websites where url = '{}' order by request_timestamp DESC".format(url2))][0]
        query = self.extract_query_using_aol_dataset(url2)
        return query, url1, meta1, title1


    def get_labeled_data_sample(self, n):
        labeled_urls = set(self.search_query_dataset.keys())
        urls = set(self.urls) & labeled_urls

        adjusted_n = min(n, len(urls))
        urls_to_consider = random.sample(urls, k = adjusted_n)
        positives = urls_to_consider[:adjusted_n//2]
        negatives = urls_to_consider[adjusted_n//2:]
        negative_pages = random.sample(urls, k = len(negatives))

        data = []

        with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name)) as conn_disk:
            for url in positives:
                query, _, meta, title = self.get_positive_data_point(url, conn_disk)
                data.append({'query':query, 'meta': meta, 'title':title, 'url':url, 'target':1})

            for url1, url2 in zip(negatives, negative_pages):
                query, url, meta, title = self.get_negative_data_point(conn_disk, url1, url2)
                data.append({'query':query, 'meta': meta, 'title':title, 'url':url, 'target':0})

        return pd.DataFrame(data)


def get_distance(v1, v2):
    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist


def is_absolute(url):
    return bool(parse.urlparse(url).netloc)


def tokenize(s):
    return str(s).lower().translate(translator).split()


def clean_text(s):
    return ' '.join(tokenize(s))


def is_home_page(r):
    if len(r['path']) < 2 and len(r['params']) < 2 and len(r['query']) < 2 and len(r['fragment']) < 2:
        return 1
    return 0


def is_https(r):
    if r['scheme'] == 'https':
        return 1
    return 0


def is_link_external(test_url, net_loc):
    '''

    :param search_link:
    :param test_link:
    :return:
    '''

    allowed_overlaps = ['www', 'net', 'com', 'org', 'gov', 'en']

    split_netloc1 = [i for i in net_loc.split('.') if i not in allowed_overlaps]
    test_url_netloc = parse.urlparse(test_url).netloc


    if test_url_netloc:
        split_test_url_netloc = [i for i in test_url_netloc.split('.') if i not in allowed_overlaps]
        if not set(split_test_url_netloc) & set(split_netloc1):
            return 1
    return 0

def extract_title(soup):
    title_tag = soup.find('title')
    if title_tag:
        return clean_text(title_tag.get_text())
    return ''

def extract_keywords(soup):
    keywords_tag = soup.find('keywords')
    if keywords_tag:
        return clean_text(keywords_tag.get_text())
    return ''

def extract_description(soup):
    description_tag = soup.find('description')
    if description_tag:
        return clean_text(description_tag.get_text())
    return ''


def extract_meta_information(soup):
    header = soup.find('head')
    texts = []

    if header:
        for tag in header.children:
            try:
                texts.append(tag.get('content', ''))
                texts.append(tag.get_text())

            except AttributeError:
                pass
    return clean_text(' '.join(texts))

def extract_page_text(soup):
    return soup.get_text()


def get_initial_website_list():
    try:
        with open('{dir_loc}/search_query_dataset_dict.pkl'.format(dir_loc=dir_loc), 'rb') as f:
            return list(pickle.load(f).keys())
    except:
        with open('{dir_loc}/{initial_website_file_name}'.format(dir_loc=dir_loc, initial_website_file_name=initial_website_file_name), 'r') as f:
            urls = f.readlines()
            urls = [i.strip() for i in urls]
            urls = [i for i in urls]
            random.shuffle(urls)
            return urls


def process_saved_data():
    with open('{dir}/{url_record_file_name}'.format(dir=dir_loc, url_record_file_name=url_record_file_name), 'rb') as f:
        links = pickle.load(f)

    for i in links:
        links[i]['num_of_links_to'] = len(links[i]['urls_linking_to_page'])

    df = pd.DataFrame.from_dict(list(links.values()))
    df = df[df['scraped'] == 1]

    with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name)) as conn_disk:
        query = '''Select * from websites where url = ?'''

        for k, v in df.iterrows():
            res = conn_disk.execute(query, (v['url'],))
            res_list = list(res)
            if res_list:
                res_html = res_list[0][1]
                soup = BeautifulSoup(res_html)
                meta_info = extract_meta_information(soup)
                page_text = extract_meta_information(soup)

                df.loc[df['url'] == v['url'], 'response'] = res_list[0][1]



def get_data():
    with open('{dir}/{url_record_file_name}'.format(dir=dir_loc, url_record_file_name=url_record_file_name), 'rb') as f:
        links = pickle.load(f)

    for i in links:
        links[i]['num_of_links_to'] = len(links[i]['urls_linking_to_page'])

    df = pd.DataFrame.from_dict(list(links.values()))
    df = df[df['scraped'] == 1]

    with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name)) as conn_disk:
        query = '''Select * from websites where url = ?'''

        for k, v in df.iterrows():
            res = conn_disk.execute(query, (v['url'],))
            res_list = list(res)
            if res_list:
                df.loc[df['url'] == v['url'], 'response'] = res_list[0][1]
    df = df.dropna(subset=['response'])
    df = df.drop_duplicates(subset = ['url'])
    df = df.reset_index(drop = True)

    df['is_home_page'] = df.apply(is_home_page, axis = 1)
    df['is_https'] = df.apply(is_https, axis = 1)
    print(df.shape)
    print(df.describe())
    return df


def get_adjacency_matrix():
    with open('{dir}/{url_record_file_name}'.format(dir=dir_loc, url_record_file_name=url_record_file_name), 'rb') as f:
        links = pickle.load(f)

    links = {i: links[i] for i in links if links[i]['scraped'] == 1}
    urls = list(links.keys())
    df = pd.DataFrame(data = np.zeros((len(urls), len(urls))),
                      columns = urls,
                      index = urls)

    for i in links:
        for j in links[i]['page_links']:
            df[j, i] = 1
    return df


if __name__ == '__main__':
    import time
    st = time.time()
    dm = DataManager()
    t1 = time.time()
    print('here')
    print(dm.sample_queries(10))
    t2 = time.time()
    print(t1-st, t2-t1)
