
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

# dir_loc = '/home/td/Documents/web_data'
# dir_loc = '/media/td/Samsung_T5/web_data'
dir_loc = 'E:/web_data'
url_record_file_name = 'links.pkl'
url_record_backup_file_name = 'links_backup.pkl'
db_name = 'website.db'
db_name2 =  'website2.db'
index1_db_name = 'index1.db'
initial_website_file_name = 'web_crawler_input.txt'

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))


class DataManager():
    def __init__(self, minimum_query_words = 2, maximum_query_words = 5):
        query = '''Select url from websites'''
        with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name2)) as conn_disk:
            self.urls = list(set([i[0] for i in conn_disk.execute(query)]))
        self.implemented_query_generation_options = ['random_title_extract', 'random_metas_extract']
        self.minimum_query_words = minimum_query_words
        self.maximum_query_words = maximum_query_words



    def extract_possible_query(self, url, title, meta):
        option = random.choice(self.implemented_query_generation_options)
        if option == 'random_title_extract':
            tokenized_title = tokenize(title)
            if len(tokenized_title) <= self.minimum_query_words:
                return ' '.join(tokenized_title)
            else:
                num_of_words = random.randint(self.minimum_query_words, min(len(tokenized_title), self.maximum_query_words))
                return ' '.join(random.sample(tokenized_title, k =  num_of_words))

        if option == 'random_metas_extract':
            tokenized_meta = tokenize(meta)
            if len(tokenized_meta) <= self.minimum_query_words:
                return ' '.join(tokenized_meta)
            else:
                num_of_words = random.randint(self.minimum_query_words, min(len(tokenized_meta), self.maximum_query_words))
                return ' '.join(random.sample(tokenized_meta, k =  num_of_words))

    def sample_queries(self, n):
        urls = random.sample(self.urls, k = n)
        results = []

        with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name2)) as conn_disk:
            for url in tqdm.tqdm(urls):
                (url, meta, title,) = [i for i in conn_disk.execute(
                    "select url, meta, title from websites where url = '{}' order by request_timestamp DESC".format(
                        url))][0]
                results.append(self.extract_possible_query(url, meta, title))
        return results

    def sample_col(self, n, col):
        urls = random.sample(self.urls, k = n)
        results = []

        with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name2)) as conn_disk:
            for url in tqdm.tqdm(urls):
                res = [i for i in conn_disk.execute(
                    "select {0} from websites where url = '{1}' order by request_timestamp DESC".format(col, url))][0]
                results.append(res[0])
        return results

    def get_titles(self, n):
        urls = random.sample(self.urls)
        results = []

        with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name2)) as conn_disk:
            for url in urls:
                url, meta, title = [i for i in conn_disk.execute(
                    "select url, meta, title from websites where url = '{}' order by request_timestamp DESC".format(
                        url))][0]
                results.append(self.extract_possible_query(url, meta, title))
        return results

    def get_positive(self, conn, url):
        _, meta, title = [i for i in conn.execute("select url, meta, title from websites where url = '{}' order by request_timestamp DESC".format(url))][0]
        query = self.extract_possible_query(url, meta, title)
        return query, url, meta, title

    def get_negative(self, conn, url1, url2 ):
        _, meta1, title1 = [i for i in conn.execute(
            "select url, meta, title from websites where url = '{}' order by request_timestamp DESC".format(url1))][0]
        _, meta2, title2 = [i for i in conn.execute(
            "select url, meta, title from websites where url = '{}' order by request_timestamp DESC".format(url2))][0]
        query = self.extract_possible_query(url2, meta2, title2)
        return query, url1, meta1, title1


    def get_sample(self, n):
        urls_to_consider = random.sample(self.urls, k = n)
        positives = urls_to_consider[:n//2]
        negatives = urls_to_consider[n//2:]

        urls_to_sample_negatives_from = [i for i in self.urls if i not in positives and i not in negatives]
        negative_pages = random.sample(urls_to_sample_negatives_from, k = len(negatives))

        data = []

        with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name2)) as conn_disk:
            for url in positives:
                query, _, meta, title = self.get_positive(conn_disk, url)
                data.append({'query':query, 'meta': meta, 'title':title, 'url':url, 'target':1})

            for url1, url2 in zip(negatives, negative_pages):
                query, url, meta, title = self.get_negative(conn_disk, url1, url2)
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
    with open('{dir_loc}/{initial_website_file_name}'.format(dir_loc=dir_loc, initial_website_file_name=initial_website_file_name), 'r') as f:
        urls = f.readlines()
        urls = [i.strip() for i in urls]
        urls = [i for i in urls]
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
    print(DataManager().sample_queries(10))

