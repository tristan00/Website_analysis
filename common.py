
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
import traceback
import copy
import ast

# dir_loc = '/home/td/Documents/web_data'
dir_loc = '/media/td/Samsung_T5/web_data'
# dir_loc = 'E:/web_data'
num_of_file_chunks = 1024
# dir_loc = '/home/td/Documents/data/website_data'
url_record_file_name = 'links.pkl'
url_record_backup_file_name = 'links_backup.pkl'
db_name = 'website.db'
db_name2 =  'website2.db'
index1_db_name = 'index1.db'
initial_website_file_name = 'web_crawler_input.txt'
sep_char = '|'
max_websites_per_file = 10000
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))


class DataManager():
    def __init__(self, get_top_n = None, n = 10000, pagerank_iterations = 10):
        query = '''Select url, request_timestamp, file_name from websites'''
        with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name)) as conn_disk:
            res = list(conn_disk.execute(query))

            self.url_index = dict()
            for i in res:
                url, request_timestamp, file_name = i
                self.url_index.setdefault(url, dict())
                self.url_index[url][request_timestamp] = file_name
        self.urls = list(self.url_index)
        print(f'num of urls in db: {len(self.urls)}')
        del res

        if get_top_n:
            top_urls = run_page_rank(num_of_iterations=pagerank_iterations)

            for i in reversed(top_urls):
                if self.url_index.get(i):
                    del self.url_index[i]
                    self.urls.remove(i)
                if len(self.urls) <= n:
                    break


    def get_dataset_of_meta_matching_page_text(self, max_dataset_size = None):
        if max_dataset_size and len(self.urls) > max_dataset_size:
            sample_urls = random.sample(self.urls, max_dataset_size)
        else:
            sample_urls = self.urls

        output = list()
        for url in tqdm.tqdm(sample_urls):
            timestamps = list(self.url_index[url].keys())
            if not timestamps:
                continue
            timestamp = random.choice(timestamps)
            meta_record = self.get_record('meta', url, self.url_index[url][timestamp])
            text_record = self.get_record('text', url, self.url_index[url][timestamp])
            if meta_record and text_record:
                output.append({'url':url,
                               'request_timestamp':timestamp,
                               'file_name':self.url_index[url][timestamp],
                               'meta':meta_record,
                               'text':text_record,
                               'meta_matches_text':1})
        return pd.DataFrame(output)

    def get_dataset_of_meta_not_matching_page_text(self, max_dataset_size = None):
        if max_dataset_size and len(self.urls) > max_dataset_size:
            sample_urls_1 = random.sample(self.urls, max_dataset_size)
            sample_urls_2 = random.sample(self.urls, max_dataset_size)

        else:
            sample_urls_1 = copy.deepcopy(self.urls)
            sample_urls_2 = copy.deepcopy(self.urls)

        random.shuffle(sample_urls_1)
        random.shuffle(sample_urls_2)

        output = list()
        for url1, url2 in tqdm.tqdm(zip(sample_urls_1, sample_urls_2)):
            timestamps1 = list(self.url_index[url1].keys())
            timestamps2 = list(self.url_index[url2].keys())

            if not timestamps1 or not timestamps2:
                continue

            timestamp1 = random.choice(timestamps1)
            timestamp2 = random.choice(timestamps2)

            meta_record = self.get_record('meta', url1, self.url_index[url1][timestamp1])
            text_record = self.get_record('text', url2, self.url_index[url2][timestamp2])
            if meta_record and text_record:
                output.append({'url':url1,
                               'request_timestamp':timestamp1,
                               'file_name':self.url_index[url1][timestamp1],
                               'meta':meta_record,
                               'text':text_record,
                               'meta_matches_text':0})
        return pd.DataFrame(output)

    def get_record(self, r_type, url, file_name):
        '''
        :param r_type: can be html, text, or meta
        '''
        try:
            with open(f'{dir_loc}/all_{r_type}_chunks/{file_name}.txt', 'r') as f:
                # print(f'file {file_name} found')
                # lines = f.readlines()
                for row in f:
                    row_split = row.split('|')
                    if len(row_split) == 2 and row_split[0] == url:
                        return row_split[1]
        except FileNotFoundError:
            print(f'file {file_name} not found')


def run_page_rank(n=None, num_of_iterations=1):
    # TODO:  automate stopping condition
    ranking_dict1 = dict()
    ranking_dict2 = dict()

    query = '''Select url, page_external_links from websites'''
    with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name)) as conn_disk:
        cursor = conn_disk.cursor()
        res = cursor.execute(query)
        count = 0
        for res_temp in tqdm.tqdm(res):
            count += 1
            ranking_dict1[res_temp[0]] = ast.literal_eval(res_temp[1])

    default_iteration_dict = {'iteration{0}'.format(i): 0 for i in range(num_of_iterations + 1)}
    default_iteration_dict['iteration0'] = 1

    for iteration in range(num_of_iterations):
        print('page rank iteration: {}'.format(iteration))

        for i in tqdm.tqdm(ranking_dict1):
            num_of_links = len(ranking_dict1[i]) + 1
            for j in ranking_dict1[i] + [i]:
                if iteration == 0:
                    ranking_dict_2_default = copy.deepcopy(default_iteration_dict)
                    ranking_dict_1_default = copy.deepcopy(default_iteration_dict)
                    ranking_dict_2_default['url'] = j
                    ranking_dict_1_default['url'] = i
                    ranking_dict2.setdefault(j, ranking_dict_2_default)
                    ranking_dict2.setdefault(i, ranking_dict_1_default)

                ranking_dict2[j]['iteration{}'.format(iteration + 1)] += (
                        ranking_dict2[i]['iteration{}'.format(iteration)] / num_of_links)

    page_rank = pd.DataFrame.from_dict(list(ranking_dict2.values()))

    total = page_rank['iteration{}'.format(num_of_iterations)].sum()
    page_rank['page_rank'] = page_rank['iteration{}'.format(num_of_iterations)] / total
    page_rank = page_rank.sort_values('page_rank', ascending=False)
    page_rank = page_rank[['url', 'page_rank']]
    print(page_rank['url'][:20].tolist())
    if n:
        return page_rank['url'][:n].tolist()


def clean_url(url):
    return url.replace('/', '')


def get_directory_hash(url):
    value = 0
    c_url = clean_url(url)
    for i in c_url:
        value *= max(ord(i), 1)
        value += max(ord(i), 1)
    return str(value % num_of_file_chunks).zfill(len(str(num_of_file_chunks + 1)))


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


def get_data(n):

    df = pd.DataFrame(columns=['url', 'page_text, path, params, query, fragment', 'scheme'])
    with sqlite3.connect('{dir}/{db_name}'.format(dir=dir_loc, db_name=db_name)) as conn_disk:
        query = '''Select url, page_text, path, params, query, fragment, scheme from websites limit {}'''.format(n)

        res = conn_disk.execute(query)
        for i in tqdm.tqdm(res):
            df = df.append(pd.Series({'url':i[0], 'page_text':i[1], 'path':i[2], 'params':i[3], 'query':i[4], 'fragment':i[5], 'scheme':i[6]}), ignore_index=True)
    df = df.dropna(subset=['page_text'])
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
