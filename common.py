
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
import time
from bs4.element import Comment
import functools
import operator


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
    def __init__(self,
                 min_html_word_len = 100,
                 min_text_word_len = 25,
                 min_meta_word_len = 5):

        self.min_char_dict = {'text': min_text_word_len,
                              'html':min_html_word_len,
                              'meta':min_meta_word_len}

        self.load_url_file_indices()
        self.query_dataset = pd.read_csv(f'{dir_loc}/external_data/aol_dataset_consolidated.csv', sep = '|', low_memory=False)

    ###########################################################################
    # Sampling methods:

    def get_dataset_of_meta_matching_page_text(self, max_dataset_size = None):
        print('generating dataset of matching meta and page texts')
        if max_dataset_size and len(self.urls) > max_dataset_size:
            sample_urls = self.sample_urls(max_dataset_size)
        else:
            sample_urls = self.urls

        files = set()
        output = list()

        print('identifying files')
        for url in tqdm.tqdm(sample_urls):
            timestamps = list(self.url_index[url].keys())
            if not timestamps:
                continue
            timestamp = random.choice(timestamps)
            files.add(self.url_index[url][timestamp])

        meta_records = dict()
        text_records = dict()

        print('reading data from data files')
        for f in tqdm.tqdm(files):
            meta_records.update(self.get_records_from_file('meta', set(sample_urls), f))
            text_records.update(self.get_records_from_file('text', set(sample_urls), f))

        print('producing output')
        for url in tqdm.tqdm(sample_urls):
            meta_record = meta_records.get(url)
            text_record = text_records.get(url)
            if meta_record and text_record:
                output.append({'url':url,
                               'meta':meta_record,
                               'text':text_record,
                               'meta_matches_text':1})
        return pd.DataFrame(output)

    def get_dataset_of_meta_not_matching_page_text(self, max_dataset_size = None):
        print('generating dataset of mismatching meta and page texts')

        if max_dataset_size and len(self.urls) > max_dataset_size:
            sample_urls_1 = self.sample_urls(max_dataset_size)
            sample_urls_2 = self.sample_urls(max_dataset_size)

        else:
            sample_urls_1 = copy.deepcopy(self.urls)
            sample_urls_2 = copy.deepcopy(self.urls)

        random.shuffle(sample_urls_1)
        random.shuffle(sample_urls_2)

        files = set()
        output = list()

        print('identifying files')
        for url in tqdm.tqdm(set(sample_urls_1) | set(sample_urls_2)):
            timestamps = list(self.url_index[url].keys())
            timestamp = random.choice(timestamps)
            files.add(self.url_index[url][timestamp])

        meta_records = dict()
        text_records = dict()

        print('reading data from data files')
        for f in tqdm.tqdm(files):
            meta_records.update(self.get_records_from_file('meta', set(sample_urls_1) | set(sample_urls_2), f))
            text_records.update(self.get_records_from_file('text', set(sample_urls_1) | set(sample_urls_2), f))

        print('producing output')
        for url1, url2 in zip(sample_urls_1, sample_urls_2):
            meta_record = meta_records.get(url1)
            text_record = text_records.get(url2)
            if meta_record and text_record:
                output.append({'url':url1,
                               'meta':meta_record,
                               'text':text_record,
                               'meta_matches_text':0})
        return pd.DataFrame(output)

    def get_query_dataset(self, max_dataset_size = None, balance = True, text_types = ('text', )):
        df_true = self.get_dataset_of_aol_search_query_matching_text_of_clicked_link(max_dataset_size=max_dataset_size//2, text_types = text_types)
        df_false = self.get_dataset_of_aol_search_query_not_matching_text_of_clicked_link(max_dataset_size=max_dataset_size//2, text_types = text_types)

        if balance:
            if df_true.shape[0] > df_false.shape[0]:
                df_true = df_true.sample(n = df_false.shape[0])
            if df_false.shape[0] > df_true.shape[0]:
                df_false = df_false.sample(n = df_true.shape[0])
        if df_true.shape[0] > max_dataset_size//2:
            df_true = df_true.sample(n = max_dataset_size//2)
        if df_false.shape[0] > max_dataset_size//2:
            df_false = df_false.sample(n = max_dataset_size//2)
        df = pd.concat([df_false, df_true])
        for text_type in text_types:
            df[text_type] = df[text_type].fillna('')
        return df


    def get_dataset_of_aol_search_query_matching_text_of_clicked_link(self, max_dataset_size = 10000, text_types = ('text', )):
        files = set()
        sample_urls = self.sample_urls_from_url_set(set(self.query_dataset['ClickURL']), max_dataset_size)
        sample_df = self.query_dataset[self.query_dataset['ClickURL'].isin(sample_urls)]

        for url in tqdm.tqdm(sample_urls):
            if url in self.url_index:
                timestamps = list(self.url_index[url].keys())
                if not timestamps:
                    continue
                timestamp = random.choice(timestamps)
                files.add(self.url_index[url][timestamp])

        records = dict()
        print('reading data from data files')
        for text_type in text_types:
            print(f'reading data from {text_type} files')
            records[text_type] = dict()
            for f in tqdm.tqdm(files):
                try:
                    records[text_type].update(self.get_records_from_file(text_type, set(sample_urls), f))
                except:
                    traceback.print_exc()

        output_dicts = []
        for k, v in sample_df.iterrows():
            output_dict = dict()
            matched_text_data = False
            for text_type in text_types:
                output_dict[text_type] = records[text_type].get(v['ClickURL'])
                if output_dict[text_type]:
                    matched_text_data = True
            if matched_text_data:
                output_dict['query'] = v['Query']
                output_dict['target'] = 1
                output_dicts.append(output_dict)
        if len(output_dicts) > max_dataset_size:
            output_dicts = random.sample(output_dicts, max_dataset_size)

        return pd.DataFrame.from_dict(output_dicts)

    def get_dataset_of_aol_search_query_not_matching_text_of_clicked_link(self, max_dataset_size = None, text_types = ('text', )):
        sample_urls = self.sample_urls_from_url_set(set(self.query_dataset['ClickURL']), max_dataset_size)
        sample_df = self.query_dataset[self.query_dataset['ClickURL'].isin(sample_urls)]
        mismatch_sample_urls = self.sample_urls(max_dataset_size)

        files = set()
        for url in tqdm.tqdm(mismatch_sample_urls):
            if url in self.url_index:
                timestamps = list(self.url_index[url].keys())
                if not timestamps:
                    continue
                timestamp = random.choice(timestamps)
                files.add(self.url_index[url][timestamp])

        records = dict()
        print('reading data from data files')
        for text_type in text_types:
            print(f'reading data from {text_type} files')
            records[text_type] = dict()
            for f in tqdm.tqdm(files):
                records[text_type].update(self.get_records_from_file(text_type, set(mismatch_sample_urls), f))

        output_dicts = []
        for k, v in sample_df.iterrows():
            incorrect_url = random.choice(list(records[random.choice(text_types)].keys()))

            output_dict = dict()
            matched_text_data = False
            for text_type in text_types:
                output_dict[text_type] = records[text_type].get(incorrect_url)
                if output_dict[text_type]:
                    matched_text_data = True
            if matched_text_data:
                output_dict['query'] = v['Query']
                output_dict['target'] = 0
                output_dicts.append(output_dict)

        if len(output_dicts) > max_dataset_size:
            output_dicts = random.sample(output_dicts, max_dataset_size)

        return pd.DataFrame.from_dict(output_dicts)

    def data_generator(self, r_types, n):
        files = list(self.file_name_dict.keys())
        random.shuffle(files)
        sample_urls = set()
        counter = 0

        for f in files:
            next_url_dict = dict()
            next_urls_set = set()
            for r_type in r_types:
                next_url_type_data = self.get_records_from_file(r_type, self.urls, f)
                next_url_dict[r_type] = next_url_type_data
                next_urls_set.update(set(next_url_type_data.keys()))

            for i in next_urls_set:
                if i in sample_urls:
                    continue
                next_output = {'url': i}
                for r_type in r_types:
                    next_output[r_type] = next_url_dict[r_type].get(i, '')
                yield next_output
                counter += 1
                sample_urls.add(i)

                if isinstance(n, int) and counter > n:
                    break
            if isinstance(n, int) and counter > n:
                break


    ###########################################################################
    # Helper methods:

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
                    if len(row_split) == 2 and row_split[0] == url and len(tokenize(row_split[1])) >= self.min_char_dict[r_type]:
                        return row_split[1]

        except FileNotFoundError:
            print(f'file {file_name} not found')

    def get_records_from_file(self, r_type, url_set, file_name):
        output_dict = dict()

        try:
            with open(f'{dir_loc}/all_{r_type}_chunks/{file_name}.txt', 'r') as f:
                for row in f:
                    row_split = row.split('|')
                    if len(row_split) == 2 and row_split[0] in url_set and len(tokenize(row_split[1])) >= self.min_char_dict[r_type]:
                        output_dict[row_split[0] ] = row_split[1]
            return output_dict
        except FileNotFoundError:
            print(f'file {file_name} not found')
            return dict()

    def sample_urls(self, n):
        sample_urls = set()
        files = list(self.file_name_dict.keys())
        random.shuffle(files)

        for f in files:
            sample_urls = sample_urls | self.file_name_dict[f]

            if len(sample_urls) > n:
                break
        if len(sample_urls) > n:
            return random.sample(list(sample_urls), n)
        else:
            return list(sample_urls)

    def sample_urls_from_url_set(self, urls, n):
        sample_urls = set()
        files = list(self.file_name_dict.keys())
        random.shuffle(files)

        for f in files:
            sample_urls = sample_urls | (self.file_name_dict[f] & urls)

            if len(sample_urls) > n:
                break
        if len(sample_urls) > n:
            return random.sample(list(sample_urls), n)
        else:
            return list(sample_urls)

    def load_url_file_indices(self):

        self.file_name_dict = dict()
        query = '''Select url, request_timestamp, file_name from websites where html_word_len > ? and text_word_len > ? and meta_word_len > ?'''
        with sqlite3.connect('{dir}/dbs/{db_name}'.format(dir=dir_loc, db_name=db_name)) as conn_disk:
            res = list(conn_disk.execute(query, ( self.min_char_dict['html'], self.min_char_dict['text'], self.min_char_dict['meta'])))
            self.url_index = dict()
            for i in res:
                url, request_timestamp, file_name = i
                self.url_index.setdefault(url, dict())
                self.url_index[url][request_timestamp] = file_name
                self.file_name_dict.setdefault(file_name, set())
                self.file_name_dict[file_name].add(url)
        self.urls = list(self.url_index)
        print(f'num of valid scraped websites in db: {len(self.urls)}')
        del res


def run_page_rank(n=None, num_of_iterations=1):
    # TODO:  automate stopping condition
    ranking_dict1 = dict()
    ranking_dict2 = dict()

    query = '''Select url, page_external_links from websites'''
    with sqlite3.connect('{dir}/dbs/{db_name}'.format(dir=dir_loc, db_name=db_name)) as conn_disk:
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
    else:
        return page_rank['url'].tolist()


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def get_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u". ".join(t.strip() for t in visible_texts)


def get_meta_info_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    metas = soup.find_all('meta')
    return '. '.join([meta.attrs['content'] for meta in metas if 'content' in meta.attrs])


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
    return clean_text(s).split()


def clean_text(s):
    return str(s).lower().replace('\n', ' ').translate(translator)


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
        df = pd.read_csv(f'{dir_loc}/external_data/aol_dataset_consolidated.csv', sep = '|')
        return set(df['ClickURL'])

        # with open('{dir_loc}/search_query_dataset_dict.pkl'.format(dir_loc=dir_loc), 'rb') as f:
        #     return list(pickle.load(f).keys())
    except:
        traceback.print_exc()
        load_aol_dataset()
        df = pd.read_csv(f'{dir_loc}/external_data/aol_dataset_consolidated.csv', sep = '|')
        return set(df['ClickURL'])
        # with open('{dir_loc}/{initial_website_file_name}'.format(dir_loc=dir_loc, initial_website_file_name=initial_website_file_name), 'r') as f:
        #     urls = f.readlines()
        #     urls = [i.strip() for i in urls]
        #     urls = [i for i in urls]
        #     random.shuffle(urls)
        #     return urls


def load_aol_dataset():
    files = glob.glob(f'{dir_loc}/external_data/AOL-user-ct-collection/*')

    dfs = []
    for f in files:
        start_time = time.time()
        print(f)
        df = pd.read_csv(f, sep = '\t')
        print(df.shape, time.time() - start_time)
        df = df.dropna(subset = ['ClickURL'])
        print(df.shape, time.time() - start_time)
        df = df[['Query', 'ClickURL']]
        print(df.shape, time.time() - start_time)
        df = df.drop_duplicates()
        print(df.shape, time.time() - start_time)
        dfs.append(df)
        print()

    df = pd.concat(dfs)
    print(df.shape)
    df = df.drop_duplicates(subset = ['ClickURL', 'Query'])
    print(df.shape)
    df.to_csv(f'{dir_loc}/aol_dataset_consolidated.csv', index = False, sep = '|')

if __name__ == '__main__':
    dm = DataManager()
    df = dm.get_query_dataset(max_dataset_size = 1000, balance = True, text_types = ('text', 'meta', 'html'))
    print(df.columns.tolist())
    print(df['target'].mean())

