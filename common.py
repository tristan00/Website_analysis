
import pickle
import pandas as pd
import sqlite3
import numpy as np
import string
from urllib import parse

# dir_loc = '/home/td/Documents/web_data'
dir_loc = '/media/td/Samsung_T5/web_data'
url_record_file_name = 'links.pkl'
url_record_backup_file_name = 'links_backup.pkl'
db_name = 'website.db'
index1_db_name = 'index1.db'
initial_website_file_name = 'web_crawler_input.txt'

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

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


def is_link_external(test_link, net_loc):
    '''

    :param search_link:
    :param test_link:
    :return:
    '''
    if is_absolute(test_link):
        return 1
    return 0


def get_initial_website_list():
    with open('{dir_loc}/{initial_website_file_name}'.format(dir_loc=dir_loc, initial_website_file_name=initial_website_file_name), 'r') as f:
        urls = f.readlines()
        urls = [i.strip() for i in urls]
        urls = [i for i in urls]
        return urls


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
    df = get_data()
    df = df.sort_values('num_of_links_to')
    df.shape

