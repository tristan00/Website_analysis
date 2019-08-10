# non-common crawl version
import pandas as pd
import requests
import time
import os
from urllib import parse
import random
from bs4 import BeautifulSoup
import copy
import pickle
import uuid
import traceback
import sqlite3
import numpy as np
import tqdm
import ast
from common import dir_loc, db_name, is_link_external, is_absolute, extract_meta_information, extract_page_text, extract_title, get_initial_website_list, extract_keywords, extract_description
import asyncio
import aiohttp
import multiprocessing

web_timeout = 5
ignore_domains = []
max_page_size = 10000000

def request_html(url):
    s = requests.Session()
    s.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',}
    s.get(url, allow_redirects=False)


def generate_link_dict(url):
    parsed_url = parse.urlparse(url)
    record = dict()
    record['url'] = url
    record['scheme'] = parsed_url.scheme
    record['netloc'] = parsed_url.netloc
    record['path'] = parsed_url.path
    record['params'] = parsed_url.params
    record['query'] = parsed_url.query
    record['fragment'] = parsed_url.fragment
    record['scraped'] = 0
    record['timestamp_last_scraped'] = 0
    return record


def process_html(r_text, r_time, url):
    if r_text and len(r_text) <= max_page_size:
        record = generate_link_dict(url)

        # print('received  response from: {0} in {1} seconds'.format(url, r_time))

        scrape_successful = True
        soup = BeautifulSoup(r_text, 'lxml')
        new_links = [i['href'] for i in soup.find_all('a', href=True)]
        if new_links:
            new_abs_links = [i for i in new_links if is_link_external(i, record['netloc'])]
            new_rel_links1 = [i for i in new_links if not is_link_external(i, record['netloc']) and is_absolute(i)]
            new_rel_links2 = [i for i in new_links if not is_link_external(i, record['netloc']) and not is_absolute(i)]

            new_rel_links_joined = [parse.urljoin(url, i) for i in new_rel_links2]

            combined_links = new_abs_links + new_rel_links_joined + new_rel_links1
            record['page_external_links'] = str(new_abs_links)
            record['page_internal_links'] = str(new_rel_links_joined)
            record['page_links'] = str(combined_links)
            record['request_time'] = r_time
            record['request_timestamp'] = time.time()
            record['html'] = r_text
            record['meta'] = extract_meta_information(soup)
            record['page_text'] = extract_page_text(soup)
            record['title'] = extract_title(soup)
            record['keywords'] = extract_keywords(soup)
            record['description'] = extract_description(soup)

            record_df = pd.DataFrame.from_dict([record])
            record_df = record_df.set_index('url')

            with sqlite3.connect('{dir}/{db_name}'.format(dir = dir_loc, db_name=db_name)) as conn_disk:
                record_df.to_sql('websites', conn_disk, if_exists='append', index  = True)


def scrape_url(url):
    try:
        s = requests.Session()
        s.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',}
        start_time = time.time()

        r = s.get(url, timeout = (5, 5), allow_redirects=True)
        t2 = time.time()
        if r.status_code == 200:
            process_html(r.text, t2- start_time, url)
            # print('processed {}'.format(url))
    except requests.exceptions.MissingSchema:
        pass
        # print('invalid url: {0}'.format(url))
    except requests.exceptions.InvalidSchema:
        pass
        # print('invalid url: {0}'.format(url))
    except requests.exceptions.ConnectionError:
        pass
        # print('ConnectionError: {0}'.format(url))
    except TimeoutError:
        pass
        # print('TimeoutError: {0}'.format(url))
    except Exception:
        pass
        # print(traceback.format_exc())


def process_urls(q):
    while True:
        url = q.get()

        if url:
            scrape_url(url)
        else:
            break


class Crawler():

    def __init__(self, domain_time_delay = 3600, url_time_delay = 2419200, max_website_len = 500000, verbose = False, num_of_processes = 30):
        self.num_of_processes = num_of_processes
        self.max_website_len = max_website_len
        self.verbose = verbose
        self.start_time = time.time()
        self.crawler_id = str(uuid.uuid4())
        self.domain_time_delay_record_keeper = dict()
        self.url_time_delay_record_keeper = dict()
        self.request_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',}

        self.domains = dict()
        self.visited_links = set()
        self.domain_time_delay = domain_time_delay # needed to not tax any domain too much
        self.url_time_delay =url_time_delay
        self.data_folder_loc = dir_loc
        self.load_past_data()

        os.system('mkdir {0}'.format(self.data_folder_loc))
        self.get_new_session()
        self.make_queue_and_pool()


    def make_queue_and_pool(self):
        self.q = multiprocessing.Queue()
        self.pool = [multiprocessing.Process(target=process_urls, args=(self.q, )) for _ in range(self.num_of_processes)]
        [p.start() for p in self.pool]


    def refresh_pool_and_queue(self):
        [self.q.put(None) for _ in range(self.num_of_processes)]
        [p.join() for p in self.pool]
        del self.pool, self.q
        self.make_queue_and_pool()


    def print(self, s, force_verbose = False):
        if self.verbose or force_verbose:
            print(s)

    def get_n_random_urls(self, n):
        url_list = []
        self.visited_links = set()
        query = '''Select url, page_links from websites'''
        with sqlite3.connect('{dir}/{db_name}'.format(dir=self.data_folder_loc, db_name=db_name)) as conn_disk:
            cursor = conn_disk.cursor()
            res = cursor.execute(query)
            for res_temp in tqdm.tqdm(res):
                url_list.extend(ast.literal_eval(res_temp[1]))
                self.visited_links.add(res_temp[0])
        url_list = list(set(url_list))
        print(len(url_list), len(self.visited_links))
        if len(url_list) > n:
            return random.sample(url_list, n)
        return url_list


    def run_page_rank(self, n, num_of_iterations = 1):
        #TODO:  automate stopping condition
        ranking_dict1 = dict()
        ranking_dict2 = dict()

        query = '''Select url, page_external_links from websites'''
        with sqlite3.connect('{dir}/{db_name}'.format(dir=self.data_folder_loc, db_name=db_name)) as conn_disk:
            cursor = conn_disk.cursor()
            res = cursor.execute(query)
            count = 0
            for res_temp in  tqdm.tqdm(res):
                # res_temp = res.fetchone()
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

                    ranking_dict2[j]['iteration{}'.format(iteration+1)] += (ranking_dict2[i]['iteration{}'.format(iteration)]/num_of_links)

        self.page_rank = pd.DataFrame.from_dict(list(ranking_dict2.values()))

        total = self.page_rank['iteration{}'.format(num_of_iterations)].sum()
        self.page_rank['page_rank'] = self.page_rank['iteration{}'.format(num_of_iterations)] / total
        self.page_rank = self.page_rank.sort_values('page_rank', ascending=False)
        self.page_rank = self.page_rank[['url', 'page_rank']]
        print(self.page_rank['url'][:20].tolist())
        print(self.page_rank.shape, len(ranking_dict1))
        return self.page_rank['url'][:n].tolist()


    def load_past_data(self):
        try:

            with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='domain_time_delay_record_keeper'), 'rb') as f:
                self.domain_time_delay_record_keeper = pickle.load(f)
            with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='url_time_delay_record_keeper'), 'rb') as f:
                self.url_time_delay_record_keeper = pickle.load(f)
            with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='visited_links'), 'rb') as f:
                self.visited_links = pickle.load(f)

        except:
            traceback.print_exc()

    def save_data(self):
        # print('starting to save data')
        with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='domain_time_delay_record_keeper'), 'wb') as f:
            pickle.dump(self.domain_time_delay_record_keeper, f)
        with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='url_time_delay_record_keeper'), 'wb') as f:
            pickle.dump(self.url_time_delay_record_keeper, f)
        with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='visited_links'), 'wb') as f:
            pickle.dump(self.visited_links, f)
        # print('data saved')


    def get_new_session(self):
        self.s = requests.Session()
        self.s.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',}


    def generate_link_dict(self, url):
        parsed_url = parse.urlparse(url)
        record = dict()
        record['url'] = url
        record['scheme'] = parsed_url.scheme
        record['netloc'] = parsed_url.netloc
        record['path'] = parsed_url.path
        record['params'] = parsed_url.params
        record['query'] = parsed_url.query
        record['fragment'] = parsed_url.fragment
        record['scraped'] = 0
        record['timestamp_last_scraped'] = 0
        return record


    def scrape_url(self, next_link):
        scrape_successful = False
        self.domain_time_delay_record_keeper.setdefault(next_link['netloc'], 0)
        self.url_time_delay_record_keeper.setdefault(next_link['url'], 0)

        if next_link['url'] and time.time() - self.domain_time_delay_record_keeper[next_link['netloc']] > self.domain_time_delay and time.time() - self.url_time_delay_record_keeper[next_link['url']] > self.url_time_delay:
            try:
                self.domain_time_delay_record_keeper[next_link['netloc']] = time.time()
                self.url_time_delay_record_keeper[next_link['url']] = time.time()
                self.q.put(next_link['url'])
                scrape_successful = True

            except requests.exceptions.MissingSchema:
                self.print('invalid url: {0}'.format(next_link['url']))
            except requests.exceptions.InvalidSchema:
                self.print('invalid url: {0}'.format(next_link['url']))
            except requests.exceptions.ConnectionError:
                self.print('ConnectionError: {0}'.format(next_link['url']))
            except TimeoutError:
                self.print('TimeoutError: {0}'.format(next_link['url']))
            except Exception:
                self.print(traceback.format_exc(), force_verbose=False)
        # else:
        #     self.print('skipping url {}'.format(next_link['url']), force_verbose=True)
        return scrape_successful

    def scrape_list(self, website_list):
        for c, i in enumerate(website_list):
            rec = self.generate_link_dict(i)
            self.scrape_url(rec)

        self.save_data()
        self.refresh_pool_and_queue()

    def crawl(self, batch_size = 1000, prob_of_random_url_choice = .1, randomize_page_rank_polling = True, num_of_batches = 10, num_page_rank_iterations = 5):
        for batch in range(num_of_batches):

            if random.random() > prob_of_random_url_choice:
                self.print('', force_verbose=True)
                self.print('running page rank', force_verbose=True)
                self.print('', force_verbose=True)
                self.print('using {} iterations'.format(num_page_rank_iterations))
                next_urls_list = self.run_page_rank(batch_size, num_of_iterations=num_page_rank_iterations)
                # self.run_page_rank(num_of_iterations=num_page_rank_iterations)
                # next_urls_set = set()
                # if randomize_page_rank_polling:
                #     while len(next_urls_set) <= min(self.page_rank.shape[0], batch_size):
                #         next_urls = random.choices(self.page_rank['url'], weights = self.page_rank['page_rank'], k = batch_size)
                #         next_urls_set.update(set(next_urls))
                #     next_urls_list = random.sample(list(next_urls_set), k = batch_size)
                # else:
                #     next_urls_list = self.page_rank['url'].tolist()[:batch_size]
            else:
                self.print('', force_verbose=True)
                self.print('running random urls', force_verbose=True)
                self.print('', force_verbose=True)
                next_urls_list = list(set(self.get_n_random_urls(batch_size)))

            self.print('running scrape', force_verbose=True)
            self.print('', force_verbose=True)
            for c, next_url in enumerate(next_urls_list):
                next_link = self.generate_link_dict(next_url)
                self.domain_time_delay_record_keeper.setdefault(next_link['netloc'], 0)
                self.scrape_url(next_link)

            self.save_data()
            self.refresh_pool_and_queue()


if __name__ == '__main__':
    c = Crawler()
    while True:
        print('here')

        # initial_sites = get_initial_website_list()
        # initial_sites2 = random.sample(initial_sites, k = 1000000)
        # c.scrape_list(initial_sites2)
        c.crawl(num_of_batches=1, batch_size=10000000, prob_of_random_url_choice=1.0)
        c.crawl(num_of_batches=1, batch_size=100000, prob_of_random_url_choice=0.0)



