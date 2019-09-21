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
import tqdm
import ast
import socket
from common import (dir_loc, db_name,
                    is_link_external,
                    get_initial_website_list,
                    sep_char,
                    max_websites_per_file)
from website_text_extraction import (get_meta_info_from_html,
                                     get_text_from_html)
import multiprocessing
import hashlib
import datetime

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
    return record


def process_html(r_text, r_time, url, timestamp, file_name):
    if r_text:
        record = generate_link_dict(url)
        soup = BeautifulSoup(r_text, 'lxml')
        new_links = [i['href'] for i in soup.find_all('a', href=True)]
        if new_links:
            new_abs_links = [i for i in new_links if is_link_external(i, record['netloc'])]
            record['page_external_links'] = str(new_abs_links)
            record['request_time'] = r_time
            record['request_timestamp'] = timestamp

            meta_data = get_meta_info_from_html(r_text)
            page_text = get_text_from_html(r_text)

            with open(f'{dir_loc}/all_html_chunks/{file_name}.txt', 'a') as f:
                f.write(f'{url}{sep_char}{str(r_text).replace(sep_char, "")}' + "\n")
            with open(f'{dir_loc}/all_meta_chunks/{file_name}.txt', 'a') as f:
                f.write(f'{url}{sep_char}{str(meta_data).replace(sep_char, "")}' + "\n")
            with open(f'{dir_loc}/all_text_chunks/{file_name}.txt', 'a') as f:
                f.write(f'{url}{sep_char}{str(page_text).replace(sep_char, "")}' + "\n")

            record['file_name'] = str(file_name)
            record_df = pd.DataFrame.from_dict([record])
            record_df = record_df.set_index('url')

            with sqlite3.connect(f'{dir_loc}/{db_name}') as conn_disk:
                record_df.to_sql('websites', conn_disk, if_exists='append', index=True)


def scrape_url(url, file_name):
    try:
        s = requests.Session()
        s.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36', }
        start_time = time.time()

        r = s.get(url, timeout=(5, 5), allow_redirects=True)
        t2 = time.time()
        if r.status_code == 200:
            process_html(r.text, t2 - start_time, url, start_time, file_name)

    except (socket.timeout,
            requests.exceptions.ReadTimeout,
            TimeoutError,
            requests.exceptions.ConnectionError,
            requests.exceptions.InvalidSchema,
            requests.exceptions.MissingSchema,
            requests.exceptions.ReadTimeout,
            ValueError,
            requests.exceptions.TooManyRedirects
            ):
        pass
    except Exception:
        pass
        print(traceback.format_exc())


def process_urls(q):
    process = multiprocessing.current_process()
    pid = process.pid
    file_name = ''

    counter = 0
    while True:

        if counter % max_websites_per_file == 0:
            hashGen = hashlib.sha512()
            hashGen.update(f"{pid}{time.time()}".encode('utf-8'))
            file_name = hashGen.hexdigest()

            with open(f'{dir_loc}/all_html_chunks/{file_name}.txt', 'w') as f:
                f.write(f'url{sep_char}data' + "\n")
            with open(f'{dir_loc}/all_text_chunks/{file_name}.txt', 'w') as f:
                f.write(f'url{sep_char}data' + "\n")
            with open(f'{dir_loc}/all_meta_chunks/{file_name}.txt', 'w') as f:
                f.write(f'url{sep_char}data' + "\n")

        url = q.get()

        if url:
            scrape_url(url, file_name)
        else:
            break

        counter += 1


class Crawler():

    def __init__(self,
                 domain_time_delay=3600,
                 url_time_delay=2419200,
                 max_website_len=500000,
                 verbose=False,
                 num_of_processes=4,
                 max_average_time_per_website=1.0):
        self.website_queue_counter = 0
        self.max_average_time_per_website = max_average_time_per_website
        self.num_of_processes = num_of_processes
        self.max_website_len = max_website_len
        self.verbose = verbose
        self.start_time = time.time()
        self.crawler_id = str(uuid.uuid4())
        self.domain_time_delay_record_keeper = dict()
        self.url_time_delay_record_keeper = dict()
        self.request_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36', }

        self.domains = dict()
        self.visited_links = set()
        self.domain_time_delay = domain_time_delay  # needed to not tax any domain too much
        self.url_time_delay = url_time_delay
        self.data_folder_loc = dir_loc
        self.set_up_dirs()
        self.load_past_data()
        self.get_new_session()
        self.make_queue_and_pool()

    ##############################################################################################################
    #High level functions:

    def crawl(self, batch_size=1000, page_rank=False, num_of_batches=1, num_page_rank_iterations=5):
        for batch in range(num_of_batches):
            self.load_past_data()

            if page_rank:
                self.print('', force_verbose=True)
                self.print('running page rank', force_verbose=True)
                self.print('', force_verbose=True)
                self.print('using {} iterations'.format(num_page_rank_iterations))
                next_urls_list = self.run_page_rank(batch_size, num_of_iterations=num_page_rank_iterations)

            else:
                self.print('', force_verbose=True)
                self.print('running random urls', force_verbose=True)
                self.print('', force_verbose=True)
                next_urls_list = list(set(self.get_n_random_urls(batch_size)))

            self.print('running scrape', force_verbose=True)
            self.print('', force_verbose=True)
            for c, next_url in enumerate(next_urls_list):
                next_link = generate_link_dict(next_url)
                self.domain_time_delay_record_keeper.setdefault(next_link['netloc'], 0)
                self.scrape_url(next_link)

            self.refresh_pool_and_queue()

    def scrape_url(self, next_link):
        scrape_successful = False
        self.domain_time_delay_record_keeper.setdefault(next_link['netloc'], 0)
        self.url_time_delay_record_keeper.setdefault(next_link['url'], 0)

        if next_link['url'] and \
                time.time() - self.domain_time_delay_record_keeper[next_link['netloc']] > self.domain_time_delay and \
                time.time() - self.url_time_delay_record_keeper[next_link['url']] > self.url_time_delay:
            try:
                self.website_queue_counter += 1
                self.domain_time_delay_record_keeper[next_link['netloc']] = time.time()
                self.url_time_delay_record_keeper[next_link['url']] = time.time()
                self.q.put(next_link['url'])
                scrape_successful = True

            except Exception:
                self.print(traceback.format_exc(), force_verbose=False)
        return scrape_successful

    def scrape_list(self, website_list):
        for c, i in enumerate(website_list):
            rec = generate_link_dict(i)
            self.scrape_url(rec)
        self.refresh_pool_and_queue()

    ##############################################################################################################
    #Url sampling functions:
    def get_n_random_urls(self, n):
        url_list = []
        self.visited_links = set()
        query = '''Select url, page_external_links from websites'''
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

    def run_page_rank(self, n, num_of_iterations=1):
        # TODO:  automate stopping condition
        ranking_dict1 = dict()
        ranking_dict2 = dict()

        query = '''Select url, page_external_links from websites'''
        with sqlite3.connect('{dir}/{db_name}'.format(dir=self.data_folder_loc, db_name=db_name)) as conn_disk:
            cursor = conn_disk.cursor()
            res = cursor.execute(query)
            count = 0
            for res_temp in tqdm.tqdm(res):
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

                    ranking_dict2[j]['iteration{}'.format(iteration + 1)] += (
                                ranking_dict2[i]['iteration{}'.format(iteration)] / num_of_links)

        self.page_rank = pd.DataFrame.from_dict(list(ranking_dict2.values()))

        total = self.page_rank['iteration{}'.format(num_of_iterations)].sum()
        self.page_rank['page_rank'] = self.page_rank['iteration{}'.format(num_of_iterations)] / total
        self.page_rank = self.page_rank.sort_values('page_rank', ascending=False)
        self.page_rank = self.page_rank[['url', 'page_rank']]
        print(self.page_rank['url'][:20].tolist())
        print(self.page_rank.shape, len(ranking_dict1))
        return self.page_rank['url'][:n].tolist()

    ##############################################################################################################
    #Utility functions:
    def set_up_dirs(self):
        if not os.path.exists(f'{dir_loc}'):
            os.makedirs(f'{dir_loc}')
        if not os.path.exists(f'{self.data_folder_loc}'):
            os.makedirs(f'{self.data_folder_loc}')
        if not os.path.exists(f'{dir_loc}/all_html_chunks'):
            os.makedirs(f'{dir_loc}/all_html_chunks')
        if not os.path.exists(f'{dir_loc}/all_text_chunks'):
            os.makedirs(f'{dir_loc}/all_text_chunks')
        if not os.path.exists(f'{dir_loc}/all_meta_chunks'):
            os.makedirs(f'{dir_loc}/all_meta_chunks')

    def make_queue_and_pool(self):
        self.q = multiprocessing.Queue()
        self.pool = [multiprocessing.Process(target=process_urls, args=(self.q,)) for _ in range(self.num_of_processes)]
        [p.start() for p in self.pool]
        self.website_queue_counter = 0

    def refresh_pool_and_queue(self):
        [self.q.put(None) for _ in range(self.num_of_processes)]
        # [p.join() for p in self.pool]
        timeout = (self.max_average_time_per_website * self.website_queue_counter) / len(self.pool)
        start = time.time()
        formatted_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        print(f'Pool of {len(self.pool)} processes has {timeout} seconds to scrape {self.website_queue_counter} urls. Staring at {formatted_timestamp}.')

        while time.time() - start <= timeout:
            if any(p.is_alive() for p in self.pool):
                time.sleep(.1)
            else:
                break
        else:
            print("Timed out, killing all processes")
            for p in self.pool:
                p.terminate()
                p.join()

        del self.pool, self.q
        self.make_queue_and_pool()

    def print(self, s, force_verbose=False):
        if self.verbose or force_verbose:
            print(s)


    def load_past_data(self):
        try:
            with sqlite3.connect(f'{dir_loc}/{db_name}') as conn_disk:
                res = conn_disk.execute('select url, netloc, request_timestamp from websites')

                self.domain_time_delay_record_keeper = dict()
                self.url_time_delay_record_keeper = dict()
                self.visited_links = set()

                for i in res:
                    url, netloc, request_timestamp = i

                    self.domain_time_delay_record_keeper.setdefault(netloc, float(request_timestamp))
                    self.domain_time_delay_record_keeper[netloc] = max(self.domain_time_delay_record_keeper[netloc],
                                                                       float(request_timestamp))

                    self.url_time_delay_record_keeper.setdefault(url, float(request_timestamp))
                    self.url_time_delay_record_keeper[url] = max(self.url_time_delay_record_keeper[url],
                                                                 float(request_timestamp))

                    self.visited_links.add(url)

        except:
            self.domain_time_delay_record_keeper = dict()
            self.url_time_delay_record_keeper = dict()
            self.visited_links = set()
            traceback.print_exc()

    def get_new_session(self):
        self.s = requests.Session()
        self.s.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36', }


if __name__ == '__main__':
    c = Crawler(num_of_processes=16)
    initial_sites = get_initial_website_list()
    initial_sites_sample = random.sample(initial_sites, k=10000)
    c.scrape_list(initial_sites_sample)

    while True:
        c.crawl(num_of_batches=1, batch_size=100000, page_rank=False)
        c.crawl(num_of_batches=1, batch_size=10000, page_rank=True, num_page_rank_iterations=random.randint(2, 10))
