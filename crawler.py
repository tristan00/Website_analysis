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
                    max_websites_per_file,
                    run_page_rank,
                    get_meta_info_from_html,
                    get_text_from_html,
                    tokenize)
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
        new_abs_links = [i for i in new_links if is_link_external(i, record['netloc'])]
        record['page_external_links'] = str(new_abs_links)
        record['request_time'] = r_time
        record['request_timestamp'] = timestamp

        meta_data = get_meta_info_from_html(r_text)
        page_text = get_text_from_html(r_text)

        record['html_char_len'] = len(r_text)
        record['text_char_len'] = len(page_text)
        record['meta_char_len'] = len(meta_data)
        record['html_word_len'] = len(tokenize(r_text))
        record['text_word_len'] = len(tokenize(page_text))
        record['meta_word_len'] = len(tokenize(meta_data))

        with open(f'{dir_loc}/all_html_chunks/{file_name}.txt', 'a') as f:
            f.write(f'{url}{sep_char}{str(r_text).replace(sep_char, "")}' + "\n")
        with open(f'{dir_loc}/all_meta_chunks/{file_name}.txt', 'a') as f:
            f.write(f'{url}{sep_char}{str(meta_data).replace(sep_char, "")}' + "\n")
        with open(f'{dir_loc}/all_text_chunks/{file_name}.txt', 'a') as f:
            f.write(f'{url}{sep_char}{str(page_text).replace(sep_char, "")}' + "\n")

        record['file_name'] = str(file_name)
        record_df = pd.DataFrame.from_dict([record])
        record_df = record_df.set_index('url')

        while True:
            try:
                with sqlite3.connect(f'{dir_loc}/dbs/{db_name}') as conn_disk:
                    record_df.to_sql('websites', conn_disk, if_exists='append', index=True)
                break
            except sqlite3.OperationalError:
                time.sleep(5)
                print('db locked')


def scrape_url(url, file_name):
    scraped_successfully = False
    start_time = time.time()

    try:
        s = requests.Session()
        s.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36', }

        r = s.get(url, timeout=(5, 5), allow_redirects=False)
        t2 = time.time()
        if r.status_code == 200:
            process_html(r.text, t2 - start_time, url, start_time, file_name)
            scraped_successfully = True

    except (socket.timeout,
            requests.RequestException,
            TimeoutError,
            ValueError,
            TypeError,
            ):
        pass
    except Exception:
        pass
        print(traceback.format_exc())
    if not scraped_successfully:
        t2 = time.time()
        process_html('<html></html>', t2 - start_time, url, start_time, file_name)


def process_urls(q):
    process = multiprocessing.current_process()
    pid = process.pid

    hashGen = hashlib.sha512()
    hashGen.update(f"{pid}{time.time()}".encode('utf-8'))
    file_name = hashGen.hexdigest()
    with open(f'{dir_loc}/all_html_chunks/{file_name}.txt', 'w') as f:
        f.write(f'url{sep_char}data' + "\n")
    with open(f'{dir_loc}/all_text_chunks/{file_name}.txt', 'w') as f:
        f.write(f'url{sep_char}data' + "\n")
    with open(f'{dir_loc}/all_meta_chunks/{file_name}.txt', 'w') as f:
        f.write(f'url{sep_char}data' + "\n")
    while True:
        url = q.get()

        if url:
            scrape_url(url, file_name)
        else:
            break


class Crawler():

    def __init__(self,
                 domain_time_delay=3600,
                 url_time_delay=2419200,
                 max_website_len=500000,
                 verbose=False,
                 num_of_processes=4,
                 max_average_time_per_website=4.0,
                 pool_time_limit_base = 10.0):
        self.pool_time_limit_base = pool_time_limit_base
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
        self.set_up_dirs()
        self.load_past_data()
        self.get_new_session()
        self.make_queue_and_pool()

    ##############################################################################################################
    # High level functions:

    def crawl(self, batch_size=1000, page_rank=False, num_of_batches=1, num_page_rank_iterations=5):
        for batch in range(num_of_batches):
            self.load_past_data()

            if page_rank:
                print('')
                print('running page rank')
                print('')
                print('using {} iterations'.format(num_page_rank_iterations))
                next_urls_list = run_page_rank(batch_size, num_of_iterations=num_page_rank_iterations)

            else:
                print('')
                print('running random urls')
                print('')
                next_urls_list = list(set(self.get_n_random_urls(batch_size)))

            print('running scrape')
            print('')
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
                print(traceback.format_exc())
        return scrape_successful

    def scrape_list(self, website_list):
        for c, i in enumerate(website_list):
            rec = generate_link_dict(i)
            self.scrape_url(rec)
        self.refresh_pool_and_queue()

    ##############################################################################################################
    # Url sampling functions:
    def get_n_random_urls(self, n):
        url_list = []
        self.visited_links = set()
        query = '''Select url, page_external_links from websites'''
        with sqlite3.connect('{dir}/dbs/{db_name}'.format(dir=dir_loc, db_name=db_name)) as conn_disk:
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


    ##############################################################################################################
    # Utility functions:
    def set_up_dirs(self):
        if not os.path.exists(f'{dir_loc}'):
            os.makedirs(f'{dir_loc}')
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
        timeout = self.pool_time_limit_base + ((self.max_average_time_per_website * self.website_queue_counter) / len(self.pool))
        start = time.time()
        formatted_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        print(f'''Pool of {len(
            self.pool)} processes has {timeout} seconds to scrape {self.website_queue_counter} urls. Starting at {formatted_timestamp}.''')

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

    def load_past_data(self):
        try:
            with sqlite3.connect(f'{dir_loc}/dbs/{db_name}') as conn_disk:
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


def wipe_db():
    with sqlite3.connect(f'{dir_loc}/dbs/{db_name}') as conn_disk:
        conn_disk.execute('delete from websites')


if __name__ == '__main__':
    # wipe_db()

    crawler = Crawler(num_of_processes=16)
    initial_sites = list(set(get_initial_website_list()))
    random.shuffle(initial_sites)

    num_of_chunks = 4000
    chunks = [set() for _ in range(num_of_chunks)]

    for counter, i in enumerate(initial_sites):
        chunks[counter%num_of_chunks].add(i)

    start_time = time.time()
    for counter, i in enumerate(chunks):
        crawler.scrape_list(i)
        print(f'iteration: {counter}, time: {time.time() - start_time}, average time per chunk: {(time.time() - start_time)/(counter + 1)}')

    while True:
        crawler.crawl(num_of_batches=1, batch_size=10000, page_rank=True, num_page_rank_iterations=3)
        crawler.crawl(num_of_batches=1, batch_size=10000, page_rank=False)
