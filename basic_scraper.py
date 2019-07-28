# non-common crawl version

import requests
import time
import os
from urllib import parse
import random
from bs4 import BeautifulSoup
import copy
import pickle
import uuid
import signal
import traceback
import sqlite3
import numpy as np

from common import get_initial_website_list, dir_loc, url_record_file_name, db_name, url_record_backup_file_name, is_link_external, is_absolute




class Timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class Crawler():

    def __init__(self, domain_time_delay = 3600, max_website_len = 500000):
        self.max_website_len = max_website_len
        self.start_time = time.time()
        self.crawler_id = str(uuid.uuid4())
        self.domain_time_delay_record_keeper = dict()
        self.domains = dict()
        self.links = dict()
        self.visited_links = set()
        self.domain_time_delay = domain_time_delay # needed to not tax any domain too much
        self.data_folder_loc = dir_loc
        self.load_past_data()

        os.system('mkdir {0}'.format(self.data_folder_loc))
        self.get_new_session()

        with sqlite3.connect('{dir}/{db_name}'.format(dir = self.data_folder_loc, db_name=db_name)) as conn_disk:
            conn_disk.execute('''
            CREATE TABLE IF NOT EXISTS websites (
             url TEXT PRIMARY KEY,
             page_text TEXT
            );
            ''')

    def load_past_data(self):
        try:
            with open('{dir}/{file}'.format(dir=self.data_folder_loc, file=url_record_backup_file_name), 'rb') as f:
                self.links = pickle.load(f)
            with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='domain_time_delay_record_keeper'), 'rb') as f:
                self.domain_time_delay_record_keeper = pickle.load(f)
            with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='visited_links'), 'rb') as f:
                self.visited_links = pickle.load(f)
            for i in self.visited_links:
                self.links[i]['scraped'] = 1
        except:
            traceback.print_exc()
            self.links = dict()

    def save_data(self):
        with open('{dir}/{file}'.format(dir=self.data_folder_loc, file=url_record_file_name), 'wb') as f:
            pickle.dump(self.links, f)
        with open('{dir}/{file}'.format(dir=self.data_folder_loc, file=url_record_backup_file_name), 'wb') as f:
            pickle.dump(self.links, f)

        with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='domain_time_delay_record_keeper'), 'wb') as f:
            pickle.dump(self.domain_time_delay_record_keeper, f)
        with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='visited_links'), 'wb') as f:
            pickle.dump(self.visited_links, f)


    def get_new_session(self):
        self.s = requests.Session()
        self.s.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',}


    def generate_link_dict(self, url):
        if url not in self.links:
            parsed_url = parse.urlparse(url)
            record = dict()
            record['url'] = url
            record['scheme'] = parsed_url.scheme
            record['netloc'] = parsed_url.netloc
            record['path'] = parsed_url.path
            record['params'] = parsed_url.params
            record['query'] = parsed_url.query
            record['fragment'] = parsed_url.fragment
            record['page_links'] = []
            record['page_links'] = []
            record['page_links'] = []
            record['urls_linking_to_page'] = []
            record['scraped'] = 0
            return record

        return self.links[url]


    def add_list_of_links_to_input(self, web_link_list):
        for i in web_link_list:
            self.links[i] = self.generate_link_dict(i)


    def scrape_url(self, next_link):
        scrape_successful = False

        t1 = time.time()
        self.domain_time_delay_record_keeper.setdefault(next_link['netloc'], 0)
        if time.time() - self.domain_time_delay_record_keeper[next_link['netloc']] > self.domain_time_delay and next_link['scraped'] == 0:
            t2 = time.time()
            try:
                self.domain_time_delay_record_keeper[next_link['netloc']] = time.time()
                r_time = np.nan
                t3 = time.time()
                print('attempting to scrape: {0}'.format(next_link['url']))

                with Timeout(5):
                    r_start_time = time.time()
                    r = self.s.get(next_link['url'])
                    r_time = time.time() - r_start_time
                print('received  response from: {0} in {1} seconds'.format(next_link['url'], r_time))

                scrape_successful = True
                self.visited_links.add(next_link['url'])
                t4 = time.time()

                soup = BeautifulSoup(r.text)
                new_links = [i['href'] for i in soup.find_all('a', href=True)]
                new_abs_links = [i for i in new_links if is_link_external(i, next_link['netloc'])]
                new_rel_links1 = [i for i in new_links if not is_link_external(i, next_link['netloc']) and is_absolute(i)]
                new_rel_links2 = [i for i in new_links if not is_link_external(i, next_link['netloc']) and not is_absolute(i)]

                new_rel_links_joined = [parse.urljoin(next_link['url'], i) for i in new_rel_links2]

                combined_links = new_abs_links + new_rel_links_joined + new_rel_links1
                self.add_list_of_links_to_input(combined_links)

                for i in combined_links:
                    self.links[i]['urls_linking_to_page'].append(next_link['url'])
                t6 = time.time()

                self.links[next_link['url']]['page_external_links'] = new_abs_links
                self.links[next_link['url']]['page_internal_links'] = new_rel_links_joined
                self.links[next_link['url']]['page_links'] = combined_links
                self.links[next_link['url']]['request_time'] = r_time

                try:
                    with sqlite3.connect('{dir}/{db_name}'.format(dir = self.data_folder_loc, db_name=db_name)) as conn_disk:
                        conn_disk.execute(''' INSERT INTO websites (url, page_text)
                        VALUES(?,?) ''', (next_link['url'], r.text[:self.max_website_len]))
                except sqlite3.IntegrityError:
                    pass
                t7 = time.time()


                print('''scraped link: {link}, total links scraped: {total_links_scraped}, total time: {total_time}, time per link scraped: {time_per_link_scraped}, links to scrape: {links_to_scrape}'''.format(link=next_link['url'],
                                                                                                                                                                                                                    total_links_scraped = len(self.visited_links),
                                                                                                                                                                                                                    total_time = time.time() - self.start_time,
                                                                                                                                                                                    time_per_link_scraped = (time.time() - self.start_time)/len(self.visited_links),
                                                                                                                                                                                                                    links_to_scrape = len(self.links)))
            except requests.exceptions.MissingSchema:
                print('invalid url: {0}'.format(next_link['url']))
            except requests.exceptions.InvalidSchema:
                print('invalid url: {0}'.format(next_link['url']))
            except requests.exceptions.ConnectionError:
                print('ConnectionError: {0}'.format(next_link['url']))
            except TimeoutError:
                print('TimeoutError: {0}'.format(next_link['url']))
            except Exception:
                traceback.print_exc()
        self.links[next_link['url']]['scraped'] = 1 if scrape_successful else 0


        return scrape_successful

    def scrape_list(self, website_list):
        for i in website_list:
            self.links[i] = self.generate_link_dict(i)
            self.scrape_url(self.links[i])

        self.save_data()

    def crawl(self, maximum_sites = 1000000, save_frequency = 100):
        while len(self.visited_links) < maximum_sites:
            sorting_start_time = time.time()
            all_links = list(self.links.values())
            all_links1 = [i for i in all_links if i['scraped'] == 1]
            all_links = [i for i in all_links if i['scraped'] == 0]
            random.shuffle(all_links)
            unscraped_links = sorted(all_links, key = lambda x: len(x['urls_linking_to_page']), reverse= True)
            sorting_end_time = time.time()

            print('sorting took {} seconds'.format(sorting_end_time - sorting_start_time))

            for next_link in unscraped_links:
                self.domain_time_delay_record_keeper.setdefault(next_link['netloc'], 0)
                scrape_successful = self.scrape_url(next_link)

                if not scrape_successful:
                    self.get_new_session()

                if scrape_successful and len(self.visited_links) % save_frequency == 0:
                    self.save_data()


        self.save_data()


if __name__ == '__main__':
    c = Crawler()
    initial_sites = get_initial_website_list()
    c.scrape_list(initial_sites)
    c.crawl()

