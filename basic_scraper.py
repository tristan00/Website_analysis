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
from common import get_initial_website_list, dir_loc, url_record_file_name, db_name, url_record_backup_file_name, is_link_external, is_absolute, extract_meta_information, extract_page_text

web_timeout = (5, 5)


class Crawler():

    def __init__(self, domain_time_delay = 3600, url_time_delay = 2419200, max_website_len = 500000, verbose = False):
        self.max_website_len = max_website_len
        self.verbose = verbose
        self.start_time = time.time()
        self.crawler_id = str(uuid.uuid4())
        self.domain_time_delay_record_keeper = dict()
        self.url_time_delay_record_keeper = dict()

        self.domains = dict()
        self.visited_links = set()
        self.domain_time_delay = domain_time_delay # needed to not tax any domain too much
        self.url_time_delay =url_time_delay
        self.data_folder_loc = dir_loc
        self.load_past_data()

        os.system('mkdir {0}'.format(self.data_folder_loc))
        self.get_new_session()

    def print(self, s, force_verbose = False):
        if self.verbose or force_verbose:
            print(s)

    def run_page_rank(self):
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


        for i in tqdm.tqdm(ranking_dict1):
            num_of_links = len(ranking_dict1[i])
            for j in ranking_dict1[i]:
                ranking_dict2.setdefault(j, {'url': j,
                                             'iteration1': 1,
                                             'iteration2': 0,
                                             'iteration3': 0,
                                             'page_rank': 0})
                ranking_dict2.setdefault(i, {'url': i,
                                             'iteration1': 1,
                                             'iteration2': 0,
                                             'iteration3': 0,
                                             'page_rank': 0})

                ranking_dict2[j]['iteration2'] += (ranking_dict2[i]['iteration1']/num_of_links)
        for i in tqdm.tqdm(ranking_dict1):
            num_of_links = len(ranking_dict1[i])
            for j in ranking_dict1[i]:
                ranking_dict2[j]['iteration3'] += (ranking_dict2[i]['iteration2']/num_of_links)

        self.page_rank = pd.DataFrame.from_dict(list(ranking_dict2.values()))
        self.page_rank['page_rank'] = self.page_rank['iteration3'] / self.page_rank['iteration3'].sum()
        self.page_rank = self.page_rank.sort_values('page_rank', ascending=False)
        self.page_rank = self.page_rank[['url', 'page_rank']]
        print(self.page_rank['url'][:10].tolist())
        print(self.page_rank.shape, len(ranking_dict1))


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
        print('starting to save data')
        with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='domain_time_delay_record_keeper'), 'wb') as f:
            pickle.dump(self.domain_time_delay_record_keeper, f)
        with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='url_time_delay_record_keeper'), 'wb') as f:
            pickle.dump(self.url_time_delay_record_keeper, f)
        with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='visited_links'), 'wb') as f:
            pickle.dump(self.visited_links, f)
        print('data saved')


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

        t1 = time.time()
        self.domain_time_delay_record_keeper.setdefault(next_link['netloc'], 0)
        self.url_time_delay_record_keeper.setdefault(next_link['url'], 0)
        if time.time() - self.domain_time_delay_record_keeper[next_link['netloc']] > self.domain_time_delay and time.time() - self.url_time_delay_record_keeper[next_link['url']] > self.url_time_delay:
            t2 = time.time()
            try:
                self.domain_time_delay_record_keeper[next_link['netloc']] = time.time()
                self.url_time_delay_record_keeper[next_link['url']] = time.time()
                r_time = np.nan
                t3 = time.time()
                self.print('attempting to scrape: {0}'.format(next_link['url']))

                r_start_time = time.time()
                r = self.s.get(next_link['url'], timeout = web_timeout)
                r_time = time.time() - r_start_time
                self.print('received  response from: {0} in {1} seconds'.format(next_link['url'], r_time))

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
                record = copy.deepcopy(next_link)
                record['page_external_links'] = str(new_abs_links)
                record['page_internal_links'] = str(new_rel_links_joined)
                record['page_links'] = str(combined_links)
                record['request_time'] = r_time
                record['request_timestamp'] = self.url_time_delay_record_keeper[next_link['url']]
                record['html'] = r.text
                record['meta'] = extract_meta_information(soup)
                record['page_text'] = extract_page_text(soup)
                record_df = pd.DataFrame.from_dict([record])
                with sqlite3.connect('{dir}/{db_name}'.format(dir = self.data_folder_loc, db_name=db_name)) as conn_disk:
                    record_df.to_sql('websites', conn_disk, if_exists='append')
                t7 = time.time()


            except requests.exceptions.MissingSchema:
                self.print('invalid url: {0}'.format(next_link['url']))
            except requests.exceptions.InvalidSchema:
                self.print('invalid url: {0}'.format(next_link['url']))
            except requests.exceptions.ConnectionError:
                self.print('ConnectionError: {0}'.format(next_link['url']))
            except TimeoutError:
                self.print('TimeoutError: {0}'.format(next_link['url']))
            except Exception:
                self.print(traceback.format_exc())
        return scrape_successful

    def scrape_list(self, website_list):
        for i in website_list:
            rec = self.generate_link_dict(i)
            self.scrape_url(rec)

        self.save_data()

    def crawl(self, maximum_sites = 1000000, sample_size = 1000):
        while len(self.visited_links) < maximum_sites:

            print()
            print('running page rank')
            print()
            self.run_page_rank()
            next_urls = random.choices(self.page_rank['url'], weights = self.page_rank['page_rank'], k = sample_size)
            next_urls = set(next_urls)

            print('running scrape')
            print()
            for next_url in tqdm.tqdm(next_urls):
                next_link = self.generate_link_dict(next_url)
                self.domain_time_delay_record_keeper.setdefault(next_link['netloc'], 0)
                scrape_successful = self.scrape_url(next_link)

                if not scrape_successful:
                    self.get_new_session()

            self.save_data()

if __name__ == '__main__':
    c = Crawler()
    initial_sites = get_initial_website_list()
    # c.run_page_rank()
    c.scrape_list(initial_sites)
    c.crawl()

