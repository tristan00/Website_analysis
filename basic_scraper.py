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

initial_sites = [
                'https://old.reddit.com/r/findareddit/wiki/directory',
                'https://en.wikipedia.org/wiki/Main_Page',
                'https://moz.com/top500',
                'https://www.alexa.com/topsites',
                'https://www.similarweb.com/top-websites/united-states',
                'https://en.wikipedia.org/wiki/List_of_most_popular_websites',
                'https://www.rankranger.com/top-websites',
                'https://www.lifewire.com/most-popular-sites-3483140',
                'https://www.lemonde.fr/',
                'https://www.bbc.com/',
                'https://www.foxnews.com/',
                'https://www.nytimes.com/',
                'https://www.cnn.com/',
                'https://www.huffpost.com/',
                'https://news.yahoo.com/',
                'https://www.ycombinator.com/',
                'https://timesofindia.indiatimes.com/',
                'http://www.chinadaily.com.cn/',
                'http://www.espn.com/espn/latestnews',
                'https://www.cbssports.com/',
                'https://www.independent.co.uk/us',
                'https://www.apnews.com/',
                'https://www.wsj.com/',
                'https://www.news.com.au/',
                'https://www.zerohedge.com/',
                'https://www.africanews.com/',
                'https://tass.com/',
                'https://www.aljazeera.com/topics/country/russia.html']


def is_absolute(url):
    return bool(parse.urlparse(url).netloc)


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

        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.data_folder_loc = os.path.join(dir_path, 'web_data')
        # self.load_past_data()

        # os.system('rm -rf {0}'.format(self.data_folder_loc))
        os.system('mkdir {0}'.format(self.data_folder_loc))
        self.get_new_session()

        with sqlite3.connect('{dir}/website.db'.format(dir = self.data_folder_loc)) as conn_disk:
            conn_disk.execute('''
            CREATE TABLE IF NOT EXISTS websites (
             url TEXT PRIMARY KEY,
             page_text TEXT
            );
            ''')

    def load_past_data(self):
        try:
            with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='links'), 'rb') as f:
                self.links = pickle.load(f)
            with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='domain_time_delay_record_keeper'), 'rb') as f:
                self.domain_time_delay_record_keeper = pickle.load(f)
            with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='visited_links'), 'rb') as f:
                self.visited_links = pickle.load(f)
        except:
            traceback.print_exc()
            self.links = dict()

    def save_data(self):
        with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='links'), 'wb') as f:
            pickle.dump(self.links, f)
        with open('{dir}/{file}.pkl'.format(dir=self.data_folder_loc, file='links_backup'), 'wb') as f:
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
            record['urls_linking_to_page'] = []
            record['scraped'] = 0
            return record
        return self.links[url]


    def add_list_of_links_to_input(self, web_link_list):
        for i in web_link_list:
            self.links[i] = self.generate_link_dict(i)


    def scrape_url(self, next_link):
        scrape_successful = False
        self.domain_time_delay_record_keeper.setdefault(next_link['netloc'], 0)
        if time.time() - self.domain_time_delay_record_keeper[next_link['netloc']] > self.domain_time_delay and next_link['scraped'] == 0:
            try:
                self.domain_time_delay_record_keeper[next_link['netloc']] = time.time()
                r_time = np.nan
                with Timeout(5):
                    r_start_time = time.time()
                    r = self.s.get(next_link['url'])
                    r_time = time.time() - r_start_time
                scrape_successful = True
                self.visited_links.add(next_link['url'])

                soup = BeautifulSoup(r.text)
                new_links = [i['href'] for i in soup.find_all('a', href=True)]
                new_abs_links = [i for i in new_links if is_absolute(i)]
                new_rel_links = [i for i in new_links if not is_absolute(i)]
                new_rel_links_joined = [parse.urljoin(next_link['url'], i) for i in new_rel_links]
                new_rel_links_joined = [i for i in new_rel_links_joined if is_absolute(i)]

                combined_links = new_abs_links + new_rel_links_joined
                self.add_list_of_links_to_input(combined_links)

                for i in combined_links:
                    self.links[i]['urls_linking_to_page'].append(next_link['url'])

                self.links[next_link['url']]['page_links'] = combined_links
                self.links[next_link['url']]['request_time'] = r_time

                try:
                    with sqlite3.connect('{dir}/website.db'.format(dir = self.data_folder_loc)) as conn_disk:
                        conn_disk.execute(''' INSERT INTO websites (url, page_text)
                        VALUES(?,?) ''', (next_link['url'], r.text[:self.max_website_len]))
                except sqlite3.IntegrityError:
                    pass


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

    def crawl(self, maximum_sites = 1000000, save_frequency = 10):
        while len(self.visited_links) < maximum_sites:
            next_key = random.choice(list(self.links.keys()))
            next_link = self.links[next_key]

            current_time = time.time()
            self.domain_time_delay_record_keeper.setdefault(next_link['netloc'], 0)
            scrape_successful = False

            if current_time - self.domain_time_delay_record_keeper[next_link['netloc']] > self.domain_time_delay and next_link['url'] not in self.visited_links:
                scrape_successful = self.scrape_url(next_link)

            if not scrape_successful:
                self.links[next_link['url']] = next_link
                self.get_new_session()

            if scrape_successful and len(self.visited_links) % save_frequency == 0:
                self.save_data()


if __name__ == '__main__':
    c = Crawler()
    c.scrape_list(initial_sites)
    c.crawl()

