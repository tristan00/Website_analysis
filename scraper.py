import requests
import random



cc_path = r'C:\Users\trist\Documents\cc_project'
spammer_path = r'C:\Users\trist\Documents\cc_project\spammer_list'
index = 'CC-MAIN-2019-22'
cc_search_url = "http://index.commoncrawl.org/{0}?url={1}&matchType=domain&output=json"
index_url = ' https://commoncrawl.s3.amazonaws.com/{0}'


class Crawler():

    def load_labeled_websites(self, max_num = 100000, balanced = False):
        with open('{0}/{1}'.format(cc_path, '{0}-index-locs/cc-index.paths'.format(index))) as f:
            data = f.readlines()
            data = [i.strip() for i in data if '.gz' in i]
            print(data)

    def get_scammer_domains(self):
        pass


if __name__ == '__main__':
    c = Crawler()
    c.load_labeled_websites()